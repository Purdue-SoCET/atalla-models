"""Compose standalone and captured-training kernels into one AtallaC program.

Each kernel keeps its own configuration and staging buffers. DMA copies connect
live graph values to those buffers, including broadcasting and GEMM packing.
This deliberately favors correctness over buffer reuse or transfer efficiency.
No reference forward execution or reference intermediates are used.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.fx import GraphModule, Node

from codegen.c_emitter import ADDR_TABLE, DRAMWriter, LayerEmission, emit_node, _ek
from graph.fx_capture import get_node_shape
from graph.tile_planner import TileConfig
from codegen.schedule_ops import (ADAPTERS, op_for, check_extra, prepare_extra, matmul_dimensions,
                                 gemm_emission, binary_emission, softmax_emission)


@dataclass
class Buffer:
    address: int
    shape: tuple
    dtype: torch.dtype = torch.bfloat16


@dataclass
class KernelSchedule:
    source: str
    dram: DRAMWriter
    manifest: dict

    def emission(self) -> LayerEmission:
        """Use the same compilation/assembly path as validation."""
        emission = LayerEmission()
        emission.c_source = self.source
        emission.dram = self.dram
        return emission

    def write(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "graph_schedule.c").write_text(self.source)
        words = self.dram.to_u32_words(include_zeros=True, stride=4)
        image = bytearray(max(words, default=0) + 4)
        for address, word in words.items():
            image[address:address + 4] = word.to_bytes(4, "little")
        (path / "dram.bin").write_bytes(image)
        (path / "schedule.json").write_text(json.dumps(self.manifest, indent=2) + "\n")


def _callable_source(source: str, name: str, config: int) -> str:
    """Adapt the entry point/configuration, leaving kernel arithmetic unchanged."""
    functions = re.findall(r"^\s*(?:int|void)\s+(\w+)\s*\(", source, re.MULTILINE)
    if functions.count("main") != 1:
        raise ValueError(f"{name}: expected one standalone kernel main()")
    # Namespace helper functions too (e.g. GEMM's load_weight_tile).
    for function in functions:
        replacement = name if function == "main" else f"{name}_{function}"
        source = re.sub(r"\b" + re.escape(function) + r"\b", replacement, source)
    source, replacements = re.subn(
        rf"\bint (cfg|cfg_tbl) = {ADDR_TABLE};",
        lambda match: f"int {match[1]} = {config};", source,
    )
    source, macros = re.subn(r"(#define\s+ADDR_TABLE\s+)60\b", rf"\g<1>{config}", source)
    for macro in re.findall(r"^#define\s+(\w+)", source, re.MULTILINE):
        source = re.sub(r"\b" + re.escape(macro) + r"\b", f"{name}_{macro}", source)
    if not (replacements or macros) or not any(
            halt in source for halt in ('asm("halt");', 'atalla_halt();')):
        raise ValueError(f"{name}: unrecognized kernel configuration/termination")
    return source.replace('asm("halt");', "/* Return to the graph schedule. */").replace('atalla_halt();', '')


# Copy a BF16 element without converting it through scalar arithmetic. This also
# works at odd BF16 indices, where a scalar 32-bit load/store would clobber data.
_COPY_SOURCE = """void schedule_copy_bf16(int src, int dst) {
    int ctl = 0;
    scpad_load(0, src, ctl);
    scpad_store(0, dst, ctl);
}
void schedule_bf16_to_f32(int src, int dst) {
    int aligned = src & -4;
    int word = *(int*)aligned;
    int bits = (word >> ((src & 2) * 8)) & 65535;
    *(int*)dst = bits << 16;
}
void schedule_f32_to_bf16(int src, int dst) {
    int bits = *(int*)src;
    int rounded = bits + 32767 + ((bits >> 16) & 1);
    int half = (rounded >> 16) & 65535;
    int aligned = dst & -4;
    int word = *(int*)aligned;
    int shift = (dst & 2) * 8;
    int mask = 65535 << shift;
    *(int*)aligned = (word & ~mask) | (half << shift);
}
void schedule_copy_f32(int src, int dst) {
    *(int*)dst = *(int*)src;
}
"""


_MAX_SOURCE = """void schedule_max_bf16(int lhs, int rhs, int dst) {
    int ctl0 = 0;
    int ctl1;
    asm("li_s %0, 1073741824" : "=r"(ctl1));
    scpad_load(0, lhs, ctl0);
    scpad_load(0, rhs, ctl1);
    vec a = vector_load(0, 0, 0, 0);
    vec b = vector_load(0, 0, 0, 1);
    int greater = make_mask(">", b, a, 1);
    /* Only lane zero contains a scalar candidate; ignore other mask bits. */
    if ((greater & 1) != 0) {
        scpad_store(0, dst, ctl1);
    } else {
        scpad_store(0, dst, ctl0);
    }
}
"""


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def _argument(node, gm, name, position, default):
    if node.op == "call_module":
        return getattr(gm.get_submodule(node.target), name, default)
    return node.kwargs.get(name, node.args[position] if len(node.args) > position else default)


def _check_spatial(node, gm, op, input_shape, output_shape):
    """Reject configurations not represented by validation's spatial planners."""
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise NotImplementedError(f"{node.name}: {op} schedule currently requires batch-one NCHW")
    if op == "conv":
        stride = _pair(_argument(node, gm, "stride", 3, 1))
        padding = _pair(_argument(node, gm, "padding", 4, 0))
        dilation = _pair(_argument(node, gm, "dilation", 5, 1))
        groups = _argument(node, gm, "groups", 6, 1)
        if stride[0] != stride[1] or padding[0] != padding[1] or dilation != (1, 1) or groups != 1:
            raise NotImplementedError(f"{node.name}: convolution requires equal spatial strides/padding, "
                                      "dilation=1, and groups=1")
        if node.op != "call_function" or node.args[1].op != "get_attr":
            raise NotImplementedError(f"{node.name}: convolution requires lowered attribute weights")
    else:
        pool = _pair(_argument(node, gm, "kernel_size", 1, 2))
        raw_stride = _argument(node, gm, "stride", 2, None)
        stride = pool if raw_stride is None else _pair(raw_stride)
        padding = _pair(_argument(node, gm, "padding", 3, 0))
        dilation = _pair(_argument(node, gm, "dilation", 4, 1))
        ceil_mode = _argument(node, gm, "ceil_mode", 5, False)
        if (pool[0] != pool[1] or stride[0] != stride[1] or padding != (0, 0)
                or dilation != (1, 1) or ceil_mode or max(input_shape[-2:]) > 32):
            raise NotImplementedError(f"{node.name}: maxpool requires H/W<=32, a square window, "
                                      "equal strides, no padding, dilation=1, and ceil_mode=False")
    p = node.meta["tile_config"].params
    if op == "conv" and any(p[dim] > 32 and p[dim] % 32 for dim in ("M", "N")):
        raise NotImplementedError(f"{node.name}: convolution GEMM needs complete M/N tiles "
                                  "when a dimension exceeds 32")
    expected = ((1, p["N"], p["Ho"], p["Wo"]) if op == "conv" else
                (1, p["channels"], p["H_out"], p["W_out"]))
    if tuple(output_shape) != expected:
        raise NotImplementedError(f"{node.name}: output {output_shape} disagrees with spatial plan {expected}")


def _index(shape: tuple, output_shape: tuple, index: str) -> str:
    """Row-major source index under NumPy/PyTorch broadcasting."""
    if len(shape) > len(output_shape):
        raise ValueError(f"Cannot broadcast {shape} to {output_shape}")
    padded = (1,) * (len(output_shape) - len(shape)) + shape
    terms = []
    for axis, (size, out_size) in enumerate(zip(padded, output_shape)):
        if size not in (1, out_size):
            raise ValueError(f"Cannot broadcast {shape} to {output_shape}")
        if size != 1:
            outer_stride = math.prod(output_shape[axis + 1:])
            inner_stride = math.prod(padded[axis + 1:])
            terms.append(f"(({index} / {outer_stride}) % {out_size}) * {inner_stride}")
    return " + ".join(terms) or "0"


def permutation_index(shape, axes):
    output = tuple(shape[axis] for axis in axes)
    return " + ".join(
        f"((i / {math.prod(output[axis + 1:])}) % {output[axis]})"
        f" * {math.prod(shape[source_axis + 1:])}"
        for axis, source_axis in enumerate(axes)
    ) or "0"


def schedule_issues(gm):
    """Collect every missing operation/configuration before emitting any source."""
    issues = []
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        op = op_for(node)
        shape = get_node_shape(node)
        meta = node.meta.get("tensor_meta")
        if op == "getitem":
            source = node.args[0]
            index = node.args[1]
            source_meta = source.meta.get("tensor_meta") if isinstance(source, Node) else None
            if (not isinstance(source_meta, (tuple, list)) or hasattr(source_meta, "shape")
                    or not isinstance(index, int) or not -len(source_meta) <= index < len(source_meta)):
                issues.append(f"{node.name}: getitem requires a constant index into a tensor tuple")
            continue
        if op in _ek.TRAINING_OPS:
            def check_meta(value):
                if value is None:
                    return
                if hasattr(value, "shape"):
                    allowed = (torch.bfloat16, torch.float32, torch.int64) if op == "max_pool2d_with_indices" else (torch.bfloat16, torch.float32)
                    if value.dtype not in allowed:
                        raise NotImplementedError(f"unsupported dtype {value.dtype}")
                    if any(d <= 0 for d in value.shape):
                        raise NotImplementedError("expected nonempty tensor result")
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        check_meta(child)
                else:
                    raise NotImplementedError("expected tensor metadata")
            try:
                if meta is None:
                    raise ValueError("missing tensor metadata")
                check_meta(meta)
                check_extra(node, gm, op)
            except (NotImplementedError, ValueError, TypeError) as error:
                issues.append(f"{node.name} ({op}): {error}")
            continue
        if shape is None or any(d <= 0 for d in shape):
            issues.append(f"{node.name}: expected one nonempty tensor result (tuple outputs need a kernel adapter)")
            continue
        if meta.dtype not in (torch.bfloat16, torch.float32):
            issues.append(f"{node.name}: unsupported dtype {meta.dtype}")
            continue
        if node.op in ("placeholder", "get_attr"):
            continue
        if op not in ADAPTERS:
            issues.append(f"No schedule adapter for {node.name} ({op}, {node.target})")
            continue
        try:
            check_extra(node, gm, op)
            if op in ("conv", "maxpool"):
                _check_spatial(node, gm, op, get_node_shape(node.args[0]), shape)
            if op == "add" and node.kwargs.get("alpha", 1) != 1:
                raise NotImplementedError("add alpha other than 1 is unsupported")
            if (meta.dtype != torch.bfloat16 and op not in
                    ("cast", "ones_like", "mse_loss", "mse_loss_backward", "flatten", "transpose", "permute", "detach", "contiguous", "expand")):
                raise NotImplementedError("this compute kernel requires BF16 output")
        except (NotImplementedError, ValueError, TypeError) as error:
            issues.append(f"{node.name} ({op}): {error}")
    return issues


def build_schedule(gm: GraphModule, placeholder_data: dict) -> KernelSchedule:
    """Build a program using the registered forward/backward kernel adapters.

    Unsupported computations fail rather than generating a partial program.
    Existing backward kernels and explicit BF16/FP32 conversions are supported;
    missing derivative kernels are reported together by the capability check.
    """
    issues = schedule_issues(gm)
    if issues:
        raise NotImplementedError("Unsupported schedule operations/configurations:\n  - " + "\n  - ".join(issues))
    dram = DRAMWriter()
    next_address = 0x1000  # Keep low memory clear for compiler/runtime use.
    buffers = {}
    cache = {}
    bodies = []
    calls = []
    manifest = {"backend": "validation-kernels", "inputs": {}, "kernels": [], "outputs": None}

    def allocate(size):
        nonlocal next_address
        address = (next_address + 63) // 64 * 64
        next_address = address + max(size, 64)
        dram.write_zeros(address, max(size, 64))
        return address

    def copy_loop(source, destination, count, source_index="i", dest_index="i",
                  source_dtype=torch.bfloat16, dest_dtype=torch.bfloat16):
        conversion = {
            (torch.bfloat16, torch.bfloat16): "schedule_copy_bf16",
            (torch.bfloat16, torch.float32): "schedule_bf16_to_f32",
            (torch.float32, torch.bfloat16): "schedule_f32_to_bf16",
            (torch.float32, torch.float32): "schedule_copy_f32",
        }[source_dtype, dest_dtype]
        src_bytes = 2 if source_dtype == torch.bfloat16 else 4
        dst_bytes = 2 if dest_dtype == torch.bfloat16 else 4
        calls.extend([
            "    { int i = 0;",
            f"      while (i < {count}) {{",
            f"        {conversion}({source} + {src_bytes} * ({source_index}), "
            f"{destination} + {dst_bytes} * ({dest_index}));",
            "        i = i + 1;",
            "      }",
            "    }",
        ])

    def describe(node):
        def descriptor(buffer, path=()):
            if buffer is None:
                return None
            if isinstance(buffer, (tuple, list)):
                return [descriptor(child, (*path, index)) for index, child in enumerate(buffer)]
            return {"node": node.name + "".join(f"[{i}]" for i in path),
                    "source_node": node.name, "result_path": list(path), "address": buffer.address,
                    "shape": list(buffer.shape), "dtype": str(buffer.dtype).removeprefix("torch.")}
        return descriptor(buffers[node])

    def install_part(emission, function, adapter):
        words = emission.dram.to_u32_words(include_zeros=True, stride=4)
        base = allocate(max(max(words, default=0) + 4,
                            emission.output_addr + emission.output_elements * 2))
        for address, word in words.items():
            dram.u32(base + address, word)
        pointers = {}
        for offset in ADAPTERS[adapter].pointers:
            pointers[offset] = base + words[ADDR_TABLE + offset]
            dram.u32(base + ADDR_TABLE + offset, pointers[offset])
        bodies.append(_callable_source(emission.c_source, function, base + ADDR_TABLE))
        return base, pointers, base + emission.output_addr

    def outputs(value):
        if isinstance(value, Node):
            return describe(value)
        if isinstance(value, (tuple, list)):
            return [outputs(v) for v in value]
        if isinstance(value, dict):
            return {k: outputs(v) for k, v in value.items()}
        return value

    for node in gm.graph.nodes:
        if node.op == "output":
            manifest["outputs"] = outputs(node.args[0])
            continue
        op = op_for(node)
        if op == "getitem":
            buffers[node] = buffers[node.args[0]][node.args[1]]
            cache[node.name] = cache[node.args[0].name][node.args[1]]
            continue
        if op in _ek.TRAINING_OPS:
            def allocate_result(meta):
                if meta is None:
                    return None, None
                if not hasattr(meta, "shape"):
                    children = [allocate_result(child) for child in meta]
                    return tuple(c[0] for c in children), tuple(c[1] for c in children)
                shape = tuple(meta.shape)
                element_bytes = {torch.bfloat16: 2, torch.float32: 4, torch.int64: 8}[meta.dtype]
                return (Buffer(allocate(math.prod(shape) * element_bytes), shape, meta.dtype),
                        np.zeros(shape, dtype=np.int64 if meta.dtype == torch.int64 else np.float32))

            buffers[node], cache[node.name] = allocate_result(node.meta.get("tensor_meta"))
            name = f"kernel_{node.name}"
            body = _ek.training_body(node, op, buffers, buffers[node])
            bodies.append(f"void {name}() {{\n{body}\n}}\n")
            calls.append(f"    {name}();")
            manifest["kernels"].append({"node": node.name, "operation": op, "function": name,
                                         "functions": [name], "source": ADAPTERS[op].source,
                                         "config_address": None, "output": describe(node)})
            continue
        shape = get_node_shape(node)
        meta = node.meta.get("tensor_meta")
        if shape is None or any(d <= 0 for d in shape):
            raise NotImplementedError(f"Schedule node {node.name} requires a nonempty tensor shape")
        if meta.dtype not in (torch.bfloat16, torch.float32):
            raise NotImplementedError(f"Schedule node {node.name} has unsupported dtype {meta.dtype}")

        if node.op in ("placeholder", "get_attr"):
            if node.op == "placeholder":
                if node.target not in placeholder_data:
                    raise ValueError(f"Missing placeholder data for {node.target}")
                value = placeholder_data[node.target]
            else:
                value = gm
                for part in node.target.split("."):
                    value = getattr(value, part)
            if tuple(value.shape) != shape or value.dtype != meta.dtype:
                raise ValueError(f"{node.name}: expected {meta.dtype} tensor with shape {shape}")
            data = value.detach().float().cpu().numpy()
            element_bytes = value.element_size()
            address = allocate(data.size * element_bytes)
            for i, element in enumerate(data.ravel()):
                (dram.bf16 if meta.dtype == torch.bfloat16 else dram.f32)(
                    address + element_bytes * i, float(element))
            buffers[node] = Buffer(address, shape, meta.dtype)
            cache[node.name] = data
            if node.op == "placeholder":
                manifest["inputs"][node.target] = describe(node)
            continue

        op = op_for(node)
        if op == "ones_like":
            address = allocate(math.prod(shape) * (2 if meta.dtype == torch.bfloat16 else 4))
            for i in range(math.prod(shape)):
                if meta.dtype == torch.bfloat16:
                    dram.bf16(address + 2 * i, 1.0)
                else:
                    dram.f32(address + 4 * i, 1.0)
            buffers[node] = Buffer(address, shape, meta.dtype)
            cache[node.name] = np.ones(shape, dtype=np.float32)
            continue
        if op == "cast":
            source = buffers[node.args[0]]
            address = allocate(math.prod(shape) * (2 if meta.dtype == torch.bfloat16 else 4))
            copy_loop(source.address, address, math.prod(shape), source_dtype=source.dtype, dest_dtype=meta.dtype)
            buffers[node] = Buffer(address, shape, meta.dtype)
            cache[node.name] = np.zeros(shape, dtype=np.float32)
            continue
        if op == "expand":
            source = buffers[node.args[0]]
            address = allocate(math.prod(shape) * (2 if source.dtype == torch.bfloat16 else 4))
            copy_loop(source.address, address, math.prod(shape), _index(source.shape, shape, "i"),
                      source_dtype=source.dtype, dest_dtype=source.dtype)
            buffers[node] = Buffer(address, shape, source.dtype)
            cache[node.name] = np.zeros(shape, dtype=np.float32)
            continue
        if op in ("flatten", "dropout", "detach", "contiguous"):
            source = buffers[node.args[0]]
            if math.prod(source.shape) != math.prod(shape):
                raise ValueError(f"{node.name}: reshape changes element count")
            buffers[node] = Buffer(source.address, shape, source.dtype)
            cache[node.name] = cache[node.args[0].name].reshape(shape)
            continue
        if op in ("transpose", "permute"):
            source = buffers[node.args[0]]
            if not source.shape:
                buffers[node] = Buffer(source.address, shape, source.dtype)
                cache[node.name] = cache[node.args[0].name]
                continue
            axes = list(range(len(source.shape)))
            if op == "transpose":
                d0, d1 = int(node.args[1]), int(node.args[2])
                axes[d0], axes[d1] = axes[d1], axes[d0]
            else:
                axes = list(node.args[1]) if isinstance(node.args[1], (tuple, list)) else list(node.args[1:])
            address = allocate(math.prod(shape) * (2 if source.dtype == torch.bfloat16 else 4))
            terms = []
            for axis, source_axis in enumerate(axes):
                terms.append(f"((i / {math.prod(shape[axis + 1:])}) % {shape[axis]})"
                             f" * {math.prod(source.shape[source_axis + 1:])}")
            copy_loop(source.address, address, math.prod(shape), " + ".join(terms),
                      source_dtype=source.dtype, dest_dtype=source.dtype)
            buffers[node] = Buffer(address, shape, source.dtype)
            cache[node.name] = cache[node.args[0].name].transpose(axes).copy()
            continue

        if op not in ADAPTERS:
            raise NotImplementedError(f"No schedule adapter for {node.name} ({op}, {node.target})")
        if op == "add" and node.kwargs.get("alpha", 1) != 1:
            raise NotImplementedError(f"{node.name}: add alpha other than 1 is unsupported")
        if op in ("conv", "maxpool"):
            _check_spatial(node, gm, op, buffers[node.args[0]].shape, shape)
        if op == "atalla_sdpa":
            p = node.meta["tile_config"].params
            n, d = p["N"], p["D"]
            name = f"kernel_{node.name}"
            scores_name, scale_name, softmax_name = (f"{name}_{suffix}" for suffix in ("scores", "scale", "softmax"))
            scores = gemm_emission(n, n, d)
            _, pointers, score_output = install_part(scores, scores_name, "matmul")
            stride = math.ceil(d / 32) * 32
            copy_loop(buffers[node.args[0]].address, pointers[0], n * d,
                      dest_index=f"(i / {d}) * {stride} + i % {d}")
            copy_loop(buffers[node.args[1]].address, pointers[4], n * d,
                      dest_index=f"(i / {d}) * {stride} + i % {d}")
            calls.append(f"    {scores_name}();")
            scaled = binary_emission("mul", n * n, p["inv_sqrt_d"])
            _, pointers, scale_output = install_part(scaled, scale_name, "mul")
            copy_loop(score_output, pointers[0], n * n)
            calls.append(f"    {scale_name}();")
            softmax = softmax_emission(n, n)
            _, pointers, probability = install_part(softmax, softmax_name, "softmax")
            copy_loop(scale_output, pointers[0], n * n)
            calls.append(f"    {softmax_name}();")
            result = gemm_emission(n, d, n)
            base, pointers, raw_output = install_part(result, name, "matmul")
            stride = math.ceil(n / 32) * 32
            copy_loop(probability, pointers[0], n * n,
                      dest_index=f"(i / {n}) * {stride} + i % {n}")
            copy_loop(buffers[node.args[2]].address, pointers[4], n * d,
                      dest_index=f"(i % {d}) * {stride} + i / {d}")
            calls.append(f"    {name}();")
            output = raw_output
            if result.output_shape[-1] != d:
                output = allocate(n * d * 2)
                copy_loop(raw_output, output, n * d,
                          f"(i / {d}) * {result.output_shape[-1]} + i % {d}")
            buffers[node] = Buffer(output, shape)
            cache[node.name] = np.zeros(shape, dtype=np.float32)
            manifest["kernels"].append({"node": node.name, "operation": op, "function": scores_name,
                                         "functions": [scores_name, scale_name, softmax_name, name],
                                         "source": ADAPTERS[op].source,
                                         "config_address": base + ADDR_TABLE, "output": describe(node)})
            continue
        emission = prepare_extra(node, gm, cache, op)
        if emission is None:
            if op == "add":
                # Schedule execution has no NumPy fallback. Select the same real
                # add kernel even when validation's FFN fallback is enabled.
                from codegen.c_emitter import emit_add
                tc = node.meta["tile_config"]
                emission = emit_add(node, cache, TileConfig(tc.kernel_type, {**tc.params, "use_numpy": False}))
            else:
                emission = emit_node(node, gm, cache)
        if emission is None or emission.skip_emulator or not emission.c_source:
            raise NotImplementedError(f"{node.name}: validation has no AtallaC kernel for this configuration")

        # Relocate the validation emitter's private data image into shared DRAM.
        words = emission.dram.to_u32_words(include_zeros=True, stride=4)
        size = max(max(words, default=0) + 4,
                   emission.output_addr + emission.output_elements * 2)
        if op in ("add", "mul", "relu", "threshold_backward", "mse_loss_backward"):
            total = math.prod(shape)
            width = 32 if op in ("threshold_backward", "mse_loss_backward") else min(total, 32)
            rows = math.ceil(total / width)
            sp_rows = min(rows, 32)
            span = math.ceil(rows / sp_rows) * sp_rows * width * 2
            size = max(size, emission.output_addr + span)
        base = allocate(size)
        for address, word in words.items():
            dram.u32(base + address, word)
        pointer_offsets = ADAPTERS[op].pointers
        pointers = {}
        for offset in pointer_offsets:
            local = words[ADDR_TABLE + offset]
            pointers[offset] = base + local
            dram.u32(base + ADDR_TABLE + offset, pointers[offset])

        name = f"kernel_{node.name}"
        bodies.append(_callable_source(emission.c_source, name, base + ADDR_TABLE))
        calls.append(f"    /* {node.name}: {op}, output {shape} */")
        if op == "conv":
            p = node.meta["tile_config"].params
            m, n, k = p["M"], p["N"], p["K"]
            h, w, ci, r, s = p["H"], p["W"], p["C_in"], p["R"], p["S"]
            stride = math.ceil(k / 32) * 32
            # NCHW -> im2col, with K ordered (kernel-row, kernel-col, channel).
            # Out-of-bounds and padded-K entries remain zero in the staging image.
            calls.extend([
                "    { int i = 0;",
                f"      while (i < {m * k}) {{",
                f"        int patch = i / {k};",
                f"        int lane = i % {k};",
                f"        int ih = (patch / {p['Wo']}) * {p['stride']} + lane / {s * ci} - {p['pad']};",
                f"        int iw = (patch % {p['Wo']}) * {p['stride']} + (lane / {ci}) % {s} - {p['pad']};",
                f"        if (ih >= 0 && ih < {h} && iw >= 0 && iw < {w}) {{",
                f"          schedule_copy_bf16({buffers[node.args[0]].address} + "
                f"2 * ((lane % {ci}) * {h * w} + ih * {w} + iw), "
                f"{pointers[0]} + 2 * (patch * {stride} + lane));",
                "        }",
                "        i = i + 1;",
                "      }",
                "    }",
            ])
            # OIHW -> (output-channel, padded K), with the same K ordering.
            copy_loop(buffers[node.args[1]].address, pointers[4], n * k,
                      source_index=f"(i / {k}) * {k} + ((i % {k}) % {ci}) * {r * s} + (i % {k}) / {ci}",
                      dest_index=f"(i / {k}) * {stride} + i % {k}")
            if len(node.args) > 2 and isinstance(node.args[2], Node):
                copy_loop(buffers[node.args[2]].address, pointers[8], m * n, f"i % {n}")
        elif op in ("matmul", "adaptive_avg_pool"):
            if op == "matmul":
                m, n, k = matmul_dimensions(node)
            else:
                source_shape = buffers[node.args[0]].shape
                m, n, k = source_shape[1], 1, math.prod(source_shape[2:])
            stride = math.ceil(k / 32) * 32
            copy_loop(buffers[node.args[0]].address, pointers[0], m * k,
                      dest_index=f"(i / {k}) * {stride} + i % {k}")
            if op == "matmul":
                copy_loop(buffers[node.args[1]].address, pointers[4], k * n,
                          dest_index=f"(i % {n}) * {stride} + i / {n}")
        elif op == "layernorm":
            copy_loop(buffers[node.args[0]].address, pointers[0], math.prod(shape))
            for index, offset in ((2, 8), (3, 12)):
                if len(node.args) > index and isinstance(node.args[index], Node):
                    arg = buffers[node.args[index]]
                    copy_loop(arg.address, pointers[offset], math.prod(arg.shape))
        elif op == "batchnorm":
            source = buffers[node.args[0]]
            area = math.prod(source.shape[2:])
            ch_stride = words[ADDR_TABLE + 36] // 2
            copy_loop(source.address, pointers[0], math.prod(source.shape),
                      dest_index=f"(i / {area}) * {ch_stride} + i % {area}")
        elif op == "softmax":
            source = buffers[node.args[0]]
            axes = list(range(len(shape)))
            axis = node.meta["tile_config"].params.get("softmax_dim", -1)
            axes[axis], axes[-1] = axes[-1], axes[axis]
            permuted_shape = tuple(shape[i] for i in axes)
            index = permutation_index(source.shape, axes)
            copy_loop(source.address, pointers[0], math.prod(shape), index)
        elif op in ("mse_loss", "mse_loss_backward", "threshold_backward"):
            args = node.args[:2] if op != "mse_loss_backward" else node.args[1:3]
            for index, arg in enumerate(args):
                source = buffers[arg]
                copy_loop(source.address, pointers[index * 4], math.prod(source.shape), source_dtype=source.dtype)
        elif op == "maxpool":
            copy_loop(buffers[node.args[0]].address, pointers[0], math.prod(buffers[node.args[0]].shape))
        elif op in ("avgpool", "upsample", "sum_dim"):
            source = buffers[node.args[0]]
            copy_loop(source.address, pointers[0], math.prod(source.shape))
        else:
            unary = ("relu", "gelu", "upsample", "avgpool", "sum_dim")
            for index, arg in enumerate(node.args[:1 if op in unary else 2]):
                if isinstance(arg, Node):
                    src = buffers[arg]
                    copy_loop(src.address, pointers[index * 4], math.prod(shape),
                              _index(src.shape, shape, "i"))
                elif not isinstance(arg, (int, float)):
                    raise NotImplementedError(f"{node.name}: unsupported argument {arg}")
        calls.append(f"    {name}();")
        raw_output = base + emission.output_addr
        extra_functions = []
        if op == "mse_loss_backward":
            # The standalone kernel computes 2*(pred-target)/N. Autograd also
            # supplies dL/dloss; apply that live scalar, including non-unit seeds.
            total = math.prod(shape)
            width = min(total, 32)
            rows = math.ceil(total / width)
            span = math.ceil(rows / min(rows, 32)) * min(rows, 32) * width * 2
            config = allocate(12)
            seed_input = allocate(span)
            scaled = allocate(span)
            dram.u32(config, raw_output)
            dram.u32(config + 4, seed_input)
            dram.u32(config + 8, scaled)
            seed = buffers[node.args[0]]
            copy_loop(seed.address, seed_input, total, "0", source_dtype=seed.dtype)
            scale_name = f"{name}_grad_output"
            bodies.append(_callable_source(_ek.mul_c(total, width), scale_name, config))
            calls.append(f"    {scale_name}();")
            extra_functions.append(scale_name)
            raw_output = scaled
        if op in ("matmul", "adaptive_avg_pool") and emission.output_shape != (m, n):
            final_output = allocate(math.prod(shape) * 2)
            raw_n = emission.output_shape[-1]
            copy_loop(raw_output, final_output, m * n, f"(i / {n}) * {raw_n} + i % {n}")
        elif op == "batchnorm":
            final_output = allocate(math.prod(shape) * 2)
            copy_loop(raw_output, final_output, math.prod(shape), f"(i / {area}) * {ch_stride} + i % {area}")
        elif op == "softmax" and tuple(axes) != tuple(range(len(shape))):
            final_output = allocate(math.prod(shape) * 2)
            inverse = list(np.argsort(axes))
            copy_loop(raw_output, final_output, math.prod(shape), permutation_index(permuted_shape, inverse))
        elif op == "conv":
            # Validation's conv_post is NHWC GEMM -> contiguous NCHW.
            final_output = allocate(math.prod(shape) * 2)
            copy_loop(raw_output, final_output, math.prod(shape),
                      f"(i % {m}) * {n} + i / {m}")
        elif op == "maxpool":
            # Validation computes horizontal max on the host. In a schedule it
            # must execute on the simulator as well, using the vertical result.
            p = node.meta["tile_config"].params
            final_output = allocate(math.prod(shape) * 2)
            raw_index = f"(i / {p['W_out']}) * {p['W']} + (i % {p['W_out']}) * {p['stride']}"
            copy_loop(raw_output, final_output, math.prod(shape), raw_index)
            for column in range(1, p["pool"]):
                calls.extend([
                    "    { int i = 0;",
                    f"      while (i < {math.prod(shape)}) {{",
                    f"        schedule_max_bf16({final_output} + 2 * i, "
                    f"{raw_output} + 2 * ({raw_index} + {column}), {final_output} + 2 * i);",
                    "        i = i + 1;",
                    "      }",
                    "    }",
                ])
        else:
            final_output = raw_output
        if meta.dtype == torch.float32:
            converted = allocate(math.prod(shape) * 4)
            copy_loop(final_output, converted, math.prod(shape), dest_dtype=torch.float32)
            final_output = converted
        buffers[node] = Buffer(final_output, shape, meta.dtype)
        # Only shape placeholders for downstream emitters, never reference results.
        cache[node.name] = np.zeros(shape, dtype=np.float32)
        manifest["kernels"].append({"node": node.name, "operation": op, "function": name,
                                     "functions": [name, *extra_functions],
                                     "source": ADAPTERS[op].source,
                                     "config_address": base + ADDR_TABLE,
                                     "output": describe(node)})

    source = (
        "/* Generated AtallaC: device kernels connected in graph order.\n"
        " * Load dram.bin before entry at main(); see schedule.json for I/O. */\n\n"
        + _COPY_SOURCE
        + (_ek.TRAINING_HELPERS if any(k["operation"] in _ek.TRAINING_OPS for k in manifest["kernels"]) else "")
        + (_ek.EXP_HELPER if any(k["operation"] == "gelu_backward" for k in manifest["kernels"]) else "")
        + (_MAX_SOURCE if any(k["operation"] == "maxpool" for k in manifest["kernels"]) else "")
        + "\n" + "\n".join(bodies)
        + "\nint main() {\n" + "\n".join(calls)
        + '\n    asm("halt");\n    return 0;\n}\n'
    )
    return KernelSchedule(source, dram, manifest)
