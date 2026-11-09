"""Compose validation's standalone kernels into a forward AtallaC program.

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

from codegen.c_emitter import ADDR_TABLE, DRAMWriter, LayerEmission, emit_node
from graph.fx_capture import get_node_shape


@dataclass
class Buffer:
    address: int
    shape: tuple


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
    if not replacements or 'asm("halt");' not in source:
        raise ValueError(f"{name}: unrecognized kernel configuration/termination")
    return source.replace('asm("halt");', "/* Return to the graph schedule. */")


# Copy a BF16 element without converting it through scalar arithmetic. This also
# works at odd BF16 indices, where a scalar 32-bit load/store would clobber data.
_COPY_SOURCE = """void schedule_copy_bf16(int src, int dst) {
    int ctl = 0;
    scpad_load(0, src, ctl);
    scpad_store(0, dst, ctl);
}
"""


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


def build_schedule(gm: GraphModule, placeholder_data: dict) -> KernelSchedule:
    """Build a single forward program using validation's matmul/add/mul/ReLU.

    Unsupported computations fail rather than generating a partial program.
    Backward/mixed-dtype execution is intentionally still a separate task.
    """
    if gm.meta.get("include_backward"):
        raise NotImplementedError(
            "Schedule mode still needs backward kernels and mixed-dtype support."
        )
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

    def copy_loop(source, destination, count, source_index="i", dest_index="i"):
        calls.extend([
            "    { int i = 0;",
            f"      while (i < {count}) {{",
            f"        schedule_copy_bf16({source} + 2 * ({source_index}), "
            f"{destination} + 2 * ({dest_index}));",
            "        i = i + 1;",
            "      }",
            "    }",
        ])

    def describe(node):
        buffer = buffers[node]
        return {"node": node.name, "address": buffer.address,
                "shape": list(buffer.shape), "dtype": "bfloat16"}

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
        shape = get_node_shape(node)
        meta = node.meta.get("tensor_meta")
        if shape is None or not shape or any(d <= 0 for d in shape):
            raise NotImplementedError(f"Schedule node {node.name} requires a nonempty tensor shape")
        if meta.dtype != torch.bfloat16:
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
            if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
                raise ValueError(f"{node.name}: expected BF16 tensor with shape {shape}")
            data = value.detach().float().cpu().numpy()
            address = allocate(data.size * 2)
            for i, element in enumerate(data.ravel()):
                dram.bf16(address + 2 * i, float(element))
            buffers[node] = Buffer(address, shape)
            cache[node.name] = data
            if node.op == "placeholder":
                manifest["inputs"][node.target] = describe(node)
            continue

        op = node.meta.get("atalla_op")
        if op == "flatten":
            source = buffers[node.args[0]]
            if math.prod(source.shape) != math.prod(shape):
                raise ValueError(f"{node.name}: reshape changes element count")
            buffers[node] = Buffer(source.address, shape)
            cache[node.name] = cache[node.args[0].name].reshape(shape)
            continue
        if op == "transpose":
            source = buffers[node.args[0]]
            axes = list(range(len(source.shape)))
            d0, d1 = int(node.args[1]), int(node.args[2])
            axes[d0], axes[d1] = axes[d1], axes[d0]
            address = allocate(math.prod(shape) * 2)
            terms = []
            for axis, source_axis in enumerate(axes):
                terms.append(f"((i / {math.prod(shape[axis + 1:])}) % {shape[axis]})"
                             f" * {math.prod(source.shape[source_axis + 1:])}")
            copy_loop(source.address, address, math.prod(shape), " + ".join(terms))
            buffers[node] = Buffer(address, shape)
            cache[node.name] = cache[node.args[0].name].transpose(axes).copy()
            continue

        if op not in ("matmul", "relu", "add", "mul"):
            raise NotImplementedError(f"No schedule adapter for {node.name} ({op}, {node.target})")
        if op == "add" and node.kwargs.get("alpha", 1) != 1:
            raise NotImplementedError(f"{node.name}: add alpha other than 1 is unsupported")
        if op == "matmul":
            if len(buffers[node.args[0]].shape) != 2 or len(buffers[node.args[1]].shape) != 2:
                raise NotImplementedError(f"{node.name}: schedule GEMM currently requires 2D operands")
        emission = emit_node(node, gm, cache)
        if emission is None or emission.skip_emulator or not emission.c_source:
            raise NotImplementedError(f"{node.name}: validation has no AtallaC kernel for this configuration")

        # Relocate the validation emitter's private data image into shared DRAM.
        words = emission.dram.to_u32_words(include_zeros=True, stride=4)
        size = max(max(words, default=0) + 4,
                   emission.output_addr + emission.output_elements * 2)
        base = allocate(size)
        for address, word in words.items():
            dram.u32(base + address, word)
        pointer_offsets = (0, 4, 8, 40) if op == "matmul" else ((0, 4) if op == "relu" else (0, 4, 8))
        pointers = {}
        for offset in pointer_offsets:
            local = words[ADDR_TABLE + offset]
            pointers[offset] = base + local
            dram.u32(base + ADDR_TABLE + offset, pointers[offset])

        name = f"kernel_{node.name}"
        bodies.append(_callable_source(emission.c_source, name, base + ADDR_TABLE))
        calls.append(f"    /* {node.name}: {op}, output {shape} */")
        if op == "matmul":
            p = node.meta["tile_config"].params
            m, n, k = p["M"], p["N"], p["K"]
            stride = math.ceil(k / 32) * 32
            copy_loop(buffers[node.args[0]].address, pointers[0], m * k,
                      dest_index=f"(i / {k}) * {stride} + i % {k}")
            copy_loop(buffers[node.args[1]].address, pointers[4], k * n,
                      dest_index=f"(i % {n}) * {stride} + i / {n}")
        else:
            for index, arg in enumerate(node.args[:1 if op == "relu" else 2]):
                if isinstance(arg, Node):
                    src = buffers[arg]
                    copy_loop(src.address, pointers[index * 4], math.prod(shape),
                              _index(src.shape, shape, "i"))
                elif not isinstance(arg, (int, float)):
                    raise NotImplementedError(f"{node.name}: unsupported argument {arg}")
        calls.append(f"    {name}();")
        buffers[node] = Buffer(base + emission.output_addr, shape)
        # Only shape placeholders for downstream emitters, never reference results.
        cache[node.name] = np.zeros(shape, dtype=np.float32)
        manifest["kernels"].append({"node": node.name, "operation": op, "function": name,
                                     "source": "gemm.py" if op == "matmul" else f"{op}.py",
                                     "config_address": base + ADDR_TABLE,
                                     "output": describe(node)})

    source = (
        "/* Generated AtallaC: validation kernels connected in graph order.\n"
        " * Load dram.bin before entry at main(); see schedule.json for I/O. */\n\n"
        + _COPY_SOURCE + "\n" + "\n".join(bodies)
        + "\nint main() {\n" + "\n".join(calls)
        + '\n    asm("halt");\n    return 0;\n}\n'
    )
    return KernelSchedule(source, dram, manifest)
