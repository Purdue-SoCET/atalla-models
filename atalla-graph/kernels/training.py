"""Correctness-first AtallaC kernels for captured training operators.

These kernels read live device buffers. Scalar reductions accumulate in the
scalar floating-point registers; BF16 rounding happens when writing a tensor.
They favor general shapes and explicit semantics over vector throughput.
"""
from __future__ import annotations

import math
import re
import struct
from decimal import Decimal

import torch
from torch.fx import Node


TRAINING_OPS = {
    "native_layer_norm", "native_layer_norm_backward", "convolution_backward",
    "max_pool2d_with_indices", "max_pool2d_with_indices_backward", "sum_dim",
    "gelu_backward", "softmax_backward",
}


# Scalar float loads/stores in AtallaC operate on 32-bit register bit patterns.
# Packed BF16 tensors need explicit extraction and round-to-nearest-even stores.
TRAINING_HELPERS = r'''
union schedule_scalar { int bits; float value; };
float schedule_read_bf16(int address) {
    int word = *(int*)(address & -4);
    int bits = ((word >> ((address & 2) * 8)) & 65535) << 16;
    return *(float*)&bits;
}
void schedule_write_bf16(int address, int value_address) {
    int bits = *(int*)value_address;
    int half;
    if ((bits & 2147483647) > 2139095040) {
        half = (bits >> 16) | 64;
    } else {
        half = ((bits + 32767 + ((bits >> 16) & 1)) >> 16) & 65535;
    }
    int aligned = address & -4;
    int shift = (address & 2) * 8;
    int word = *(int*)aligned;
    *(int*)aligned = (word & ~(65535 << shift)) | (half << shift);
}
'''


def read(buffer, index):
    if buffer.dtype == torch.bfloat16:
        return f"schedule_read_bf16({buffer.address} + 2 * ({index}))"
    return f"*(float*)({buffer.address} + 4 * ({index}))"


def write(buffer, index, value):
    if buffer.dtype == torch.bfloat16:
        return (f"{{ union schedule_scalar stored; stored.value = {value}; "
                f"schedule_write_bf16({buffer.address} + 2 * ({index}), (int)&stored); }}")
    return f"*(float*)({buffer.address} + 4 * ({index})) = {value};"


def literal(value):
    # The AtallaC lexer does not accept scientific notation.
    result = format(Decimal(str(float(value))), "f")
    return result if "." in result else result + ".0"


def pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def reduction_dims(node):
    rank = len(node.args[0].meta["tensor_meta"].shape)
    if rank == 0:
        return []
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    dims = [dims] if isinstance(dims, int) else dims
    return sorted({d % rank for d in dims}) if dims else list(range(rank))


def check_training(node, op):
    """Validate the operator schema, including configurations not yet handled."""
    for index, arg in enumerate(node.args):
        if isinstance(arg, Node) and not (op == "max_pool2d_with_indices_backward" and index == 7):
            meta = arg.meta.get("tensor_meta")
            if meta is None or not hasattr(meta, "shape") or meta.dtype not in (torch.bfloat16, torch.float32):
                raise NotImplementedError("compute inputs must be BF16 or FP32 tensors")
    if op in ("native_layer_norm", "native_layer_norm_backward"):
        position = 1 if op == "native_layer_norm" else 2
        normalized = tuple(node.args[position])
        shape = tuple(node.args[0 if position == 1 else 1].meta["tensor_meta"].shape)
        if not normalized or tuple(shape[-len(normalized):]) != normalized:
            raise NotImplementedError("LayerNorm requires trailing normalized axes")
    elif op == "convolution_backward":
        if len(node.args[1].meta["tensor_meta"].shape) != 4:
            raise NotImplementedError("convolution backward currently supports 2D NCHW")
        if node.args[7] or any(node.args[8]):
            raise NotImplementedError("transposed convolution backward is not implemented")
    elif op in ("max_pool2d_with_indices", "max_pool2d_with_indices_backward"):
        position = 0 if op == "max_pool2d_with_indices" else 1
        if len(node.args[position].meta["tensor_meta"].shape) != 4:
            raise NotImplementedError("max-pool currently supports 2D NCHW")
    elif op == "softmax_backward":
        if node.args[3] != torch.bfloat16:
            raise NotImplementedError("softmax backward currently requires BF16 input dtype")
    elif op == "gelu_backward":
        if node.kwargs.get("approximate", "none") not in ("none", "tanh"):
            raise NotImplementedError("unknown GELU approximation")


def sum_body(node, buffers, output):
    source = buffers[node.args[0]]
    dims = reduction_dims(node)
    kept = [d for d in range(len(source.shape)) if d not in dims]
    reduced_shape = tuple(source.shape[d] for d in dims)
    kept_shape = tuple(source.shape[d] for d in kept)
    terms = []
    for axes, shape, variable in ((dims, reduced_shape, "r"), (kept, kept_shape, "i")):
        for axis, size, position in zip(axes, shape, range(len(shape))):
            terms.append(f"(({variable} / {math.prod(shape[position + 1:])}) % {size})"
                         f" * {math.prod(source.shape[axis + 1:])}")
    index = " + ".join(terms) or "0"
    return f'''int i = 0;
    while (i < {math.prod(output.shape)}) {{
        float acc = 0.0;
        int r = 0;
        while (r < {math.prod(reduced_shape)}) {{
            acc = acc + {read(source, index)};
            r = r + 1;
        }}
        {write(output, 'i', 'acc')}
        i = i + 1;
    }}'''


def layernorm_body(node, buffers, outputs, backward=False):
    x = buffers[node.args[1 if backward else 0]]
    d = math.prod(node.args[2 if backward else 1])
    rows = math.prod(x.shape) // d
    if not backward:
        y, mean, rstd = outputs
        gamma = buffers[node.args[2]] if node.args[2] is not None else None
        beta = buffers[node.args[3]] if node.args[3] is not None else None
        return f'''int row = 0;
        while (row < {rows}) {{
            float acc = 0.0;
            int j = 0;
            while (j < {d}) {{
                acc = acc + {read(x, f'row * {d} + j')};
                j = j + 1;
            }}
            float mu = acc * {literal(1 / d)};
            float variance = 0.0;
            j = 0;
            while (j < {d}) {{
                float centered = {read(x, f'row * {d} + j')} - mu;
                variance = variance + centered * centered;
                j = j + 1;
            }}
            float inv = 1.0 / sqrt(variance * {literal(1 / d)} + {literal(node.args[4])});
            {write(mean, 'row', 'mu')}
            {write(rstd, 'row', 'inv')}
            j = 0;
            while (j < {d}) {{
                float normalized = ({read(x, f'row * {d} + j')} - mu) * inv;
                float result = normalized * {read(gamma, 'j') if gamma else '1.0'} + {read(beta, 'j') if beta else '0.0'};
                {write(y, f'row * {d} + j', 'result')}
                j = j + 1;
            }}
            row = row + 1;
        }}'''
    grad, mean, rstd = (buffers[node.args[i]] for i in (0, 3, 4))
    gamma = buffers[node.args[5]] if node.args[5] is not None else None
    dx, dg, db = outputs
    body = ""
    if dx is not None:
        body += f'''int row = 0;
        while (row < {rows}) {{
            float mu = {read(mean, 'row')};
            float inv = {read(rstd, 'row')};
            float s1 = 0.0;
            float s2 = 0.0;
            int j = 0;
            while (j < {d}) {{
                float weighted = {read(grad, f'row * {d} + j')} * {read(gamma, 'j') if gamma else '1.0'};
                float centered = {read(x, f'row * {d} + j')} - mu;
                s1 = s1 + weighted;
                s2 = s2 + weighted * centered;
                j = j + 1;
            }}
            j = 0;
            while (j < {d}) {{
                float weighted = {read(grad, f'row * {d} + j')} * {read(gamma, 'j') if gamma else '1.0'};
                float centered = {read(x, f'row * {d} + j')} - mu;
                float result = inv * (weighted - s1 * {literal(1 / d)} - centered * inv * inv * s2 * {literal(1 / d)});
                {write(dx, f'row * {d} + j', 'result')}
                j = j + 1;
            }}
            row = row + 1;
        }}\n'''
    if dg is not None or db is not None:
        body += f'''int j = 0;
        while (j < {d}) {{
            float sg = 0.0;
            float sb = 0.0;
            int row = 0;
            while (row < {rows}) {{
                float g = {read(grad, f'row * {d} + j')};
                float normalized = ({read(x, f'row * {d} + j')} - {read(mean, 'row')}) * {read(rstd, 'row')};
                sg = sg + g * normalized;
                sb = sb + g;
                row = row + 1;
            }}
            {write(dg, 'j', 'sg') if dg else ''}
            {write(db, 'j', 'sb') if db else ''}
            j = j + 1;
        }}'''
    return body


def pool_body(node, buffers, outputs, backward=False):
    x = buffers[node.args[1 if backward else 0]]
    n, channels, h, w = x.shape
    if backward:
        grad, indices = buffers[node.args[0]], buffers[node.args[7]]
        oh, ow = grad.shape[-2:]
        kh, kw = pair(node.args[2])
        sh, sw = pair(node.args[3]) if node.args[3] else (kh, kw)
        ph, pw = pair(node.args[4])
        dh, dw = pair(node.args[5])
        # Gather rather than scatter so overlapping windows accumulate in a
        # register and round once, with no BF16 atomic or host postprocessing.
        return f'''int i = 0;
        while (i < {n * channels * h * w}) {{
            int plane = i / {h * w};
            int location = i % {h * w};
            int r = 0;
            float acc = 0.0;
            while (r < {kh * kw}) {{
                int gh = location / {w} + {ph} - (r / {kw}) * {dh};
                int gw = location % {w} + {pw} - (r % {kw}) * {dw};
                if (gh >= 0 && gw >= 0 && gh % {sh} == 0 && gw % {sw} == 0 && gh / {sh} < {oh} && gw / {sw} < {ow}) {{
                    int out_index = plane * {oh * ow} + (gh / {sh}) * {ow} + gw / {sw};
                    int index = *(int*)({indices.address} + 8 * out_index);
                    if (index == location) {{
                        acc = acc + {read(grad, 'out_index')};
                    }}
                }}
                r = r + 1;
            }}
            {write(outputs, 'i', 'acc')}
            i = i + 1;
        }}'''
    y, indices = outputs
    oh, ow = y.shape[-2:]
    kh, kw = pair(node.args[1])
    sh, sw = pair(node.args[2]) if len(node.args) > 2 and node.args[2] else (kh, kw)
    ph, pw = pair(node.args[3]) if len(node.args) > 3 else (0, 0)
    dh, dw = pair(node.args[4]) if len(node.args) > 4 else (1, 1)
    return f'''int i = 0;
    while (i < {math.prod(y.shape)}) {{
        int plane = i / {oh * ow};
        int top = (i / {ow}) % {oh} * {sh} - {ph};
        int left = i % {ow} * {sw} - {pw};
        int found = 0;
        int selected = -1;
        float best = 0.0;
        int r = 0;
        while (r < {kh * kw}) {{
            int ih = top + (r / {kw}) * {dh};
            int iw = left + (r % {kw}) * {dw};
            if (ih >= 0 && ih < {h} && iw >= 0 && iw < {w}) {{
                float value = {read(x, f'plane * {h * w} + ih * {w} + iw')};
                int take = found == 0;
                int candidate_bits = *(int*)&value;
                int best_bits = *(int*)&best;
                int candidate_abs = candidate_bits & 2147483647;
                int best_abs = best_bits & 2147483647;
                int candidate_sign = (candidate_bits >> 31) & 1;
                int best_sign = (best_bits >> 31) & 1;
                if (best_abs <= 2139095040 && (candidate_abs != 0 || best_abs != 0)) {{
                    if (candidate_sign == 0 && best_sign != 0) {{ take = 1; }}
                    if (candidate_sign == 0 && best_sign == 0 && candidate_abs > best_abs) {{ take = 1; }}
                    if (candidate_sign != 0 && best_sign != 0 && candidate_abs < best_abs) {{ take = 1; }}
                }}
                if (candidate_abs > 2139095040) {{ take = 1; }}
                if (take) {{
                    best = value;
                    selected = ih * {w} + iw;
                    found = 1;
                }}
            }}
            r = r + 1;
        }}
        {write(y, 'i', 'best')}
        *(int*)({indices.address} + 8 * i) = selected;
        *(int*)({indices.address} + 8 * i + 4) = 0;
        i = i + 1;
    }}'''


def convolution_body(node, buffers, outputs):
    grad, x, weight = (buffers[arg] for arg in node.args[:3])
    dx, dg, db = outputs
    batch, ci, h, w = x.shape
    co, cig, kh, kw = weight.shape
    oh, ow = grad.shape[-2:]
    sh, sw = pair(node.args[4])
    ph, pw = pair(node.args[5])
    dh, dw = pair(node.args[6])
    groups = node.args[9]
    cog = co // groups
    body = ""
    if dx is not None:
        body += f'''int i = 0;
        while (i < {math.prod(x.shape)}) {{
            int n = i / {ci * h * w};
            int c = (i / {h * w}) % {ci};
            int ih = (i / {w}) % {h};
            int iw = i % {w};
            float acc = 0.0;
            int r = 0;
            while (r < {cog * kh * kw}) {{
                int oc = (c / {cig}) * {cog} + r / {kh * kw};
                int rh = (r / {kw}) % {kh};
                int rw = r % {kw};
                int gh = ih + {ph} - rh * {dh};
                int gw = iw + {pw} - rw * {dw};
                if (gh >= 0 && gw >= 0 && gh % {sh} == 0 && gw % {sw} == 0 && gh / {sh} < {oh} && gw / {sw} < {ow}) {{
                    float g = {read(grad, f'((n * {co} + oc) * {oh} + gh / {sh}) * {ow} + gw / {sw}')};
                    float v = {read(weight, f'((oc * {cig} + c % {cig}) * {kh} + rh) * {kw} + rw')};
                    acc = acc + g * v;
                }}
                r = r + 1;
            }}
            {write(dx, 'i', 'acc')}
            i = i + 1;
        }}\n'''
    if dg is not None:
        body += f'''int wi = 0;
        while (wi < {math.prod(weight.shape)}) {{
            int oc = wi / {cig * kh * kw};
            int c = (oc / {cog}) * {cig} + (wi / {kh * kw}) % {cig};
            int rh = (wi / {kw}) % {kh};
            int rw = wi % {kw};
            float acc = 0.0;
            int r = 0;
            while (r < {batch * oh * ow}) {{
                int n = r / {oh * ow};
                int gh = (r / {ow}) % {oh};
                int gw = r % {ow};
                int ih = gh * {sh} - {ph} + rh * {dh};
                int iw = gw * {sw} - {pw} + rw * {dw};
                if (ih >= 0 && ih < {h} && iw >= 0 && iw < {w}) {{
                    float g = {read(grad, f'((n * {co} + oc) * {oh} + gh) * {ow} + gw')};
                    float v = {read(x, f'((n * {ci} + c) * {h} + ih) * {w} + iw')};
                    acc = acc + g * v;
                }}
                r = r + 1;
            }}
            {write(dg, 'wi', 'acc')}
            wi = wi + 1;
        }}\n'''
    if db is not None:
        body += f'''int oc = 0;
        while (oc < {co}) {{
            float acc = 0.0;
            int r = 0;
            while (r < {batch * oh * ow}) {{
                acc = acc + {read(grad, f'(r / {oh * ow} * {co} + oc) * {oh * ow} + r % {oh * ow}')};
                r = r + 1;
            }}
            {write(db, 'oc', 'acc')}
            oc = oc + 1;
        }}'''
    return body


def softmax_backward_body(node, buffers, output):
    grad, y = (buffers[arg] for arg in node.args[:2])
    axis = node.args[2] % len(y.shape)
    width = y.shape[axis]
    outer, inner = math.prod(y.shape[:axis]), math.prod(y.shape[axis + 1:])
    idx = f'(row / {inner}) * {width * inner} + j * {inner} + row % {inner}'
    return f'''int row = 0;
    while (row < {outer * inner}) {{
        float dot = 0.0;
        int j = 0;
        while (j < {width}) {{
            dot = dot + {read(grad, idx)} * {read(y, idx)};
            j = j + 1;
        }}
        j = 0;
        while (j < {width}) {{
            float result = {read(y, idx)} * ({read(grad, idx)} - dot);
            {write(output, idx, 'result')}
            j = j + 1;
        }}
        row = row + 1;
    }}'''


def gelu_backward_body(node, buffers, output):
    grad, x = (buffers[arg] for arg in node.args[:2])
    # erf polynomial (Abramowitz-Stegun 7.1.26), absolute error <1.5e-7
    # in real arithmetic. exp uses the device vector EXP instruction, lane 0.
    approximate = node.kwargs.get("approximate", "none")
    if approximate == "tanh":
        derivative = '''float z = 0.7978845608028654 * (v + 0.044715 * v * v * v);
            float t = 0.0;
            int z_bits = *(int*)&z;
            if (((z_bits >> 31) & 1) == 0 && z_bits > 1092616192) { t = 1.0; }
            else if (((z_bits >> 31) & 1) != 0 && (z_bits & 2147483647) > 1092616192) { t = -1.0; }
            else {
                float exponential = schedule_exp_scalar(0.0 - 2.0 * z);
                t = 2.0 / (1.0 + exponential) - 1.0;
            }
            float derivative = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * v * v);'''
    else:
        derivative = '''float a = v * 0.7071067811865475;
            float sign = 1.0;
            if (((*(int*)&a >> 31) & 1) != 0) { a = 0.0 - a; sign = -1.0; }
            float t = 1.0 / (1.0 + 0.3275911 * a);
            float p = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
            float density = schedule_exp_scalar(0.0 - 0.5 * v * v);
            float erf_value = sign * (1.0 - p * t * density);
            float derivative = 0.5 * (1.0 + erf_value) + v * 0.3989422804014327 * density;'''
    return f'''int i = 0;
    while (i < {math.prod(output.shape)}) {{
        float v = {read(x, 'i')};
        {derivative}
        float result = {read(grad, 'i')} * derivative;
        {write(output, 'i', 'result')}
        i = i + 1;
    }}'''


EXP_HELPER = '''float schedule_exp_scalar(float value) {
    vec zero = vector_load(0, 0, 0, 0);
    zero = vec_op_masked("*", zero, 0.0, 1);
    vec v = vec_op_masked("+", zero, value, 1);
    vec result = vec_op_masked("EXP", v, 0.0, 1);
    return result[0];
}
'''


def _training_body(node, op, buffers, output):
    if op == "sum_dim":
        return sum_body(node, buffers, output)
    if op in ("native_layer_norm", "native_layer_norm_backward"):
        return layernorm_body(node, buffers, output, op.endswith("backward"))
    if op in ("max_pool2d_with_indices", "max_pool2d_with_indices_backward"):
        return pool_body(node, buffers, output, op.endswith("backward"))
    if op == "convolution_backward":
        return convolution_body(node, buffers, output)
    if op == "softmax_backward":
        return softmax_backward_body(node, buffers, output)
    if op == "gelu_backward":
        return gelu_backward_body(node, buffers, output)
    raise ValueError(op)


def training_body(node, op, buffers, output):
    body = _training_body(node, op, buffers, output)
    # PPCI's AtallaC float has sizeof=2, but scalar register loads/stores are
    # 32 bits. Reserve a full word for each accumulator to avoid overlapping
    # stack slots, without changing the compiler or its packed BF16 ABI.
    declarations = re.compile(r"\bfloat\s+(\w+)\s*=\s*([^;]+);")
    names = {match[1] for match in declarations.finditer(body)}
    body = declarations.sub(lambda m: f"union schedule_scalar slot_{m[1]}; slot_{m[1]}.value = {m[2]};", body)
    if names:
        body = re.sub(r"(?<!\.)\b(" + "|".join(sorted(names, key=len, reverse=True)) + r")\b",
                      lambda m: f"slot_{m[0]}.value", body)
    # Keep tensor reads in the caller. PPCI does not preserve expression
    # temporaries across a scalar helper call, so f(x) + f(y) is unsafe here.
    lines = []
    sequence = 0
    for line in body.splitlines():
        prefix = []
        while "schedule_read_bf16(" in line:
            start = line.index("schedule_read_bf16(")
            arg_start = start + len("schedule_read_bf16(")
            depth, end = 1, arg_start
            while depth:
                depth += (line[end] == "(") - (line[end] == ")")
                end += 1
            address = line[arg_start:end - 1]
            name = f"loaded_{sequence}"
            sequence += 1
            prefix.append(f"int {name}_addr = {address};")
            prefix.append(f"int {name}_word = *(int*)({name}_addr & -4);")
            prefix.append(f"union schedule_scalar {name};")
            prefix.append(f"{name}.bits = (({name}_word >> (({name}_addr & 2) * 8)) & 65535) << 16;")
            line = line[:start] + f"{name}.value" + line[end:]
        lines.extend(prefix)
        lines.append(line)
    body = "\n".join(lines)
    # C float literals are quantized by PPCI's BF16 literal lowering. Load the
    # exact FP32 bit patterns instead, especially for reduction reciprocals.
    constants = {}
    def constant(match):
        value = match[0]
        if value not in constants:
            constants[value] = f"constant_{len(constants)}"
        return constants[value] + ".value"
    body = re.sub(r"\b\d+\.\d+\b", constant, body)
    declarations = []
    for value, name in constants.items():
        bits = struct.unpack("<i", struct.pack("<f", float(value)))[0]
        declarations.append(f"union schedule_scalar {name}; {name}.bits = {bits};")
    return "\n".join(declarations) + "\n" + body
