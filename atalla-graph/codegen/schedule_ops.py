"""Schedule adapter registry and standalone-kernel configurations.

The registry describes real device implementations, never host reference
fallbacks. Shape restrictions are checked before source generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from torch.fx import Node

from codegen.c_emitter import (
    ADDR_TABLE, LayerEmission, _ek, _gemm_c, _write_gemm_params,
    emit_layernorm, emit_softmax, sdma_ctl_val,
    emit_threshold_backward, emit_mse_loss, emit_mse_loss_backward,
)
from graph.fx_capture import get_node_shape


@dataclass(frozen=True)
class Adapter:
    source: str
    pointers: tuple = ()
    note: str = ""


ADAPTERS = {
    "matmul": Adapter("gemm.py", (0, 4, 8, 40)),
    "linear": Adapter("gemm.py", note="lowered to matmul and bias add by the frontend"),
    "conv": Adapter("gemm.py", (0, 4, 8, 40)),
    "relu": Adapter("relu.py", (0, 4)),
    "add": Adapter("add.py", (0, 4, 8)),
    "mul": Adapter("mul.py", (0, 4, 8)),
    "maxpool": Adapter("maxpool.py", (0, 4)),
    "softmax": Adapter("softmax.py", (0,)),
    "layernorm": Adapter("layernorm.py", (0, 4, 8, 12)),
    "gelu": Adapter("gelu_tanh.c", (0, 4), "polynomial GELU approximation"),
    "batchnorm": Adapter("batchnorm.c", (0, 4, 8, 12), "evaluation mode, folded constant statistics"),
    "avgpool": Adapter("avgpool.c", (0, 4)),
    "adaptive_avg_pool": Adapter("gemm.py", (0, 4, 8, 40), "global spatial mean via GEMM"),
    "upsample": Adapter("upsample.c", (0, 4)),
    "atalla_sdpa": Adapter("gemm.py + mul.py + softmax.py", note="QK^T, scale, softmax, PV"),
    "flatten": Adapter("layout"),
    "transpose": Adapter("layout"),
    "permute": Adapter("layout"),
    "dropout": Adapter("alias", note="evaluation mode only"),
    "detach": Adapter("alias"),
    "contiguous": Adapter("alias"),
    "expand": Adapter("layout"),
    "cast": Adapter("dtype conversion"),
    "ones_like": Adapter("constant initialization"),
    "threshold_backward": Adapter("threshold_backward.c", (0, 4, 8)),
    "mse_loss": Adapter("mse_loss.c", (0, 4, 8)),
    "mse_loss_backward": Adapter("mse_loss_backward.c", (0, 4, 8)),
    "getitem": Adapter("tuple alias"),
    **{op: Adapter("training.py") for op in _ek.TRAINING_OPS},
}


def argument(node, gm, name, position, default):
    if node.op == "call_module":
        return getattr(gm.get_submodule(node.target), name, default)
    return node.kwargs.get(name, node.args[position] if len(node.args) > position else default)


def pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def op_for(node):
    aliases = {"aten.detach.default": "detach", "aten.clone.default": "contiguous",
               "aten._to_copy.default": "cast", "aten.ones_like.default": "ones_like",
               **{f"aten.{op}.default": op for op in
                  ("mse_loss", "mse_loss_backward", "threshold_backward")},
               "aten.sum.dim_IntList": "sum_dim"}
    aliases.update({f"aten.{op}.default": op for op in _ek.TRAINING_OPS})
    aliases["aten._softmax_backward_data.default"] = "softmax_backward"
    return aliases.get(node.meta.get("atalla_op"), node.meta.get("atalla_op"))


def matmul_dimensions(node):
    a, b = get_node_shape(node.args[0]), get_node_shape(node.args[1])
    if len(a) < 2 or len(b) < 2 or (len(b) > 2 and math.prod(b[:-2]) != 1):
        raise NotImplementedError("GEMM needs a shared 2D RHS (singleton batch axes are allowed)")
    m, k, n = math.prod(a[:-1]), a[-1], b[-1]
    if b[-2] != k or math.prod(get_node_shape(node)) != m * n:
        raise NotImplementedError("batched matmul output cannot be represented by a shared RHS")
    return m, n, k


def check_extra(node, gm, op):
    """Return no result on support; raise a specific capability error otherwise."""
    if op in _ek.TRAINING_OPS:
        return _ek.check_training(node, op)
    p = node.meta.get("tile_config")
    params = p.params if p is not None else {}
    shape = get_node_shape(node)
    src = get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], Node) else None
    if op == "matmul":
        matmul_dimensions(node)
    elif op == "linear":
        raise NotImplementedError("linear must be lowered to matmul and add before scheduling")
    elif op == "layernorm" and (params.get("D", 0) < 32 or params["D"] % 32):
        raise NotImplementedError("LayerNorm normalized size must be a positive multiple of 32")
    elif op == "softmax" and params.get("row_len", 33) > 32:
        raise NotImplementedError("softmax reduction axis must have at most 32 elements")
    elif op == "atalla_sdpa":
        if params.get("B") != 1 or not 1 <= params.get("N", 0) <= 32:
            raise NotImplementedError("attention requires B=1 and N<=32 (softmax kernel limit)")
    elif op == "batchnorm":
        if len(src) != 4 or src[0] != 1 or argument(node, gm, "training", 5, False):
            raise NotImplementedError("BatchNorm currently needs evaluation-mode batch-one NCHW")
        if node.op != "call_function":
            raise NotImplementedError("BatchNorm must be lowered before scheduling")
        if any(isinstance(a, Node) and a.op != "get_attr" for a in node.args[1:5]):
            raise NotImplementedError("BatchNorm statistics and affine parameters must be constant attributes")
        if node.args[1] is None or node.args[2] is None:
            raise NotImplementedError("BatchNorm requires running mean and variance")
    elif op == "avgpool":
        kernel = pair(argument(node, gm, "kernel_size", 1, 2))
        stride = argument(node, gm, "stride", 2, None)
        stride = kernel if stride is None else pair(stride)
        if (len(src) != 4 or src[0] != 1 or src[-1] != 8 or src[-2] % 2 or src[-2] > 32
                or kernel != (2, 2) or stride != (2, 2)
                or pair(argument(node, gm, "padding", 3, 0)) != (0, 0)
                or argument(node, gm, "ceil_mode", 4, False)
                or argument(node, gm, "divisor_override", 6, None) is not None):
            raise NotImplementedError("standalone avgpool needs batch-one NCHW, even H<=32, W=8, "
                                      "kernel=stride=2, no padding/ceil/divisor override")
    elif op == "adaptive_avg_pool":
        if len(src) != 4 or src[0] != 1 or tuple(shape[-2:]) != (1, 1):
            raise NotImplementedError("adaptive average pooling currently supports global batch-one NCHW -> 1x1")
    elif op == "upsample":
        if (len(src) != 4 or src[0] != 1 or src[-1] % 2
                or argument(node, gm, "mode", 3, "nearest") != "nearest"
                or tuple(shape) != (1, src[1], 2 * src[2], 2 * src[3])):
            raise NotImplementedError("standalone upsample needs batch-one NCHW, even input W, nearest 2x")
    elif op == "dropout":
        if argument(node, gm, "training", 2, True) and argument(node, gm, "p", 1, 0.5) != 0:
            raise NotImplementedError("training-mode dropout requires a random-number kernel")
    elif op == "threshold_backward" and (len(node.args) < 3 or node.args[2] != 0):
        raise NotImplementedError("threshold backward kernel supports threshold=0 (ReLU)")
    elif op in ("mse_loss", "mse_loss_backward"):
        position = 2 if op == "mse_loss" else 3
        reduction = argument(node, gm, "reduction", position, 1)
        if reduction not in (1, "mean"):
            raise NotImplementedError("MSE kernels currently support mean reduction only")


def gemm_emission(m, n, k):
    """Validation's GEMM with complete large output tiles and padded K storage."""
    mp = m if m <= 32 else math.ceil(m / 32) * 32
    np_ = n if n <= 32 else math.ceil(n / 32) * 32
    ks = math.ceil(k / 32) * 32
    a = 0x1000
    w = a + mp * ks * 2
    c = w + np_ * ks * 2
    z = c + mp * np_ * 2
    em = LayerEmission()
    em.dram.write_zeros(a, z + 64 - a)
    _write_gemm_params(em.dram, a, w, c, mp, np_, k, z)
    em.c_source = _gemm_c(mp, np_, k)
    em.output_addr, em.output_elements = c, mp * np_
    em.output_shape = (mp, np_)
    return em


def binary_emission(op, total, constant=None):
    width = min(total, 32)
    rows = math.ceil(total / width)
    sp_rows = min(rows, 32)
    size = math.ceil(rows / sp_rows) * sp_rows * width * 2
    a, b, c = 0x1000, 0x1000 + size, 0x1000 + 2 * size
    em = LayerEmission()
    em.dram.write_zeros(a, size * 3)
    for offset, address in ((0, a), (4, b), (8, c)):
        em.dram.u32(ADDR_TABLE + offset, address)
    if constant is not None:
        for i in range(total):
            em.dram.bf16(b + i * 2, constant)
    em.c_source = (_ek.mul_c if op == "mul" else _ek.add_c)(total, width)
    em.output_addr, em.output_elements = c, total
    return em


def softmax_emission(rows, width):
    em = LayerEmission()
    em.c_source = _ek.softmax_c_batched(rows, width)
    em.dram.u32(ADDR_TABLE, 0x1000)
    em.dram.u32(ADDR_TABLE + 4, 0)
    em.dram.write_zeros(0x1000, rows * width * 2)
    em.output_addr, em.output_elements = 0x1000, rows * width
    return em


def standalone_emission(node, gm, op):
    shape = get_node_shape(node)
    src = get_node_shape(node.args[0])
    total = math.prod(shape)
    em = LayerEmission()
    image = em.dram
    a = 0x1000
    if op == "gelu":
        rows = math.ceil(total / 32)
        sp_rows = min(rows, 32)
        tiles = math.ceil(rows / sp_rows)
        size = tiles * sp_rows * 64
        b = a + size
        values = [a, b, tiles, sp_rows, sdma_ctl_val(0, sp_rows, 32, 32), sp_rows * 64]
        constants = [0.797885, 0.035677, -0.002561, 0.039209, -0.237744, 0.968467,
                     1.0, -1.0, 0.5, 2.5, -2.5]
        for i, value in enumerate(constants):
            image.f32(ADDR_TABLE + 24 + 4 * i, value)
        em.c_source = _ek.gelu_c(method="tanh")
        image.write_zeros(a, 2 * size)
    elif op == "batchnorm":
        channels, area = src[1], math.prod(src[2:])
        rows = math.ceil(area / 32)
        sp_rows = min(rows, 32)
        tiles = math.ceil(rows / sp_rows)
        ch_bytes = tiles * sp_rows * 64
        b = a + channels * ch_bytes
        scale = b + channels * ch_bytes
        bias = scale + channels * 4
        values = [a, b, scale, bias, channels, sp_rows, tiles,
                  sdma_ctl_val(0, sp_rows, 32, 32), sp_rows * 64, ch_bytes]
        image.write_zeros(a, bias + channels * 4 - a)

        def attribute(arg, default):
            if arg is None:
                return np.full(channels, default, dtype=np.float32)
            value = gm
            for part in arg.target.split("."):
                value = getattr(value, part)
            return value.detach().float().cpu().numpy().reshape(-1)

        mean, var = attribute(node.args[1], 0), attribute(node.args[2], 1)
        gamma, beta = attribute(node.args[3], 1), attribute(node.args[4], 0)
        eps = float(argument(node, gm, "eps", 7, 1e-5))
        scales = gamma / np.sqrt(var + eps)
        biases = beta - mean * scales
        for i in range(channels):
            image.f32(scale + 4 * i, float(scales[i]))
            image.f32(bias + 4 * i, float(biases[i]))
        em.c_source = _ek.batchnorm_c()
    elif op == "upsample":
        b = a + math.prod(src) * 2
        values = [a, b, src[1] * src[2], src[3] // 2, src[3] * 2, src[3] * 4]
        image.write_zeros(a, (math.prod(src) + total) * 2)
        em.c_source = _ek.upsample_c()
    elif op == "avgpool":
        b = a + math.prod(src) * 2
        values = [a, b, src[1], src[2], shape[2],
                  sdma_ctl_val(0, src[2], 8, 8), sdma_ctl_val(0, shape[2], 4, 4),
                  src[2] * 8 * 2, shape[2] * 4 * 2]
        image.write_zeros(a, (math.prod(src) + total) * 2)
        image.f32(ADDR_TABLE + 36, 0.25)
        em.c_source = _ek.avgpool_c()
    else:
        raise ValueError(op)
    for i, value in enumerate(values):
        image.u32(ADDR_TABLE + 4 * i, value)
    em.output_addr, em.output_elements, em.output_shape = b, total, shape
    return em


def prepare_extra(node, gm, cache, op):
    if op in ("gelu", "batchnorm", "upsample", "avgpool"):
        return standalone_emission(node, gm, op)
    if op == "matmul":
        return gemm_emission(*matmul_dimensions(node))
    if op == "adaptive_avg_pool":
        shape = get_node_shape(node.args[0])
        em = gemm_emission(shape[1], 1, math.prod(shape[2:]))
        words = em.dram.to_u32_words(stride=4)
        weight = words[ADDR_TABLE + 4]
        for i in range(math.prod(shape[2:])):
            em.dram.bf16(weight + 2 * i, 1.0 / math.prod(shape[2:]))
        return em
    tc = node.meta["tile_config"]
    if op == "layernorm":
        return emit_layernorm(node, gm, cache[node.args[0].name], cache, tc)
    if op == "softmax":
        return emit_softmax(node, cache[node.args[0].name], tc)
    if op == "threshold_backward":
        return emit_threshold_backward(node, cache, tc)
    if op == "mse_loss":
        return emit_mse_loss(node, cache, tc)
    if op == "mse_loss_backward":
        return emit_mse_loss_backward(node, cache, tc)
    return None
