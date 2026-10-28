"""FX Graph capture + op normalization for Atalla pipeline.

Traces a PyTorch nn.Module via torch.fx, propagates shapes,
and normalizes FX ops to a fixed set of Atalla primitives.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, symbolic_trace

from model.atalla_ops import AtallaSdpa
from torch.fx.passes.shape_prop import ShapeProp


ATALLA_OPS = {
    "matmul", "linear", "conv", "relu", "add", "mul", "maxpool",
    "softmax", "flatten", "adaptive_avg_pool", "dropout", "bias_add",
    "layernorm", "gelu", "atalla_sdpa",
}

_OP_MAP: Dict[object, str] = {
    torch.matmul: "matmul",
    torch.mm: "matmul",
    torch.bmm: "matmul",
    torch.ops.aten.matmul.default: "matmul",
    torch.ops.aten.mm.default: "matmul",
    torch.ops.aten.bmm.default: "matmul",
    torch.ops.aten.t.default: "transpose",
    torch.ops.aten.transpose.int: "transpose",
    torch.ops.aten.view.default: "flatten",
    torch.ops.aten.reshape.default: "flatten",
    F.relu: "relu",
    torch.relu: "relu",
    torch.ops.aten.relu.default: "relu",
    F.softmax: "softmax",
    F.max_pool2d: "maxpool",
    F.conv2d: "conv",
    F.linear: "linear",
    F.adaptive_avg_pool2d: "adaptive_avg_pool",
    F.dropout: "dropout",
    torch.flatten: "flatten",
    torch.add: "add",
    operator.add: "add",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.add_.Tensor: "add",
    operator.mul: "mul",
    torch.mul: "mul",
    torch.ops.aten.mul.Tensor: "mul",
    F.batch_norm: "batchnorm",
    F.layer_norm: "layernorm",
    F.gelu: "gelu",
    F.avg_pool2d: "avgpool",
    F.interpolate: "upsample",
}

_METHOD_MAP: Dict[str, str] = {
    "relu": "relu",
    "flatten": "flatten",
    "add": "add",
    "matmul": "matmul",
    "view": "flatten",
    "reshape": "flatten",
    "transpose": "transpose",
    "mean": "avgpool",
}

_MODULE_MAP: Dict[type, str] = {
    nn.Conv2d: "conv",
    nn.Linear: "linear",
    nn.ReLU: "relu",
    nn.MaxPool2d: "maxpool",
    nn.AdaptiveAvgPool2d: "adaptive_avg_pool",
    nn.Dropout: "dropout",
    nn.Softmax: "softmax",
    nn.BatchNorm2d: "batchnorm",
    nn.Flatten: "flatten",
    nn.LayerNorm: "layernorm",
    nn.GELU: "gelu",
    AtallaSdpa: "atalla_sdpa",
}


def _resolve_module_type(gm: GraphModule, node: Node) -> Optional[str]:
    """Resolve a call_module node's target to an Atalla op name."""
    parts = node.target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    for cls, name in _MODULE_MAP.items():
        if isinstance(mod, cls):
            return name
    return None


def _get_module(gm: GraphModule, target: str) -> nn.Module:
    parts = target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p)
    return mod


def normalize_ops(gm: GraphModule) -> GraphModule:
    """Canonicalize operations and tag each node with its Atalla primitive."""
    for node in gm.graph.nodes:
        op_name = None

        if node.op == "call_function":
            if node.target == torch.ops.aten.t.default:
                shape = get_node_shape(node.args[0])
                if shape is None:
                    raise ValueError("aten.t normalization requires shape propagation first")
                if len(shape) > 2:
                    raise ValueError("aten.t only supports tensors with at most two dimensions")
                # t() swaps matrix axes, but is an identity for vectors/scalars.
                node.target = torch.ops.aten.transpose.int
                node.args = (node.args[0], 0, 1 if len(shape) == 2 else 0)
            op_name = _OP_MAP.get(node.target)
            if op_name is None and isinstance(node.target, torch._ops.OpOverload):
                # Preserve the exact operator/overload for kernels not implemented
                # yet. In particular, addmm has (bias, input, weight) arguments:
                # labeling it "linear" would mislead the existing tile planner.
                op_name = str(node.target)
            elif op_name is None and hasattr(node.target, "__name__"):
                fname = node.target.__name__.lower()
                if "addmm" in fname:
                    op_name = "linear"
                elif "matmul" in fname or "mm" in fname:
                    op_name = "matmul"
                elif gm.meta.get("include_backward"):
                    op_name = node.target.__name__

        elif node.op == "call_method":
            op_name = _METHOD_MAP.get(node.target)

        elif node.op == "call_module":
            op_name = _resolve_module_type(gm, node)

        elif node.op in ("placeholder", "get_attr", "output"):
            op_name = node.op

        node.meta["atalla_op"] = op_name

    gm.graph.lint()
    gm.recompile()
    return gm


def capture(model: nn.Module, example_input: torch.Tensor) -> GraphModule:
    """Trace, propagate shapes, and normalize ops."""
    model = model.bfloat16()
    example_input = example_input.bfloat16()

    gm = symbolic_trace(model)
    ShapeProp(gm).propagate(example_input)
    gm = normalize_ops(gm)
    return gm


def get_node_shape(node: Node) -> Optional[Tuple[int, ...]]:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return None
    if hasattr(tm, "shape"):
        return tuple(int(d) for d in tm.shape)
    if isinstance(tm, tuple) and all(isinstance(d, int) for d in tm):
        return tuple(int(d) for d in tm)
    return None
