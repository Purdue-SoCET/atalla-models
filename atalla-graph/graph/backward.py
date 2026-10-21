"""Capture a lowered forward graph, scalar loss, and parameter gradients together."""
from __future__ import annotations

from typing import Callable

import torch
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx


def capture_backward(
    forward: GraphModule,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    example_target: torch.Tensor,
    loss_fn: Callable,
) -> GraphModule:
    """Keep weights as get_attr nodes; expose only data and target as inputs.

    The result returns (prediction, loss, {original_parameter_name: gradient}).
    Frozen parameters are excluded and unused parameters have gradient None.
    Gradients are returned explicitly, without changing parameter .grad fields.
    """
    named_params = tuple(
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    )
    if not named_params:
        raise ValueError("Backward capture requires at least one trainable parameter")
    params = tuple(p for _, p in named_params)

    def forward_loss_backward(x, target):
        prediction = forward(x)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Backward capture currently requires a single tensor prediction")
        # CPU BF16 MSE backward is unavailable. Keep model/gradients in BF16,
        # but evaluate the loss in FP32 (integer class labels stay integers).
        loss_target = target.float() if target.is_floating_point() else target
        loss = loss_fn(prediction.float(), loss_target)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor (use a reduction)")
        if not loss.requires_grad:
            raise ValueError("loss_fn must depend on trainable model parameters")
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        return prediction, loss, {
            name: grad for (name, _), grad in zip(named_params, gradients)
        }

    with torch.enable_grad():
        joint = make_fx(forward_loss_backward)(example_input, example_target)

    # Give callers stable argument names instead of make_fx's x_1/target_1.
    placeholders = [n for n in joint.graph.nodes if n.op == "placeholder"]
    for node, name in zip(placeholders, ("x", "target")):
        node.name = node.target = name

    parameter_names = {id(p): name for name, p in model.named_parameters()}
    buffer_names = {id(b): name for name, b in model.named_buffers()}
    for node in joint.graph.nodes:
        if node.op != "get_attr":
            continue
        value = joint
        for part in node.target.split("."):
            value = getattr(value, part)
        if id(value) in parameter_names:
            node.meta["parameter_name"] = parameter_names[id(value)]
        elif id(value) in buffer_names:
            node.meta["buffer_name"] = buffer_names[id(value)]

    # Label stages by dependencies, without relying on operator-name heuristics:
    # matrix multiplication occurs in both the forward and backward portions.
    output = next(n for n in joint.graph.nodes if n.op == "output")
    prediction_node, loss_node, _ = output.args[0]

    def ancestors(root):
        found = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if isinstance(node, Node) and node not in found:
                found.add(node)
                pending.extend(node.all_input_nodes)
        return found

    forward_nodes = ancestors(prediction_node)
    loss_nodes = ancestors(loss_node) - forward_nodes
    for node in joint.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            phase = node.op
        else:
            phase = "forward" if node in forward_nodes else (
                "loss" if node in loss_nodes else "backward"
            )
        node.meta["phase"] = phase

    joint.meta["include_backward"] = True
    joint.meta["gradient_names"] = [name for name, _ in named_params]
    joint.graph.lint()
    joint.recompile()
    return joint
