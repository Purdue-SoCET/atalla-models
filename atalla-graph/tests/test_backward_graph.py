"""CPU graph-capture tests; accelerator backward kernels are not required.

Run (from repo root):
    python -m pytest atalla-graph/tests -q
or (from atalla-graph/):
    python -m unittest discover -s tests -v

``load_model`` returns ``(model, example_input, expected_output_or_None)``;
``_load`` below also tolerates the older 2-tuple form and skips models that
the current branch does not register (``ValueError: Unknown model``).
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_graph import build_graph, load_model, run_schedule, run_validate
from graph.fx_capture import get_node_shape
from graph.lower_modules import lower_linear_modules
from torch.fx import symbolic_trace


def _load(name: str):
    """Return (model, example_input) for *name*; skip if the model is not registered."""
    try:
        loaded = load_model(name)
    except ValueError as exc:
        if "Unknown model" in str(exc):
            raise unittest.SkipTest(f"model {name!r} not registered in run_graph.load_model")
        raise
    model, x = loaded[0], loaded[1]
    return model, x


class BackwardGraphTests(unittest.TestCase):
    def assert_gradients(self, model, x, target, loss_fn):
        gm = build_graph(model, x, verbose=False, include_backward=True,
                         example_target=target, loss_fn=loss_fn)
        # The existing BF16 lowering splits fused linear into matmul + add.
        # Its extra rounding is part of the compiler's forward semantics.
        reference_forward = build_graph(model, x, verbose=False)
        # Capture does not populate .grad fields or move parameters to inputs.
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual([n.target for n in gm.graph.nodes if n.op == "placeholder"],
                         ["x", "target"])
        self.assertTrue(any(n.op == "get_attr" and "parameter_name" in n.meta
                            for n in gm.graph.nodes))
        self.assertTrue(all(n.meta.get("atalla_op") for n in gm.graph.nodes))
        self.assertTrue(any(n.meta.get("phase") == "backward" for n in gm.graph.nodes))

        # Reuse capture with different data to catch baked-in values.
        for inputs in (x.bfloat16(), (x + 0.25).bfloat16()):
            prediction, loss, gradients = gm(inputs, target)
            reference = reference_forward(inputs)
            loss_target = target.float() if target.is_floating_point() else target
            reference_loss = loss_fn(reference.float(), loss_target)
            params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            expected = torch.autograd.grad(reference_loss, [p for _, p in params],
                                           allow_unused=True)
            torch.testing.assert_close(prediction, reference)
            torch.testing.assert_close(loss, reference_loss)
            self.assertEqual(list(gradients), [name for name, _ in params])
            for (name, _), grad in zip(params, expected):
                if grad is None:
                    self.assertIsNone(gradients[name])
                else:
                    torch.testing.assert_close(gradients[name], grad)

        for node in gm.graph.nodes:
            if node.op == "output":
                continue
            self.assertIn("dram_addr", node.meta)
            total = sum(output["dram_bytes"] for output in node.meta["dram_outputs"])
            self.assertLessEqual(total, node.meta["dram_bytes"])
        return gm

    def test_existing_forward_path(self):
        model, x = _load("basic")
        gm = build_graph(model, x, verbose=False)
        self.assertFalse(gm.meta.get("include_backward", False))
        reference = lower_linear_modules(symbolic_trace(model))
        torch.testing.assert_close(gm(x.bfloat16()), reference(x.bfloat16()))
        self.assertEqual(len([n for n in gm.graph.nodes if n.op == "placeholder"]), 1)

    def test_residual_and_bias_gradients(self):
        torch.manual_seed(1)
        model, x = _load("basic")
        gm = self.assert_gradients(model, x, torch.randn(1, 32), F.mse_loss)
        # threshold_backward has a tile planner (_plan_threshold_backward), so it
        # must not be reported as pending; the work list itself must exist.
        self.assertIn("pending_kernel_ops", gm.meta)
        self.assertNotIn("aten.threshold_backward.default", gm.meta["pending_kernel_ops"])
        # Existing weights are live graph attributes, not frozen trace constants.
        with torch.no_grad():
            model.output.bias.add_(1)
        reference = build_graph(model, x, verbose=False)
        torch.testing.assert_close(gm(x.bfloat16(), torch.zeros(1, 32))[0],
                                   reference(x.bfloat16()))

    def test_conv_pool_and_integer_targets(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Linear(8, 3))
        gm = self.assert_gradients(model, torch.randn(2, 1, 6, 6),
                                   torch.tensor([0, 2]), F.cross_entropy)
        tuple_nodes = [n for n in gm.graph.nodes
                       if len(n.meta.get("dram_outputs", [])) > 1]
        self.assertTrue(tuple_nodes)
        self.assertTrue(all(get_node_shape(n) is None for n in tuple_nodes))
        self.assertTrue(any(output["dtype"] == "torch.int64"
                            for n in gm.graph.nodes
                            for output in n.meta.get("dram_outputs", [])))

    def test_attention_and_normalization(self):
        for name in ("layernorm_smoke", "vit_micro"):
            with self.subTest(model=name):
                model, x = _load(name)
                self.assert_gradients(model, x, torch.zeros_like(x), F.mse_loss)

    def test_frozen_unused_and_buffers(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)
                self.bn = nn.BatchNorm1d(3)
                self.unused = nn.Parameter(torch.ones(2))
                self.fc.bias.requires_grad_(False)

            def forward(self, x):
                return self.bn(self.fc(x))

        model = Model()
        gm = self.assert_gradients(model, torch.randn(2, 4), torch.zeros(2, 3), F.mse_loss)
        self.assertNotIn("fc.bias", gm.meta["gradient_names"])
        self.assertIn("unused", gm.meta["gradient_names"])
        self.assertTrue(any("buffer_name" in n.meta for n in gm.graph.nodes))
        torch.testing.assert_close(model.bn.running_mean, torch.zeros_like(model.bn.running_mean))

    def test_validation_and_backend_boundaries(self):
        model, x = _load("basic")
        with self.assertRaisesRegex(ValueError, "example_target and loss_fn"):
            build_graph(model, x, verbose=False, include_backward=True)
        with self.assertRaisesRegex(ValueError, "scalar tensor"):
            build_graph(model, x, verbose=False, include_backward=True,
                        example_target=x, loss_fn=lambda p, t: (p - t).square())
        with torch.no_grad():
            gm = build_graph(model, x, verbose=False, include_backward=True,
                             example_target=x, loss_fn=F.mse_loss)
            prediction, loss, gradients = gm(x.bfloat16(), x)
            self.assertTrue(gradients)
        # Schedule emission of combined graphs is still rejected.
        with self.assertRaisesRegex(NotImplementedError, "backward kernels"):
            run_schedule(gm, x, "unused")
        # Gradient validation is implemented (--training); it requires the target.
        with self.assertRaisesRegex(ValueError, "expected_output"):
            run_validate(gm, model, x, "unused")


if __name__ == "__main__":
    unittest.main()
