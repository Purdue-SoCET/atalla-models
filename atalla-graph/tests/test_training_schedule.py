"""Execute captured training kernels with live inputs in the device simulator."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codegen.kernel_schedule import build_schedule, schedule_issues
from run_graph import build_graph, load_model, verify_schedule


def verify(module, x, path, *, training=False, target=None, change_input=True):
    gm = build_graph(module, x, verbose=False, include_backward=training,
                     example_target=target, loss_fn=nn.MSELoss() if training else None)
    assert not schedule_issues(gm), schedule_issues(gm)
    name = next(n.target for n in gm.graph.nodes if n.op == "placeholder")
    inputs = {name: x.bfloat16()}
    if training:
        inputs["target"] = target
    schedule = build_schedule(gm, inputs)
    # Change exported input after generation to detect cached host results.
    changed = (x * -0.5 + 0.125).bfloat16() if change_input else x.bfloat16()
    address = schedule.manifest["inputs"][name]["address"]
    for index, value in enumerate(changed.float().flatten()):
        schedule.dram.bf16(address + 2 * index, float(value))
    schedule.write(str(path))
    report = verify_schedule(schedule, gm, changed, str(path), verbose=False, expected_output=target)
    assert report["passed"], report
    return report


@pytest.mark.parametrize("model_name", ["alexnet_small", "layernorm_smoke", "vit_micro"])
def test_training_models_have_complete_adapters(model_name):
    torch.manual_seed(42)
    model, x, target = load_model(model_name)
    gm = build_graph(model, x, verbose=False, include_backward=True,
                     example_target=target, loss_fn=nn.MSELoss())
    assert not schedule_issues(gm), schedule_issues(gm)
    schedule = build_schedule(gm, {"x": x.bfloat16(), "target": target})
    assert "schedule_read_bf16" in schedule.source
    assert schedule.manifest["outputs"][2]


@pytest.mark.parametrize("normalized", [(7,), (2, 4)])
def test_native_layernorm_training(tmp_path, normalized):
    torch.manual_seed(4)
    x = torch.randn(2, *normalized)
    verify(nn.LayerNorm(normalized), x, tmp_path, training=True, target=torch.randn_like(x))


@pytest.mark.parametrize("mask", [(True, True, True), (False, True, False)])
def test_convolution_backward_masks_stride_padding_dilation(tmp_path, mask):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(2, 1, 2, 2).bfloat16())
            self.register_buffer("grad", torch.randn(1, 2, 2, 2).bfloat16())

        def forward(self, x):
            return torch.ops.aten.convolution_backward.default(
                self.grad, x, self.weight, [2], [2, 2], [1, 1], [2, 2],
                False, [0, 0], 1, list(mask))

    torch.manual_seed(7)
    verify(Model(), torch.randn(1, 1, 4, 4), tmp_path)


def test_convolution_backward_batch_and_groups(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(2, 1, 2, 2).bfloat16())
            self.register_buffer("grad", torch.randn(2, 2, 2, 2).bfloat16())

        def forward(self, x):
            return torch.ops.aten.convolution_backward.default(
                self.grad, x, self.weight, [2], [1, 1], [0, 0], [1, 1],
                False, [0, 0], 2, [True, True, True])

    torch.manual_seed(9)
    verify(Model(), torch.randn(2, 2, 3, 3), tmp_path)


def test_native_layernorm_without_affine_and_masked_gradients(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("grad", torch.randn(2, 5).bfloat16())

        def forward(self, x):
            y, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [5], None, None, 1e-5)
            dx, dg, db = torch.ops.aten.native_layer_norm_backward.default(
                self.grad, x, [5], mean, rstd, None, None, [True, False, False])
            return y, mean, rstd, dx, dg, db

    torch.manual_seed(10)
    verify(Model(), torch.randn(2, 5), tmp_path)


@pytest.mark.parametrize("sign", [-1, 1])
def test_maxpool_indices_overlap_ties_padding_and_backward(tmp_path, sign):
    class Model(nn.Module):
        def forward(self, x):
            pooled, indices = torch.ops.aten.max_pool2d_with_indices.default(
                x, [2, 2], [1, 1], [1, 1], [1, 1], True)
            grad = torch.ops.aten.max_pool2d_with_indices_backward.default(
                pooled, x, [2, 2], [1, 1], [1, 1], [1, 1], True, indices)
            return pooled, indices, grad

    x = torch.tensor([[[[-2.0, -2.0, -4.0], [-3.0, -1.0, -1.0], [-4.0, -1.0, -5.0]]]])
    report = verify(Model(), x * sign, tmp_path)
    indices = report["outputs"][1]
    assert report["comparisons"][indices["node"]]["max_abs_error"] == 0


@pytest.mark.parametrize("dims", [(0,), (0, 2), (1,), (-1,)])
def test_sum_large_reductions_and_partial_columns(tmp_path, dims):
    class Model(nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dims, keepdim=True)

    torch.manual_seed(12)
    verify(Model(), torch.randn(35, 2, 3), tmp_path)


def test_maxpool_signed_zero_does_not_hide_positive_maximum(tmp_path):
    class Model(nn.Module):
        def forward(self, x):
            return torch.ops.aten.max_pool2d_with_indices.default(x, [2, 2])

    x = torch.tensor([[[[-0.0, 2.0], [1.0, 0.0]]]])
    verify(Model(), x, tmp_path, change_input=False)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_backward_axes(tmp_path, dim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("grad", torch.randn(2, 3, 4).bfloat16())

        def forward(self, x):
            y = torch.softmax(x, dim=dim)
            return torch.ops.aten._softmax_backward_data.default(self.grad, y, dim, torch.bfloat16)

    torch.manual_seed(8)
    verify(Model(), torch.randn(2, 3, 4), tmp_path)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward(tmp_path, approximate):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("grad", torch.linspace(-1, 1, 17).bfloat16())

        def forward(self, x):
            return torch.ops.aten.gelu_backward.default(self.grad, x, approximate=approximate)

    verify(Model(), torch.linspace(-5, 5, 17), tmp_path)


def test_sum_integer_output_is_rejected_before_generation():
    class Model(nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=0, dtype=torch.int64)

    gm = build_graph(Model(), torch.ones(2, 3), verbose=False)
    issues = schedule_issues(gm)
    assert any("unsupported dtype torch.int64" in issue for issue in issues)
    with pytest.raises(NotImplementedError, match="unsupported dtype torch.int64"):
        build_schedule(gm, {"x": torch.ones(2, 3).bfloat16()})
