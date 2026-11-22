"""End-to-end tests for the single-program schedule (requires compiler/simulator)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codegen.kernel_schedule import build_schedule, schedule_issues
from run_graph import build_graph, load_model, run_schedule, verify_schedule


def test_basic_schedule_uses_changed_input(tmp_path):
    torch.manual_seed(42)
    model, x, _ = load_model("basic")
    gm = build_graph(model, x, verbose=False)
    schedule = build_schedule(gm, {"x": x.bfloat16()})
    # Change only the exported input buffer after code generation. A program
    # containing precomputed reference intermediates would fail this comparison.
    changed = (x * -1.5 + 0.25).bfloat16()
    address = schedule.manifest["inputs"]["x"]["address"]
    for index, value in enumerate(changed.float().flatten()):
        schedule.dram.bf16(address + 2 * index, float(value))
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, changed, str(tmp_path), verbose=False)
    assert report["passed"], report
    assert len(report["comparisons"]) == 10
    assert all(c["max_abs_error"] == 0 for c in report["comparisons"].values())
    arrays = np.load(tmp_path / "schedule_outputs.npz")
    assert not np.array_equal(arrays["result_add_3"], gm(x.bfloat16()).detach().float().numpy())


def test_nonsquare_gemm_padding_and_batch_broadcast(tmp_path):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    x = torch.randn(2, 5)
    gm = build_graph(model, x, verbose=False)
    schedule = build_schedule(gm, {"input": x.bfloat16()})
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, x, str(tmp_path), verbose=False)
    assert report["passed"], report


def test_incorrect_kernel_result_fails_verification(tmp_path):
    class Model(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    x = torch.ones(1, 32)
    gm = build_graph(Model(), x, verbose=False)
    schedule = build_schedule(gm, {"x": x.bfloat16()})
    # Omit computation: the unwritten output remains zero and must fail.
    schedule.source = schedule.source.replace("    kernel_relu();", "")
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, x, str(tmp_path), verbose=False)
    assert not report["passed"]
    assert report["comparisons"]["relu"]["rel_l2_error"] > 0.9


def test_schedule_lists_calls(tmp_path, capsys):
    torch.manual_seed(42)
    model, x, _ = load_model("basic")
    gm = build_graph(model, x, verbose=False)
    run_schedule(gm, x, str(tmp_path))
    output = capsys.readouterr().out
    assert "matmul -> kernel_matmul()" in output
    assert "kernels/gemm.py" in output
    assert "add_3: add -> kernel_add_3()" in output


def test_unsupported_operation_is_not_silently_omitted():
    class Model(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    x = torch.randn(1, 32)
    gm = build_graph(Model(), x, verbose=False)
    with pytest.raises(NotImplementedError, match="No schedule adapter"):
        build_schedule(gm, {"x": x.bfloat16()})


def test_conv_pool_linear_uses_live_input(tmp_path):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2), nn.Flatten(), nn.Linear(12, 3))
    x = torch.randn(1, 2, 4, 4)
    gm = build_graph(model, x, verbose=False)
    schedule = build_schedule(gm, {"input": x.bfloat16()})
    changed = (x * -0.5 + 0.25).bfloat16()
    address = schedule.manifest["inputs"]["input"]["address"]
    for index, value in enumerate(changed.float().flatten()):
        schedule.dram.bf16(address + 2 * index, float(value))
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, changed, str(tmp_path), verbose=False)
    assert report["passed"], report


def test_conv_stride_padding_without_bias(tmp_path):
    torch.manual_seed(3)
    model = nn.Conv2d(2, 3, 3, stride=2, padding=1, bias=False)
    x = torch.randn(1, 2, 5, 7)
    gm = build_graph(model, x, verbose=False)
    schedule = build_schedule(gm, {"input": x.bfloat16()})
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, x, str(tmp_path), verbose=False)
    assert report["passed"], report


@pytest.mark.parametrize("pool", [2, 3])
def test_maxpool_negative_values_and_overlapping_windows(tmp_path, pool):
    class Model(nn.Module):
        def forward(self, x):
            return torch.nn.functional.max_pool2d(x, pool, stride=1)

    torch.manual_seed(11)
    x = -torch.rand(1, 2, 4, 5)
    gm = build_graph(Model(), x, verbose=False)
    schedule = build_schedule(gm, {"x": x.bfloat16()})
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, x, str(tmp_path), verbose=False)
    assert report["passed"], report
    assert all(c["max_abs_error"] == 0 for c in report["comparisons"].values())


def test_grouped_conv_is_rejected():
    model = nn.Conv2d(2, 2, 3, groups=2)
    x = torch.randn(1, 2, 4, 4)
    gm = build_graph(model, x, verbose=False)
    with pytest.raises(NotImplementedError, match="groups=1"):
        build_schedule(gm, {"input": x.bfloat16()})


def test_conv_partial_large_gemm_tile_is_rejected():
    model = nn.Conv2d(1, 1, 1)
    x = torch.randn(1, 1, 7, 7)  # M=49 exceeds one GEMM tile and has a partial tail.
    gm = build_graph(model, x, verbose=False)
    with pytest.raises(NotImplementedError, match="complete M/N tiles"):
        build_schedule(gm, {"input": x.bfloat16()})


def _verify_module(module, x, directory):
    gm = build_graph(module, x, verbose=False)
    name = next(n.target for n in gm.graph.nodes if n.op == "placeholder")
    schedule = build_schedule(gm, {name: x.bfloat16()})
    schedule.write(str(directory))
    report = verify_schedule(schedule, gm, x, str(directory), verbose=False)
    assert report["passed"], report
    return report


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_axes(tmp_path, dim):
    torch.manual_seed(4)
    _verify_module(nn.Softmax(dim=dim), torch.randn(3, 8) * 0.25, tmp_path)


@pytest.mark.parametrize("normalized", [(32,), (2, 16)])
def test_layernorm_shapes(tmp_path, normalized):
    torch.manual_seed(4)
    _verify_module(nn.LayerNorm(normalized), torch.randn(2, *normalized), tmp_path)


def test_gelu_standalone_kernel(tmp_path):
    _verify_module(nn.GELU(approximate="tanh"), torch.linspace(-3, 3, 66).reshape(2, 33), tmp_path)


def test_batchnorm_padded_channels(tmp_path):
    torch.manual_seed(4)
    model = nn.BatchNorm2d(2)
    with torch.no_grad():
        model.running_mean.copy_(torch.tensor([0.25, -0.5]))
        model.running_var.copy_(torch.tensor([0.5, 2.0]))
        model.weight.copy_(torch.tensor([0.5, 1.25]))
        model.bias.copy_(torch.tensor([0.125, -0.25]))
    _verify_module(nn.Sequential(model), torch.randn(1, 2, 3, 6), tmp_path)


def test_avgpool_standalone_kernel(tmp_path):
    x = torch.arange(64).reshape(1, 2, 4, 8).float() / 8
    _verify_module(nn.AvgPool2d(2), x, tmp_path)


def test_global_adaptive_pool(tmp_path):
    x = torch.arange(64).reshape(1, 2, 4, 8).float() / 16
    _verify_module(nn.AdaptiveAvgPool2d(1), x, tmp_path)


def test_nearest_upsample(tmp_path):
    x = torch.arange(-12, 12).reshape(1, 2, 3, 4).float() / 8
    _verify_module(nn.Upsample(scale_factor=2, mode="nearest"), x, tmp_path)


def test_attention_lowering(tmp_path):
    from model.atalla_ops import AtallaSdpa

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = AtallaSdpa(8)

        def forward(self, x):
            return self.attn(x, x, x)

    torch.manual_seed(3)
    _verify_module(Model(), torch.randn(1, 8, 8) * 0.5, tmp_path)


@pytest.mark.parametrize("op", ["add", "mul", "relu"])
def test_elementwise_multiple_tiles_and_tail(tmp_path, op):
    class Model(nn.Module):
        def forward(self, x):
            if op == "add":
                return x + 0.5
            if op == "mul":
                return x * -0.5
            return torch.relu(x)

    x = torch.linspace(-2, 2, 1089).reshape(33, 33)
    report = _verify_module(Model(), x, tmp_path)
    assert all(c["max_abs_error"] == 0 for c in report["comparisons"].values())


def test_basic_backward_schedule(tmp_path):
    from model.basic import BasicModule
    torch.manual_seed(42)
    model = BasicModule(dim=32, depth=2)
    x, target = torch.randn(2, 32), torch.randn(2, 32)
    gm = build_graph(model, x, verbose=False, include_backward=True,
                     example_target=target, loss_fn=nn.MSELoss())
    schedule = build_schedule(gm, {"x": x.bfloat16(), "target": target})
    schedule.write(str(tmp_path))
    report = verify_schedule(schedule, gm, x, str(tmp_path), verbose=False, expected_output=target)
    assert report["passed"], report
    assert "mse_loss" in report["comparisons"]
    assert len(report["outputs"][2]) == 6


def test_mse_backward_uses_nonunit_seed(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("seed", torch.tensor(0.5))
            self.register_buffer("target", torch.zeros(1, 64))

        def forward(self, x):
            return torch.ops.aten.mse_loss_backward.default(self.seed.float(), x.float(), self.target.float(), 1)

    _verify_module(Model(), torch.linspace(-2, 2, 64).reshape(1, 64), tmp_path)


def test_capabilities_collect_all_missing_operations():
    class Model(nn.Module):
        def forward(self, x):
            return torch.sin(x) + torch.cos(x)

    gm = build_graph(Model(), torch.ones(1, 32), verbose=False)
    issues = schedule_issues(gm)
    assert len(issues) == 2
    assert "sin" in issues[0] and "cos" in issues[1]


def test_captured_layernorm_has_schedule_adapters():
    model = nn.LayerNorm(32)
    x, target = torch.randn(1, 32), torch.zeros(1, 32)
    gm = build_graph(model, x, verbose=False, include_backward=True,
                     example_target=target, loss_fn=nn.MSELoss())
    issues = schedule_issues(gm)
    assert not issues, issues


def test_layout_operations(tmp_path):
    class Model(nn.Module):
        def forward(self, x):
            y = x.permute(2, 0, 1).contiguous().reshape(4, 6)
            return torch.relu(y.unsqueeze(0).expand(2, 4, 6))

    x = torch.arange(-12, 12).reshape(2, 3, 4).float() / 4
    _verify_module(Model(), x, tmp_path)


def test_leading_axis_sum(tmp_path):
    class Model(nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=(0, 1), keepdim=True)

    x = torch.arange(32).reshape(1, 4, 8).float() / 8
    _verify_module(Model(), x, tmp_path)


def test_cast_roundtrip_without_compute_kernels(tmp_path):
    class Model(nn.Module):
        def forward(self, x):
            return x.float().bfloat16()

    _verify_module(Model(), torch.linspace(-2, 2, 33), tmp_path)


def test_captured_conv_uses_existing_forward_adapter():
    model = nn.Sequential(nn.Conv2d(1, 1, 1))
    x, target = torch.randn(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    gm = build_graph(model, x, verbose=False, include_backward=True,
                     example_target=target, loss_fn=nn.MSELoss())
    issues = schedule_issues(gm)
    assert not issues, issues
