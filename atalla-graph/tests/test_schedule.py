"""End-to-end tests for the single-program schedule (requires compiler/simulator)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codegen.kernel_schedule import build_schedule
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
