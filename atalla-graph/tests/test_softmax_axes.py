"""Packed softmax execution must restore axes before feeding the next node."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_graph import build_graph, run_validate


@pytest.mark.parametrize('axis', [0, 1, -2, -1])
def test_chained_softmax_preserves_output_axes(axis, tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x):
            return F.softmax(x, dim=axis) * 2.0

    x = torch.tensor(np.random.default_rng(17).normal(size=(2, 3, 5)), dtype=torch.float32)
    model = Model()
    gm = build_graph(model, x, verbose=False)
    report = run_validate(gm, model, x, str(tmp_path), verbose=False,
                          strict_end_to_end=True, validate_inputs='chained')
    expected = model(x.bfloat16()).float().numpy()
    np.testing.assert_allclose(report['emulator_output'], expected, atol=.01, rtol=.03)
    assert report['strict_pass']
    assert report['stats']['emulated'] == 2
    assert report['stats']['numpy'] == 0
    assert report['stats']['reference_fallback'] == 0
