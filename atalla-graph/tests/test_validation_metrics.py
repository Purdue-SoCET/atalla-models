"""Comparison safety is shared by forward kernels and training validation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_graph import _layer_compare_metrics, _resolve_transpose


@pytest.mark.parametrize('reference,result', [
    (np.ones((1,2)),np.ones(2)),
    (np.ones(2),np.ones(3)),
    (np.array([np.nan]),np.ones(1)),
    (np.ones(1),np.array([np.nan])),
    (np.array([np.inf]),np.ones(1)),
    (np.ones(1),np.array([-np.inf])),
])
def test_invalid_comparisons_are_rejected(reference,result):
    with pytest.raises(ValueError):
        _layer_compare_metrics(reference,result)


@pytest.mark.parametrize('shape', [(3,), (0,)])
def test_identical_zero_outputs_compare_as_equal(shape):
    result = _layer_compare_metrics(np.zeros(shape),np.zeros(shape))
    assert result['cos_sim'] == 1.
    assert result['rel_l2_error'] == 0.
    assert result['max_abs_error'] == 0.


def test_scalar_transpose_has_no_axes():
    assert _resolve_transpose(0,(0,0)) == ()
