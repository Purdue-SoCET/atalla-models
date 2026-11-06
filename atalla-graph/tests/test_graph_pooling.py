"""Graph maxpool must emit final values without a host horizontal reduction."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.c_emitter import emit_maxpool
from graph.fx_capture import capture
from graph.tile_planner import plan_tiles
import run_graph as runner


class PoolThenAdd(torch.nn.Module):
    def __init__(self, options, functional=False):
        super().__init__()
        self.options = options
        self.functional = functional
        self.pool = torch.nn.MaxPool2d(**options)

    def forward(self, x):
        y = F.max_pool2d(x, **self.options) if self.functional else self.pool(x)
        return y + y


@pytest.mark.parametrize('functional', [False, True])
@pytest.mark.parametrize('shape,options', [
    ((2, 2, 5, 35), dict(kernel_size=2)),
    ((2, 2, 5, 35), dict(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1), dilation=(2, 1), ceil_mode=True)),
    ((1, 2, 5, 9), dict(kernel_size=3, stride=None, padding=1, ceil_mode=True)),
    ((1, 1, 6, 10), dict(kernel_size=2, stride=3, ceil_mode=True)),
    ((2, 5, 35), dict(kernel_size=2)),
])
def test_complete_pooling_runs_in_emulator(shape, options, functional, tmp_path):
    x = ((torch.arange(np.prod(shape)).reshape(shape) % 29 - 14) / 8).bfloat16()
    model = PoolThenAdd(options, functional)
    gm = plan_tiles(capture(model, x))
    pool = next(node for node in gm.graph.nodes if node.meta.get('atalla_op') == 'maxpool')
    emission = emit_maxpool(pool, x.float().numpy(), pool.meta['tile_config'])
    assert not emission.skip_emulator
    assert emission.maxpool_post is None
    expected = model(x).float().numpy()
    assert emission.output_shape == expected.shape
    assert emission.output_elements == expected.size
    assert emission.layer_metrics['kernel_source'] == 'pooling_nchw_max.c'
    report = runner.run_validate(gm, model, x, str(tmp_path), verbose=False,
                                 validate_inputs='chained', strict_end_to_end=True)
    np.testing.assert_array_equal(report['emulator_output'], expected)
    assert report['strict_pass']
    assert report['stats']['emulated'] == 2
    assert report['stats']['numpy'] == report['stats']['host'] == report['stats']['reference_fallback'] == 0
    metric = next(k for k in report['kernel_metrics'] if k['op'] == 'maxpool')
    assert metric['backend'] == 'emulator'
    assert metric['elems'] == expected.size


def test_pooling_uses_execution_values_when_reference_is_wrong(tmp_path, monkeypatch):
    x = torch.arange(1, 1 + 4 * 35).reshape(1, 1, 4, 35).bfloat16()
    model = PoolThenAdd(dict(kernel_size=2))
    gm = plan_tiles(capture(model, x))
    pool = next(node for node in gm.graph.nodes if node.meta.get('atalla_op') == 'maxpool')
    original = runner.extract_input_data

    def wrong_reference(*args, **kwargs):
        values = original(*args, **kwargs)
        values[pool.name] = np.zeros_like(values[pool.name])
        return values

    monkeypatch.setattr(runner, 'extract_input_data', wrong_reference)
    report = runner.run_validate(gm, model, x, str(tmp_path), verbose=False,
                                 validate_inputs='chained', strict_end_to_end=True)
    np.testing.assert_array_equal(report['emulator_output'], model(x).float().numpy())
    assert report['stats']['reference_fallback'] == report['stats']['numpy'] == 0
    assert report['strict_pass']


@pytest.mark.parametrize('functional', [False, True])
def test_pooling_indices_are_explicitly_unsupported(functional):
    class IndexPool(torch.nn.Module):
        def forward(self, x):
            return F.max_pool2d(x, 2, return_indices=True)

    model = IndexPool() if functional else torch.nn.Sequential(torch.nn.MaxPool2d(2, return_indices=True))
    gm = capture(model, torch.ones(1, 1, 4, 4))
    with pytest.raises(NotImplementedError, match='return_indices'):
        plan_tiles(gm)
