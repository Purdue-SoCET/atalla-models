"""Chained validation must never replace execution results with golden values."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import ShapeProp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_graph as runner


def graph_for(target, x, **kwargs):
    graph = Graph()
    inp = graph.placeholder('x')
    result = graph.call_function(target, (inp,), kwargs)
    result.meta['atalla_op'] = str(target)
    graph.output(result)
    gm = GraphModule({}, graph)
    ShapeProp(gm).propagate(x)
    return gm


def test_alias_preserves_error_instead_of_resetting_to_reference(tmp_path, monkeypatch):
    x = torch.tensor([[1., -2.]])
    gm = graph_for(torch.ops.aten.alias.default, x)
    # Inject a disagreement between the actual input and golden values.
    monkeypatch.setattr(runner, 'extract_input_data', lambda *a: {
        'x': np.zeros((1, 2)), 'alias_default': np.zeros((1, 2)),
        'output': np.zeros((1, 2))})
    report = runner.run_validate(gm, torch.nn.Identity(), x, str(tmp_path),
                                 verbose=False, strict_end_to_end=True)
    np.testing.assert_array_equal(report['emulator_output'], [[1., -2.]])
    assert report['strict_pass'] is False
    assert report['stats']['reference_fallback'] == 0


@pytest.mark.parametrize('target,kwargs', [
    (torch.ops.aten._to_copy.default, {'dtype': torch.bfloat16}),
    (torch.ops.aten._to_copy.default, {'dtype': torch.float32}),
    (torch.ops.aten.ones_like.default, {}),
])
def test_host_operations_use_actual_cached_values(tmp_path, target, kwargs):
    x = torch.tensor([[1.00390625, -2.015625]])
    gm = graph_for(target, x, **kwargs)
    report = runner.run_validate(gm, torch.nn.Identity(), x, str(tmp_path), verbose=False)
    expected = target(x.bfloat16(), **kwargs).float().numpy()
    np.testing.assert_array_equal(report['emulator_output'], expected)
    assert report['stats']['host'] == 1


def test_unknown_operation_fails_chained_but_is_explicit_in_oracle(tmp_path):
    x = torch.tensor([[1., 2.]])
    gm = graph_for(torch.ops.aten.sin.default, x)
    with pytest.raises(NotImplementedError, match='sin'):
        runner.run_validate(gm, torch.nn.Identity(), x, str(tmp_path), verbose=False)
    report = runner.run_validate(gm, torch.nn.Identity(), x, str(tmp_path),
                                 verbose=False, validate_inputs='oracle')
    assert report['stats']['reference_fallback'] == 1


def test_bf16_conversion_rounds_actual_fp32_values():
    x = torch.tensor([[1.00390625, 1.01171875]])
    gm = graph_for(torch.ops.aten._to_copy.default, x, dtype=torch.bfloat16)
    node = next(n for n in gm.graph.nodes if n.op == 'call_function')
    # Both are exact halfway cases: BF16 round-to-nearest, ties-to-even.
    result = runner._execute_host_tensor_op(node, {'x': x.numpy()})
    np.testing.assert_array_equal(result, [[1.0, 1.015625]])


def test_scalar_ones_is_not_the_input_value(tmp_path):
    x = torch.tensor(7.)
    gm = graph_for(torch.ops.aten.ones_like.default, x)
    report = runner.run_validate(gm, torch.nn.Identity(), x, str(tmp_path), verbose=False)
    assert report['emulator_output'].shape == ()
    assert report['emulator_output'].item() == 1.
