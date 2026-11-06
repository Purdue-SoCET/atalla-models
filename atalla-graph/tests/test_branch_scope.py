"""The kernel branch has a single-input forward validation interface."""
from pathlib import Path
import sys

import pytest
import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import ShapeProp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.tile_planner import plan_tiles
from run_graph import run_validate


def test_validation_rejects_multiple_placeholders(tmp_path):
    graph = Graph()
    left = graph.placeholder('left')
    right = graph.placeholder('right')
    result = graph.call_function(torch.add, (left, right))
    result.meta['atalla_op'] = 'add'
    graph.output(result)
    module = GraphModule({}, graph)
    x = torch.ones(1, 4)
    ShapeProp(module).propagate(x, x)
    plan_tiles(module)
    with pytest.raises(ValueError, match='exactly one input placeholder'):
        run_validate(module, torch.nn.Identity(), x, str(tmp_path), verbose=False)


def test_forward_json_keeps_strict_metrics_without_training_fields(tmp_path):
    import json
    from graph.fx_capture import capture
    model = torch.nn.Identity()
    x = torch.tensor([[1., -2.]])
    module = plan_tiles(capture(model, x))
    metrics = tmp_path/'metrics.json'
    result = run_validate(module, model, x, str(tmp_path), verbose=False,
                          strict_end_to_end=True, metrics_json_path=str(metrics))
    saved = json.loads(metrics.read_text())
    assert result['strict_pass'] is True
    assert saved['strict_pass'] is True
    assert saved['end_to_end'] == result['end_to_end']
    assert saved['strict_fail_reasons'] == []
    assert 'training_metrics' not in saved
