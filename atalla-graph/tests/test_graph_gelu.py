"""GELU graph lowering must use vector kernels, never host PyTorch arithmetic."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from torch.fx import Graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.c_emitter import emit_gelu, compile_and_assemble, render_in_file
from graph.tile_planner import TileConfig
from run_graph import _run_emulator, _read_bf16


@pytest.mark.parametrize('approximate', ['none', 'tanh'])
@pytest.mark.parametrize('shape', [(), (0,), (1,), (7,), (33,), (3, 353)])
def test_gelu_executes_vectors_and_matches_pytorch(shape, approximate, tmp_path):
    total = int(np.prod(shape))
    x = torch.linspace(-5, 5, total).reshape(shape).bfloat16()
    graph = Graph()
    inp = graph.placeholder('x')
    node = graph.call_function(torch.nn.functional.gelu, (inp,), {'approximate': approximate})
    node.meta['tensor_meta'] = SimpleNamespace(shape=shape)
    em = emit_gelu(node, x.float().numpy(), TileConfig('gelu', dict(total_elements=total, approximate=approximate)))
    assert not em.skip_emulator, 'graph GELU must execute on Atalla'
    assert em.numpy_result is None
    assert em.output_shape == shape
    compile_and_assemble(em, str(tmp_path), 'gelu')
    image = tmp_path/'gelu.in'
    image.write_text(render_in_file(em))
    memory, unit = _run_emulator(str(image), str(tmp_path), 'gelu')
    actual = _read_bf16(memory, em.output_addr, total).reshape(shape)
    expected = torch.nn.functional.gelu(x, approximate=approximate).float().numpy()
    np.testing.assert_allclose(actual, expected, atol=.02, rtol=.02)
    assert unit.perf_metrics.metrics['dyn_retired_vector_alu'] > 0


def execute_values(x, approximate, tmp_path):
    graph = Graph()
    inp = graph.placeholder('x')
    node = graph.call_function(torch.nn.functional.gelu, (inp,), {'approximate': approximate})
    node.meta['tensor_meta'] = SimpleNamespace(shape=tuple(x.shape))
    em = emit_gelu(node, x.float().numpy(), TileConfig('gelu', dict(total_elements=x.numel(), approximate=approximate)))
    assert not em.skip_emulator and em.numpy_result is None
    compile_and_assemble(em, str(tmp_path), 'gelu')
    image = tmp_path/'gelu.in'
    image.write_text(render_in_file(em))
    memory, unit = _run_emulator(str(image), str(tmp_path), 'gelu')
    assert unit.perf_metrics.metrics['dyn_retired_vector_alu'] > 0
    return _read_bf16(memory, em.output_addr, x.numel()).reshape(tuple(x.shape))


@pytest.mark.parametrize('approximate', ['none', 'tanh'])
def test_gelu_dense_negative_tail_and_boundaries(approximate, tmp_path):
    x = torch.linspace(-8, 8, 4097).bfloat16()
    actual = execute_values(x, approximate, tmp_path)
    expected = torch.nn.functional.gelu(x, approximate=approximate).float().numpy()
    np.testing.assert_allclose(actual, expected, atol=.02, rtol=.02)
    # Negative outputs are small: use a tighter absolute bound than the global one.
    negative = x.float().numpy() < 0
    np.testing.assert_allclose(actual[negative], expected[negative], atol=.002, rtol=.02)
    assert np.max(actual[negative]) <= 0
    assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < .005


@pytest.mark.parametrize('approximate', ['none', 'tanh'])
def test_gelu_tiny_values_signed_zeros_and_nonfinite(approximate, tmp_path):
    values = [-float('inf'), -torch.finfo(torch.bfloat16).max, -8., -4., -3., -2., -1.,
              -1e-8, -1e-30, -0., 0., 1e-30, 1e-8, 1., 2., 3., 4., 8.,
              torch.finfo(torch.bfloat16).max / 2, 2.**127,
              torch.finfo(torch.bfloat16).max, float('inf'), float('nan')]
    x = torch.tensor(values).bfloat16()
    actual = execute_values(x, approximate, tmp_path)
    expected = torch.nn.functional.gelu(x, approximate=approximate).float().numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=.002, rtol=.02)
    np.testing.assert_array_equal(actual[np.isinf(expected)], expected[np.isinf(expected)])
    zeros = x.float().numpy() == 0
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
    tiny = (np.abs(x.float().numpy()) < 1e-7) & ~zeros
    np.testing.assert_allclose(actual[tiny], expected[tiny], atol=0, rtol=.02)


@pytest.mark.parametrize('functional', [False, True])
@pytest.mark.parametrize('approximate', ['none', 'tanh'])
def test_gelu_chained_graph_uses_emulator(functional, approximate, tmp_path):
    from graph.fx_capture import capture
    from graph.tile_planner import plan_tiles
    import run_graph as runner

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = torch.nn.GELU(approximate=approximate)

        def forward(self, x):
            y = (torch.nn.functional.gelu(x, approximate=approximate) if functional else self.activation(x))
            return y + y

    x = torch.linspace(-3.5, 3.5, 210).reshape(2, 3, 35).bfloat16()
    model = Model()
    gm = plan_tiles(capture(model, x))
    node = next(n for n in gm.graph.nodes if n.meta.get('atalla_op') == 'gelu')
    assert node.meta['tile_config'].params['approximate'] == approximate
    report = runner.run_validate(gm, model, x, str(tmp_path), verbose=False,
                                 validate_inputs='chained', strict_end_to_end=True)
    assert report['strict_pass']
    assert report['stats']['emulated'] == 2
    assert report['stats']['numpy'] == report['stats']['host'] == report['stats']['reference_fallback'] == 0
    metric = next(k for k in report['kernel_metrics'] if k['op'] == 'gelu')
    assert metric['backend'] == 'emulator'
    assert metric['kernel_source'] == 'gelu_piecewise.c'
