"""Compile and emulate attention against independent PyTorch BF16 semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.fx import Graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen import c_emitter as emit
from graph.tile_planner import TileConfig
from run_graph import _read_bf16, _run_emulator


@pytest.mark.parametrize('case', ['nonuniform_scores', 'nonsymmetric_values', 'random'])
def test_flash_sdpa_layout(case, tmp_path):
    if case == 'nonuniform_scores':
        q = np.zeros((32, 32), dtype=np.float32)
        q[:, 0] = 1
        k = np.zeros_like(q)
        k[:, 0] = np.linspace(-2, 2, 32)
        v = np.eye(32, dtype=np.float32)
    elif case == 'nonsymmetric_values':
        q = k = np.zeros((32, 32), dtype=np.float32)
        v = np.broadcast_to(np.arange(32, dtype=np.float32)[:, None], (32, 32)).copy()
    else:
        rng = np.random.default_rng(7)
        q, k, v = [rng.normal(0, .5, (32, 32)).astype(np.float32) for _ in range(3)]
    tq, tk, tv = [torch.tensor(x).bfloat16().unsqueeze(0) for x in (q, k, v)]
    expected = (torch.softmax((tq @ tk.transpose(-2, -1)) * (32 ** -.5), dim=-1)
                @ tv).float().numpy()
    graph = Graph()
    args = [graph.placeholder(name) for name in ('q', 'k', 'v')]
    node = graph.call_function(torch.nn.functional.scaled_dot_product_attention, tuple(args))
    cache = {node.name: value.float().numpy() for node, value in zip(args, (tq, tk, tv))}
    emission = emit.emit_sdpa(node, None, cache, TileConfig(
        'atalla_sdpa', dict(B=1, N=32, D=32, use_flash=True)))
    assert not emission.skip_emulator
    emit.compile_and_assemble(emission, str(tmp_path), 'sdpa')
    image = tmp_path / 'sdpa.in'
    image.write_text(emit.render_in_file(emission))
    memory, _ = _run_emulator(str(image), str(tmp_path), 'sdpa')
    actual = _read_bf16(memory, emission.output_addr, emission.output_elements).reshape(1, 32, 32)
    if case == 'nonsymmetric_values':
        # Uniform attention must average V rows, not V columns; result is exact.
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=.005, rtol=.03)
