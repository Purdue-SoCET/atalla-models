"""Numerical integration tests for shape-independent C kernels and their config ABI."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch.fx import Graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen import c_emitter as emit
from graph.tile_planner import TileConfig
from run_graph import _run_emulator, _read_bf16


def bf16(x):
    return torch.as_tensor(np.asarray(x).copy()).to(torch.bfloat16).float().numpy()


def node_for(shape, args_count=1):
    graph = Graph()
    inputs = [graph.placeholder(f'x{i}') for i in range(args_count)]
    node = graph.call_function(torch.add if args_count == 2 else torch.relu, tuple(inputs))
    node.meta['tensor_meta'] = SimpleNamespace(shape=shape)
    return node, inputs


def execute(emission, tmp_path):
    emit.compile_and_assemble(emission, str(tmp_path), 'runtime')
    image = tmp_path / 'runtime.in'
    image.write_text(emit.render_in_file(emission))
    memory, _ = _run_emulator(str(image), str(tmp_path), 'runtime')
    return _read_bf16(memory, emission.output_addr, emission.output_elements)


@pytest.mark.parametrize('op,total', [('add', 7), ('add', 35), ('add', 2048),
                                      ('mul', 7), ('mul', 1056), ('relu', 7), ('relu', 1056)])
def test_elementwise_runtime_shapes(op, total, tmp_path):
    x = bf16(np.linspace(-2, 2, total, dtype=np.float32))
    y = bf16(np.linspace(1, -1, total, dtype=np.float32))
    if op == 'relu':
        node, _ = node_for((total,))
        emission = emit.emit_relu(node, x, TileConfig(op, dict(total_elements=total, width=min(total, 32))))
        expected = np.maximum(x, 0)
    else:
        node, args = node_for((total,), 2)
        emission = getattr(emit, 'emit_' + op)(node, {args[0].name:x, args[1].name:y},
                                               TileConfig(op, dict(total_elements=total)))
        expected = bf16(x + y if op == 'add' else x * y)
    np.testing.assert_array_equal(execute(emission, tmp_path), expected)


@pytest.mark.parametrize('shape', [(1, 7, 4), (4, 4, 4), (1, 32, 32), (32, 32, 64), (32,64,32)])
def test_gemm_runtime_shapes(shape, tmp_path):
    m, n, k = shape
    # Small integers keep products and sums exact in BF16.
    a = (np.arange(m*k).reshape(m,k) % 3 - 1).astype(np.float32)
    b = (np.arange(k*n).reshape(k,n) % 3 - 1).astype(np.float32)
    node, inputs = node_for((m,n), 2)
    emission = emit.emit_matmul(node, None, a, {inputs[0].name:a, inputs[1].name:b},
                                 TileConfig('matmul', dict(M=m, N=n, K=k)))
    np.testing.assert_array_equal(execute(emission, tmp_path).reshape(m,n), a @ b)


@pytest.mark.parametrize('rows,width', [(1,7),(3,7),(3,32)])
def test_softmax_runtime_shapes(rows, width, tmp_path):
    x = bf16(np.linspace(-1,1,rows*width,dtype=np.float32).reshape(rows,width))
    node, _ = node_for(x.shape)
    emission = emit.emit_softmax(node, x, TileConfig('softmax', dict(num_rows=rows,row_len=width)))
    expected = torch.softmax(torch.tensor(x),dim=-1).numpy()
    np.testing.assert_allclose(execute(emission,tmp_path).reshape(x.shape),
                               expected, atol=.005, rtol=.03)


@pytest.mark.parametrize('width,pool', [(8,2),(8,3),(32,2)])
def test_maxpool_runtime_window(width, pool, tmp_path):
    channels, height, stride = 2, 5, 1
    h_out, w_out = height-pool+1, width-pool+1
    x = (np.arange(channels*height*width).reshape(channels,height,width)%17-8).astype(np.float32)
    node, _ = node_for((1,channels,h_out,w_out))
    emission = emit.emit_maxpool(node, x, TileConfig('maxpool',dict(H=height,W=width,channels=channels,
                                    pool=pool,stride=stride,H_out=h_out,W_out=w_out)))
    raw = execute(emission,tmp_path).reshape(channels,h_out,width)
    expected = np.stack([x[:,r:r+pool,:].max(axis=1) for r in range(h_out)],axis=1)
    np.testing.assert_array_equal(raw,expected)


@pytest.mark.parametrize('groups,features,eps', [(1,32,1e-5),(2,64,.25)])
def test_layernorm_runtime_dimensions(groups, features, eps, tmp_path):
    x = np.tile(np.array([-1.,1.],dtype=np.float32), (groups,features//2))
    node, _ = node_for(x.shape)
    emission = emit.emit_layernorm(node,None,x,{},TileConfig('layernorm',dict(M=groups,D=features,eps=eps)))
    np.testing.assert_allclose(execute(emission,tmp_path).reshape(x.shape),x/np.sqrt(1+eps),atol=.02)
