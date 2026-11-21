"""Named linalg.conv_* kernels against independent goldens on both simulators.

Each case builds one ELF with the compiler; the functional simulator runs the
.in written from it, atalla-sim runs the ELF. The golden walks each op's named
index map directly (not graph/conv.py's canonical strides) with Linalg's
reduction order: per-operation bf16 rounding, or i32 arithmetic modulo 2^32.

Set ATALLA_CONV_CYCLE=0 to skip atalla-sim. Objects are cached per kernel in
ATALLA_OBJ_CACHE (a session temporary directory by default).
"""
import itertools
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.layout import TensorView
from graph.conv import OPS, PLAIN, ConvOp, lower_conv
from conv_runtime import execute_conv

CYCLE = os.environ.get('ATALLA_CONV_CYCLE', '1') != '0'
STANDARD = [n for n, v in OPS.items() if not v[5]]
QUANTIZED = [n for n, v in OPS.items() if v[5]]
NP = {'i8': '<i1', 'i16': '<i2', 'i32': '<i4'}


@pytest.fixture(scope='session', autouse=True)
def object_cache(tmp_path_factory):
    if not os.environ.get('ATALLA_OBJ_CACHE'):
        os.environ['ATALLA_OBJ_CACHE'] = str(tmp_path_factory.mktemp('objcache'))


def bf16_round(x):
    return torch.tensor(np.asarray(x, np.float32)).bfloat16().float().numpy()


def encode(x, dtype):
    if dtype == 'bf16':
        return torch.tensor(np.asarray(x, np.float32)).bfloat16().view(torch.int16).numpy().astype('<i2').tobytes()
    return np.asarray(x).astype(NP[dtype]).tobytes()


def decode(raw, dtype, shape):
    if dtype == 'bf16':
        return (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(raw, NP[dtype]).reshape(shape)


def geometry(name, n=1, g=2, c=3, f=2, out=(3, 4, 5), k=(2, 3, 2)):
    """Logical operand shapes for ``name`` from its axis tokens."""
    rank, i_ax, k_ax, o_ax = OPS[name][:4]
    out, k = out[3 - rank:], k[3 - rank:]
    def size(tok, steps, dil):
        if tok[0] == 'S':
            d = int(tok[1])
            return (out[d] - 1) * steps[d] + (k[d] - 1) * dil[d] + 1
        if tok[0] == 'K':
            return k[int(tok[1])]
        if tok[0] == 'O':
            return out[int(tok[1])]
        return {'N': n, 'G': g, 'C': c, 'F': f}[tok]
    return rank, i_ax.split(), k_ax.split(), o_ax.split(), size, out, k


def golden(name, x, w, y, steps, dilations, dtype, izp=0, kzp=0):
    rank, i_ax, k_ax, o_ax, red = OPS[name][:5]
    i_ax, k_ax, o_ax, red = i_ax.split(), k_ax.split(), o_ax.split(), red.split()
    sizes = dict(zip(k_ax, w.shape)) | dict(zip(i_ax, x.shape))
    sizes.update(dict(zip(o_ax, y.shape)))
    result = y.copy()
    for oi in np.ndindex(y.shape):
        idx = dict(zip(o_ax, oi))
        acc = y[oi]
        for ri in itertools.product(*(range(sizes[a]) for a in red)):
            idx.update(zip(red, ri))
            xi = tuple(idx[f'O{t[1]}'] * steps[int(t[1])] + idx[f'K{t[1]}'] * dilations[int(t[1])]
                       if t[0] == 'S' else idx[t] for t in i_ax)
            a, b = x[xi], w[tuple(idx[t] for t in k_ax)]
            if dtype == 'bf16':
                prod = bf16_round(np.float32(a) * np.float32(b))
                acc = bf16_round(np.float32(acc) + prod)
            else:
                acc = (int(acc) + (int(a) - izp) * (int(b) - kzp)) & 0xffffffff
                acc = acc - (1 << 32) if acc >= 1 << 31 else acc
        result[oi] = acc
    return result


def make_case(name, rng, dtype='bf16', out_dtype=None, steps=None, dilations=None, values='int',
              **shape):
    rank, i_ax, k_ax, o_ax, size, out, k = geometry(name, **shape)
    steps = steps or (1,) * rank
    dilations = dilations or (1,) * rank
    xs = tuple(size(t, steps, dilations) for t in i_ax)
    ws = tuple(size(t, steps, dilations) for t in k_ax)
    ys = tuple(size(t, steps, dilations) for t in o_ax)
    if values == 'small':      # every partial sum stays an exact bf16 integer (< 256)
        x = rng.integers(-1, 2, xs).astype(np.float32)
        w = rng.integers(-1, 2, ws).astype(np.float32)
        y = rng.integers(-1, 2, ys).astype(np.float32)
    elif values == 'int':
        x = rng.integers(-3, 4, xs).astype(np.float32)
        w = rng.integers(-2, 3, ws).astype(np.float32)
        y = rng.integers(-2, 3, ys).astype(np.float32)
    else:
        x = bf16_round(rng.standard_normal(xs))
        w = bf16_round(rng.standard_normal(ws) / 3)
        y = bf16_round(rng.standard_normal(ys))
    if dtype != 'bf16':
        x, w, y = x.astype(np.int64), w.astype(np.int64), y.astype(np.int64)
    attrs = {} if name in PLAIN else dict(strides=steps, dilations=dilations)
    return x, w, y, attrs, steps, dilations


def run(name, x, w, y, attrs, tmp_path, dtype='bf16', out_dtype=None, cycle=CYCLE, **zp):
    out_dtype = out_dtype or dtype
    op = ConvOp(name, TensorView('x', x.shape, dtype), TensorView('w', w.shape, dtype),
                TensorView('y', y.shape, out_dtype), **attrs, **zp)
    bufs = {'x': encode(x, dtype), 'w': encode(w, dtype), 'y': encode(y, out_dtype)}
    r = execute_conv(op, bufs, tmp_path, tag='conv', cycle=cycle, dram='fixed')
    got = decode(r.buffers['y'], out_dtype, y.shape)
    assert r.buffers['x'] == bufs['x'] and r.buffers['w'] == bufs['w']
    if cycle:
        assert r.cycle.rtl_violations == 0
        assert r.cycle_buffers['y'] == r.buffers['y'], 'atalla-sim and functional sim disagree'
    return got, r


@pytest.mark.parametrize('name', STANDARD)
def test_named_conv_bf16(name, tmp_path):
    """Every standard op on its own default layout; small integers are exact in bf16."""
    rng = np.random.default_rng(len(name))
    x, w, y, attrs, steps, dil = make_case(name, rng)
    got, r = run(name, x, w, y, attrs, tmp_path)
    np.testing.assert_array_equal(got, golden(name, x, w, y, steps, dil, 'bf16'))
    assert r.plan.path == ('lanes' if name in PLAIN else 'array')
    assert r.metrics['dyn_retired_gemm_systolic' if name not in PLAIN else 'dyn_retired_vector_alu'] > 0


@pytest.mark.parametrize('name', QUANTIZED)
def test_named_conv_quantized(name, tmp_path):
    rng = np.random.default_rng(len(name))
    x, w, y, attrs, steps, dil = make_case(name, rng, dtype='i8')
    x = rng.integers(-128, 128, x.shape)
    w = rng.integers(-128, 128, w.shape)
    y = rng.integers(-1000, 1000, y.shape)
    got, r = run(name, x, w, y, attrs, tmp_path, dtype='i8', out_dtype='i32',
                 input_zero_point=-7, filter_zero_point=5)
    np.testing.assert_array_equal(got, golden(name, x, w, y, steps, dil, 'i32', -7, 5))
    assert r.plan.path == 'scalar'


@pytest.mark.parametrize('name', ['conv_2d_nhwc_hwcf', 'conv_2d_nchw_fchw', 'conv_2d_ngchw_gfchw',
                                  'conv_3d_ndhwc_dhwcf', 'conv_1d_ncw_fcw'])
def test_tiles_tails_strides_dilations(name, tmp_path):
    """Channel and filter tails, several pixel tiles, steps and dilations."""
    rng = np.random.default_rng(3)
    rank = OPS[name][0]
    shape = dict(n=1, g=1, c=37, f=35, out=(2, 2, 40) if rank == 3 else (1, 3, 40), k=(2, 2, 3))
    if 'ngchw' in name:
        shape.update(g=2, c=33, f=3)
    x, w, y, attrs, steps, dil = make_case(name, rng, steps=(1, 2, 2)[3 - rank:],
                                           dilations=(1, 1, 2)[3 - rank:], values='small', **shape)
    got, r = run(name, x, w, y, attrs, tmp_path)
    assert r.plan.path == 'array'
    np.testing.assert_array_equal(got, golden_float64(name, x, w, y, steps, dil))


@pytest.mark.parametrize('name', ['conv_2d_nhwc_fhwc', 'conv_2d_nchw_fchw'])
def test_array_fractional_accuracy(name, tmp_path):
    """The array rounds once per (tap, channel tile) step instead of per product:
    the error is within one bf16 half-ulp of the absolute sum per rounding."""
    rng = np.random.default_rng(5)
    x, w, y, attrs, steps, dil = make_case(name, rng, values='normal', n=1, c=48, f=8,
                                           out=(1, 3, 6), k=(1, 3, 3))
    got, _ = run(name, x, w, y, attrs, tmp_path)
    exact = golden_float64(name, x, w, y, steps, dil)
    roundings = 9 * 2 + 1
    bound = roundings * 2 ** -8 * golden_abs(name, x, w, y, steps, dil)
    assert np.all(np.abs(got - exact) <= bound)


def golden_float64(name, x, w, y, steps, dil, absolute=False):
    rank, i_ax, k_ax, o_ax, red = OPS[name][:5]
    f = np.abs if absolute else (lambda v: v)
    i_ax, k_ax, o_ax, red = i_ax.split(), k_ax.split(), o_ax.split(), red.split()
    sizes = dict(zip(k_ax, w.shape)) | dict(zip(o_ax, y.shape))
    out = f(y.astype(np.float64))
    for oi in np.ndindex(y.shape):
        idx = dict(zip(o_ax, oi))
        for ri in itertools.product(*(range(sizes[a]) for a in red)):
            idx.update(zip(red, ri))
            xi = tuple(idx[f'O{t[1]}'] * steps[int(t[1])] + idx[f'K{t[1]}'] * dil[int(t[1])]
                       if t[0] == 'S' else idx[t] for t in i_ax)
            out[oi] += f(np.float64(x[xi]) * np.float64(w[tuple(idx[t] for t in k_ax)]))
    return out


def golden_abs(name, x, w, y, steps, dil):
    return golden_float64(name, x, w, y, steps, dil, absolute=True)


@pytest.mark.parametrize('name', ['conv_1d', 'conv_2d', 'conv_3d'])
def test_lanes_bf16_rounding_is_per_operation(name, tmp_path):
    rng = np.random.default_rng(7)
    x, w, y, attrs, steps, dil = make_case(name, rng, values='normal', out=(2, 3, 37), k=(2, 2, 3))
    got, _ = run(name, x, w, y, attrs, tmp_path)
    np.testing.assert_array_equal(got, golden(name, x, w, y, steps, dil, 'bf16'))


def test_restage_is_exact_for_non_finite_values(tmp_path):
    """NaN/Inf in a restaged filter reach only the outputs Linalg says they reach."""
    name = 'conv_2d_nchw_fchw'
    rng = np.random.default_rng(9)
    x, w, y, attrs, steps, dil = make_case(name, rng, n=1, c=4, f=3, out=(1, 3, 3), k=(1, 2, 2))
    w[1, 2, 0, 1] = np.nan
    w[2, 0, 1, 1] = np.inf
    got, _ = run(name, x, w, y, attrs, tmp_path)
    want = golden(name, x, w, y, steps, dil, 'bf16')
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    finite = ~np.isnan(want)
    np.testing.assert_array_equal(got[finite], want[finite])


def test_i32_wraps_modulo_32(tmp_path):
    name = 'conv_2d_nhwc_hwcf'
    x = np.full((1, 1, 1, 2), 2147483647, np.int64)
    w = np.array([[[[3], [-2]]]], np.int64)
    y = np.array([[[[11]]]], np.int64)
    got, r = run(name, x, w, y, dict(strides=(1, 1), dilations=(1, 1)), tmp_path, dtype='i32')
    assert r.plan.path == 'scalar'
    np.testing.assert_array_equal(got, golden(name, x, w, y, (1, 1), (1, 1), 'i32'))


def test_i16_inputs_sign_extend(tmp_path):
    name = 'conv_2d_nchw_fchw_q'
    rng = np.random.default_rng(11)
    x, w, y, attrs, steps, dil = make_case(name, rng, dtype='i16', n=1, c=2, f=2, out=(1, 2, 2), k=(1, 2, 2))
    x = rng.integers(-30000, 30000, x.shape)
    w = rng.integers(-30000, 30000, w.shape)
    got, _ = run(name, x, w, y, attrs, tmp_path, dtype='i16', out_dtype='i32',
                 input_zero_point=123, filter_zero_point=-456, cycle=False)
    np.testing.assert_array_equal(got, golden(name, x, w, y, steps, dil, 'i32', 123, -456))


def test_aliased_input_reads_initial_values(tmp_path):
    """Output and input share one buffer: the kernel reads a snapshot."""
    name = 'conv_1d'
    x = np.arange(8, dtype=np.float32)
    w = np.array([1., -1., 2.], np.float32)
    op = ConvOp(name, TensorView('a', (8,), 'bf16'), TensorView('w', (3,), 'bf16'),
                TensorView('a', (6,), 'bf16', offset_bytes=4))
    bufs = {'a': encode(x, 'bf16'), 'w': encode(w, 'bf16')}
    r = execute_conv(op, bufs, tmp_path, cycle=False)
    want = x.copy()
    want[2:] = golden(name, x, w, x[2:].copy(), (1,), (1,), 'bf16')
    np.testing.assert_array_equal(decode(r.buffers['a'], 'bf16', (8,)), want)
    assert r.plan.path == 'scalar'


def test_views_with_offsets_keep_neighbouring_bytes(tmp_path):
    name = 'conv_2d_nhwc_fhwc'
    rng = np.random.default_rng(13)
    x, w, y, attrs, steps, dil = make_case(name, rng)
    canary = b'\x35\x71'
    op = ConvOp(name, TensorView('x', x.shape, 'bf16', offset_bytes=2),
                TensorView('w', w.shape, 'bf16', offset_bytes=2),
                TensorView('y', y.shape, 'bf16', offset_bytes=2), **attrs)
    bufs = {k: canary + encode(v, 'bf16') + canary for k, v in (('x', x), ('w', w), ('y', y))}
    r = execute_conv(op, bufs, tmp_path, cycle=False)
    want = golden(name, x, w, y, steps, dil, 'bf16')
    assert r.buffers['y'] == canary + encode(want, 'bf16') + canary
    assert r.plan.path == 'array'


def test_empty_reduction_preserves_output(tmp_path):
    name = 'conv_2d_nhwc_hwcf'
    op = ConvOp(name, TensorView('x', (1, 2, 2, 0), 'bf16'), TensorView('w', (1, 1, 0, 2), 'bf16'),
                TensorView('y', (1, 2, 2, 2), 'bf16'), strides=(1, 1), dilations=(1, 1))
    bufs = {'x': b'', 'w': b'', 'y': b'\xff\x7f\x00\x80' * 4}
    r = execute_conv(op, bufs, tmp_path, cycle=False)
    assert r.buffers['y'] == bufs['y'] and r.plan.path == 'none'


def test_plans_and_errors():
    names = iter(range(100))
    v = lambda shape, dt='bf16', **kw: TensorView(f't{next(names)}', shape, dt, **kw)
    bufs = lambda *views: {x.buffer: bytes(4 * int(np.prod(x.shape)) + 8) for x in views}
    x, w, y = v((1, 3, 6, 6)), v((4, 3, 3, 3)), v((1, 4, 4, 4))
    assert lower_conv(ConvOp('linalg.conv_2d_nchw_fchw', x, w, y), bufs(x, w, y)).path == 'array'
    big = v((1, 4, 5, 4))
    with pytest.raises(ValueError):
        lower_conv(ConvOp('conv_2d_nchw_fchw', x, w, big), bufs(x, w, big))
    with pytest.raises(NotImplementedError):
        lower_conv(ConvOp('conv_2d_nchw_fchw_q', x, w, y, input_zero_point=0, filter_zero_point=0),
                   bufs(x, w, y))
    a, b, c = v((4, 4)), v((2, 2)), v((3, 3))
    with pytest.raises(ValueError):
        lower_conv(ConvOp('conv_2d', a, b, c, strides=(1, 1)), bufs(a, b, c))
    odd, ww, yy = v((1, 6, 6, 3), offset_bytes=1), v((4, 3, 3, 3)), v((1, 4, 4, 4))
    assert lower_conv(ConvOp('conv_2d_nhwc_fhwc', odd, ww, yy), bufs(odd, ww, yy)).path == 'scalar'
