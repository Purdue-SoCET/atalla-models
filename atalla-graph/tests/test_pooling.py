"""Standalone pooling algorithms: compiled C and independent numerical goldens."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.layout import TensorView
from graph.pooling import PoolingOp
from codegen.pooling import emit_pooling
from codegen.c_emitter import compile_and_assemble, render_in_file
from run_graph import _run_emulator


def encode(array, dtype='bf16'):
    if dtype == 'bf16':
        return torch.tensor(np.asarray(array).copy()).bfloat16().view(torch.int16).numpy().astype('<i2').tobytes()
    return np.asarray(array, dtype={'f16': '<f2', 'f32': '<f4', 'i32': '<i4', 'u32': '<u4'}[dtype]).tobytes()


def decode(data, dtype='bf16'):
    if dtype == 'bf16':
        return (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(data, dtype={'f16': '<f2', 'f32': '<f4', 'i32': '<i4', 'u32': '<u4'}[dtype])


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    # Every named kernel is independently compiled; reuse only within that name.
    programs = {}

    def compile_for(emission, name):
        name = name.removeprefix('linalg.')
        if name not in programs:
            work = tmp_path_factory.mktemp(name)
            compile_and_assemble(emission, str(work), 'pooling')
            programs[name] = emission.instr_text
        return programs[name]

    return compile_for


def execute(op, buffers, compiled, tmp_path, metrics=None):
    packed = emit_pooling(op, buffers)
    packed.emission.instr_text = compiled(packed.emission, op.name)
    image = tmp_path / 'pooling.in'
    image.write_text(render_in_file(packed.emission))
    memory, unit = _run_emulator(str(image), str(tmp_path), 'pooling')
    if metrics is not None:
        metrics.update(unit.perf_metrics.metrics)
    return {name: bytes((memory.read_data((address + i) & -4) >> (8 * ((address + i) & 3))) & 255
                        for i in range(packed.buffer_sizes[name]))
            for name, address in packed.buffer_addresses.items()}


def test_nhwc_min_with_initial_output_and_rectangular_window(compiled, tmp_path):
    x = (np.arange(60).reshape(1, 4, 5, 3) % 19 - 9).astype(np.float32)
    initial = np.full((1, 3, 2, 3), 100, dtype=np.float32)
    initial[..., 0] = -100
    expected = initial.copy()
    for h in range(3):
        for w in range(2):
            expected[0, h, w] = np.minimum(initial[0, h, w], x[0, h:h + 2, 2*w:2*w + 3].min(axis=(0, 1)))
    op = PoolingOp('linalg.pooling_nhwc_min', TensorView('x', x.shape, 'bf16'),
                   TensorView('y', initial.shape, 'bf16'), (2, 3), strides=(1, 2))
    buffers = {'x': encode(x), 'y': encode(initial)}
    result = execute(op, buffers, compiled, tmp_path)
    np.testing.assert_array_equal(decode(result['y']).reshape(initial.shape), expected)
    assert result['x'] == buffers['x']


def test_nhwc_min_uses_asymmetric_dilation(compiled, tmp_path):
    x = np.arange(1 * 5 * 7 * 2, dtype=np.float32).reshape(1, 5, 7, 2)[..., ::-1].copy()
    x[0, 2, 4, 0] = -50
    initial = np.full((1, 2, 3, 2), 100, dtype=np.float32)
    expected = initial.copy()
    for h in range(2):
        for w in range(3):
            expected[0, h, w] = x[0, 2*h:2*h + 3:2, w:w + 5:2].min(axis=(0, 1))
    op = PoolingOp('pooling_nhwc_min', TensorView('x', x.shape, 'bf16'),
                   TensorView('y', initial.shape, 'bf16'), (2, 3), strides=(2, 1), dilations=(2, 2))
    result = execute(op, {'x': encode(x), 'y': encode(initial)}, compiled, tmp_path)
    np.testing.assert_array_equal(decode(result['y']).reshape(initial.shape), expected)


def test_min_propagates_nan_and_prefers_negative_zero(compiled, tmp_path):
    x = np.array([[[[0., -0., np.nan, 2.], [-0., 0., 1., np.nan]]]], dtype=np.float32)
    initial = np.array([[[[0., 0., 3., 4.]]]], dtype=np.float32)
    op = PoolingOp('pooling_nhwc_min', TensorView('x', x.shape, 'bf16'),
                   TensorView('y', initial.shape, 'bf16'), (1, 2))
    result = execute(op, {'x': encode(x), 'y': encode(initial)}, compiled, tmp_path)
    actual = decode(result['y'])
    np.testing.assert_array_equal(actual, [-0., -0., np.nan, np.nan])
    assert np.signbit(actual[:2]).all()


# Independent literal inventory, not generated from the implementation registry.
POOL_CASES = [
    ('pooling_ncw_max', 'NCW', 'max'), ('pooling_ncw_sum', 'NCW', 'sum'),
    ('pooling_nwc_max', 'NWC', 'max'), ('pooling_nwc_min', 'NWC', 'min'),
    ('pooling_nwc_sum', 'NWC', 'sum'),
    ('pooling_nwc_max_unsigned', 'NWC', 'max'), ('pooling_nwc_min_unsigned', 'NWC', 'min'),
    ('pooling_nchw_max', 'NCHW', 'max'), ('pooling_nchw_sum', 'NCHW', 'sum'),
    ('pooling_nhwc_max', 'NHWC', 'max'), ('pooling_nhwc_min', 'NHWC', 'min'),
    ('pooling_nhwc_sum', 'NHWC', 'sum'),
    ('pooling_nhwc_max_unsigned', 'NHWC', 'max'), ('pooling_nhwc_min_unsigned', 'NHWC', 'min'),
    ('pooling_ndhwc_max', 'NDHWC', 'max'), ('pooling_ndhwc_min', 'NDHWC', 'min'),
    ('pooling_ndhwc_sum', 'NDHWC', 'sum'),
]


@pytest.mark.parametrize('name,layout,mode,dtype', [
    (name, layout, mode, dtype) for name, layout, mode in POOL_CASES
    for dtype in (['u32'] if name.endswith('_unsigned') else ['bf16', 'i32'])
])
def test_named_pooling_variants(name, layout, mode, dtype, compiled, tmp_path):
    import itertools
    rank = len(layout) - 2
    spatial, window, out, strides = {
        1: ((6,), (3,), (2,), (2,)),
        2: ((4, 5), (2, 3), (2, 3), (2, 1)),
        3: ((3, 4, 5), (2, 2, 2), (1, 2, 2), (1, 1, 2)),
    }[rank]
    shape = (2, *spatial, 2)
    source = (np.arange(np.prod(shape)).reshape(shape) % 13 - 6).astype(np.int64)
    if dtype == 'u32':
        source = np.where(source < 0, source + (1 << 32), source)
    initial = np.full((2, *out, 2), 1 if mode == 'sum' else (100 if mode == 'min' else -100), dtype=np.int64)
    if dtype == 'u32' and mode == 'max':
        initial.fill(0)
    # A channel with a stronger initial bound must survive the reduction.
    if mode == 'min':
        initial[..., 0] = 0 if dtype == 'u32' else -100
    elif mode == 'max':
        initial[..., 0] = (1 << 32) - 1 if dtype == 'u32' else 100
    expected = initial.copy()
    for n in range(2):
        for index in itertools.product(*(range(size) for size in out)):
            region = source[(n, *(slice(i*s, i*s+k) for i, s, k in zip(index, strides, window)), slice(None))]
            values = region.reshape(-1, 2)
            target = (n, *index, slice(None))
            if mode == 'min':
                expected[target] = np.minimum(initial[target], values.min(axis=0))
            elif mode == 'max':
                expected[target] = np.maximum(initial[target], values.max(axis=0))
            else:
                expected[target] += values.sum(axis=0)
    if layout[1] == 'C':
        source = np.moveaxis(source, -1, 1)
        initial = np.moveaxis(initial, -1, 1)
        expected = np.moveaxis(expected, -1, 1)
    op = PoolingOp('linalg.' + name, TensorView('x', source.shape, dtype),
                   TensorView('y', initial.shape, dtype), window, strides=strides)
    buffers = {'x': encode(source, dtype), 'y': encode(initial, dtype)}
    result = execute(op, buffers, compiled, tmp_path)
    np.testing.assert_array_equal(decode(result['y'], dtype).reshape(initial.shape), expected)
    assert result['x'] == buffers['x']


@pytest.mark.parametrize('dtype', ['bf16', 'f16', 'f32'])
@pytest.mark.parametrize('mode', ['min', 'max'])
def test_float_extrema_special_values(dtype, mode, compiled, tmp_path):
    x = np.array([[[0., -0., np.nan, np.inf, -np.inf, -1.e-40],
                   [-0., 0., 2., -np.inf, np.inf, 1.e-40]]], dtype=np.float32)
    initial = np.array([[[0., -0., 3., 0., 0., 0.]]], dtype=np.float32)
    op = PoolingOp('pooling_nwc_' + mode, TensorView('x', x.shape, dtype),
                   TensorView('y', initial.shape, dtype), (2,))
    result = execute(op, {'x': encode(x, dtype), 'y': encode(initial, dtype)}, compiled, tmp_path)
    actual = decode(result['y'], dtype)
    assert np.isnan(actual[2])
    quantized = decode(encode(x, dtype), dtype).reshape(x.shape)
    expected_tail = quantized[0, :, 5].min() if mode == 'min' else quantized[0, :, 5].max()
    assert actual[5] == expected_tail
    if mode == 'min':
        assert np.signbit(actual[:2]).all()
        np.testing.assert_array_equal(actual[3:5], [-np.inf, -np.inf])
        assert actual[5] <= 0
    else:
        assert not np.signbit(actual[:2]).any()
        np.testing.assert_array_equal(actual[3:5], [np.inf, np.inf])
        assert actual[5] >= 0


@pytest.mark.parametrize('mode', ['min', 'max'])
def test_unsigned_extrema_cross_sign_bit(mode, compiled, tmp_path):
    x = np.array([[[0xFFFFFFFF, 0x80000000, 0x7FFFFFFF], [1, 0x7FFFFFFF, 0x80000000]]], dtype=np.uint32)
    initial = np.full((1, 1, 3), 0xFFFFFFFF if mode == 'min' else 0, dtype=np.uint32)
    expected = x.min(axis=1, keepdims=True) if mode == 'min' else x.max(axis=1, keepdims=True)
    op = PoolingOp('pooling_nwc_' + mode + '_unsigned', TensorView('x', x.shape, 'u32'),
                   TensorView('y', initial.shape, 'u32'), (2,))
    result = execute(op, {'x': encode(x, 'u32'), 'y': encode(initial, 'u32')}, compiled, tmp_path)
    np.testing.assert_array_equal(decode(result['y'], 'u32').reshape(initial.shape), expected)


@pytest.mark.parametrize('dtype,initial_value,expected', [('i32', 2147483647, -2147483647),
                                                       ('u32', 0xFFFFFFFF, 1)])
def test_integer_sum_wraps(dtype, initial_value, expected, compiled, tmp_path):
    op = PoolingOp('pooling_nwc_sum', TensorView('x', (1, 2, 1), dtype),
                   TensorView('y', (1, 1, 1), dtype), (2,))
    result = execute(op, {'x': encode([1, 1], dtype), 'y': encode([initial_value], dtype)}, compiled, tmp_path)
    assert decode(result['y'], dtype)[0] == expected


def test_bf16_sum_rounds_each_addition(compiled, tmp_path):
    op = PoolingOp('pooling_nwc_sum', TensorView('x', (1, 2, 1), 'bf16'),
                   TensorView('y', (1, 1, 1), 'bf16'), (2,))
    result = execute(op, {'x': encode([.00390625, .00390625]), 'y': encode([1.])}, compiled, tmp_path)
    assert decode(result['y'])[0] == 1.0  # Two ties-to-even steps, not one widened sum.


def test_pooling_negative_strides_and_padded_output(compiled, tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    initial = bytearray(b'!' * 32)
    np.ndarray((1, 2, 3), dtype='<u2', buffer=initial, offset=1, strides=(24, 12, 3))[...] = 0x7F80
    op = PoolingOp('pooling_nwc_min', TensorView('x', x.shape, 'bf16', strides_bytes=(24, -6, 2), offset_bytes=18),
                   TensorView('y', (1, 2, 3), 'bf16', strides_bytes=(24, 12, 3), offset_bytes=1), (2,), strides=(2,))
    expected = bytearray(initial)
    expected_bits = torch.tensor(np.stack([x[0, 2:4].min(axis=0), x[0, :2].min(axis=0)])[None]).bfloat16().view(torch.int16).numpy().view(np.uint16)
    np.ndarray((1, 2, 3), dtype='<u2', buffer=expected, offset=1, strides=(24, 12, 3))[...] = expected_bits
    result = execute(op, {'x': encode(x), 'y': bytes(initial)}, compiled, tmp_path)
    assert result['y'] == bytes(expected)


def test_aliased_pooling_reads_original_input(compiled, tmp_path):
    # Reversed input view: an early output write would corrupt a later window.
    original = encode([9., 8., 7., 6., 2., 1.])
    op = PoolingOp('pooling_nwc_min', TensorView('a', (1, 6, 1), 'bf16', strides_bytes=(12, -2, 2), offset_bytes=10),
                   TensorView('a', (1, 3, 1), 'bf16'), (2,), strides=(2,))
    expected = bytearray(original)
    # Initial destination values [9,8,7] also participate.
    expected[:6] = encode([1., 6., 7.])
    assert execute(op, {'a': original}, compiled, tmp_path)['a'] == bytes(expected)


@pytest.mark.parametrize('empty_kernel', [False, True])
def test_empty_pooling_preserves_initial_output(empty_kernel, compiled, tmp_path):
    xshape = (1, 3, 1)
    yshape = (1, 2, 1) if empty_kernel else (1, 0, 1)
    op = PoolingOp('pooling_nwc_sum', TensorView('x', xshape, 'bf16'),
                   TensorView('y', yshape, 'bf16'), (0,) if empty_kernel else (2,))
    buffers = {'x': encode([1., 2., 3.]), 'y': encode([7., 9.]) if empty_kernel else b''}
    assert execute(op, buffers, compiled, tmp_path) == buffers


@pytest.mark.parametrize('channels', [1, 31, 32, 33, 65])
def test_channel_counts_cross_vector_boundaries(channels, compiled, tmp_path):
    x = (np.arange(2 * channels).reshape(1, 2, channels) % 23 - 11).astype(np.float32)
    y = np.full((1, 1, channels), 100, dtype=np.float32)
    op = PoolingOp('pooling_nwc_min', TensorView('x', x.shape, 'bf16'),
                   TensorView('y', y.shape, 'bf16'), (2,))
    result = execute(op, {'x': encode(x), 'y': encode(y)}, compiled, tmp_path)
    np.testing.assert_array_equal(decode(result['y']), x.min(axis=1).ravel())


@pytest.mark.parametrize('changes,error,message', [
    ({'name': 'pooling_nchw_min'}, NotImplementedError, 'operation'),
    ({'source': TensorView('x', (1, 3, 4, 2), 'f32')}, NotImplementedError, 'dtype'),
    ({'name': 'pooling_nhwc_sum', 'source': TensorView('x', (1, 3, 4, 2), 'f32'),
      'destination': TensorView('y', (1, 2, 3, 2), 'f32')}, NotImplementedError, 'dtype'),
    ({'name': 'pooling_nhwc_min_unsigned', 'source': TensorView('x', (1, 3, 4, 2), 'i32'),
      'destination': TensorView('y', (1, 2, 3, 2), 'i32')}, NotImplementedError, 'unsigned'),
    ({'strides': (0, 1)}, ValueError, 'strides'),
    ({'dilations': (1, -1)}, ValueError, 'dilations'),
    ({'kernel_shape': (2,)}, ValueError, 'kernel'),
    ({'kernel_shape': (-1, 2)}, ValueError, 'kernel'),
    ({'destination': TensorView('y', (2, 2, 3, 2), 'bf16')}, ValueError, 'batch'),
    ({'destination': TensorView('y', (1, 3, 4, 2), 'bf16')}, ValueError, 'window'),
    ({'destination': TensorView('y', (1, 2, 3, 2), 'bf16', strides_bytes=(0, 0, 0, 0))}, ValueError, 'overlap'),
    ({'strides': (1 << 31, 1)}, OverflowError, '32-bit'),
])
def test_invalid_pooling_configuration(changes, error, message):
    from graph.pooling import lower_pooling
    values = dict(name='pooling_nhwc_min', source=TensorView('x', (1, 3, 4, 2), 'bf16'),
                  destination=TensorView('y', (1, 2, 3, 2), 'bf16'), kernel_shape=(2, 2))
    values.update(changes)
    with pytest.raises(error, match=message):
        lower_pooling(PoolingOp(**values), {'x': bytes(256), 'y': bytes(256)})


def test_public_pooling_runtime_returns_execution_buffers(tmp_path):
    from pooling_runtime import execute_pooling
    op = PoolingOp('pooling_nhwc_min', TensorView('x', (1, 1, 2, 1), 'bf16'),
                   TensorView('y', (1, 1, 1, 1), 'bf16'), (1, 2))
    result = execute_pooling(op, {'x': encode([3., -2.]), 'y': encode([10.])}, tmp_path)
    assert result.backend == 'emulator'
    assert decode(result.buffers['y'])[0] == -2.


@pytest.mark.parametrize('mode', ['min', 'max'])
def test_signed_integer_limits(mode, compiled, tmp_path):
    x = np.array([[[-2147483648, 2147483647], [-1, 0]]], dtype=np.int32)
    y = np.full((1, 1, 2), 2147483647 if mode == 'min' else -2147483648, dtype=np.int32)
    op = PoolingOp('pooling_nwc_' + mode, TensorView('x', x.shape, 'i32'),
                   TensorView('y', y.shape, 'i32'), (2,))
    result = execute(op, {'x': encode(x, 'i32'), 'y': encode(y, 'i32')}, compiled, tmp_path)
    expected = x.min(axis=1) if mode == 'min' else x.max(axis=1)
    np.testing.assert_array_equal(decode(result['y'], 'i32'), expected.ravel())


def test_image_marker_is_included_in_address_budget(monkeypatch):
    from codegen import pooling
    monkeypatch.setattr(pooling, 'MAX_DATA_ADDRESS', 4106)
    op = PoolingOp('pooling_nwc_min', TensorView('x', (1, 1, 1), 'bf16'),
                   TensorView('y', (1, 1, 1), 'bf16'), (1,))
    with pytest.raises(OverflowError, match='budget'):
        pooling.emit_pooling(op, {'x': bytes(2), 'y': bytes(2)})


@pytest.mark.parametrize('dtype,width', [('bf16', 2), ('f16', 2), ('f32', 4)])
@pytest.mark.parametrize('mode', ['min', 'max'])
def test_fractional_extrema_with_negative_and_output_strides(dtype, width, mode, compiled, tmp_path):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 4, 5, 3)).astype(np.float32)
    initial = rng.normal(size=(1, 2, 2, 3)).astype(np.float32)
    source_bytes = encode(x, dtype)
    logical_x = decode(source_bytes, dtype).reshape(x.shape)[:, ::-1]
    initial_bytes = encode(initial, dtype)
    expected_values = decode(initial_bytes, dtype).reshape(initial.shape).copy()
    for h in range(2):
        for w in range(2):
            patch = logical_x[0, h:h + 3:2, 2*w:2*w + 2]
            reduction = patch.min(axis=(0, 1)) if mode == 'min' else patch.max(axis=(0, 1))
            expected_values[0, h, w] = (np.minimum if mode == 'min' else np.maximum)(expected_values[0, h, w], reduction)
    storage_dtype = '<u2' if width == 2 else '<u4'
    dest_strides = (128 * width, 32 * width, 8 * width, 2 * width)
    destination = bytearray(b'!' * (4 + 48 * width))
    output_view = np.ndarray(initial.shape, dtype=storage_dtype, buffer=destination, offset=1, strides=dest_strides)
    output_view[...] = np.frombuffer(initial_bytes, dtype=storage_dtype).reshape(initial.shape)
    expected = bytearray(destination)
    np.ndarray(initial.shape, dtype=storage_dtype, buffer=expected, offset=1, strides=dest_strides)[...] = np.frombuffer(encode(expected_values, dtype), dtype=storage_dtype).reshape(initial.shape)
    op = PoolingOp('pooling_nhwc_' + mode,
                   TensorView('x', x.shape, dtype, strides_bytes=(60 * width, -15 * width, 3 * width, width), offset_bytes=45 * width),
                   TensorView('y', initial.shape, dtype, strides_bytes=dest_strides, offset_bytes=1),
                   (2, 2), strides=(1, 2), dilations=(2, 1))
    result = execute(op, {'x': source_bytes, 'y': bytes(destination)}, compiled, tmp_path)
    assert result == {'x': source_bytes, 'y': bytes(expected)}


@pytest.mark.parametrize('values,initial,expected', [
    ([.00390625, .00390625], 1.0078125, 1.015625),
    ([np.nan, 1.], 0., np.nan),
    ([np.inf, -np.inf], 0., np.nan),
    ([1., 2.], np.inf, np.inf),
    ([1., 2.], -np.inf, -np.inf),
    ([2. ** -133, 2. ** -133], 0., 2. ** -132),
    ([float.fromhex('0x1.fep127'), 0.], float.fromhex('0x1.fep127'), np.inf),
])
def test_bf16_sum_special_values_and_odd_ties(values, initial, expected, compiled, tmp_path):
    op = PoolingOp('pooling_nwc_sum', TensorView('x', (1, 2, 1), 'bf16'),
                   TensorView('y', (1, 1, 1), 'bf16'), (2,))
    with np.errstate(over='ignore', invalid='ignore'):
        result = execute(op, {'x': encode(values), 'y': encode([initial])}, compiled, tmp_path)
    actual = decode(result['y'])[0]
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize('mode,expected', [('min', -2.), ('max', 3.), ('sum', 1.)])
def test_operation_is_fixed_by_compiled_image(mode, expected, tmp_path):
    source = TensorView('x', (1, 2, 1), 'bf16')
    destination = TensorView('y', (1, 1, 1), 'bf16')
    buffers = {'x': encode([-2., 3.]), 'y': encode([0.])}
    common = emit_pooling(PoolingOp('pooling_nwc_min', source, destination, (2,)), buffers)
    selected = emit_pooling(PoolingOp('pooling_nwc_' + mode, source, destination, (2,)), buffers)
    compile_and_assemble(selected.emission, str(tmp_path), mode)
    # Identical runtime data/config for all three independently compiled images.
    # The image itself, rather than a runtime selector, must define the algorithm.
    selected.emission.dram = common.emission.dram
    image = tmp_path / f'{mode}.in'
    image.write_text(render_in_file(selected.emission))
    memory, _ = _run_emulator(str(image), str(tmp_path), mode)
    address = selected.buffer_addresses['y']
    bits = memory.read_bf16_le(address)
    actual = np.array([bits << 16], dtype=np.uint32).view(np.float32)[0]
    assert actual == expected


@pytest.mark.parametrize('layout', ['NCW', 'NWC', 'NCHW', 'NHWC', 'NDHWC'])
@pytest.mark.parametrize('mode', ['sum', 'max', 'min'])
def test_bf16_vector_pooling(layout, mode, compiled, tmp_path):
    if mode == 'min' and layout in ('NCW', 'NCHW'):
        pytest.skip('no named channel-first min pooling operation')
    rank = len(layout) - 2
    channel_first = layout[1] == 'C'
    # Width/channels cross two complete vectors and a tail.
    spatial = (3,) * (rank - 1) + ((67,) if channel_first else (3,))
    channels = 2 if channel_first else 65
    shape = (1, channels, *spatial) if channel_first else (1, *spatial, channels)
    x = ((np.arange(np.prod(shape)).reshape(shape) % 29) - 14).astype(np.float32) / 8
    axes = tuple(range(2, len(shape))) if channel_first else tuple(range(1, len(shape) - 1))
    out_shape = list(shape)
    for axis in axes:
        out_shape[axis] -= 1
    initial = np.full(out_shape, {'sum': .125, 'max': -100, 'min': 100}[mode], np.float32)
    expected = torch.tensor(initial).bfloat16()
    import itertools
    for offset in itertools.product(range(2), repeat=rank):
        slices = [slice(None)] * len(shape)
        for axis, start in zip(axes, offset):
            slices[axis] = slice(start, start + out_shape[axis])
        v = torch.tensor(x[tuple(slices)].copy()).bfloat16()
        expected = expected + v if mode == 'sum' else (
            torch.minimum(expected, v) if mode == 'min' else torch.maximum(expected, v))
    op = PoolingOp('pooling_' + layout.lower() + '_' + mode,
                   TensorView('x', shape, 'bf16'), TensorView('y', tuple(out_shape), 'bf16'), (2,) * rank)
    packed = emit_pooling(op, {'x': encode(x), 'y': encode(initial)})
    program = compiled(packed.emission, op.name)
    assert 'add.vv' in program or 'mul.v' in program, 'pooling reduction must use vector arithmetic'
    actual = execute(op, {'x': encode(x), 'y': encode(initial)}, compiled, tmp_path)
    np.testing.assert_array_equal(decode(actual['y']), expected.float().numpy().ravel())
    assert actual['x'] == encode(x)


@pytest.mark.parametrize('mode', ['min', 'max', 'sum'])
@pytest.mark.parametrize('channels', [1, 31, 32, 33, 65])
def test_vector_tails_preserve_neighbor_bytes(mode, channels, compiled, tmp_path):
    x = np.resize(np.array([-0., 0., -2., 3., np.nan, np.inf, -np.inf], np.float32),
                  (1, 2, channels))
    initial = np.full((1, 1, channels), {'min': 100., 'max': -100., 'sum': .125}[mode], np.float32)
    expected = torch.tensor(initial).bfloat16()
    for row in range(2):
        value = torch.tensor(x[:, row:row+1]).bfloat16()
        expected = expected + value if mode == 'sum' else (
            torch.minimum(expected, value) if mode == 'min' else torch.maximum(expected, value))
    sentinel = b'\x35\x71'
    buffers = {'x': sentinel + encode(x) + sentinel, 'y': sentinel + encode(initial) + sentinel}
    op = PoolingOp('pooling_nwc_' + mode,
                   TensorView('x', x.shape, 'bf16', offset_bytes=2),
                   TensorView('y', initial.shape, 'bf16', offset_bytes=2), (2,))
    actual = execute(op, buffers, compiled, tmp_path)
    assert actual['x'] == buffers['x']
    assert actual['y'][:2] == sentinel and actual['y'][-2:] == sentinel
    result = decode(actual['y'][2:-2])
    np.testing.assert_array_equal(result, expected.float().numpy().ravel())
    # PyTorch's zero-tie preference depends on operand order; the kernel contract
    # explicitly requires -0 for minimum and +0 for maximum.
    if mode != 'sum':
        for channel in np.flatnonzero(result == 0):
            values = x[0, :, channel]
            signs = np.signbit(values[values == 0])
            negative = signs.any() if mode == 'min' else signs.all()
            assert bool(np.signbit(result[channel])) == bool(negative)
    bits = np.frombuffer(actual['y'][2:-2], dtype='<u2')
    assert np.all(bits[np.isnan(result)] == 0x7fc0)


@pytest.mark.parametrize('mode', ['sum', 'min', 'max'])
@pytest.mark.parametrize('payload', [0x7fff, 0xffff, 0x7f81])
@pytest.mark.parametrize('location', ['source', 'initial'])
def test_vector_pooling_canonicalizes_raw_nan_payloads(mode, payload, location, compiled, tmp_path):
    source = np.array([payload if location == 'source' else 0x3f80, 0x4000], dtype='<u2').tobytes()
    initial = np.array([payload if location == 'initial' else 0x3f80], dtype='<u2').tobytes()
    op = PoolingOp('pooling_nwc_' + mode, TensorView('x', (1, 2, 1), 'bf16'),
                   TensorView('y', (1, 1, 1), 'bf16'), (2,))
    actual = execute(op, {'x': source, 'y': initial}, compiled, tmp_path)
    assert actual['y'] == b'\xc0\x7f'


@pytest.mark.parametrize('mode,baseline', [('min', 90180), ('max', 90180)])
def test_zero_cleanup_reduces_packets(mode, baseline, compiled, tmp_path):
    x = np.zeros((1, 3, 65), np.float32)
    y = np.full((1, 2, 65), 100 if mode == 'min' else -100, np.float32)
    op = PoolingOp('pooling_nwc_' + mode, TensorView('x', x.shape, 'bf16'),
                   TensorView('y', y.shape, 'bf16'), (2,))
    metrics = {}
    actual = execute(op, {'x': encode(x), 'y': encode(y)}, compiled, tmp_path, metrics)
    assert actual['y'] == bytes(y.size * 2)
    # Same fixture as the recorded 9a49cd0 mixed/zero benchmark.
    assert metrics['packets_executed'] < baseline * .8


@pytest.mark.parametrize('layout', ['NCW', 'NCHW'])
@pytest.mark.parametrize('mode', ['max', 'sum'])
@pytest.mark.parametrize('out_width', [1, 17, 33])
@pytest.mark.parametrize('alias', [False, True])
def test_channel_first_stride_two_vectors(layout, mode, out_width, alias, compiled, tmp_path):
    import itertools
    width = 2 * (out_width - 1) + 5
    shape = (2, 2, width) if layout == 'NCW' else (2, 2, 5, width)
    yshape = (2, 2, out_width) if layout == 'NCW' else (2, 2, 2, out_width)
    rank = len(shape) - 2
    kernel = (3,) if rank == 1 else (2, 3)
    x = ((np.arange(np.prod(shape)).reshape(shape) % 19 - 9) / 8).astype(np.float32)
    x.flat[:2] = [-0., 0.]
    if x.size > 12:
        x.flat[6], x.flat[8], x.flat[10] = np.nan, np.inf, -np.inf
    initial = (x.ravel()[:np.prod(yshape)].reshape(yshape).copy() if alias else
               np.full(yshape, -100 if mode == 'max' else .125, np.float32))
    expected = torch.tensor(initial).bfloat16()
    for offset in itertools.product(*(range(k) for k in kernel)):
        slices = (slice(None), slice(None), *(slice(k*2, k*2 + size*2, 2)
                                            for k, size in zip(offset, yshape[2:])))
        v = torch.tensor(x[slices].copy()).bfloat16()
        if mode == 'sum':
            expected = expected + v
        else:
            zeros = (expected == 0) & (v == 0)
            expected = torch.where(zeros, expected + v, torch.maximum(expected, v))
    source_name = 'a' if alias else 'x'
    destination_name = 'a' if alias else 'y'
    sentinel = b'\x35\x71'
    buffers = {source_name: sentinel + encode(x) + sentinel}
    if not alias:
        buffers[destination_name] = sentinel + encode(initial) + sentinel
    op = PoolingOp('pooling_' + layout.lower() + '_' + mode,
                   TensorView(source_name, shape, 'bf16', offset_bytes=2),
                   TensorView(destination_name, yshape, 'bf16', offset_bytes=2),
                   kernel, strides=(2,)*rank, dilations=(2,)*rank)
    metrics = {}
    actual = execute(op, buffers, compiled, tmp_path, metrics)
    bits = expected.view(torch.int16).numpy().astype('<u2')
    bits[np.isnan(expected.float().numpy())] = 0x7fc0
    expected_buffer = bytearray(buffers[destination_name])
    expected_buffer[2:2 + bits.nbytes] = bits.tobytes()
    assert actual[destination_name] == bytes(expected_buffer)
    if not alias:
        assert actual[source_name] == buffers[source_name]
    assert metrics['dyn_retired_vector_alu'] > 0, 'stride-two reduction must execute vectors'


@pytest.mark.parametrize('layout', ['ncw', 'nchw'])
@pytest.mark.parametrize('mode', ['max', 'sum'])
def test_stride_two_raw_special_values(layout, mode, compiled, tmp_path):
    source = np.array([0x8000, 0x8000, 0x8000, 0, 0x7fff, 0x3f80, 0x7f80, 0xff80], dtype='<u2')
    initial = np.full(4, 0xff80 if mode == 'max' else 0x8000, dtype='<u2')
    shape = (1, 1, 8) if layout == 'ncw' else (1, 1, 1, 8)
    yshape = (1, 1, 4) if layout == 'ncw' else (1, 1, 1, 4)
    kernel = (2,) if layout == 'ncw' else (1, 2)
    steps = (2,) if layout == 'ncw' else (1, 2)
    sentinel = b'\x35\x71'
    buffers = {'x': sentinel + source.tobytes() + sentinel,
               'y': sentinel + initial.tobytes() + sentinel}
    op = PoolingOp('pooling_' + layout + '_' + mode,
                   TensorView('x', shape, 'bf16', offset_bytes=2),
                   TensorView('y', yshape, 'bf16', offset_bytes=2), kernel, strides=steps)
    metrics = {}
    actual = execute(op, buffers, compiled, tmp_path, metrics)
    expected = np.array([0x8000, 0, 0x7fc0, 0x7f80 if mode == 'max' else 0x7fc0], dtype='<u2')
    assert actual['y'] == sentinel + expected.tobytes() + sentinel
    assert actual['x'] == buffers['x']
    assert metrics['dyn_retired_vector_alu'] > 0


@pytest.mark.parametrize('layout', ['ncw', 'nchw'])
def test_stride_two_sum_rounds_each_addition(layout, compiled, tmp_path):
    shape = (1, 1, 5) if layout == 'ncw' else (1, 1, 1, 5)
    yshape = (1, 1, 1) if layout == 'ncw' else (1, 1, 1, 1)
    kernel = (3,) if layout == 'ncw' else (1, 3)
    steps = (2,) if layout == 'ncw' else (1, 2)
    op = PoolingOp('pooling_' + layout + '_sum', TensorView('x', shape, 'bf16'),
                   TensorView('y', yshape, 'bf16'), kernel, strides=steps, dilations=steps)
    metrics = {}
    actual = execute(op, {'x': encode([256., 13., 1., 17., -256.]), 'y': encode([0.])},
                     compiled, tmp_path, metrics)
    # BF16 rounds 256 + 1 to 256, so sequential addition returns zero, not one.
    assert actual['y'] == encode([0.])
    assert metrics['dyn_retired_vector_alu'] > 0


def test_stride_two_workspace_respects_address_budget(monkeypatch):
    import codegen.pooling as packing
    monkeypatch.setattr(packing, 'MAX_DATA_ADDRESS', 0x1040)
    op = PoolingOp('pooling_ncw_max', TensorView('x', (1, 1, 2), 'bf16'),
                   TensorView('y', (1, 1, 1), 'bf16'), (2,), strides=(2,))
    with pytest.raises(OverflowError, match='vector staging'):
        emit_pooling(op, {'x': encode([1., 2.]), 'y': encode([0.])})
