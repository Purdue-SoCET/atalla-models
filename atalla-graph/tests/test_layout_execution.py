"""Compiled data movement, with independent byte and NumPy goldens."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.layout import LayoutOp, TensorView
from layout_runtime import execute_layout


DTYPES = [('i8', 1), ('u8', 1), ('i16', 2), ('u16', 2), ('f16', 2), ('bf16', 2),
          ('i32', 4), ('u32', 4), ('f32', 4), ('i64', 8), ('u64', 8), ('f64', 8)]


@pytest.mark.parametrize('dtype,width', DTYPES)
def test_copy_preserves_bits_and_unaligned_neighbors(dtype, width, tmp_path):
    # Includes negative zero/NaN-like patterns; movement must never convert them.
    pattern = bytes.fromhex('0080ff7f0000c07fffffffff0100000080')
    payload = (pattern * 4)[:5 * width]
    original = b'abc' + payload + b'xy'
    destination = b'\xa5' * (len(payload) + 10)
    op = LayoutOp('linalg.copy', TensorView('a', (5,), dtype, offset_bytes=3),
                  TensorView('b', (5,), dtype, offset_bytes=5))
    result = execute_layout(op, {'a': original, 'b': destination}, tmp_path)
    assert result.backend == 'emulator'
    assert result.buffers == {'a': original, 'b': b'\xa5' * 5 + payload + b'\xa5' * 5}


@pytest.mark.parametrize('dtype,width', [('u8', 1), ('bf16', 2), ('f32', 4), ('f64', 8)])
def test_fill_strided_output_preserves_padding(dtype, width, tmp_path):
    value = bytes.fromhex('ff80c07f00000080')[:width]
    initial = b'\x93' * (3 + 14 * width)
    expected = bytearray(initial)
    view = np.ndarray((2, 3), dtype=f'V{width}', buffer=expected, offset=1,
                      strides=(8 * width, 2 * width))
    view[...] = np.void(value)
    op = LayoutOp('fill', TensorView('s', (), dtype),
                  TensorView('o', (2, 3), dtype, offset_bytes=1,
                             strides_bytes=(8 * width, 2 * width)))
    result = execute_layout(op, {'s': value, 'o': initial}, tmp_path)
    assert result.buffers == {'s': value, 'o': bytes(expected)}


@pytest.mark.parametrize('dtype,numpy_dtype,width', [('i16', '<i2', 2), ('f32', '<f4', 4), ('u64', '<u8', 8)])
def test_transpose_negative_source_strides_materializes_permutation(dtype, numpy_dtype, width, tmp_path):
    array = np.arange(24, dtype=numpy_dtype).reshape(2, 3, 4)
    expected = array[:, :, ::-1].transpose(2, 0, 1).tobytes()
    op = LayoutOp('transpose', TensorView('a', (2, 3, 4), dtype,
                                        strides_bytes=(12 * width, 4 * width, -width), offset_bytes=3 * width),
                  TensorView('b', (4, 2, 3), dtype, offset_bytes=4), permutation=(2, 0, 1))
    result = execute_layout(op, {'a': array.tobytes(), 'b': b'!' * (24 * width + 8)}, tmp_path)
    assert result.buffers['b'] == b'!' * 4 + expected + b'!' * 4
    assert result.buffers['a'] == array.tobytes()


@pytest.mark.parametrize('dtype,numpy_dtype,width', [('u8', 'u1', 1), ('u16', '<u2', 2), ('f64', '<f8', 8)])
def test_broadcast_inserts_multiple_output_axes(dtype, numpy_dtype, width, tmp_path):
    a = np.arange(6, dtype=numpy_dtype).reshape(2, 3)
    op = LayoutOp('broadcast', TensorView('a', (2, 3), dtype),
                  TensorView('b', (4, 2, 5, 3), dtype), dimensions=(0, 2))
    result = execute_layout(op, {'a': a.tobytes(), 'b': b'!' * (120 * width)}, tmp_path)
    assert result.buffers['b'] == np.broadcast_to(a[None, :, None, :], (4, 2, 5, 3)).tobytes()


@pytest.mark.parametrize('src,dst', [(0, 2), (2, 0)])
def test_overlapping_copy_uses_source_snapshot(src, dst, tmp_path):
    original = bytes(range(11))
    expected = bytearray(original)
    expected[dst:dst + 7] = original[src:src + 7]
    op = LayoutOp('copy', TensorView('a', (7,), 'u8', offset_bytes=src),
                  TensorView('a', (7,), 'u8', offset_bytes=dst))
    assert execute_layout(op, {'a': original}, tmp_path).buffers['a'] == bytes(expected)


def test_in_place_nonsquare_transpose_uses_source_snapshot(tmp_path):
    a = np.arange(6, dtype='<u4').reshape(2, 3)
    op = LayoutOp('transpose', TensorView('a', (2, 3), 'u32'),
                  TensorView('a', (3, 2), 'u32'), permutation=(1, 0))
    assert execute_layout(op, {'a': a.tobytes()}, tmp_path).buffers['a'] == a.T.tobytes()


def test_fill_scalar_partially_aliases_first_output_element(tmp_path):
    op = LayoutOp('fill', TensorView('a', (), 'u16', offset_bytes=1),
                  TensorView('a', (4,), 'u16'))
    assert execute_layout(op, {'a': b'abcdefghi'}, tmp_path).buffers['a'] == b'bc' * 4 + b'i'


def test_copy_zero_stride_source_and_negative_stride_destination(tmp_path):
    op = LayoutOp('copy', TensorView('a', (4,), 'u8', strides_bytes=(0,), offset_bytes=1),
                  TensorView('b', (4,), 'u8', strides_bytes=(-1,), offset_bytes=4))
    assert execute_layout(op, {'a': b'xy', 'b': b'!!!!!!'}, tmp_path).buffers['b'] == b'!yyyy!'


@pytest.mark.parametrize('name', ['copy', 'fill', 'broadcast', 'transpose'])
def test_empty_outputs_do_not_write(name, tmp_path):
    source = TensorView('a', (), 'f32') if name == 'fill' else TensorView('a', (0,), 'f32', offset_bytes=4)
    destination = TensorView('b', (0,), 'f32', offset_bytes=4)
    op = LayoutOp(name, source, destination, permutation=(0,) if name == 'transpose' else ())
    buffers = {'a': b'1234', 'b': b'5678'}
    assert execute_layout(op, buffers, tmp_path).buffers == buffers


@pytest.mark.parametrize('name', ['copy', 'fill', 'broadcast', 'transpose'])
def test_rank_zero_output(name, tmp_path):
    op = LayoutOp(name, TensorView('a', (), 'u8'), TensorView('b', (), 'u8', offset_bytes=1))
    assert execute_layout(op, {'a': b'Z', 'b': b'abc'}, tmp_path).buffers == {'a': b'Z', 'b': b'aZc'}


def test_layout_chain_consumes_previous_execution_buffers(tmp_path):
    a = np.arange(6, dtype='u1').reshape(2, 3)
    buffers = {'a': a.tobytes(), 'b': b'!' * 6, 'c': b'!' * 6, 'd': b'!' * 12}
    operations = [
        LayoutOp('copy', TensorView('a', (2, 3), 'u8'), TensorView('b', (2, 3), 'u8')),
        LayoutOp('transpose', TensorView('b', (2, 3), 'u8'), TensorView('c', (3, 2), 'u8'), permutation=(1, 0)),
        LayoutOp('broadcast', TensorView('c', (3, 2), 'u8'), TensorView('d', (2, 3, 2), 'u8'), dimensions=(0,)),
    ]
    for i, operation in enumerate(operations):
        result = execute_layout(operation, buffers, tmp_path, tag=f'chain{i}')
        assert result.backend == 'emulator'
        buffers = result.buffers
        # Later operations must use the actual copy result, not the original input.
        buffers['a'] = b'\xff' * 6
    assert buffers['d'] == np.broadcast_to(a.T[None], (2, 3, 2)).tobytes()
