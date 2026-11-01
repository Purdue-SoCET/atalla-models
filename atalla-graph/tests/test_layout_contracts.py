"""Normalized Linalg movement contracts, without compiler or simulator imports."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.layout import LayoutOp, TensorView, lower_layout


def test_copy_preserves_explicit_source_and_destination_strides():
    source = TensorView('a', (2, 3), 'i16', strides_bytes=(8, -2), offset_bytes=4)
    destination = TensorView('b', (2, 3), 'i16')
    plan = lower_layout(LayoutOp('linalg.copy', source, destination), {'a': bytes(14), 'b': bytes(12)})
    assert plan.source_strides_bytes == (8, -2)
    assert plan.destination.strides_bytes == (6, 2)
    assert (plan.elements, plan.element_bytes) == (6, 2)


def test_broadcast_dimensions_are_inserted_output_axes():
    op = LayoutOp('linalg.broadcast', TensorView('a', (2, 3), 'u8'),
                  TensorView('b', (4, 2, 5, 3), 'u8'), dimensions=(0, 2))
    plan = lower_layout(op, {'a': bytes(6), 'b': bytes(120)})
    assert plan.source_strides_bytes == (0, 3, 0, 1)
    assert plan.elements == 120


def test_transpose_maps_source_strides_in_permutation_order():
    op = LayoutOp('linalg.transpose', TensorView('a', (2, 3, 4), 'f32'),
                  TensorView('b', (4, 2, 3), 'f32'), permutation=(2, 0, 1))
    assert lower_layout(op, {'a': bytes(96), 'b': bytes(96)}).source_strides_bytes == (4, 48, 16)


def test_fill_repeats_scalar_for_any_rank():
    op = LayoutOp('linalg.fill', TensorView('s', (), 'bf16'), TensorView('b', (2, 3), 'bf16'))
    assert lower_layout(op, {'s': bytes(2), 'b': bytes(12)}).source_strides_bytes == (0, 0)


def test_empty_output_allows_one_past_end_and_skips_overlap_check():
    view = TensorView('a', (2, 0, 3), 'u8', strides_bytes=(0, 0, 0), offset_bytes=1)
    plan = lower_layout(LayoutOp('copy', view, view), {'a': b'!'})
    assert plan.elements == 0


def test_read_zero_strides_and_negative_destination_strides_are_valid():
    op = LayoutOp('copy', TensorView('a', (3,), 'i32', strides_bytes=(0,)),
                  TensorView('b', (3,), 'i32', strides_bytes=(-4,), offset_bytes=8))
    assert lower_layout(op, {'a': bytes(4), 'b': bytes(12)}).source_strides_bytes == (0,)


@pytest.mark.parametrize('op,buffers,message', [
    (LayoutOp('copy', TensorView('a', (2,), 'i8'), TensorView('b', (3,), 'i8')),
     {'a': bytes(2), 'b': bytes(3)}, 'shape'),
    (LayoutOp('fill', TensorView('a', (1,), 'i8'), TensorView('b', (3,), 'i8')),
     {'a': bytes(1), 'b': bytes(3)}, 'scalar'),
    (LayoutOp('broadcast', TensorView('a', (1,), 'i8'), TensorView('b', (3,), 'i8')),
     {'a': bytes(1), 'b': bytes(3)}, 'shape'),
    (LayoutOp('broadcast', TensorView('a', (), 'i8'), TensorView('b', (2, 3), 'i8'), dimensions=(0, 0)),
     {'a': b'!', 'b': bytes(6)}, 'dimensions'),
    (LayoutOp('broadcast', TensorView('a', (2,), 'i8'), TensorView('b', (2, 3), 'i8'), dimensions=(2,)),
     {'a': bytes(2), 'b': bytes(6)}, 'dimensions'),
    (LayoutOp('transpose', TensorView('a', (2, 3), 'i8'), TensorView('b', (3, 2), 'i8'), permutation=(0, 0)),
     {'a': bytes(6), 'b': bytes(6)}, 'permutation'),
    (LayoutOp('transpose', TensorView('a', (2, 3), 'i8'), TensorView('b', (2, 3), 'i8'), permutation=(1, 0)),
     {'a': bytes(6), 'b': bytes(6)}, 'shape'),
    (LayoutOp('copy', TensorView('a', (3,), 'i8', strides_bytes=(-1,)), TensorView('b', (3,), 'i8')),
     {'a': bytes(3), 'b': bytes(3)}, 'bounds'),
    (LayoutOp('copy', TensorView('a', (3,), 'i16'), TensorView('b', (3,), 'i16')),
     {'a': bytes(5), 'b': bytes(6)}, 'bounds'),
    (LayoutOp('copy', TensorView('a', (3,), 'i8'), TensorView('b', (3,), 'i8', strides_bytes=(0,))),
     {'a': bytes(3), 'b': bytes(3)}, 'overlap'),
    (LayoutOp('copy', TensorView('a', (2,), 'i16'), TensorView('b', (2,), 'i16', strides_bytes=(1,))),
     {'a': bytes(4), 'b': bytes(4)}, 'overlap'),
    (LayoutOp('copy', TensorView('a', (2,), 'u8'), TensorView('b', (2,), 'u8'), dimensions=(0,)),
     {'a': bytes(2), 'b': bytes(2)}, 'attributes'),
])
def test_invalid_operation_contracts_fail_explicitly(op, buffers, message):
    with pytest.raises(ValueError, match=message):
        lower_layout(op, buffers)


def test_cross_dtype_copy_is_explicitly_unimplemented():
    op = LayoutOp('copy', TensorView('a', (1,), 'i32'), TensorView('b', (1,), 'f32'))
    with pytest.raises(NotImplementedError, match='dtype'):
        lower_layout(op, {'a': bytes(4), 'b': bytes(4)})


@pytest.mark.parametrize('kwargs', [dict(shape=(-1,), dtype='i8'),
                                   dict(shape=(2.5,), dtype='i8'),
                                   dict(shape=(2,), dtype='i8', strides_bytes=(1.5,)),
                                   dict(shape=(2,), dtype='i8', offset_bytes=.5),
                                   dict(shape=(2,), dtype='i8', strides_bytes=()),
                                   dict(shape=(2,), dtype='i1')])
def test_invalid_view_metadata_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        TensorView('a', **kwargs)


@pytest.mark.parametrize('kwargs', [dict(shape=(1 << 31,), dtype='u8'),
                                   dict(shape=(65536, 65536), dtype='u8'),
                                   dict(shape=(1,), dtype='u8', strides_bytes=(1 << 31,))])
def test_index_overflow_is_rejected(kwargs):
    with pytest.raises(OverflowError):
        TensorView('a', **kwargs)


def test_missing_buffer_and_unknown_operation_are_rejected():
    view = TensorView('a', (), 'i8')
    with pytest.raises(ValueError, match='buffer'):
        lower_layout(LayoutOp('copy', view, view), {})
    with pytest.raises(ValueError, match='operation'):
        lower_layout(LayoutOp('pack', view, view), {'a': b'!'})
