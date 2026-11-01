"""Typed, strided contracts for normalized Linalg data movement.

Strides and offsets are bytes. These contracts describe materialized operations,
not metadata aliases. Numerical casts and sub-byte/vector types are unsupported.
"""
from dataclasses import dataclass
import math
import operator
from typing import Mapping


ELEMENT_BYTES = {
    'i8': 1, 'u8': 1, 'i16': 2, 'u16': 2, 'f16': 2, 'bf16': 2,
    'i32': 4, 'u32': 4, 'f32': 4, 'i64': 8, 'u64': 8, 'f64': 8,
}
INT_MAX = (1 << 31) - 1


def _integer(value):
    if isinstance(value, bool):
        raise TypeError('boolean is not integer layout metadata')
    return operator.index(value)


def _integers(values):
    return tuple(_integer(value) for value in values)


def _index_range(value):
    if not -INT_MAX - 1 <= value <= INT_MAX:
        raise OverflowError('layout metadata exceeds signed 32-bit index range')
    return value


@dataclass(frozen=True)
class TensorView:
    buffer: str
    shape: tuple
    dtype: str
    strides_bytes: tuple | None = None
    offset_bytes: int = 0

    def __post_init__(self):
        if not isinstance(self.buffer, str) or not self.buffer:
            raise ValueError('buffer must be a nonempty allocation name')
        if self.dtype not in ELEMENT_BYTES:
            raise ValueError(f'unsupported scalar storage dtype: {self.dtype}')
        shape = _integers(self.shape)
        if any(size < 0 for size in shape):
            raise ValueError('shape extents must be nonnegative')
        for size in shape:
            _index_range(size)
        _index_range(math.prod(shape))
        if self.strides_bytes is None:
            strides = []
            span = ELEMENT_BYTES[self.dtype]
            for size in reversed(shape):
                strides.append(span)
                span *= size
            strides = tuple(reversed(strides))
        else:
            strides = _integers(self.strides_bytes)
        if len(strides) != len(shape):
            raise ValueError('strides rank must match shape rank')
        for stride in strides:
            _index_range(stride)
        offset = _index_range(_integer(self.offset_bytes))
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'strides_bytes', strides)
        object.__setattr__(self, 'offset_bytes', offset)


@dataclass(frozen=True)
class LayoutOp:
    name: str
    source: TensorView
    destination: TensorView
    dimensions: tuple = ()
    permutation: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'dimensions', _integers(self.dimensions))
        object.__setattr__(self, 'permutation', _integers(self.permutation))


@dataclass(frozen=True)
class LayoutPlan:
    source: TensorView
    destination: TensorView
    source_strides_bytes: tuple
    elements: int
    element_bytes: int


def _check_bounds(view, length):
    if math.prod(view.shape) == 0:
        start = end = view.offset_bytes
    else:
        start = end = view.offset_bytes
        for size, stride in zip(view.shape, view.strides_bytes):
            extent = (size - 1) * stride
            start += min(0, extent)
            end += max(0, extent)
        end += ELEMENT_BYTES[view.dtype]
    if start < 0 or end > length:
        raise ValueError(f'view for buffer {view.buffer!r} exceeds allocation bounds')
    _index_range(start)
    _index_range(end)


def _check_output_overlap(view):
    if math.prod(view.shape) == 0:
        return
    span = ELEMENT_BYTES[view.dtype]
    axes = sorted((abs(stride), size) for size, stride in
                  zip(view.shape, view.strides_bytes) if size > 1)
    for stride, size in axes:
        if stride < span:
            raise ValueError('destination elements may overlap; layout not provably nonoverlapping')
        span += (size - 1) * stride


def validate_view(view: TensorView, buffers: Mapping[str, bytes], *, writable=False):
    """Check storage bounds and, for outputs, prove element nonoverlap."""
    if view.buffer not in buffers:
        raise ValueError(f'missing buffer: {view.buffer}')
    if not isinstance(buffers[view.buffer], (bytes, bytearray)):
        raise TypeError('buffer allocations must contain bytes or bytearray')
    _check_bounds(view, len(buffers[view.buffer]))
    if writable:
        _check_output_overlap(view)


def lower_layout(op: LayoutOp, buffers: Mapping[str, bytes]) -> LayoutPlan:
    """Validate one operation and map source byte strides to output axes.

    No buffer payload is read or transformed here. Overlapping source/destination
    allocations are valid; the execution backend must snapshot before writing.
    """
    name = op.name.removeprefix('linalg.')
    if name not in ('copy', 'fill', 'broadcast', 'transpose'):
        raise ValueError(f'unsupported layout operation: {op.name}')
    src, dst = op.source, op.destination
    validate_view(src, buffers)
    validate_view(dst, buffers, writable=True)
    if src.dtype != dst.dtype:
        raise NotImplementedError('cross-dtype layout operations require numerical cast lowering')
    rank = len(dst.shape)
    if name in ('copy', 'fill'):
        if op.dimensions or op.permutation:
            raise ValueError(f'{name} does not accept layout attributes')
        if name == 'fill':
            if src.shape != ():
                raise ValueError('fill requires a rank-zero scalar source')
            strides = (0,) * rank
        else:
            if src.shape != dst.shape:
                raise ValueError('copy source and destination shape must match')
            strides = src.strides_bytes
    elif name == 'broadcast':
        dimensions = op.dimensions
        if op.permutation:
            raise ValueError('broadcast does not accept permutation attributes')
        if len(set(dimensions)) != len(dimensions) or any(d < 0 or d >= rank for d in dimensions):
            raise ValueError('broadcast dimensions must be unique output axes')
        remaining = tuple(i for i in range(rank) if i not in dimensions)
        if tuple(dst.shape[i] for i in remaining) != src.shape:
            raise ValueError('broadcast shape must match after removing inserted dimensions')
        mapping = dict(zip(remaining, src.strides_bytes))
        strides = tuple(mapping.get(i, 0) for i in range(rank))
    else:
        permutation = op.permutation
        if op.dimensions:
            raise ValueError('transpose does not accept dimensions attributes')
        if len(permutation) != len(src.shape) or sorted(permutation) != list(range(len(src.shape))):
            raise ValueError('transpose permutation must be a bijection of source axes')
        if tuple(src.shape[i] for i in permutation) != dst.shape:
            raise ValueError('transpose destination shape must follow permutation')
        strides = tuple(src.strides_bytes[i] for i in permutation)
    elements = math.prod(dst.shape)
    width = ELEMENT_BYTES[dst.dtype]
    _index_range(elements * width)
    return LayoutPlan(src, dst, strides, elements, width)
