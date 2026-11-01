"""Standalone pooling geometry; this is not graph/MLIR dispatch integration."""
from dataclasses import dataclass
import math

from graph.layout import ELEMENT_BYTES, TensorView, _integers, _index_range, validate_view


# Name -> physical layout, spatial rank, reduction, unsigned comparison.
POOLING_OPS = {
    'pooling_ncw_max': ('NCW', 1, 1, False),
    'pooling_ncw_sum': ('NCW', 1, 2, False),
    'pooling_nwc_max': ('NWC', 1, 1, False),
    'pooling_nwc_min': ('NWC', 1, 0, False),
    'pooling_nwc_sum': ('NWC', 1, 2, False),
    'pooling_nwc_max_unsigned': ('NWC', 1, 1, True),
    'pooling_nwc_min_unsigned': ('NWC', 1, 0, True),
    'pooling_nchw_max': ('NCHW', 2, 1, False),
    'pooling_nchw_sum': ('NCHW', 2, 2, False),
    'pooling_nhwc_max': ('NHWC', 2, 1, False),
    'pooling_nhwc_min': ('NHWC', 2, 0, False),
    'pooling_nhwc_sum': ('NHWC', 2, 2, False),
    'pooling_nhwc_max_unsigned': ('NHWC', 2, 1, True),
    'pooling_nhwc_min_unsigned': ('NHWC', 2, 0, True),
    'pooling_ndhwc_max': ('NDHWC', 3, 1, False),
    'pooling_ndhwc_min': ('NDHWC', 3, 0, False),
    'pooling_ndhwc_sum': ('NDHWC', 3, 2, False),
}
TYPE_IDS = {'bf16': 0, 'f32': 1, 'i32': 2, 'u32': 3, 'f16': 4}


@dataclass(frozen=True)
class PoolingOp:
    name: str
    source: TensorView
    destination: TensorView
    kernel_shape: tuple
    strides: tuple = ()
    dilations: tuple = ()

    def __post_init__(self):
        for field in ('kernel_shape', 'strides', 'dilations'):
            object.__setattr__(self, field, _integers(getattr(self, field)))


@dataclass(frozen=True)
class PoolingPlan:
    shape: tuple
    source_strides: tuple
    destination_strides: tuple
    kernel_shape: tuple
    strides: tuple
    dilations: tuple
    elements: int
    kernel_elements: int
    mode: int
    dtype: int
    element_bytes: int


def lower_pooling(op, buffers):
    name = op.name.removeprefix('linalg.')
    if name not in POOLING_OPS:
        raise NotImplementedError(f'unsupported pooling operation: {op.name}')
    layout, spatial_rank, mode, unsigned = POOLING_OPS[name]
    src, dst = op.source, op.destination
    validate_view(src, buffers)
    validate_view(dst, buffers, writable=True)
    if src.dtype != dst.dtype:
        raise NotImplementedError('pooling cross-dtype casts are not implemented')
    if src.dtype not in TYPE_IDS:
        raise NotImplementedError('pooling dtype is not implemented')
    if unsigned and src.dtype != 'u32':
        raise NotImplementedError('unsigned pooling requires u32 storage')
    if not unsigned and mode != 2 and src.dtype == 'u32':
        raise NotImplementedError('signed min/max pooling requires i32 or floating dtype')
    if mode == 2 and src.dtype not in ('bf16', 'i32', 'u32'):
        raise NotImplementedError('sum pooling dtype is not implemented')
    rank = spatial_rank + 2
    if len(src.shape) != rank or len(dst.shape) != rank:
        raise ValueError(f'{name} requires rank {rank} tensors')
    axes = (0, *range(2, rank), 1) if layout[1] == 'C' else tuple(range(rank))
    source_shape = tuple(src.shape[i] for i in axes)
    destination_shape = tuple(dst.shape[i] for i in axes)
    if source_shape[0] != destination_shape[0] or source_shape[-1] != destination_shape[-1]:
        raise ValueError('pooling source/output batch and channels must match')
    kernel = op.kernel_shape
    strides = op.strides or (1,) * spatial_rank
    dilations = op.dilations or (1,) * spatial_rank
    if len(kernel) != spatial_rank or any(k < 0 for k in kernel):
        raise ValueError('kernel shape must have nonnegative spatial extents')
    if len(strides) != spatial_rank or any(s <= 0 for s in strides):
        raise ValueError('pooling strides must be positive per spatial axis')
    if len(dilations) != spatial_rank or any(d <= 0 for d in dilations):
        raise ValueError('pooling dilations must be positive per spatial axis')
    for value in (*kernel, *strides, *dilations):
        _index_range(value)
    elements = math.prod(destination_shape)
    kernel_elements = _index_range(math.prod(kernel))
    if elements and kernel_elements:
        for size, out, k, step, dilation in zip(source_shape[1:-1], destination_shape[1:-1], kernel, strides, dilations):
            if (out - 1) * step + (k - 1) * dilation >= size:
                raise ValueError('pooling output window exceeds input shape')
    return PoolingPlan(destination_shape, tuple(src.strides_bytes[i] for i in axes),
                       tuple(dst.strides_bytes[i] for i in axes), kernel, strides, dilations,
                       elements, kernel_elements, mode, TYPE_IDS[src.dtype], ELEMENT_BYTES[src.dtype])
