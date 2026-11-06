"""Standalone named depthwise convolution contracts; no FX/MLIR dispatch."""
from dataclasses import dataclass
import math

from graph.layout import TensorView, _integers, _index_range, validate_view


# Name -> physical input layout, spatial rank, explicit multiplier axis.
DEPTHWISE_OPS = {
    'depthwise_conv_1d_ncw_cw': ('NCW', 1, False),
    'depthwise_conv_1d_nwc_wc': ('NWC', 1, False),
    'depthwise_conv_1d_nwc_wcm': ('NWC', 1, True),
    'depthwise_conv_2d_nchw_chw': ('NCHW', 2, False),
    'depthwise_conv_2d_nhwc_hwc': ('NHWC', 2, False),
    'depthwise_conv_2d_nhwc_hwcm': ('NHWC', 2, True),
    'depthwise_conv_3d_ncdhw_cdhw': ('NCDHW', 3, False),
    'depthwise_conv_3d_ndhwc_dhwc': ('NDHWC', 3, False),
    'depthwise_conv_3d_ndhwc_dhwcm': ('NDHWC', 3, True),
}


@dataclass(frozen=True)
class DepthwiseOp:
    name: str
    source: TensorView
    weights: TensorView
    destination: TensorView
    strides: tuple = ()
    dilations: tuple = ()

    def __post_init__(self):
        for field in ('strides', 'dilations'):
            object.__setattr__(self, field, _integers(getattr(self, field)))


@dataclass(frozen=True)
class DepthwisePlan:
    elements: int
    kernel_elements: int
    dtype: int
    element_bytes: int
    kernel_shape: tuple
    strides: tuple
    dilations: tuple


def lower_depthwise(op, buffers):
    name = op.name.removeprefix('linalg.')
    if name not in DEPTHWISE_OPS:
        raise NotImplementedError(f'unsupported depthwise operation: {op.name}')
    layout, rank, multiplier = DEPTHWISE_OPS[name]
    source, weights, output = op.source, op.weights, op.destination
    validate_view(source, buffers)
    validate_view(weights, buffers)
    validate_view(output, buffers, writable=True)
    if source.dtype != weights.dtype or source.dtype != output.dtype:
        raise NotImplementedError('depthwise mixed dtype casts are not implemented')
    if source.dtype not in ('bf16', 'i32'):
        raise NotImplementedError('depthwise supports matching bf16 or i32 storage')
    if (len(source.shape) != rank + 2 or len(weights.shape) != rank + 1 + multiplier
            or len(output.shape) != rank + 2 + multiplier):
        raise ValueError('depthwise tensor ranks do not match named layout')
    first = layout[1] == 'C'
    source_channel = 1 if first else rank + 1
    weight_channel = 0 if first else rank
    output_channel = 1 if first else rank + 1
    if not (source.shape[source_channel] == weights.shape[weight_channel]
            == output.shape[output_channel]):
        raise ValueError('depthwise input/filter/output channels must match')
    if source.shape[0] != output.shape[0]:
        raise ValueError('depthwise input/output batches must match')
    if multiplier and weights.shape[-1] != output.shape[-1]:
        raise ValueError('depthwise filter/output multiplier extents must match')
    spatial_axes = tuple(range(2, rank + 2)) if first else tuple(range(1, rank + 1))
    kernel_shape = weights.shape[1:] if first else weights.shape[:rank]
    strides = op.strides or (1,) * rank
    dilations = op.dilations or (1,) * rank
    if len(strides) != rank or any(step <= 0 for step in strides):
        raise ValueError('depthwise strides must be positive per spatial axis')
    if len(dilations) != rank or any(d <= 0 for d in dilations):
        raise ValueError('depthwise dilations must be positive per spatial axis')
    for value in (*strides, *dilations):
        _index_range(value)
    elements = _index_range(math.prod(output.shape))
    kernel_elements = _index_range(math.prod(kernel_shape))
    if elements and kernel_elements:
        for axis, k, step, dilation in zip(spatial_axes, kernel_shape, strides, dilations):
            if (output.shape[axis] - 1) * step + (k - 1) * dilation >= source.shape[axis]:
                raise ValueError('depthwise output window exceeds input shape')
    return DepthwisePlan(elements, kernel_elements, int(source.dtype == 'i32'),
                         2 if source.dtype == 'bf16' else 4, kernel_shape, strides, dilations)
