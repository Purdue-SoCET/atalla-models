# Kernel Support Overview

## Kernels declared in `kernels/kernels.h`
All prototypes listed below have stub implementations in `kernels/kernels.c`:
- `matmul_kernel`, `add_kernel`, `mul_kernel`, `mul_scalar_kernel`
- `conv_kernel`
- `relu_kernel`, `softmax_kernel`
- `batchnorm_kernel`, `layernorm_kernel`
- `gelu_kernel`
- `avgpool_kernel`, `maxpool_kernel`
- `upsample_kernel`

## Ops currently lowered in `scripts/generate_schedule.py`
AlexNet-only lowering currently emits just these kernels:
- `torch.matmul`, `torch.ops.aten.matmul.default`
- `operator.add` / `torch.add` (with tensor×scalar mul folded into alpha)
- Tensor×tensor or tensor×scalar `mul`
- `F.conv2d`
- `F.relu` / `nn.ReLU`
- `F.avg_pool2d`, `nn.AdaptiveAvgPool2d`, `tensor.mean(...)`
- `nn.MaxPool2d`
- `F.softmax` / `nn.Softmax`

Other PyTorch ops (BatchNorm, LayerNorm, GELU, Upsample, etc.) still have kernel prototypes but are not emitted by `generate_schedule.py`.
Unsupported ops will cause an error. 

## Normalized data/layout operations

`graph.layout` defines `TensorView`, `LayoutOp`, and `lower_layout` for
`linalg.copy`, `linalg.fill`, `linalg.broadcast`, and `linalg.transpose` (the
unprefixed names are also accepted). These are executable operation descriptors
for later compiler lowering; they do not parse textual MLIR and are not yet FX
operator adapters.

A view identifies a named allocation, logical shape, scalar storage dtype,
optional **byte strides**, and a byte offset. Default strides are contiguous
row-major. Buffers are little-endian `bytes` or `bytearray`; returned buffers are
immutable bytes. Rank-zero tensors contain one scalar; zero extents produce no
writes. Negative/zero source strides and provably nonoverlapping strided outputs
are supported. Bounds, shape, permutation, and address overflow errors fail
explicitly before execution. The conservative output-layout check can reject
some unusual nonoverlapping stride combinations.

```python
from graph.layout import LayoutOp, TensorView
from layout_runtime import execute_layout

# Materialize a 2x3 u8 matrix as a 3x2 transpose.
operation = LayoutOp(
    'linalg.transpose',
    TensorView('input', (2, 3), 'u8'),
    TensorView('output', (3, 2), 'u8'),
    permutation=(1, 0),
)
result = execute_layout(operation, {
    'input': bytes([0, 1, 2, 3, 4, 5]),
    'output': bytes(6),
}, 'out/layout')
assert result.buffers['output'] == bytes([0, 3, 1, 4, 2, 5])
# Pass result.buffers into the next operation to chain actual execution values.
```

Copy requires equal shapes. Fill takes a rank-zero scalar source. Broadcast's
`dimensions` identifies **inserted output axes**: the remaining output extents
must exactly match the input shape; this is not NumPy singleton expansion.
Transpose's `permutation` is a bijection with
`output.shape[i] == input.shape[permutation[i]]` and actually moves data.

Same-dtype bit movement supports `i8/u8/i16/u16/i32/u32/i64/u64/f16/bf16/f32/f64`,
including signed zero and NaN payloads. Copy casts, sub-byte/vector/complex types
and other Linalg operations remain unsupported. No numerical conversion occurs.
Buffer names define alias identity. Source/destination overlap uses a temporary
snapshot, including in-place transpose. Unaddressed allocation bytes are preserved.

Execution compiles the ordinary `kernels/layout.c`, with all shapes, strides and
addresses in runtime configuration. The scalar C kernel gathers source bytes
then scatters to the destination; Python only prepares config, serializes input,
and reads actual simulator memory. The reported backend is `emulator`, with no
host/reference fallback. This correctness path is not a vector/DMA optimization
or hardware timing claim. The canonical simulator must include the signed-branch
comparison correction required by reverse-axis loops.

## Standalone pooling family

`PoolingOp` and `execute_pooling` provide a standalone compiled kernel boundary.
Graph `MaxPool2d` uses the NCHW max kernel through this boundary; the other
names remain standalone, and textual-MLIR dispatch is not implemented. Each of the 17 pooling names in the
pinned Linalg inventory has its own ordinary C source, such as
`kernels/pooling_nhwc_min.c`. The supported layouts are: NCW/NWC 1D,
NCHW/NHWC 2D, and NDHWC 3D min/max/sum where those named variants exist, plus
NWC/NHWC unsigned min/max. Each kernel fixes its reduction, physical layout and
spatial rank in C, with explicit spatial loops. Python selects that source and
supplies shapes, physical byte strides, window sizes, spatial strides, dilations
and dtype through runtime configuration. Rank and reduction are not runtime
selectors. No PPCI source changes are required.

```python
import numpy as np
from graph.layout import TensorView
from graph.pooling import PoolingOp
from pooling_runtime import execute_pooling

x = np.array([[[[3.], [-2.]]]], dtype='<f4')  # NHWC
initial = np.array([[[[10.]]]], dtype='<f4')
operation = PoolingOp(
    'linalg.pooling_nhwc_min',
    TensorView('input', x.shape, 'f32'),
    TensorView('output', initial.shape, 'f32'),
    kernel_shape=(1, 2),
)
result = execute_pooling(operation, {
    'input': x.tobytes(), 'output': initial.tobytes(),
}, 'out/pooling')
assert np.frombuffer(result.buffers['output'], dtype='<f4')[0] == -2.
```

The **existing output values participate** in reduction. To perform ordinary min
pooling, initialize output to positive infinity; max uses negative infinity and
sum uses zero. Smaller/larger explicit initial values are also respected.
The output view determines iteration extents, which must fit valid windows of
the input. There is no implicit padding. Window operand values do not contribute;
its spatial shape defines the reduction. Empty windows preserve output bits;
empty outputs perform no output writes. Batch and channel dimensions must match.

| Operation subset | Supported same input/output storage |
|---|---|
| Ordinary min/max | BF16, f16, f32, i32 |
| Unsigned min/max | u32 |
| Sum | BF16, i32, u32 |

Float extrema propagate NaNs and distinguish signed-zero ties (min selects -0,
max selects +0). BF16 sum rounds to nearest-even after **each** addition in
lexicographic window order. Integer sums wrap modulo 2^32. Mixed input/output
casts, f16/f32 sums, other widths, vectors and complex values are explicitly
unsupported. This is bounded standalone coverage, not full Linalg type support.

The API uses the existing `TensorView` byte-stride/bounds contract, including
negative/zero input strides and provably nonoverlapping strided output. When
input/output share an allocation name, C snapshots the input allocation before
writing output. Unaddressed bytes are preserved. The returned buffers contain
actual simulator values and may feed another kernel invocation.

BF16 has guarded vector paths: channel-last lanes span channels, while NCW/NCHW
lanes span width for steps one and two. Width step two uses a 64-byte C staging
buffer to gather even result lanes. Scalars handle configuration, staging,
canonical NaNs and signed-zero repair; other supported dtypes/layouts retain the
scalar path. Compiler spills remain, and simulator packet counts are not hardware
cycle measurements.

Graph `MaxPool2d` emits the complete output through `pooling_nchw_max.c`, including
widths above 32. Its planner preserves batches, CHW/NCHW, rectangular kernels and
strides, padding, dilation and ceil mode. Input packing materializes negative
infinity padding, including any trailing ceil window. The runner performs no
horizontal pooling reduction, and wide tensors do not take a NumPy pooling route.
Both module and functional index-returning pooling fail explicitly: this kernel
returns values only.


## Graph GELU

Graph GELU executes `kernels/gelu_piecewise.c` in the simulator. Module and
functional calls preserve `approximate="none"` or `"tanh"`, with separate
coefficient tables for the erf and tanh references. Four centered cubic intervals
use vector arithmetic, evaluating the CDF ratio near zero and the small tail
correction elsewhere to reduce cancellation. Finite tails beyond magnitude four
use identity/signed-zero limits. The implementation is a numerical approximation,
not bit-exact PyTorch GELU. Signed zeros, NaNs and infinities are covered, including
the erf reference's extreme positive overflow ordering.

Python packs BF16 input, padded DMA tiles and constant coefficients; it does not
compute execution GELU outputs. Logical output sizes include scalar, empty and
partial tiles. FP32 coefficient words are read through four-byte union storage
because ordinary C `float` storage is BF16-sized in the current frontend.
The older Horner/tanh sources remain available for standalone experiments.

GELU reporting `emulator` does not mean every graph operation is compiled. The
validated flash ViT still has one explicit NumPy broadcast add. Simulator packets
and static spill counts are not hardware-cycle or end-to-end speedup measurements.


## Standalone depthwise convolution

`DepthwiseOp` and `execute_depthwise` run dedicated ordinary-C kernels for these
non-quantized named subsets. Each name has its own entry point and layout-specific loops. Identical byte
access, BF16 rounding and scalar multiply/add helpers live in
`kernels/depthwise_common.h`; Python shares validation, allocation and runtime
configuration. The compilation wrapper supplies the kernel include directory.
For manual compilation, pass `-I atalla-graph/kernels`; moving a C source also
requires the shared header.

| Spatial rank | Channel-first | Channel-last | Explicit multiplier |
|---|---|---|---|
| 1D | `depthwise_conv_1d_ncw_cw` | `depthwise_conv_1d_nwc_wc` | `depthwise_conv_1d_nwc_wcm` |
| 2D | `depthwise_conv_2d_nchw_chw` | `depthwise_conv_2d_nhwc_hwc` | `depthwise_conv_2d_nhwc_hwcm` |
| 3D | `depthwise_conv_3d_ncdhw_cdhw` | `depthwise_conv_3d_ndhwc_dhwc` | `depthwise_conv_3d_ndhwc_dhwcm` |

Input, filter and initialized output must have matching BF16 or i32 storage.
BF16 rounds each product and each addition separately; i32 arithmetic wraps
modulo32. Spatial steps and dilations are runtime metadata. Output windows must
fit the input; there is no implicit padding. Empty reductions preserve initial
output bits, and empty outputs write nothing. TensorViews support byte strides,
negative input/filter strides and nonoverlapping strided output. The C kernel
snapshots either operand when it aliases output, before any writes.

The basic BF16 vector paths span contiguous channels, multiplier lanes, or output
width for channel-first width step one. Other strides/alignment and i32 use scalar
C arithmetic. This boundary never substitutes a NumPy/PyTorch numerical result.
It does not implement FX/textual-MLIR dispatch, quantized zero points, mixed dtype
casts or other arithmetic dtypes. Compiler spills remain; no hardware-cycle
performance claim is made.

The pinned 3D multiplier YAML has a five-axis output shape map but a six-axis
static indexing map. This standalone contract follows the explicit indexing map:
output is `N,D,H,W,C,M`. This is a documented numerical subset, not a claim that
the inconsistent pinned declaration has been integrated into a textual compiler.

```python
import numpy as np
from graph.layout import TensorView
from graph.depthwise import DepthwiseOp
from depthwise_runtime import execute_depthwise

x = np.array([[[2], [3]]], dtype='<i4')
w = np.array([[4], [5]], dtype='<i4')
y = np.array([[[1]]], dtype='<i4')
op = DepthwiseOp(
    'linalg.depthwise_conv_1d_nwc_wc',
    TensorView('input', x.shape, 'i32'),
    TensorView('filter', w.shape, 'i32'),
    TensorView('output', y.shape, 'i32'),
)
result = execute_depthwise(op, {
    'input': x.tobytes(), 'filter': w.tobytes(), 'output': y.tobytes(),
}, 'out/depthwise')
assert np.frombuffer(result.buffers['output'], dtype='<i4')[0] == 24
```


## Kernel and training branch boundaries

`atalla-linalg-kernels` is based on `main` and contains standalone kernels and
forward graph integration. Its runner accepts one input and has no backprop
capture, training targets/gradient reporting, or `--training` CLI. Peter's
training integration and tests are preserved on `atalla-linalg-with-training`.
Reduction, loss and gradient C kernels remain available as standalone numerical
primitives; keeping those kernels does not enable a training graph pipeline.
