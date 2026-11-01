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
