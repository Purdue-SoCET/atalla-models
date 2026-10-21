# Forward and backward graph capture

`build_graph` still returns a single `torch.fx.GraphModule`. Forward-only calls
use the existing trace, module lowering, shape propagation, normalization, and
tile planning pipeline:

```python
gm = build_graph(model, example_input)
prediction = gm(example_input.bfloat16())
```

To include the loss and parameter-gradient operations in that graph:

```python
import torch
import torch.nn.functional as F
from run_graph import build_graph, load_model

model, x = load_model("basic")
labels = torch.zeros(x.shape[0], dtype=torch.long)
gm = build_graph(
    model, x,
    include_backward=True,
    example_target=labels,
    loss_fn=F.cross_entropy,
)

prediction, loss, gradients = gm(x.bfloat16(), labels)
print(gradients["output.weight"])
gm.graph.print_tabular()
print("Operators needing planners/kernels:", gm.meta["pending_kernel_ops"])
```

The backward path starts with the same FX tracing and module lowering. It then
runs that lowered forward graph, a scalar loss function, and `torch.autograd.grad`
under `make_fx` to capture the actual tensor operations for all three stages.
Shape propagation, normalization, and tile planning run on the combined graph.
ATen names such as `aten.mm.default` identify PyTorch tensor operators; they are
retained as graph nodes, including operators whose Atalla kernels do not exist
yet. This tracing also expands the current fused `AtallaSdpa` leaf into its
constituent tensor operations.

Weights and buffers remain graph attributes (`get_attr` nodes), rather than extra
function inputs. Weights are shared with the model, so changing a weight changes
subsequent graph execution. The gradient dictionary uses the original model's
parameter names. Frozen parameters are omitted; unused trainable parameters have
gradient `None`. The graph returns gradients explicitly and does not accumulate
into `.grad` or perform an optimizer update.

As before, `build_graph` converts the model to BF16 and calls `.eval()`. This
captures evaluation-mode derivatives, including evaluation behavior for dropout
and batch normalization. Model computations and parameter gradients use BF16;
the loss function receives FP32 predictions and floating targets so CPU loss
backward operations such as MSE work. Integer class labels keep their dtype.
`loss_fn(prediction, target)` must return a scalar tensor. Input shapes, dtypes,
and model mode should match the capture when reusing the graph.

The reference for gradient correctness is the **lowered forward graph**. In
particular, the existing split of BF16 linear into matmul and bias add introduces
an extra rounding step compared with the original fused PyTorch linear operator.

Each node retains `atalla_op`, shape, tile, and DRAM metadata. Combined graphs
also provide:

- `phase`: `forward`, `loss`, or `backward` for computation nodes, classified by
  output dependencies; saved-value preparation belongs to its consuming stage.
- `parameter_name` / `buffer_name` on attributes that came from the model.
- `requires_kernel=True` and a tile config with `pending=True` for operators
  without a tile planner. `gm.meta["pending_kernel_ops"]` lists these operators;
  it is a planner work list, not a complete emitter-support inventory.
- `dram_outputs`: shape, dtype, address, and size for every tensor result,
  including tuple results such as convolution backward and pooling indices.

DRAM planning conservatively keeps separate allocations for intermediate results,
including forward values used in backward. Sizes account for BF16, FP32, and
integer tensors. These annotations are for graph development; the BF16 binary
allocator and existing C emitter do not yet implement mixed-dtype backward
execution. `run_schedule` and `run_validate` reject combined graphs with an
explicit explanation. Run the graph directly in PyTorch to inspect predictions,
loss, and gradients while implementing the new kernels and backend integration.

Graph capture tests (no accelerator kernels required):

```text
python -m unittest discover -s tests -v
```

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
