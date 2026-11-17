# atalla-graph: forward and backward graph capture

Pipeline overview and directory map: `../docs/REPO_GUIDE.md`. Setup: `../docs/ENVIRONMENT.md`.
Canonical runs (from this directory, with the 3.11 venv):

```bash
python run_graph.py --model basic --mode validate --out-dir out/graph --metrics-json out/graph/m.json
python run_graph.py --model basic --mode schedule --out-dir out/graph
python run_graph.py --model basic --mode validate --training --out-dir out/train
```

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

model, x, _expected = load_model("basic")   # (model, example_input, expected_output_or_None)
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
execution in schedule mode: `run_schedule` rejects combined graphs. Validate
mode does support them: `run_graph.py --training` runs forward, loss and
backward nodes on the emulator and compares prediction, loss and every
parameter gradient (`run_validate(training=True, expected_output=...)`).
Backward kernels that exist today: `threshold_backward`, `sum_dim`, `mse_loss`,
`mse_loss_backward` (see `kernels/` and `tests/kernels/`).

Graph capture tests (no accelerator kernels required):

```text
python -m pytest tests -q            # from atalla-graph/
python -m unittest discover -s tests -v
```

Models not registered in `load_model` on the current branch are skipped.

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

## Standalone `.c` kernels (validate path)

`kernels/*.c` are static reference kernels (avgpool, batchnorm, gelu, gelu_tanh, upsample, sum_dim,
mse_loss, mse_loss_backward, threshold_backward, flash_sdpa_n32d32). They read their ADDR_TABLE through
`*(cfg_ptr + n)` and finish with `atalla_halt()`; `../tests/kernels/` compiles each with `atalla_cc -p -S`,
checks for stack spills, assembles with `build_compiler.py --prepacked`, runs the simulator and diffs
against a BF16 golden (`../run_kernel_tests.sh <name>`).

**Weight-load order:** `kernels/gemm.py` loads systolic weights first-to-last. The pending upstream
`investigate-weight-load-order` change flips `lw.vi` to last-to-first; reverse the loop when the pins move.
 
