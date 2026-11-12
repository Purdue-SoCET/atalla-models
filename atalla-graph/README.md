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

model, x, _ = load_model("basic")
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
integer tensors. Schedule execution supports BF16 tensors and FP32 storage with
explicit conversions, including basic-model forward/loss/backward execution.
Graphs requiring unavailable derivative kernels or integer computation are
rejected with a complete capability report. Validation can still use references
for unsupported computations; schedule execution never injects those references.

Graph capture tests (no accelerator kernels required):

```text
python -m unittest discover -s tests -v
```

# Schedules using available forward and backward kernels

From the repository root:

```bash
python3 atalla-graph/run_graph.py --model basic --mode schedule --out-dir out/schedule
python3 atalla-graph/run_graph.py --model basic --mode schedule --out-dir out/schedule --verify-schedule
python3 atalla-graph/run_graph.py --model alexnet_small --mode schedule --out-dir out/alexnet_small --verify-schedule
python3 atalla-graph/run_graph.py --model vit_micro --mode schedule --out-dir out/schedule-vit --verify-schedule
python3 atalla-graph/run_graph.py --model basic --training --mode schedule --out-dir out/schedule-training --verify-schedule
python3 atalla-graph/run_graph.py --model vit_micro --training --mode schedule --check-schedule
```

The schedule prints each kernel's graph node, operation, function, source file,
output shape, and output address. `scripts/generate_schedule.py` delegates to
`codegen/kernel_schedule.py`, which reuses the source and configuration emitted
by `codegen/c_emitter.py`. It namespaces the kernels, replaces their individual
entry points with callable functions, and emits one `main()` with DMA copies
connecting live graph results. GEMM inputs are packed in the layout expected by
validation, and transposes are materialized rather than ignored.
Captured training operators also use `kernels/training.py`, which emits device
code directly against live buffers. Tuple results have separate buffers for
each tensor; `getitem` aliases the selected buffer, including absent masked
gradients. Pool indices are stored as exact int64 values and verified exactly.

`codegen/schedule_ops.py` is the adapter registry. It describes kernel sources,
configuration pointer fields, and supported configurations. The schedule builder
packs live graph values for each interface. Unsupported operations are collected
by `schedule_issues()` before emission; `--check-schedule` prints that report
without compiling or running the graph. Model-specific adapter changes are not
needed when the graph only uses supported operations/configurations.

| Operation | Device implementation / limits |
|---|---|
| Matmul / linear | Validation GEMM; leading input axes are flattened when the RHS is shared. Large M/N tails are padded and unpacked. Linear is lowered to matmul plus bias add. |
| Add / multiply / ReLU | Validation kernels, including multiple tiles and padded tails. |
| Softmax | Validation kernel; any axis, axis length <=32, with layout conversion. |
| LayerNorm | Validation kernel; normalized size must be a positive multiple of 32. |
| GELU | Existing `gelu_tanh.c` polynomial approximation; accuracy remains verified against PyTorch. |
| BatchNorm | Existing `batchnorm.c`; evaluation mode, batch-one NCHW, constant running statistics/affine parameters folded at compilation. |
| Average pool | Existing `avgpool.c`; batch-one NCHW, even H<=32, W=8, kernel=stride=2, no padding/ceil/divisor override. |
| Adaptive average pool | Global batch-one NCHW ->1x1, lowered to existing GEMM with constant averaging weights. |
| Upsample | Existing `upsample.c`; batch-one NCHW, even input W, nearest-neighbor 2x. |
| Attention | QK transpose -> GEMM -> scale -> softmax -> GEMM with V. B=1, sequence length <=32. Uses existing GEMM/mul/softmax kernels. |
| ReLU backward | Existing `threshold_backward.c`, threshold=0. |
| Native LayerNorm / backward | Device scalar reductions over arbitrary trailing normalized shapes; emits output, mean, reciprocal standard deviation, and masked input/weight/bias gradients. Optional affine parameters are supported. |
| Convolution backward | Non-transposed 2D NCHW; supports batch, groups, stride, padding, dilation, and masked input/weight/bias gradients. Forward convolution retains the limits below. |
| Native max-pool / backward | 2D NCHW, exact int64 argmax indices, first-index ties, padding, dilation, ceil mode, and accumulation for overlapping windows. |
| GELU backward | `none` and `tanh` derivatives; device EXP plus scalar arithmetic. The erf form uses a polynomial approximation. |
| Softmax backward | BF16 input dtype; any reduction axis. |
| Sum | Device scalar reduction over arbitrary axes, including more than 32 rows and partial column tiles; BF16 or FP32 results. |
| MSE forward/backward | Existing kernels, mean reduction. FP32 operands are staged as BF16; backward multiplies by the live incoming loss-gradient seed. |
| Layout / initialization | Reshape, transpose, permute, expand, contiguous, detach, evaluation dropout, ones initialization, and BF16/FP32 conversion. |

Backward schedules return prediction, loss, and the named gradient dictionary.
They do not accumulate into `.grad` or perform an optimizer step. The basic MLP
with MSE and the LayerNorm smoke model are verified end to end. AlexNetSmall,
LayerNorm smoke, and ViT Micro training graphs pass the adapter capability check.
The new training kernels prioritize correctness and general shapes; their scalar
loops can be slower than the existing vector kernels. A passing capability check
does not establish numerical accuracy: use `--verify-schedule` to compare every
intermediate, loss, and gradient against PyTorch with the existing thresholds.
Schedules never substitute reference values for device computations.

Full default AlexNetSmall and ViT Micro training schedules also compile and run,
but do not yet pass every strict numerical comparison. In the seed-42 run,
AlexNet's accumulated BF16 differences change one pool winner and some ReLU
decisions, which propagate into gradients. The pool indices match the live
simulator input exactly. ViT passes 83 of 85 comparisons; the remaining reduction
and its reshape differ by at most about 9e-6 on a near-zero bias gradient. These
failures remain visible; verification thresholds have not been relaxed.

Convolution currently requires batch-one NCHW, groups=1, dilation=1, and equal
strides/padding on both spatial axes. GEMM M/N dimensions above 32 must be multiples
of 32 with the current convolution kernel. The adapter builds im2col from live inputs,
packs OIHW weights for validation's GEMM kernel, initializes the bias, and converts
the GEMM output to NCHW. Maxpool requires batch-one NCHW, H/W<=32, a square window,
equal strides, no padding, dilation=1, and ceil_mode=False. It reuses validation's
vertical max kernel and performs horizontal max on the simulator too, rather than
using validation's host-side postprocessing. These adapters cover AlexNetSmall at
the default input size and scale.

Generation writes `graph_schedule.c`, `dram.bin`, and `schedule.json` with input,
kernel configuration, and output addresses. `--verify-schedule` also writes:

- `graph_schedule.s` and `graph_schedule.in`: compilation/assembly artifacts.
- `graph_schedule_*` simulator dumps, including performance counters.
- `schedule_verify.json`: comparisons against the traced BF16 PyTorch graph.
- `schedule_outputs.npz`: `result_<node>` and `reference_<node>` tensor arrays.

The complete schedule executes once, without per-layer reference injection.
Verification checks every kernel output and graph output, requires cosine >= 0.999,
relative L2 <= 0.02, and relative maximum error <= 0.05, and fails the command on
a mismatch. Scalar loss comparisons check magnitude errors. The reference uses
lowered matmul-plus-bias semantics, as validation does, rather than the original
fused PyTorch linear operator. Prediction, loss, and parameter-gradient outputs
are all compared for backward graphs.

All intermediate checks remain enforced even if the final prediction passes.
For the default AlexNetSmall seed/scale, final relative L2 is about 0.001287, but
`relu_3` has relative L2 about 0.020042, narrowly exceeding the 0.02 limit. The
command therefore reports an accuracy failure after successfully compiling and
executing the schedule. The simulated ReLU exactly matches its simulated input's
ReLU; the difference from PyTorch is carried forward from convolution arithmetic.

Schedule regression tests (compiler and simulator required):

```bash
python3 -m pytest atalla-graph/tests/test_schedule.py -q
```

The legacy descriptor interface in `kernels/kernels.h` and the stubs in
`kernels/kernels.c` are no longer used by schedule mode.
