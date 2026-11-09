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
schedule backend does not yet implement mixed-dtype backward execution.
`run_schedule` rejects combined graphs with an explicit explanation. Validation
can compare backward graphs using the available kernels and reference fallbacks.

Graph capture tests (no accelerator kernels required):

```text
python -m unittest discover -s tests -v
```

# Forward schedule using validation kernels

From the repository root:

```bash
python3 atalla-graph/run_graph.py --model basic --mode schedule --out-dir out/schedule
python3 atalla-graph/run_graph.py --model basic --mode schedule --out-dir out/schedule --verify-schedule
```

The schedule prints each kernel's graph node, operation, function, source file,
output shape, and output address. `scripts/generate_schedule.py` delegates to
`codegen/kernel_schedule.py`, which reuses the source and configuration emitted
by `codegen/c_emitter.py`. It namespaces the kernels, replaces their individual
entry points with callable functions, and emits one `main()` with DMA copies
connecting live graph results. GEMM inputs are packed in the layout expected by
validation, and transposes are materialized rather than ignored.

Current adapters support 2D matrix multiplication, addition, multiplication,
ReLU, reshapes, and transposes with BF16 tensors. Required operations without an
adapter fail explicitly. This first implementation uses separate staging buffers
and element copies; it prioritizes correctness over transfer efficiency.

Generation writes `graph_schedule.c`, `dram.bin`, and `schedule.json` with input,
kernel configuration, and output addresses. `--verify-schedule` also writes:

- `graph_schedule.s` and `graph_schedule.in`: compilation/assembly artifacts.
- `graph_schedule_*` simulator dumps, including performance counters.
- `schedule_verify.json`: comparisons against the traced BF16 PyTorch graph.
- `schedule_outputs.npz`: `result_<node>` and `reference_<node>` tensor arrays.

The complete schedule executes once, without per-layer reference injection.
Verification checks every kernel output and graph output, requires cosine >= 0.999,
relative L2 <= 0.02, and relative maximum error <= 0.05, and fails the command on
a mismatch. The reference uses lowered matmul-plus-bias semantics, as validation
does, rather than the original fused PyTorch linear operator.

Schedule regression tests (compiler and simulator required):

```bash
python3 -m pytest atalla-graph/tests/test_schedule.py -q
```

The legacy descriptor interface in `kernels/kernels.h` and the stubs in
`kernels/kernels.c` are no longer used by schedule mode.
