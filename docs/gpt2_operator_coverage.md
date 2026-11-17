# GPT-2 block operator decomposition and kernel-coverage map (issue #66)

> **Update, later on 2026-11-04:** both failures recorded below were fixed the same day.
> The multi-tile `add` defect was a compiler immediate-range bug (`addi_s` wraps `+2048` to `-2048`),
> worked around in `kernels/{add,mul,relu}.py`; see `docs/add_kernel_multitile_fix.md`. The flash SDPA
> trial failed because K was read from the wrong scratchpad and V was not transposed; see
> `docs/flash_sdpa_investigation.md`. Authoritative post-fix numbers: `docs/baselines/README.md`.
> The tables below are kept as the pre-fix record.


Model file: `atalla-graph/model/gpt2_block.py`. Geometry: N=32 tokens, D=32, 1 head, MLP
hidden 128 (4·D), activations kept 2-D `(N, D)` (batch squeezed) so every GEMM is a plain
32-wide tile product. Attention is written explicitly (matmul → scalar mul → add causal-mask
buffer → softmax → matmul) because the only fused attention lowering (`atalla_sdpa` →
`kernels/flash_sdpa_n32d32.c`) is non-causal (and currently fails, see
`model_baselines_notes.md`). The causal mask is a registered `(N,N)` buffer of `0 / -1e4`
(bf16-exact; `-inf` would NaN in the bf16 add), so it appears as a `get_attr` constant exactly
like a bias.

All counts/accuracies below are from the 2026-11-04 seed-0 runs listed in
`docs/model_baselines_notes.md` (`--validate-inputs chained`, i.e. each kernel consumes the
previous kernel's emulator output). "cos" is the per-node cosine vs the BF16 PyTorch
reference for that node.

Execution paths
* **emu:`<kernel>`** — compiled AtallaC (`kernels/*.py` generator or `kernels/*.c`) → ppci →
  `build_compiler` → functional simulator.
* **numpy** — `codegen/c_emitter.py` returns `skip_emulator` with a NumPy/PyTorch result
  (counted as "numpy" by `run_graph.py`).
* **pass** — view/constant node, no compute (`transpose` of weights, `get_attr`, placeholder).

## GPT2MLPBlock (`--model gpt2_mlp`) — `x + c_proj(gelu(c_fc(ln_2(x))))`

Validate: 16 nodes = 5 emulated, 2 numpy, 2 passthrough (+7 placeholder/get_attr). Final cos
0.999998, rel-L2 0.002196, rel-max 0.003704.

| FX node | normalized op (source) | shape | path | per-node cos | notes |
|---|---|---|---|---|---|
| x | placeholder | (32,32) | pass | — | BF16 input |
| ln_2_weight, ln_2_bias | get_attr | (32,) | pass | — | |
| layer_norm | layernorm (`F.layer_norm`, from `nn.LayerNorm` via `lower_linear_modules`) | (32,32) | emu:`layernorm` M=32 D=32 | 0.999994 | needs D % 32 == 0 |
| c_fc_weight | get_attr | (128,32) | pass | — | |
| transpose | transpose (`call_method`, from Linear lowering) | (32,128) | pass | — | weight view |
| matmul | matmul (`torch.matmul`) | (32,128) | emu:`gemm` M=32 N=128 K=32 | 0.999991 | 4 N-tiles |
| c_fc_bias | get_attr | (128,) | pass | — | |
| add_1 | add (`operator.add`, bias) | (32,128) | **emulator** (`add`, 4 tiles; was NumPy before the add fix) | 0.999990 | pre-fix: `ATALLA_FFN_ADD_NUMPY=0` gave cos 0.4999; fixed, see docs/add_kernel_multitile_fix.md |
| gelu | gelu (`F.gelu`, approximate="tanh") | (32,128) | **numpy** (PyTorch BF16) by default; `ATALLA_GELU_KERNEL=horner\|tanh` → emu:`gelu.c`/`gelu_tanh.c` 4 tiles | 0.999986 (numpy) / 0.999822 (horner) / 0.999966 (tanh) | polynomial approx error ≈ 0.02 max-abs |
| c_proj_weight | get_attr | (32,128) | pass | — | |
| transpose_1 | transpose | (128,32) | pass | — | |
| matmul_1 | matmul | (32,32) | emu:`gemm` M=32 N=32 K=128 | 0.999977 | 4 K-tiles accumulate |
| c_proj_bias | get_attr | (32,) | pass | — | |
| add_2 | add (bias) | (32,32) | emu:`add` 1 tile | 0.999979 | |
| add | add (residual `x + h`) | (32,32) | emu:`add` 1 tile | 0.999998 | |
| output | output | (32,32) | — | — | |

Schedule (`--mode schedule`): 7 kernel calls (2 `matmul_kernel`, 3 `add_kernel`, 1
`layernorm_kernel`, 1 `gelu_kernel`) — see "Schedule mode" below.

## GPT2AttnBlock (`--model gpt2_attn`) — `x + c_proj(softmax(q kᵀ/√D + mask) v)`

Validate: 32 nodes = 15 emulated, 0 numpy, 5 passthrough (+12 placeholder/get_attr). Final cos
0.999999, rel-L2 0.001724, rel-max 0.004167. **Every compute node runs on the emulator.**

| FX node | normalized op (source) | shape | path | per-node cos | notes |
|---|---|---|---|---|---|
| x | placeholder | (32,32) | pass | — | |
| ln_1_weight, ln_1_bias | get_attr | (32,) | pass | — | |
| layer_norm | layernorm (`F.layer_norm`) | (32,32) | emu:`layernorm` | 0.999993 | |
| q_proj_weight / transpose_1 / matmul_2 / q_proj_bias / add_2 | Linear → get_attr, transpose, matmul, get_attr, add | (32,32) | pass / pass / emu:`gemm` 32³ / pass / emu:`add` | 0.999991 / 0.999990 | q |
| k_proj_weight / transpose_2 / matmul_3 / k_proj_bias / add_3 | same | (32,32) | same | 0.999991 / 0.999990 | k |
| v_proj_weight / transpose_3 / matmul_4 / v_proj_bias / add_4 | same | (32,32) | same | 0.999991 / 0.999990 | v |
| transpose | transpose (`k.transpose(-1,-2)`) | (32,32) | pass (numpy transpose of cached k) | — | activation view; no kernel |
| matmul | matmul (`q @ kᵀ`) | (32,32) | emu:`gemm` M=N=K=32 | 0.999977 | scores; largest single-layer abs err 0.0625 (values up to ~6) |
| mul | mul (`operator.mul`, scalar 1/√32 = 0.176777) | (32,32) | emu:`mul` (scalar broadcast) | 0.999974 | |
| causal_mask | get_attr (buffer) | (32,32) | pass | — | 0 / -1e4 upper-triangular |
| add | add (`s + causal_mask`) | (32,32) | emu:`add` | 1.000000 | rel-L2 0.000000 (masked entries dominate norm) |
| softmax | softmax (`F.softmax(dim=-1)`) | (32,32) | emu:`softmax` batched 32 rows × 32 | 0.999992 | row_len must be ≤ 32 for the kernel |
| matmul_1 | matmul (`p @ v`) | (32,32) | emu:`gemm` | 0.999984 | |
| c_proj_weight / transpose_4 / matmul_5 / c_proj_bias / add_5 | Linear | (32,32) | pass / pass / emu:`gemm` / pass / emu:`add` | 0.999982 / 0.999985 | output projection |
| add_1 | add (residual) | (32,32) | emu:`add` | 0.999999 | |
| output | output | (32,32) | — | — | |

Schedule: 14 kernel calls (6 `matmul_kernel`, 6 `add_kernel`, 1 `layernorm_kernel`, 1
`softmax_kernel`); `scripts/generate_schedule.py` folds `mul(scalar)` + `add(mask)` into
`add_kernel(&causal_mask, &scores, &dst, 0.176777)` — see "Schedule mode" below.

## GPT2Block (`--model gpt2_block`) — attention sub-block then MLP sub-block

Validate: 47 nodes = 20 emulated, 2 numpy, 7 passthrough (+18 placeholder/get_attr). Final cos
0.999995, rel-L2 0.003006, rel-max 0.009901. The node list is the attention table (prefixed
`attn_`) followed by the MLP table (prefixed `mlp_`); per-node cos values match the standalone
blocks to within 1e-5 (all emulated nodes ≥ 0.999975). The two numpy nodes are again
`add_7` (4096-elem FFN bias add) and `gelu`.

With `ATALLA_GELU_KERNEL=horner` (GELU on emulator, bias add on NumPy): not run for the full
block today (run for `gpt2_mlp`: final cos 0.999990). With `ATALLA_GELU_KERNEL=horner
ATALLA_FFN_ADD_NUMPY=0` (22 emulated / 0 numpy): final cos 0.985983, rel-L2 0.167 before the add fix — failure was
entirely the multi-tile add (`add_7` cos 0.491377). After the fix: 22 emulated / 0 numpy, final cos 0.999995 (`docs/baselines/README.md`, gpt2_block / gelu-emulator-chained).

Schedule: 21 kernel calls (8 `matmul_kernel`, 9 `add_kernel`, 2 `layernorm_kernel`, 1
`gelu_kernel`, 1 `softmax_kernel`) — see "Schedule mode" below.

## Measured coverage summary

| model | compute nodes | emulator | numpy | which numpy | final cos / rel-L2 |
|---|---|---|---|---|---|
| gpt2_mlp (default) | 7 | 5 | 2 | FFN bias add (4096), GELU | 0.999998 / 0.002196 |
| gpt2_mlp (`ATALLA_GELU_KERNEL=tanh`) | 7 | 6 | 1 | FFN bias add | 0.999996 / 0.002739 |
| gpt2_attn | 15 | 15 | 0 | — | 0.999999 / 0.001724 |
| gpt2_block (default) | 22 | 20 | 2 | FFN bias add, GELU | 0.999995 / 0.003006 |

## Schedule mode (`--mode schedule`) — measured 2026-11-04 (agent "schedule")

`run_graph.py --mode schedule` runs `graph.memoryallocator.allocate_memory` and then
`scripts/generate_schedule.py::emit` to write `graph_schedule.c` (one `GlobalTile` per bf16
tensor + a `run_graph()` body of `*_kernel(...)` calls against `kernels/kernels.h`) and
`dram.bin`. Until today `_render_call` returned `None` for `F.layer_norm`, `F.gelu` and
`F.softmax`, so those nodes were silently dropped. Patch applied (from the former
`docs/_handoff_models.md`, now deleted): `LAYERNORM_TARGETS` / `GELU_TARGETS` /
`SOFTMAX_TARGETS` plus three `_render_call` branches that emit

* `layernorm_kernel(&in, &weight|NULL, &bias|NULL, &out, normalized_shape[-1], eps)` —
  `eps` from the node kwarg/arg (default 1e-5);
* `gelu_kernel(&in, &out, approximate)` — `1` when the node kwarg `approximate == "tanh"`, else `0`;
* `softmax_kernel(&in, &out)` — raises if `dim` is not the last axis (the kernel only does
  row softmax).

Commands (all exit 0; outputs under `atalla-graph/out/schedule/<model>/`):

```
cd atalla-graph
for m in basic layernorm_smoke vit_micro gpt2_mlp gpt2_attn gpt2_block alexnet_small; do
  python run_graph.py --model $m --mode schedule --seed 0 --out-dir out/schedule/$m
done
```

Kernel calls per `graph_schedule.c` (`grep -c '<name>(' graph_schedule.c`), before → after
the patch (before = same commands run against the pre-patch generator this morning):

| model | calls before → after | matmul | add | relu | conv | maxpool | layernorm | gelu | softmax |
|---|---|---|---|---|---|---|---|---|---|
| basic | 9 → 9 | 3 | 4 | 2 | — | — | — | — | — |
| layernorm_smoke | 2 → 3 | 1 | 1 | — | — | — | 1 | — | — |
| vit_micro | 17 → 20 | 7 | 10 | — | — | — | 2 | 1 | — |
| gpt2_mlp | 5 → 7 | 2 | 3 | — | — | — | 1 | 1 | — |
| gpt2_attn | 12 → 14 | 6 | 6 | — | — | — | 1 | — | 1 |
| gpt2_block | 17 → 21 | 8 | 9 | — | — | — | 2 | 1 | 1 |
| alexnet_small | 21 → 21 | 3 | 3 | 7 | 5 | 3 | — | — | — |

Per-node audit (a scratch script that re-runs `allocate_memory` + `_render_call` on every
non-placeholder/get_attr/output FX node and classifies it as *call*, *folded* (scalar `mul`
absorbed into the following `add_kernel` alpha), *passthrough* (view: `transpose`,
`getattr`) or *skipped*):

| model | nodes audited | with call | folded into add | passthrough view | skipped (no kernel) |
|---|---|---|---|---|---|
| basic | 11 | 9 | 1 | 1 | 0 |
| layernorm_smoke | 4 | 3 | 0 | 1 | 0 |
| vit_micro | 28 | 20 | 0 | 7 | **1: `sdpa` (`AtallaSdpa` call_module)** |
| gpt2_mlp | 9 | 7 | 0 | 2 | 0 |
| gpt2_attn | 20 | 14 | 1 | 5 | 0 |
| gpt2_block | 29 | 21 | 1 | 7 | 0 |
| alexnet_small | 25 | 21 | 0 | 3 | 1: `flatten` (`torch.flatten`, a reshape — no compute, same tile data; not in `VIEW_FUNCTIONS` so it falls through `_render_call` rather than being recognised as a view) |

So every compute node of the three GPT-2 graphs (and `basic`, `layernorm_smoke`) now has a
kernel call. **Explicitly unsupported in schedule mode** (documented in the
`generate_schedule.py` module docstring): the fused `AtallaSdpa` leaf (validate runs it as
NumPy or `flash_sdpa_n32d32.c`), `call_method` ops other than views/`mean`, and any
`call_function` outside the `*_TARGETS` sets (`torch.flatten` is the only such node in
today's models and is a pure view).

### Compiling `graph_schedule.c` with the compiler (syntax check)

```
cd aihw-ppci-compiler
python -m ppci atalla_cc -m atalla -I ../atalla-graph -I <shim> \
    -S ../atalla-graph/out/schedule/gpt2_block/graph_schedule.c -o /tmp/claude-502/gs.s
```

* Without `-I` the include of `kernels/kernels.h` is not found; with `-I ../atalla-graph`
  the header's `#include <stddef.h>` / `<stdint.h>` are not found (ppci's `atalla_cc` has no
  default libc include path, and `librt/libc/include/stdint.h` does `#include_next`, which
  fails in both hosted and `--freestanding` mode: `typedef short int16_t;` → `Expected ";",
  got "ID"`). A 2-file shim dir (`stddef.h`: `typedef unsigned int size_t; #define NULL
  ((void*)0)`; `stdint.h`: `typedef unsigned char uint8_t; typedef unsigned int uint32_t;`)
  passed as the second `-I` resolves this; it lives in the agent scratchpad, not in the repo.
* **Two float-literal findings, both fixed in `generate_schedule.py` (`_c_float`)**: the
  `atalla_c` front-end rejects scientific notation (`1e-05` → `ValueError: could not convert
  string to float: '1e'` in `ppci/lang/atalla_c/utils.py::float_num`) and *any* `f` suffix
  (`0.5f` → `Expected ",", got "ID"`; verified with a 2-line test file: `0.5f` fails, `0.5`,
  `1.0`, `1` pass). The pre-patch generator emitted `1.0f` / `0.176777f` for every
  `add_kernel` alpha, so **no `graph_schedule.c` compiled with `atalla_cc` before today**.
  Float arguments are now rendered as plain decimals (`1.0`, `0.176777`, `0.00001`), which
  convert implicitly to the `float` parameters in `kernels/kernels.h`.
* With those fixes all seven schedules compile to assembly (exit 0; `.s` line counts: basic
  1525, layernorm_smoke 818, vit_micro 4223, gpt2_mlp 1525, gpt2_attn 2726, gpt2_block
  4135, alexnet_small 3961; `gs_gpt2_block.s` declares `global layernorm_kernel` /
  `gelu_kernel` / `softmax_kernel` and `global run_graph`). This is a front-end/codegen
  syntax check only: the kernels are the empty stubs in `kernels/kernels.c`, nothing was
  linked or run on the emulator.

Regression after the patch: `pytest atalla-graph/tests -q` → 6 passed, 2 subtests passed;
`run_graph.py --model basic --mode validate --seed 0` → 18 nodes / 10 emulated, cos 1.000000.

## Remaining gaps

1. **Causal masking approach.** Implemented as an additive `(N,N)` constant buffer + generic
   `add` kernel (one extra 2 KB DRAM read per attention). A fused causal flash kernel does not
   exist; the non-causal `flash_sdpa_n32d32.c` path currently returns wrong values
   (`ATALLA_SDPA_FLASH=1`, sdpa node cos -0.11). The mask value `-1e4` (not `-inf`) relies on
   softmax underflow in bf16, which holds for |scores| ≪ 1e4.
2. **Multi-head.** Single head only. `h` heads would need either `h` sequential
   `(N, D/h)` attention chains with D/h = 32 (so D ≥ 32·h; D=32 → 1 head) plus a concat
   (view) node, or a batched `(h, N, N)` matmul, which `_plan_matmul` does not tile (it only
   reads the last two dims).
3. **D = 768 scaling (real GPT-2 small).** LayerNorm kernel supports D % 32 == 0 (768 ✓); GEMMs
   tile M/N/K by 32 (768 = 24 tiles ✓, hidden 3072 = 96 tiles). Blockers: (a) ~~the multi-tile `add` kernel bug~~ (fixed 2026-11-04; large adds now run on the emulator by default); (b) softmax kernel row length ≤ 32, so N=1024 context would
   need a multi-tile row softmax; (c) `_plan_add` NumPy fallback and DRAM image sizes have
   not been exercised beyond 4096 elements; (d) runtime: today's emulator spends ~0.5–1 s per
   kernel call, so a D=768 block (dozens of multi-tile GEMMs) was **not measured**.
4. **GELU.** Default path is PyTorch BF16 (numpy). `ATALLA_GELU_KERNEL=horner|tanh` now runs
   Aman's polynomial kernels on the emulator (per-node cos 0.99982 / 0.99997, max-abs ≈ 0.02–
   0.03 vs exact tanh-GELU); the error is the approximation, not the pipeline. `--mode
   schedule` emits it as `gelu_kernel(src, dst, 1)` (1 = tanh approximate) since 2026-11-04.
5. **Schedule completeness.** `layer_norm`, `gelu`, `softmax` are now emitted (see "Schedule
   mode"); the only compute nodes still silently dropped by `scripts/generate_schedule.py`
   are the fused `AtallaSdpa` leaf (`vit_micro` only) — the GPT-2 graphs have no skipped
   compute node.
6. **Dropout / attention-prob dropout / weight tying / embeddings / LM head** — not modelled
   (inference-only block, no token/pos embedding, no final LN).
