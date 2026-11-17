# Model baselines — measured notes (agent "models", 2026-11-04)

> **Update, later on 2026-11-04:** both failures recorded below were fixed the same day.
> The multi-tile `add` defect was a compiler immediate-range bug (`addi_s` wraps `+2048` to `-2048`),
> worked around in `kernels/{add,mul,relu}.py`; see `docs/add_kernel_multitile_fix.md`. The flash SDPA
> trial failed because K was read from the wrong scratchpad and V was not transposed; see
> `docs/flash_sdpa_investigation.md`. Authoritative post-fix numbers: `docs/baselines/README.md`.
> The tables below are kept as the pre-fix record.


All numbers below come from runs executed on 2026-11-04 in the `robert` worktree
(`atalla-models` HEAD `3124659` + uncommitted changes listed at the bottom; submodules
`aihw-ppci-compiler 351b5fab`, `functional_sim f159f0d`). Python:
`/Users/robert.zhang/.venvs/atalla/bin/python` (3.11, torch 2.15). Every command is run from
`atalla-models-rz/atalla-graph/`. Anything not listed here was **not measured**.

`--seed` (new): `run_graph.py --seed N` now calls `torch.manual_seed(N)` and `np.random.seed(N)`
**before** `load_model`, so weights *and* the random input are reproducible. Default `0`
(previously hard-coded 42). Verified: `--mode schedule --seed 0` vs `--seed 1` produce
different `dram.bin` (weights differ), same seed -> byte-identical.

Accuracy columns are the **final output vs the traced BF16 PyTorch model** in `chained`
validate mode (each layer consumes the previous layer's emulator output). Coverage =
`emulated / numpy / passthrough` node counts printed by `run_graph.py` (`total` includes
placeholder + get_attr nodes). Packets/instructions are the emulator's dynamic counters
summed over emulated layers (`aggregate_metrics.total_emu_packets`,
`total_emu_instructions_retired`), `sched` = statically scheduled packets.

## Spring models (restored in `load_model`)

| model | command (append `--metrics-json out/<m>/metrics.json` as needed) | nodes total / emu / numpy / pass | cos | rel-L2 | rel-max | sched pk | dyn pk | retired instr | bytes ld / wr | FLOPs | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| basic | `python run_graph.py --model basic --mode validate --seed 0 --out-dir out/models_agent/basic` | 18 / 10 / 0 / 1 | 1.000000 | 0.000000 | 0.000000 | 1560 | 2908 | 3390 | 7488 / 640 | 7877 | 4.62 s |
| alexnet_small (scale 0.01) | `python run_graph.py --model alexnet_small --scale 0.01 --mode validate --seed 0 --out-dir out/models_agent/alexnet_small` | 42 / 21 / 0 / 4 | 0.999992 | 0.004199 | 0.006579 | 3677 | 101172 | 119762 | 97652 / 9276 | 3124917 | 11.36 s |
| layernorm_smoke | `python run_graph.py --model layernorm_smoke --mode validate --seed 0 --out-dir out/models_agent/layernorm_smoke` | 9 / 3 / 0 / 1 | 0.999990 | 0.004901 | 0.006803 | 639 | 1095 | 1289 | 2688 / 192 | 3018 | 1.46 s |
| vit_micro, default SDPA (NumPy ref) | `python run_graph.py --model vit_micro --mode validate --seed 0 --out-dir out/models_agent/vit_micro` | 48 / 18 / 3 / 7 | 0.999994 | 0.003433 | 0.006803 | 3376 | 36611 | 46789 | 113216 / 40960 | 663374 | 8.47 s |
| vit_micro, flash trial | `ATALLA_SDPA_FLASH=1 python run_graph.py --model vit_micro --mode validate --seed 0 --out-dir out/models_agent/vit_micro_flash` | 48 / 19 / 2 / 7 | **0.986332** | **0.165220** | **0.127551** | 3652 | 78581 | 98530 | 246400 / 43008 | 828670 | 15.01 s |

Notes
* None of the spring models broke from the kernel port (aded87b) or the submodule bump; they
  were only disabled because `load_model` had been reduced to `basic`. Restoring them needed no
  code fix beyond `load_model` + the `(model, x, expected_output_or_None)` 3-tuple (forward
  validate never dereferences `expected_output`; `--training` now exits with a clear message if
  the model returns `None`).
* `vit_micro` default: `add_8` (2048-elem FFN bias add) and `gelu` run as NumPy/PyTorch
  fallback, `sdpa` runs as the NumPy BF16 reference (`[sdpa] atalla_sdpa -> NumPy`).
* **alexnet_small FLOPs / arithmetic intensity**: 3.12 MFLOP over 106.9 KB moved = 29.2 FLOP/B
  (conv-dominated); the transformer blocks sit at 4–6 FLOP/B.

## ViT Micro: "default SDPA" vs "flash trial" — how each is selected

* Selection is the environment variable **`ATALLA_SDPA_FLASH`**, read in
  `graph/tile_planner.py::_plan_sdpa` (`os.environ.get("ATALLA_SDPA_FLASH", "0") == "1"`),
  and only honoured when `B == 1 and N == 32 and D == 32`. It sets
  `tile_config.params["use_flash"]`, which `codegen/c_emitter.py::emit_sdpa` uses to choose
  between the fused AtallaC kernel `kernels/flash_sdpa_n32d32.c` (emulator) and the NumPy
  BF16 reference (`skip_emulator`, map_kind `atalla_sdpa`, `use_flash=False`).
* The attention op itself is `model/atalla_ops.py::AtallaSdpa` (`(q @ k^T) * 1/sqrt(d) ->
  softmax -> @ v`, non-causal); `run_graph.py::_AtallaLeafTracer` keeps it as one FX leaf
  so it is a single `atalla_sdpa` node.
* Default SDPA run = no env var (or `ATALLA_SDPA_FLASH=0`). Flash trial = `ATALLA_SDPA_FLASH=1`.
* Measured today (seed 0): the flash kernel node reports
  `[sdpa] atalla_sdpa -> emulator ... cos=-0.106544, rel_l2=1.157023, relmax=1.020992`
  (emu_norm 2.86 vs ref_norm 5.90), which drags the end-to-end output to cos 0.986 /
  rel-L2 0.165. So the **flash trial still fails** exactly as in the spring report; the default
  path passes (cos 0.999994).
* Related knob: `ATALLA_FFN_ADD_NUMPY` (default now `0`; was `1` before the fix) in `_plan_add` forces any add with
  > 1024 elements onto NumPy. See the GPT-2 section for why (multi-tile add kernel is wrong).

## GPT-2 blocks (new: `model/gpt2_block.py`, issue #66)

N=32 tokens, D=32, 1 head, MLP hidden 128, 2-D `(N, D)` activations, explicit causal attention
(matmul -> scalar mul -> add mask buffer -> softmax -> matmul). Per-node detail is in
`docs/gpt2_operator_coverage.md`.

| model / variant | command | nodes total / emu / numpy / pass | cos | rel-L2 | rel-max | sched pk | dyn pk | retired instr | bytes ld / wr | FLOPs | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2_mlp (default: GELU + 4096-wide bias add on NumPy) | `python run_graph.py --model gpt2_mlp --mode validate --seed 0 --out-dir out/models_agent/gpt2_mlp` | 16 / 5 / 2 / 2 | 0.999998 | 0.002196 | 0.003704 | 1019 | 21410 | 26644 | 68096 / 22528 | 560542 | 4.06 s |
| gpt2_mlp, GELU on emulator (Horner poly, `kernels/gelu.c`) | `ATALLA_GELU_KERNEL=horner python run_graph.py --model gpt2_mlp --mode validate --seed 0 --out-dir out/models_agent/gpt2_mlp_gelu_horner` | 16 / 6 / 1 / 2 | 0.999990 | 0.004584 | 0.003704 | 1264 | 38252 | 49650 | 76288 / 30720 | 647796 | 3.92 s |
| gpt2_mlp, GELU on emulator (tanh poly, `kernels/gelu_tanh.c`) | `ATALLA_GELU_KERNEL=tanh python run_graph.py --model gpt2_mlp --mode validate --seed 0 --out-dir out/models_agent/gpt2_mlp_gelu_tanh` | 16 / 6 / 1 / 2 | 0.999996 | 0.002739 | 0.003704 | 1333 | 45560 | 60286 | 76288 / 30720 | 709496 | 4.05 s |
| gpt2_mlp, everything on emulator (**fails**: multi-tile add) | `ATALLA_GELU_KERNEL=horner ATALLA_FFN_ADD_NUMPY=0 python run_graph.py --model gpt2_mlp --mode validate --seed 0 --out-dir out/models_agent/gpt2_mlp_full_emu` | 16 / 7 / 0 / 2 | **0.987473** | **0.157804** | **0.129630** | 1363 | 42604 | 55948 | 84492 / 34822 | 660327 | — |
| gpt2_attn (all compute on emulator) | `python run_graph.py --model gpt2_attn --mode validate --seed 0 --out-dir out/models_agent/gpt2_attn` | 32 / 15 / 0 / 5 | 0.999999 | 0.001724 | 0.004167 | 2750 | 25595 | 32838 | 78208 / 30720 | 448208 | 7.93 s |
| gpt2_block (default) | `python run_graph.py --model gpt2_block --mode validate --seed 0 --out-dir out/models_agent/gpt2_block` | 47 / 20 / 2 / 7 | 0.999995 | 0.003006 | 0.009901 | 3769 | 47005 | 59482 | 146304 / 53248 | 1008750 | 15.62 s |
| gpt2_block, everything on emulator (**fails**: multi-tile add) | `ATALLA_GELU_KERNEL=horner ATALLA_FFN_ADD_NUMPY=0 python run_graph.py --model gpt2_block --mode validate --seed 0 --out-dir out/models_agent/gpt2_block_full_emu` | 47 / 22 / 0 / 7 | **0.985983** | **0.166986** | **0.230198** | 4113 | 68199 | 88786 | 162700 / 65542 | 1108535 | — |

Schedule mode (`"compile through graph compiler -> AtallaC"`), all exit 0:

```
python run_graph.py --model gpt2_mlp   --mode schedule --seed 0 --out-dir out/models_agent/gpt2_mlp_sched    # 5 kernel calls
python run_graph.py --model gpt2_attn  --mode schedule --seed 0 --out-dir out/models_agent/gpt2_attn_sched   # 12 kernel calls
python run_graph.py --model gpt2_block --mode schedule --seed 0 --out-dir out/models_agent/gpt2_block_sched  # 17 kernel calls
```

`graph_schedule.c` contains `matmul_kernel` / `add_kernel` calls for every GEMM, bias add,
residual add and the `s*scale + mask` pair (folded by `scripts/generate_schedule.py` into
`add_kernel(mask, scores, dst, 0.176777f)`). **`layer_norm`, `gelu` and `softmax` nodes are
not emitted in the schedule** (pre-existing limitation of `scripts/generate_schedule.py`,
documented in its module docstring; `kernels/kernels.h` already declares `layernorm_kernel`,
`gelu_kernel`, `softmax_kernel`). The patch to add them is in `docs/_handoff_models.md`.

### Finding (RESOLVED same day, see docs/add_kernel_multitile_fix.md): multi-tile `add` kernel was wrong on the emulator (> 1024 elements)

With `ATALLA_FFN_ADD_NUMPY=0` the 4096-element FFN bias add (`(32,128) + (128,)`, 128 rows ->
4 scratchpad tiles of 32 rows) compiles and runs but returns garbage:
`[add_1] add -> emulator (0x7000, 4096 elems) cos=0.499978, relmax=0.820988, rel_l2=0.866038`
(gpt2_mlp) and `cos=0.491377` (gpt2_block `add_7`). Single-tile adds (1024 elems) and the
4-tile **GELU** kernel (`kernels/gelu.c`, same tile loop structure, cos 0.999822) are fine, so
the defect is specific to `kernels/add.py`'s two-operand multi-tile loop (loads B through
`sdma_ctl_sp1` into scratchpad slot 1 of scpad 0). That is why `_plan_add` defaults to NumPy
above 1024 elements. Not fixed today; it is the only thing between `gpt2_block` and 100 %
emulator coverage of its compute nodes.

### GELU wiring (new, `codegen/c_emitter.py::emit_gelu`)

`ATALLA_GELU_KERNEL=horner|tanh` builds the DRAM image for Aman's standalone kernels
(`kernels/gelu.c` / `kernels/gelu_tanh.c`) using the ADDR_TABLE layout from
`tests/kernels/test_gelu.py` (`+0 IN, +4 OUT, +8 tile_count, +12 sp_rows, +16 sdma_sp0,
+20 tile_bytes, +24.. f32 coefficients`) and runs it through the emulator; unset/`0` keeps the
PyTorch BF16 fallback (default, so other agents' baselines are unchanged). Per-layer GELU error
on `(32,128)`: horner cos 0.999822 / rel-L2 0.019885 / max_abs 0.027; tanh cos 0.999966 /
rel-L2 0.008328 / max_abs 0.023 — i.e. the polynomial approximation error, as documented in
`kernels/gelu.py`.

## Files changed by this agent (uncommitted)

* `atalla-graph/run_graph.py` — `--seed`; `load_model` restored for all spring models and
  extended with `gpt2_mlp | gpt2_attn | gpt2_block`; `--training` guard for `expected_output=None`.
* `atalla-graph/model/gpt2_block.py` — new.
* `atalla-graph/codegen/c_emitter.py` — `emit_gelu` emulator path (`ATALLA_GELU_KERNEL`).
* `docs/model_baselines_notes.md`, `docs/gpt2_operator_coverage.md`, `docs/_handoff_models.md` — new.
