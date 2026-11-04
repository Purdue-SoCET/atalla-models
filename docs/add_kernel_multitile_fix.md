# Multi-tile `add` kernel fix (agent "addfix", 2026-11-04)

All numbers below come from runs executed on 2026-11-04 in the `robert` worktree
(submodules `aihw-ppci-compiler 351b5fab`, `functional_sim f159f0d`), Python
`/Users/robert.zhang/.venvs/atalla/bin/python`, every command run from `atalla-models-rz/atalla-graph/`.
Run outputs are under `atalla-graph/out/addfix/` (gitignored). Helper scripts used for the diagnosis are
copied there too (`addiff.py`, `trace_add.py`, `kernel_multitile_test.py`).

## Symptom

Any element-wise `add` with more than 1024 elements (more than one 32x32 BF16 scratchpad tile) returned
garbage on the emulator, so `_plan_add` forced them onto NumPy (`ATALLA_FFN_ADD_NUMPY`, default `1`).

Repro (before the fix), GELU still on NumPy:

```
ATALLA_FFN_ADD_NUMPY=0 python run_graph.py --model gpt2_mlp --mode validate --seed 0 --out-dir out/addfix/repro
  [add_1] add -> emulator (0x7000, 4096 elems) cos=0.499978 relmax=0.820988 rel_l2=0.866038
          emu_nz=1024 ref_nz=4095 bytes_ld=8204 bytes_st=4102
  end-to-end: cos=0.987352 rel_l2=0.158556 relmax=0.129630
```

`out/addfix/addiff.py add_1 4096 1000 4000 7000` (A at 0x1000, B at 0x4000, C at 0x7000, 128 rows of 32)
shows **tile 0 is exactly right and tiles 1-3 are all zeros**:

```
tile 0: max_err=0.0039 nz=1024/1024
tile 1: max_err=1.9937 nz=0/1024
tile 2: max_err=1.9207 nz=0/1024
tile 3: max_err=2.0898 nz=0/1024
```

## Root cause: `addi_s` immediate overflow in the compiler, not the SDMA/slot logic

`kernels/add.py` emitted the per-tile pointer bump as a literal, `A_GMEM = A_GMEM + 2048;`
(`tile_bytes = 32 rows * 32 cols * 2 B`). `aihw-ppci-compiler/ppci/arch/atalla/instructions.py`
selects `addi_s` for `ADDI32(reg, CONSTI32)` whenever `value < 4096`
(`pattern_add_i32_reg_const` / `pattern_add_i32_const_reg`, around lines 675-710), but the `addi_s`
immediate is a **signed 12-bit field**: `functional_sim/src/components/decode.py` decodes it with
`sign_extend(get_bits(instr, 34, 23), 12)`. So `+2048` is encoded as `0x800` and executes as **`-2048`**.
Every constant in `[2048, 4095]` is affected; `1024` (the 1-tile case) and everything `< 2048` is fine,
which is exactly why single-tile adds passed.

Dynamic SDMA trace of the failing kernel (`out/addfix/trace_add.py out/addfix/repro/add_1.in ...`,
which wraps `sdma_load`/`sdma_store` and prints every call):

```
LD sid=0 dram=0x1000   LD sid=1 dram=0x4000   ST sid=0 dram=0x7000     <- tile 0 correct
LD sid=0 dram=0x800    LD sid=1 dram=0x3800   ST sid=0 dram=0x6800     <- pointers went DOWN by 0x800
LD sid=0 dram=0xFFFFF800 NR=0 NC=0 fc=None ...                         <- cfg words now garbage
```

Secondary effect that made the symptom look like "stale operand / wrong slot": `run_graph._run_emulator`
puts the scalar stack right after the DRAM image (`stack_base = 0x7000` here, frame pointer
`x8 = 0x6FC8`, locals `A_GMEM`/`B_GMEM`/`C_GMEM`/`sdma_ctl_*` at `0x6FD8..0x6FFC`). The second
iteration's `scpad_store` to `0x6800..0x6FFF` **overwrote the stack frame**, so from iteration 3 on the
SDMA control words and pointers were tile data (`NR=0`, `full_cols=None`, address `0xFFFFF800`). That is
why the perf counters showed only ~2 tiles of traffic (`bytes_ld 8204 / bytes_st 4102`) even though the
loop counter reached 4 and all 6298 instructions retired.

`kernels/gelu.c` (same 4-tile loop) never hit this because it reads `tile_bytes` from the cfg table
(`*(cfg_ptr + 5)`) and therefore does a register-register `add_s`. The old comment in `add.py` blaming
"variable `sp` confused regalloc" was a red herring; the SP1/slot-1 load path and
`sdma_ctl_val` packing are correct.

## Fix (kernel side, this worktree)

`atalla-graph/kernels/add.py` and `atalla-graph/kernels/mul.py` (identical loop): materialise the stride as a
local, `int tile_bytes = 2048;` (the compiler's constant materialisation `lui_s`/`addi_s(value & 0x7F)` is
correct for large values), and bump with `A_GMEM = A_GMEM + tile_bytes;`. Generated `.s` now reads
`lui_s x9, 16` (16 << 7 = 2048) once and `add_s x9, x9, x10` in the loop, no `addi_s ..., 2048`.

`atalla-graph/graph/tile_planner.py::_plan_add`: the > 1024-element NumPy cutoff is no longer the default.
`ATALLA_FFN_ADD_NUMPY` now defaults to `0` (emulator); set `ATALLA_FFN_ADD_NUMPY=1` to get the old
behaviour (adds > 1024 elements on NumPy). `codegen/c_emitter.py::emit_add` needed no change.

**Not fixed (outside my file list):**
* The real bug is in the compiler pattern condition (`value < 4096` should be `-2048 <= value < 2048`
  for both `ADDI32(reg, CONST)` patterns, and likewise any other `I`-type immediate pattern that uses the
  same bound). Any hand-written or generated AtallaC that adds a literal in `[2048, 4095]` to an int is
  silently wrong until that is fixed.
* `atalla-graph/kernels/relu.py` has the identical `IN_GMEM = IN_GMEM + {tile_bytes};` pattern and will
  fail the same way for any relu > 1024 elements. No current model triggers it (largest relu today:
  1024 elements in `alexnet_small`; basic/alexnet relus are 32..1024), so it is latent. Same one-line fix applies.

## Verification

### Standalone kernel check (random N(0,1) BF16 inputs, real compile + emulate pipeline)

`python out/addfix/kernel_multitile_test.py out/addfix/kernel_test`:

| kernel | elems | tiles | cos vs NumPy | bytes_st |
|---|---|---|---|---|
| add | 1024 | 1 | 0.999998 | 2048 |
| add | 2048 | 2 | 0.999998 | 4096 |
| add | 3072 | 3 | 0.999998 | 6144 |
| add | 4096 | 4 | 0.999998 | 8192 |
| mul | 1024 | 1 | 0.999999 | 2048 |
| mul | 2048 | 2 | 1.000000 | 4096 |
| mul | 3072 | 3 | 1.000000 | 6144 |
| mul | 4096 | 4 | 1.000000 | 8192 |

(`max_abs` 0.016-0.031 = one BF16 ulp from the script's own rounding of the reference; every tile populated.)

### `add_1` node in gpt2_mlp, before / after

| | cos | rel_l2 | relmax | emu_nz | bytes_ld / bytes_st |
|---|---|---|---|---|---|
| before (`out/addfix/repro`) | 0.499978 | 0.866038 | 0.820988 | 1024 / 4095 | 8204 / 4102 |
| after (`out/addfix/after_mlp`) | 0.999990 | 0.004607 | 0.006173 | 4095 / 4095 | 16384 / 8192 |

Per-tile diff after: all four tiles `max_err=0.0039`.

### All-emulator GPT-2 runs (`ATALLA_GELU_KERNEL=horner ATALLA_FFN_ADD_NUMPY=0`)

| model | nodes total / emu / numpy / pass | cos | rel_l2 | relmax | bytes ld / wr | FLOPs | out dir |
|---|---|---|---|---|---|---|---|
| gpt2_mlp, before (GELU NumPy, add emu; my repro) | 16 / 6 / 1 / 2 | 0.987352 | 0.158556 | 0.129630 | 76300 / 26630 | 573073 | `out/addfix/repro` |
| gpt2_mlp, after, all emulator | 16 / **7 / 0** / 2 | **0.999991** | 0.004266 | 0.003704 | 92672 / 38912 | 657825 | `out/addfix/gpt2_mlp_full_emu` |
| gpt2_block, after, all emulator | 47 / **22 / 0** / 7 | **0.999989** | 0.004733 | 0.004950 | 170880 / 69632 | 1106033 | `out/addfix/gpt2_block_full_emu` |

(The pre-fix all-emulator gpt2_block number, cos 0.985983, is the "models" agent's run recorded in
`docs/model_baselines_notes.md`; I did not re-run it before the fix.) Per-node in the all-emulator block:
`add_7` (4096 elems) cos 0.999989 / rel_l2 0.004808, `gelu` (horner) cos 0.999822, every other add
>= 0.999818. **gpt2_block now runs 100 % of its compute nodes on the emulator** (0 NumPy fallbacks).

### Default runs with the new planner default (adds > 1024 on the emulator)

| model | command (`python run_graph.py --mode validate --seed 0 ...`) | nodes total / emu / numpy / pass | cos | rel_l2 | relmax | bytes ld / wr |
|---|---|---|---|---|---|---|
| basic | `--model basic --out-dir out/addfix/basic` | 18 / 10 / 0 / 1 | 1.000000 | 0.000000 | 0.000000 | 7488 / 640 |
| alexnet_small | `--model alexnet_small --scale 0.01 --out-dir out/addfix/alexnet_small` | 42 / 21 / 0 / 4 | 0.999992 | 0.004199 | 0.006579 | 97652 / 9276 |
| layernorm_smoke | `--model layernorm_smoke --out-dir out/addfix/layernorm_smoke` | 9 / 3 / 0 / 1 | 0.999990 | 0.004901 | 0.006803 | 2688 / 192 |
| vit_micro (default SDPA) | `--model vit_micro --out-dir out/addfix/vit_micro` | 48 / **19 / 2** / 7 (was 18 / 3 / 7) | 0.999994 | 0.003433 | 0.006803 | 121408 / 45056 |
| gpt2_mlp | `--model gpt2_mlp --out-dir out/addfix/gpt2_mlp_default` | 16 / **6 / 1** / 2 (was 5 / 2 / 2) | 0.999998 | 0.002196 | 0.003704 | 84480 / 30720 |
| gpt2_attn | `--model gpt2_attn --out-dir out/addfix/gpt2_attn` | 32 / 15 / 0 / 5 | 0.999999 | 0.001724 | 0.004167 | 78208 / 30720 |
| gpt2_block | `--model gpt2_block --out-dir out/addfix/gpt2_block_default` | 47 / **21 / 1** / 7 (was 20 / 2 / 7) | 0.999995 | 0.003006 | 0.009901 | 162688 / 61440 |

basic / alexnet_small / vit_micro / layernorm_smoke / gpt2_attn end-to-end cosines are unchanged from
`docs/model_baselines_notes.md`; the only remaining NumPy node in gpt2_mlp/gpt2_block is GELU (default
`ATALLA_GELU_KERNEL` unset), and in vit_micro GELU + the NumPy SDPA reference. `vit_micro`'s 2048-element
`add_8` now runs on the emulator: cos 0.999989 / rel_l2 0.004811.

## Files changed by this agent

* `atalla-graph/kernels/add.py` — `tile_bytes` as a local register operand (+ comment explaining why).
* `atalla-graph/kernels/mul.py` — same fix.
* `atalla-graph/graph/tile_planner.py::_plan_add` — `ATALLA_FFN_ADD_NUMPY` default `1` -> `0`.
* `docs/add_kernel_multitile_fix.md` — this file.
