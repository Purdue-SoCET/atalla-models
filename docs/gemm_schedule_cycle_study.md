# GEMM schedule study: baseline vs double_buffer vs weight_stationary

Measured 2026-11-16/17 (Robert Zhang). Pins: compiler `atalla-arch` 351b5fab, functional sim `main` f159f0d, cycle simulator ITaP `SoCETOrganization/atalla-sim` master 51c7468 (atalla-sim 1.2.0).

Schedules live in `atalla-graph/kernels/gemm.py` and are selected with `ATALLA_GEMM_SCHEDULE=baseline|double_buffer|weight_stationary|auto`. `gemm_traffic()` is the analytic traffic model; it matches measured bytes loaded/written and SDMA counts exactly on all 100 GEMM layer-runs below.

- **weight_stationary** loops `ni > ki > mi`: a weight tile is loaded with `lw.vi` once and reused across all M row tiles.
- **double_buffer** overlaps the next tile's DMA with the current tile's compute.
- **auto** picks per layer from `gemm_traffic()`.

## 1. Cycle counts (atalla-sim, one 32x32 output tile per program)

Cold icache (`cycle_runs/programs/summary.md`):

| shape (MxNxK) | schedule | cycles | Δ vs baseline | packets | lw.vi | SDMA ld/st | DRAM B scpad / icache / dcache |
|---|---|---:|---:|---:|---:|---|---|
| 32x32x128 | baseline | 39,625 | | 7,418 | 128 | 16/4 | 33,024 / 25,664 / 640 |
| 32x32x128 | double_buffer | 36,696 | **−7.4%** | 6,980 | 128 | 9/1 | 20,480 / 27,008 / 512 |
| 32x32x128 | weight_stationary | 38,546 | −2.7% | 7,068 | 128 | 12/4 | 32,768 / 27,072 / 512 |
| 128x32x32 | baseline | 39,635 | | 7,496 | 128 | 16/4 | 33,024 / 25,664 / 640 |
| 128x32x32 | double_buffer | 39,280 | −0.9% | 7,205 | 128 | 12/4 | 32,768 / 26,880 / 512 |
| 128x32x32 | weight_stationary | 28,763 | **−27.4%** | 5,421 | 32 | 9/4 | 26,624 / 11,328 / 512 |

Warm icache (`cycle_runs/programs_warm_icache/summary.md`): 32x32x128 = 39,053 / 36,136 / 38,004 and 128x32x32 = 39,053 / 38,722 / 28,220 (baseline / double_buffer / weight_stationary). Same ranking.

**Reading the numbers**

- Each schedule wins where its reuse applies: weight_stationary when weights are reused across row tiles (large M), double_buffer when there are several K steps to overlap (large K).
- weight_stationary on 128x32x32 wins mostly by issuing fewer instructions (32 vs 128 `lw.vi`, 5,421 vs 7,496 packets), not by moving less DRAM data.
- Absolute cycles are dominated by overhead because kernels are compiled without `-O` (see caveats): `frontend:refill` is about 44% of cycles, and MAC utilisation is 0.3–0.45%. Treat the deltas, not the absolute cycles, as the result.

## 2. Whole-model functional-sim comparison (seed 42, chained validate)

16 harness runs, all rc 0, in `runs/` (regenerate the table with `runs/compare.py` → `runs/compare.txt`). Functional sim has no timing: packets and instructions are counts, not cycles.

| Model | Schedule | cosine | bytes loaded | SDMA | gemm+lw.vi | dyn pkts | retired instrs |
|---|---|---:|---:|---:|---:|---:|---:|
| alexnet_small @0.01 | baseline | 0.9999992 | 97,652 | 301 | 3,208 | 101,215 | 119,805 |
| alexnet_small @0.01 | double_buffer | 0.9999992 | −0.5% | −4.0% | 0 | −2.8% | −2.2% |
| alexnet_small @0.01 | weight_stationary | 0.9999992 | −5.2% | −29.2% | −40.9% | **−40.1%** | **−37.6%** |
| alexnet_small @0.01 | auto | 0.9999992 | −5.3% | −31.2% | −40.9% | **−40.2%** | **−37.6%** |
| gpt2_block | baseline | 0.9999957 | 162,688 | 557 | 896 | 50,553 | 64,589 |
| gpt2_block | double_buffer | 0.9999957 | −4.3% (writes −10%) | −3.6% | 0 | −2.3% | −1.9% |
| gpt2_block | weight_stationary | 0.9999957 | −0.6% | −2.5% | 0 | −2.1% | −1.8% |
| gpt2_block | auto | 0.9999957 | −4.3% (writes −10%) | −3.6% | 0 | −2.3% | −1.9% |

`basic` and `gpt2_attn` move by 1–7%; full rows, the retired-instruction mix and per-GEMM-layer numbers are in `runs/compare.txt`.

- AlexNetSmall's convolutions (im2col GEMMs with many row tiles) are where weight reuse pays off: `conv2d` drops from 62,927 to 32,750 dynamic packets.
- GPT-2's 32x32 GEMMs have a single row tile, so there is little weight reuse; double_buffer helps more there. `auto` picks the better schedule for every model.
- Cosine is identical across schedules for every model: the schedules change data movement and instruction count, not arithmetic.

## 3. Correctness checks

- Scratchpad DRAM bytes from atalla-sim equal the functional sim's `bytes_moved` and `gemm_traffic()` in all 6 cycle runs.
- Outputs are bit-identical across schedules for each shape.
- **atalla-sim output is column-reversed** relative to the functional sim and NumPy (raw cosine ≈ −0.04 / −0.002). atalla-sim's `lw.vi` shifts columns in at column 0 while `gemm.py` pushes weight rows first to last. After flipping columns the atalla-sim output equals the functional-sim output bit for bit in all 6 runs (cosine vs NumPy 0.9999985 for 128x32x32, 0.9999949 for 32x32x128; re-verified 2026-11-22 from the saved `.npy` files). This does not affect cycles. functional_sim PR #3 flips `lw.vi` to last-to-first, which should resolve it once `gemm.py` reverses its weight loop and the pin is bumped.
- `tests/kernels/test_gemm_schedule.py`: 4 passed (functional sim f159f0d).

## 4. Problems found (for the compiler and atalla-sim owners)

1. **`-O2` miscompile** (compiler 0df04a93, same codegen in 351b5fab): loop counters in x9–x20 are kept live across `jal load_weight_tile`, and the callee clobbers x9–x15 without saving them, so every loop exits after one trip. Measured 5,600 cycles, output all zeros. Repro: `cycle_runs/o2_miscompile/`.
2. **atalla-sim `--golden` can't run these programs**: its bundled functional sim 4f3eb29 rejects `vreg.st` to scratchpad sid 3 (used by `-O0` vector spills) and raises "Unknown mnemonic" on the `-O2` image. Needs a functional-sim submodule bump in atalla-sim. The golden here comes from the atalla-models functional sim on the same A/W.
3. **Missing DMA dependency**: `--check` reports 12–20 places per run where a `vreg.ld` reads a pad row before its `scpad.ld` has finished; the compiler emits no dependency. Outputs were still correct with 6-cycle DRAM but could go stale at realistic latency.
4. `make native` in atalla-sim doesn't build on Apple Silicon (x86 intrinsics). Optional; the numpy fallback gives the same results.

## 5. Caveats

- Kernels are compiled without `-O` (the atalla-models flags) because of problem 1. atalla-sim's own default is `-O 2 -p -S`; `cycle_sim/run_atalla_sim.py` is `atalla-sim run` with that one change. The packetizer also differs: atalla-sim uses the compiler's (`-p`), atalla-models uses functional_sim's `schedule_program`.
- Ramulator was not built, so DRAM is a fixed 6-cycle latency. Numbers are from the atalla-0x01 performance model, not RTL- or silicon-validated.
- One output tile per program; whole-model cycle counts are not available yet.
- The team agreed on 2026-11-22 that branch/jump immediates are in packet units (imm × 20 bytes, not imm × 5). Both the compiler and RTL will change; these programs were built before that change. Expect branch encodings to differ, but cycle counts are unaffected unless that changes where branch targets fall.

## 6. Reproduce

Requires an atalla-sim checkout with its venv (set `ATALLA_SIM_PY`, default `../../atalla-sim-itap/.venv/bin/python` relative to `cycle_sim/`) and the atalla-models venv (`docs/ENVIRONMENT.md`). Each simulation takes about 7 s and 58 MB.

```bash
P=~/.venvs/atalla/bin/python
$P cycle_sim/gen_gemm.py --out cycle_runs/programs
ATALLA_SIM_PY=/path/to/atalla-sim/.venv/bin/python ./cycle_sim/run_all.sh cycle_runs/programs
# functional-sim golden, per run dir <d> = <sched>_<M>x<N>x<K>:
$P tests/kernels/test_gemm_schedule.py --sim-child M N K SCHED <d>/A.npy <d>/W.npy <d>/fsim
$P cycle_sim/analyze.py cycle_runs/programs
# sanity check of the simulator install (expects 1341 cycles):
gunzip -c tests/equivalence/programs/gemm_tiled.in.gz > /tmp/g.in && atalla-sim run /tmp/g.in --report   # inside the atalla-sim checkout
```

Whole-model runs: `bench/run_benchmark.py --model <m> --seed 42 --tag sched-<schedule> --env ATALLA_GEMM_SCHEDULE=<schedule>`, then `runs/compare.py`.

Committed artifacts are trimmed: per-run `fsim/*.out` register/memory dumps are not committed (they are regenerated by the golden step).
