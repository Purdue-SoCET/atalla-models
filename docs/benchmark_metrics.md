# Atalla benchmark metrics

This document defines every column of the benchmark table produced by
`bench/run_benchmark.py` (per run: `docs/baselines/<run>/summary.json`) and
collected by `bench/build_workbook.py` into
`output/Atalla_Benchmarking_Workbook.xlsx` and `docs/baselines/README.md`.

Every value comes from the **functional simulator** path driven by
`atalla-graph/run_graph.py --mode validate --metrics-json ...`. The harness
never computes a metric itself; it only renames and rolls up keys from the
`metrics.json` that `run_graph.py` writes. The exact source key is given for
each metric so a row can be traced back to the raw file.

## How a run is produced

```
cd atalla-models-rz
python bench/run_benchmark.py --model basic --tag strict-chained --seed 42 --extra-args="--strict-end"
python bench/build_workbook.py
```

`run_benchmark.py` executes, with `cwd = atalla-graph/`:

```
python run_graph.py --model <M> --mode validate [--scale S] [--seed N] \
    --out-dir <run>/out --metrics-json <run>/metrics.json \
    --layer-metrics-csv <run>/layers.csv [extra args]
```

Environment switches that `run_graph.py` reads (e.g. `ATALLA_SDPA_FLASH=1`
for the fused flash-attention trial on ViT Micro) are passed with
`--env KEY=VAL` and recorded in `config.json`/`summary.json.flags`.
`--dry-run` prints the command without running it.

It writes to `docs/baselines/<model>__<tag>__<YYYYMMDD-HHMM>/`:

| file | content |
|---|---|
| `run.log` | stdout+stderr of `run_graph.py` (the exact command is the first line) |
| `config.json` | provenance: model, scale, seed (requested and effective), flags, shapes of every emulated node, git SHA/branch/dirty of the atalla-models worktree and of the `aihw-ppci-compiler` and `functional_sim` submodules, python/torch/numpy versions, hostname, platform, UTC timestamp, wall-clock elapsed seconds, return code, notes |
| `metrics.json` | raw `run_graph.py --metrics-json` output |
| `layers.csv` (+ `layers_template_summary.csv`) | raw `run_graph.py --layer-metrics-csv` output |
| `summary.json` | Sprint-2 columns (below) + `layers[]` rows + `provenance` block; missing keys are `null`, never an error |
| `out/` | graph dump, generated `.c`, assembled `.s`, `.in` inputs, sim `*_mem.out` / `*_perf.out` per kernel |

If `run_graph.py` does not accept `--seed` (it is being added on the `robert`
branch), the harness retries without it and records a note; `seed` in
`summary.json` is then the effective seed (`run_graph.py` hard-codes
`torch.manual_seed(42)`), `seed_requested` is what was asked for.

## Metric definitions

Key notation: `metrics.json` path `->` `summary.json` field. `agg` =
`aggregate_metrics`, `km[]` = `kernel_metrics[]` (one entry per emulated
layer), `e2e` = `end_to_end`.

### Output accuracy

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| Final cosine | `e2e.cos_sim -> final_cosine` | Cosine similarity of the final chained Atalla output vs the PyTorch reference output | One headline number for "did the model run correctly on Atalla"; 1.0 = same direction |
| rel-L2 | `e2e.rel_l2_error -> rel_l2` | `‖emu − ref‖₂ / ‖ref‖₂` on the final output | Scale-aware error; catches magnitude errors cosine hides |
| rel-max | `e2e.rel_max_abs_error -> rel_max` | `max|emu − ref| / max|ref|` on the final output | Worst single element; catches one bad tile that averages away |
| Max abs error | `e2e.max_abs_error -> max_abs_error` | absolute worst element | Debug aid for the two above |
| Strict pass | `strict_pass`, `strict_fail_reasons` (needs `--strict-end`) | `run_graph.py` asserts cos ≥ 0.999, rel-L2 ≤ 0.02, rel-max ≤ 0.05 (and gradient/loss checks in `--training`) | Pass/fail gate for CI-style regression |

### Per-node accuracy

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| Per-layer cosine / rel-L2 / rel-max | `km[].cos_sim`, `km[].rel_l2_error`, `km[].rel_max_abs_error -> layers[].cos_sim / rel_l2 / rel_max` | Same metrics per emulated node, computed against the reference activation of that node | Localises which kernel introduces error |
| Validate-inputs mode | `validate_inputs` | `chained`: emulated output of node *i−1* feeds node *i* (true end-to-end; per-layer error accumulates). `oracle`: every node gets reference inputs (per-layer error isolated; final output can look right even when a matmul is wrong) | The two modes are **not comparable** row-to-row; the baseline table uses `chained` |

### Target coverage

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| Coverage E / N / P | `stats.emulated / stats.numpy / stats.passthrough -> coverage_emulated / coverage_numpy / coverage_passthrough` (fallback: count of `km[].backend` values, see `backend_values_seen`) | Number of FX graph nodes executed on the Atalla functional sim (**E**mulated), in the NumPy fallback kernel (**N**), or **P**assed through from the PyTorch reference (reshapes, unsupported ops) | Says how much of the model actually ran on Atalla. A perfect cosine with low E means little |
| Total nodes | `stats.total -> coverage_total_nodes` | nodes in the lowered FX graph | denominator for coverage |
| Emulated layers | `agg.emulated_layer_count -> emulated_layer_count` | length of `km[]` | should equal Coverage E |

Observed `km[].backend` values so far: `emulator`. `run_graph.py` also labels
nodes `numpy` (fallback kernels) and `passthrough` in its `stats` block; the
latter never appear in `km[]`, which is why `stats` is the authoritative source.

### Packet work

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| Scheduled packets | `agg.total_sched_packets -> sched_packets` | Static packet (VLIW bundle, ≤ 4 slots) count in the compiled program, summed over emulated kernels | Code size / static ILP proxy |
| Dynamic packets | `agg.total_emu_packets -> dyn_packets` | Packets actually issued by the functional sim (loops unrolled at run time) | Closest available proxy for issue cycles; the sim has no stall model |
| Retired instructions | `agg.total_emu_instructions_retired -> retired_instructions` | Slot-instructions retired at run time | Work actually done |
| Static fill | `agg.aggregate_static_slot_efficiency -> static_fill` | `total_sched_slots_filled / (4 × sched_packets)` | How well the scheduler packs bundles |
| Dynamic fill | `agg.aggregate_dynamic_slot_efficiency -> dynamic_fill` | `retired_instructions / (4 × dyn_packets)` | Run-time bundle utilisation |
| Slot histogram | `agg.aggregate_sched_slot_histogram -> sched_slot_histogram` | `{slots_filled: packet_count}` | Shows whether fill is dominated by 1-wide packets |
| Retired by class | `agg.total_dyn_retired_<c> -> dyn_retired_<c>` for `c ∈ {branch_control, sdma, scalar_mem, scalar_alu, vector_mem, vector_alu, gemm_systolic, move_convert}` | Retired-instruction mix per execution class | Instruction mix; the `O_*` operands of the future cost model |

### External traffic (data movement)

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| Bytes loaded / written | `agg.total_bytes_loaded / agg.total_bytes_written -> bytes_loaded / bytes_written` | Bytes DMA'd from external memory into the scratchpads / written back | DRAM bandwidth demand; denominator of arithmetic intensity |
| SP0 / SP1 split | `agg.total_bytes_loaded_sp0/sp1`, `agg.total_bytes_stored_sp0/sp1 -> bytes_loaded_sp0/sp1`, `bytes_stored_sp0/sp1` | Same bytes split by scratchpad | Scratchpad balance (weights vs activations) |
| Bytes total | `agg.total_bytes_memory_traffic -> bytes_total` | loaded + written | |

### Compute work

| Metric | Functional-sim value | Meaning | Why it matters |
|---|---|---|---|
| FLOPs | `agg.total_flops -> flops` | FLOPs derived from op geometry (2·M·N·K for GEMM, elementwise counts for vector/scalar ops), summed over emulated nodes | Numerator of arithmetic intensity |
| FLOPs by class | `Σ km[].flops_matmul / flops_vector / flops_scalar -> flops_matmul / flops_vector / flops_scalar` | Split by systolic (GEMM) / vector / scalar unit | Which unit the model stresses |
| Arithmetic intensity | `agg.aggregate_arithmetic_intensity -> arithmetic_intensity` | `flops / (bytes_loaded + bytes_written)` | Roofline x-axis |
| FLOPs per dynamic packet | `flops / dyn_packets -> flops_per_dyn_packet` (harness-derived) | useful work per issued bundle | cheap efficiency indicator |

### Provenance

| Field | Source | Why |
|---|---|---|
| model, model_class, tag, scale, seed, seed_requested, flags, validate_inputs | CLI args + `metrics.model`, `metrics.validate_inputs` | identifies the configuration |
| shapes | `km[].shape` per node | sanity check that the same graph was compiled |
| atalla_models_sha/branch, compiler_sha, functional_sim_sha | `git -C <path> rev-parse HEAD` / `--abbrev-ref HEAD`, plus `dirty` flag | rows from different compiler/sim revisions are not comparable |
| python, torch, numpy, hostname, platform, timestamp_utc, elapsed_s_wall | `platform`, module `__version__`, `socket.gethostname()` | reproducibility |

### Timing (future)

| Field | Value today |
|---|---|
| `cycles`, `stalls` | `null` — **not measured**. The functional sim has no cycle, latency or stall model. |
| `elapsed_s_run_graph`, `elapsed_s_wall` | host wall-clock seconds of the Python pipeline; **not** a hardware metric |

## Functional-sim limitations

* **Counts only, no timing.** The simulator retires packets and instructions
  but has no pipeline, latency, bank-conflict or DMA-stall model. Dynamic
  packets are an *issue-slot* proxy, not cycles. Nothing in this table can be
  turned into seconds or throughput without a cycle model.
* **Legacy `compile_asm` packetizer vs compiler `-p` path are not comparable.**
  `run_graph.py` schedules through the in-tree pipeline (`c_emitter` → ppci
  `-p` packetized assembly). Numbers produced by the older standalone
  `compile_asm` packetizer in `aihw-ppci-compiler` use a different bundling
  policy and different `sched_packets` / fill values. Only compare rows whose
  `compiler_sha` matches and that went through the same path (all rows in
  `docs/baselines/` use `run_graph.py`).
* **Chained vs oracle.** See *Validate-inputs mode* above; baseline rows are
  `chained`.
* **FLOPs are analytic**, derived from the op geometry, not counted by the
  simulator; they are the same for every backend.
* **Byte counts are DMA bytes** between external memory and the scratchpads.
  Scratchpad-internal reads/writes and register traffic are not counted.
* **Coverage is node-level.** A single passthrough node can hide most of the
  model's work (e.g. an attention op passed through from PyTorch). Always
  read E/N/P together with FLOPs.
* **Not measured:** cycles, stalls, latency, power, area, real DRAM traffic,
  host-side launch overhead.

## Cost model (future item — NOT implemented)

The Sprint-2 report proposes an estimated-cost model built from the counts
above:

```
Ĉ = c0 + c_dma · B + c_sys · O_sys + c_vec · O_vec + c_scalar · O_scalar + c_branch · O_branch + stalls
```

where

| term | source column in this table |
|---|---|
| `B` | `bytes_total` (`bytes_loaded + bytes_written`) |
| `O_sys` | `dyn_retired_gemm_systolic` |
| `O_vec` | `dyn_retired_vector_alu + dyn_retired_vector_mem` |
| `O_scalar` | `dyn_retired_scalar_alu + dyn_retired_scalar_mem` |
| `O_branch` | `dyn_retired_branch_control` |
| `c0, c_dma, c_sys, c_vec, c_scalar, c_branch` | **unknown** — must be fitted against measured cycles |
| `stalls` | **not modelled** |

**Status: not implemented.** Every operand `B`, `O_*` is already recorded in
`summary.json`, but the coefficients require measured cycles from an RTL or
cycle-accurate simulation that does not exist in this repository. Until then
the `cycles` and `stalls` columns stay `null` and no estimated-cost column is
emitted; emitting one with guessed coefficients would be fabricated data.
