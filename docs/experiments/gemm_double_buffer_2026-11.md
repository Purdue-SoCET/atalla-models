# GEMM schedule experiment: `double_buffer` vs `baseline` (2026-11-04)

One bounded scheduling / data-movement experiment in the tiled GEMM generator
(`atalla-graph/kernels/gemm.py`), measured with the benchmark harness on the
functional simulator. Everything below is a **count**; the simulator has no
cycle, latency or DMA-stall model, so nothing here is a speed claim.

Selected with `ATALLA_GEMM_SCHEDULE=baseline|double_buffer` (unset = `baseline`,
whose generated C is byte-identical to the pre-experiment generator). The
variable reaches the generator through `run_graph.py`'s environment, so the
harness records it in `config.json` / `summary.json.flags` (`bench/run_benchmark.py --env`).

Runs (seed 42, chained inputs, submodules `aihw-ppci-compiler 351b5fab` /
`functional_sim f159f0d`): `docs/baselines/<model>__sched-baseline__20261104-15xx`
and `<model>__sched-double_buffer__20261104-15xx|16xx`, rolled into
`docs/baselines/README.md` and `output/Atalla_Benchmarking_Workbook.xlsx` by
`bench/build_workbook.py`. The pre-existing baseline rows are untouched.

## 1. What the hardware allows

* Two scratchpads, SP0 and SP1, each **one 32×32 BF16 tile** (32 rows × 32 banks;
  `functional_sim/run.py` and `run_graph.py` build `Scratchpad(slots_per_bank=32)`,
  and `scpad_ls.py` addresses rows modulo 32).
* The systolic array holds one weight tile, filled by 32 `lw.vi` (`load_weights`)
  from a scratchpad; `gemm.vv` is matmul-only, the accumulate is an explicit
  `add.vv` (`ppci/arch/atalla/vector_instructions.py::patt_gemm_vv`).
* The simulator resets its `lw.vi` column counter **only on an SP1 DMA load**
  (`functional_sim/src/functional_sim.py`, "SP1 DMA loads are always a new tile").
  A weight tile pushed from SP0 after the first one would be inserted with the
  shift semantics (reversed order), so weight tiles must stream through SP1.

A 32×32×32 K step therefore has three live tiles if the next weight tile is to
be in flight while the current step computes (A, C, next W) and only two
buffers exist. True DMA/compute overlap at tile granularity is not expressible
with this ISA; the variant below is the closest schedule that is, and it is the
structure of the compiler's own hand-written reference kernels
(`aihw-ppci-compiler/test_validation_kernels/gemm_tiled_pipelined*.c`).

## 2. The two schedules

Loop order is `mi > ni > ki` in both; the arithmetic (BF16 inputs, BF16-rounded
`gemm.vv` product, BF16-rounded `add.vv` accumulate) is identical.

| | `baseline` | `double_buffer` |
|---|---|---|
| per (mi, ni) | – | **C tile → SP0 once**; `fetch_weight_tile(W(0))` → SP1 |
| per K step | zero row → SP0; W → SP1; 32× `load_weights`; A → SP0; **C → SP1**; 32 `gemm` rows; **C → DRAM** | `push_weights` (SP1 → array, unrolled ×2); A → SP1 (over the consumed W); 32 `gemm` rows (A from SP1, C in SP0); **`fetch_weight_tile(W(k+1))` → SP1 issued right after the row loop, before the loop bookkeeping** |
| after K loop | – | **C tile → DRAM once** |
| zero row for partial-N weight tiles | DMA'd every K step, even when `N ≥ 32` | DMA'd only when `tile_n < 32`, through SP1 row 0 just before the W DMA (SP0 = resident C is never touched) |
| scratchpad roles | SP0 = Z + A, SP1 = W then C | SP0 = C (resident), SP1 = W then A (streaming) |
| `load_weights` loop | rolled, 32 iterations | unrolled by 2, 16 iterations (`push_weights`) |
| pointer bumps | `a_off = mi*tile*gK + ki*tile` recomputed per step | `a_addr += k_step`, `k_step = tile_sz*2` held in a register (`addi_s` 12-bit immediate, see `docs/add_kernel_multitile_fix.md`) |

Both variants compile with `atalla_cc -S`, run on the simulator and produce
**bit-identical C** for 64×64×64 (2×2×2 tiles) and 40×20×100 (partial tiles,
4 K tiles): `tests/kernels/test_gemm_schedule.py`. Bit-identity holds because
the accumulator leaving `add.vv` is already BF16-valued, so the baseline's
per-step DRAM round trip (truncating store, reload) is lossless.

Analytic DMA traffic per GEMM, with `Mt, Nt, Kt` tile counts and `tm, tn, tk`
tile edges (BF16 = 2 bytes):

```
baseline       loads  = Mt·Nt·Kt · (64 + 2·tn·tk + 2·tm·tk + 2·tm·tn)      stores = Mt·Nt·Kt · 2·tm·tn
double_buffer  loads  = Mt·Nt·Kt · (64·[tn<32] + 2·tn·tk + 2·tm·tk) + Mt·Nt · 2·tm·tn
               stores = Mt·Nt · 2·tm·tn
saving         loads  = Mt·Nt·(Kt−1)·2·tm·tn + 64·Mt·Nt·Kt·[tn≥32]        stores = Mt·Nt·(Kt−1)·2·tm·tn
```

So the byte saving is zero unless a GEMM has more than one K tile or full-width
N tiles; at today's model sizes (D = 32) that is rare, which the table confirms.

## 3. Measured deltas (functional simulator, seed 42)

`sched` = scheduled (static) packets, `dyn` = dynamic packets, `retired` =
slot-instructions retired, bytes are SDMA bytes between DRAM and the
scratchpads. Cosine, rel-L2 and rel-max are identical to six significant
figures in every run, and every per-layer cosine is identical between the two
schedules; no non-GEMM layer's counters changed.

| Model | Schedule | sched pkts | dyn pkts | retired | bytes loaded | bytes written | static fill | dynamic fill | final cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basic | baseline | 1,593 | 2,941 | 3,423 | 7,488 | 640 | 29.6% | 29.1% | 1.000000 |
| basic | double_buffer | 1,587 (−0.4%) | 2,722 (−7.4%) | 3,192 (−6.7%) | 7,296 (−2.6%) | 640 (0) | 29.5% | 29.3% | 1.000000 |
| alexnet_small (scale 0.01) | baseline | 3,720 | 101,215 | 119,805 | 97,652 | 9,276 | 29.4% | 29.6% | 0.999999 |
| alexnet_small (scale 0.01) | double_buffer | 3,926 (**+5.5%**) | 98,421 (−2.8%) | 117,227 (−2.2%) | 97,120 (−0.5%) | 9,128 (−1.6%) | 29.4% | 29.8% | 0.999999 |
| gpt2_attn | baseline | 2,785 | 25,630 | 32,873 | 78,208 | 30,720 | 29.6% | 32.1% | 0.999999 |
| gpt2_attn | double_buffer | 2,773 (−0.4%) | 25,192 (−1.7%) | 32,411 (−1.4%) | 77,824 (−0.5%) | 30,720 (0) | 29.5% | 32.2% | 0.999999 |
| gpt2_block | baseline | 3,918 | 50,553 | 64,589 | 162,688 | 61,440 | 29.7% | 31.9% | 0.999996 |
| gpt2_block | double_buffer | 3,902 (−0.4%) | 49,375 (−2.3%) | 63,355 (−1.9%) | 155,648 (**−4.3%**) | 55,296 (**−10.0%**) | 29.6% | 32.1% | 0.999996 |

Retired-instruction mix (the `O_*` operands of the future cost model):

| Model | Schedule | SDMA | branch | scalar ALU | scalar mem | vector mem | gemm + lw.vi | SP0 ld / SP1 ld | SP0 st / SP1 st |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| basic | baseline | 34 | 318 | 1,192 | 1,351 | 392 | 99 | 832 / 6,656 | 448 / 192 |
| basic | double_buffer | 31 | 228 | 1,156 | 1,255 | 386 | 99 | 640 / 6,656 | 640 / 0 |
| alexnet_small | baseline | 301 | 10,481 | 38,951 | 40,882 | 22,479 | 3,208 | 77,588 / 20,064 | 5,396 / 3,880 |
| alexnet_small | double_buffer | 289 | 8,912 | 39,029 | 39,819 | 22,467 | 3,208 | 10,536 / 86,584 | 9,128 / 0 |
| gpt2_attn | baseline | 307 | 1,833 | 13,222 | 8,955 | 5,900 | 384 | 39,296 / 38,912 | 18,432 / 12,288 |
| gpt2_attn | double_buffer | 301 | 1,653 | 13,150 | 8,763 | 5,888 | 384 | 38,912 / 38,912 | 30,720 / 0 |
| gpt2_block | baseline | 557 | 3,759 | 25,771 | 17,706 | 11,772 | 896 | 78,720 / 83,968 | 32,768 / 28,672 |
| gpt2_block | double_buffer | 537 | 3,342 | 25,561 | 17,147 | 11,744 | 896 | 71,680 / 83,968 | 55,296 / 0 |

Per GEMM-backed layer (bytes loaded / written, dynamic packets, static packets):

| Model | node | M×N×K (tiles) | baseline ld / wr | double_buffer ld / wr | dyn pkts | sched pkts |
|---|---|---|---|---|---|---|
| basic | matmul ×3 | 1×32×32 (1·1·1) | 2,240 / 64 | 2,176 / 64 | 742 → 669 | 281 → 279 |
| alexnet_small | conv2d | 1024×1×27 (32·1·1) | 61,120 / 2,048 | 61,120 / 2,048 | 62,927 → 61,456 | 280 → 315 |
| alexnet_small | conv2d_1 | 256×1×9 (8·1·1) | 5,776 / 512 | 5,776 / 512 | 15,791 → 15,424 | 280 → 315 |
| alexnet_small | conv2d_2..4 | 64×{3,2,2}×{9,27,18} (2·1·1) | unchanged | unchanged | −91 each | 281 → 316 |
| alexnet_small | matmul | 1×40×32 (1·2·1) | 4,480 / 128 | 4,352 / 128 | 1,391 → 1,244 | 281 → 279 |
| alexnet_small | matmul_1 | 1×40×40 (1·2·2) | 8,960 / 256 | 8,576 / 128 | 2,663 → 2,268 | 281 → 279 |
| alexnet_small | matmul_2 | 1×10×40 (1·1·2) | 1,576 / 40 | 1,556 / 20 | 1,906 → 1,765 | 281 → 316 |
| gpt2_attn | matmul ×6 | 32×32×32 (1·1·1) | 6,208 / 2,048 | 6,144 / 2,048 | 1,672 → 1,599 | 281 → 279 |
| gpt2_block | matmul ×6 | 32×32×32 (1·1·1) | 6,208 / 2,048 | 6,144 / 2,048 | 1,672 → 1,599 | 281 → 279 |
| gpt2_block | matmul_6 (c_fc) | 32×128×32 (1·4·1) | 24,832 / 8,192 | 24,576 / 8,192 | 6,409 → 6,114 | 281 → 279 |
| gpt2_block | matmul_7 (c_proj) | 32×32×128 (1·1·4) | 24,832 / 8,192 | **18,432 / 2,048** | 6,370 → 5,925 | 281 → 279 |

### Which counters moved, and why

* **Bytes loaded / written.** Every delta matches the formula in §2 exactly.
  Single-K-tile, full-N GEMMs (all of `basic`, `gpt2_attn`, six of the eight
  block GEMMs) only lose the 64-byte zero row per step (−64 B each). The only
  GEMM with several K tiles at these sizes is the block's `c_proj`
  (32×32×128): C is loaded once instead of four times and written once instead
  of four times (−6,144 B loaded, −6,144 B written), which is the whole −10%
  write delta of `gpt2_block`. AlexNet's convolutions have `K ≤ 27` and
  `N ≤ 3`, so they keep the zero row and have nothing to hoist: their bytes are
  unchanged; only the 1×40×40 and 1×10×40 classifier matmuls (two K tiles) save
  a C round trip. **Arithmetic intensity** moves with the bytes
  (gpt2_block 4.546 → 4.829 FLOP/B).
* **SP0 / SP1 split.** The roles swap by design: C now lives in SP0 (all stores
  are SP0 stores; SP1 stores drop to 0) and A moves from SP0 to SP1. The split
  is a layout fact, not a saving.
* **Dynamic packets / retired instructions** (−1.4% to −7.4%). Three sources,
  all visible in the class mix: (1) the weight loop runs 16 iterations instead
  of 32 per tile, so branch retirements fall by 10–28% and scalar loop
  bookkeeping with them; (2) the per-step C load, C store and (for full tiles)
  zero-row load disappear, so SDMA retirements fall exactly by the number of
  removed DMAs (basic −3 = three zero rows; gpt2_block −20 = 14 zero rows + 3
  C loads + 3 C stores; alexnet −12); (3) the address arithmetic per step is a
  register add instead of two multiplies and adds. `gemm.vv` + `lw.vi`
  retirements are identical in every run: the same systolic work is done.
* **Scheduled (static) packets.** A full-tile GEMM gets 2 packets smaller
  (281 → 279). A partial-N GEMM gets ~35 packets larger (280 → 315) because
  `fetch_weight_tile` keeps the zero-row and padding loop and is now called
  from two sites (prologue and in-loop prefetch), and `push_weights` is a
  separate function. That is the **+5.5% code size** on AlexNet, whose five
  convolutions all have `N ≤ 3`: static code up, dynamic work down. Static
  slot fill is unchanged within 0.1 pt; dynamic fill improves by 0.1–0.2 pt.
* **FLOPs column** moves by a few tens (e.g. gpt2_block 1,018,779 → 1,018,633)
  only because `flops_scalar` is derived from retired scalar-ALU instructions;
  `flops_matmul` and `flops_vector` are identical. The harness note in
  `docs/benchmark_metrics.md` ("FLOPs from retired instructions") applies.
* **Accuracy.** Unchanged to the last printed digit at every layer; the
  standalone test shows the outputs are bit-identical.

### What this does and does not show

The experiment removes real DRAM traffic where a GEMM has several K tiles and
removes real instruction issue everywhere, with no accuracy cost. It does
**not** demonstrate DMA/compute overlap: the prefetch of `W(k+1)` is issued
after the row loop and is consumed by `push_weights` a handful of scalar
packets later, because there is no third buffer to issue it any earlier. On
hardware with a blocking `scpad.ld` the placement is irrelevant; with an
asynchronous DMA engine it hides only the loop bookkeeping.

## 4. What a timing model needs to turn these counts into cycles

The simulator retires packets with no notion of time. To convert the deltas
above into cycles the model needs, per the cost model sketched in
`docs/benchmark_metrics.md`:

1. **Issue model.** Cycles per dynamic packet (1 if the VLIW core issues one
   bundle per cycle without stalls). With that alone, `dyn_packets` *is* the
   cycle count and the deltas above are −1.7% … −7.4%. The compiler's scheduler
   already carries per-instruction latencies (`aihw-ppci-compiler/ppci/arch/
   atalla/latency.py`: `gemm_vv` 45, `lw_vi` 1, `scpad_ld` 13, `scpad_st` 2),
   which is the seed for an in-order issue + latency-chain estimate; note that
   the 13-cycle `scpad_ld` figure is a scheduler dependency latency, not a
   transfer time per byte.
2. **DMA model.** Setup latency plus bytes-per-cycle for `scpad.ld`/`scpad.st`,
   and whether the core **blocks** on a DMA or continues until a dependent
   vector load. This decides whether the removed C round trips are worth
   `(setup + 2048/BW)` cycles each and whether prefetch placement matters at
   all. The SDMA count and byte deltas per class above are the operands.
3. **Systolic weight-load cost.** Cycles per `lw.vi` and the array fill/drain
   latency, so the unchanged `gemm + lw.vi` count can be charged correctly and
   the weight-stationary reordering below can be evaluated.
4. **Scratchpad bank conflicts / port limits**, if `vreg.ld` from SP1 and
   `vreg.st` to SP0 in the same bundle contend.

With (1) and (2) measured on RTL or a cycle-level simulator, the expected gain
per model is `Σ_layers [Δdyn_packets · c_issue + Δbytes · c_dma + ΔSDMA · c_setup]`;
all three Δ terms are already in `summary.json` for every run.

## 5. Follow-ups this experiment points at

* **Weight-stationary loop order for `M_tiles > 1`** (`ni > ki > mi`): AlexNet's
  first convolution has 32 M tiles sharing every weight tile, so each weight tile
  is DMA'd and pushed 32 times today. Reordering would cut `lw.vi` by
  `(Mt−1)/Mt` for those layers at the cost of the per-step C round trip that
  `double_buffer` just removed; which wins depends on the DMA vs. `lw.vi` costs
  in §4. Both orders can live behind `ATALLA_GEMM_SCHEDULE`.
* **A real prefetch window.** Either half-tile streaming (DMA `W(k+1)[0:16]`
  into the rows of `A(k)` already consumed) or a 64-row scratchpad; the first
  is expressible now but doubles SDMA instruction count and has to be reversed
  when the `lw.vi` order flips.
* **Partial-N tiles.** Both schedules DMA `tile_n = min(32, N)` rows for every
  N tile and store `tile_n` columns with `full_cols = N`, so a trailing N tile
  (e.g. 1×40 matmul) reads rows past the matrix and writes columns past it.
  Harmless today (M = 1, and later rows overwrite in order) but worth a
  per-tile edge in the generator before larger models.
* `pytest atalla-graph/tests tests/kernels` in **one** process fails at
  collection (pre-existing): `run_graph` needs functional_sim's `kernels`
  package on the bare name while `tests/kernels/*` import atalla-graph's. The
  README runs the two directories separately; `test_gemm_schedule.py` runs its
  simulator leg in a child process for the same reason.
