# Flash SDPA kernel (`ATALLA_SDPA_FLASH=1`) — root cause and fix (agent "flash", 2026-11-04)

All numbers come from runs executed on 2026-11-04 in the `robert` worktree (submodules
`aihw-ppci-compiler 351b5fab`, `functional_sim f159f0d`), Python
`/Users/robert.zhang/.venvs/atalla/bin/python`, every command run from `atalla-models-rz/atalla-graph/`.
Run outputs are under `atalla-graph/out/flash/<name>/` (gitignored).

## TL;DR

Two independent defects, both in how the fused kernel `kernels/flash_sdpa_n32d32.c` and
its DRAM image (`codegen/c_emitter.py::emit_sdpa`) talk to the functional sim's systolic array:

1. **K weights were read from the wrong scratchpad.** The kernel DMA'd the K tile into SP0
   (`SDMA_RS3(0, …)`, sid 0) but read it back with `vector_load(sp, idx, N_M1, 1)`. The 4th
   argument of `vector_load` is the scratchpad **sid** (`vreg.ld vd, rs1, rs2, num_cols, sid`,
   `functional_sim/ASSEMBLY_SYNTAX.md`; same convention as `kernels/mse_loss.c`, which DMAs Y
   into SP1 and reads it with sid 1). SP1 was never written (`sdpa_sp1.out` is all empty
   cells → read as 0.0), so every score row was `q · 0 = 0`, softmax was uniform, and all
   32 output rows were identical (measured row-to-row spread **0.0**).
2. **V was presented to the systolic array in the wrong orientation.** The sim's `gemm.vv`
   computes `out[j] = Σ_k a[k] · G[k, j]` where `G[:, c]` is the c-th stationary weight column
   filled by `lw.vi`; with the kernel's reversed-order load in shift mode, scratchpad **row c
   becomes column c**, so lane j of the output is `activation · (tile row j)`. For
   `s = q · K[j,:]` the K tile must be row-major (it was), but for `o[j] = p · V[:, j]` the
   V tile must hold **V^T**. `emit_sdpa` wrote V row-major, so the kernel computed
   `softmax(QKᵀ/√d) · Vᵀ`. After fixing (1) alone, the emulator output matched that wrong
   expression with cos **0.999993** (and the true reference with cos −0.085).

Fix: `vector_load(..., 1)` → `vector_load(..., 0)` for the K loads (kernel), and
`emit_sdpa` stores `V^T` at `0x5000` (emitter). The sdpa node goes from cos **−0.1066** to
**0.999987**; vit_micro end-to-end goes from cos **0.9863** to **0.999992** (default NumPy-SDPA
path: 0.999994).

## Before / after (seed 0)

| run | command | sdpa node (cos / rel-L2 / relmax) | vit_micro final (cos / rel-L2 / relmax) | nodes emu / numpy |
|---|---|---|---|---|
| repro (original kernel + emitter) | `ATALLA_SDPA_FLASH=1 python run_graph.py --model vit_micro --mode validate --seed 0 --out-dir out/flash/repro` | −0.106544 / 1.157023 / 1.020992 (emu_norm 2.86 vs ref 5.90) | 0.986332 / 0.165220 / 0.127551 | 19 / 2 |
| fix 1 only (K sid 1→0) | same, `--out-dir out/flash/fix1` | −0.085744 / 1.165719 / 1.062977 | 0.985918 / 0.167999 / 0.132653 | — |
| **fix 1 + fix 2 (V^T in DRAM)** | same, `--out-dir out/flash/fix2` | **0.999987 / 0.007101 / 0.011450** (emu_norm 5.925 vs ref 5.897) | **0.999992 / 0.003962 / 0.013605** | 20 / 1 |
| default path (no env var), for reference | `python run_graph.py --model vit_micro --mode validate --seed 0 --out-dir out/flash/vit_default` | NumPy ref: 0.999992 / 0.004296 / 0.006223 | 0.999994 / 0.003433 / 0.006803 | 19 / 2 |
| baseline `basic` (unchanged) | `python run_graph.py --model basic --mode validate --seed 0 --out-dir out/flash/basic` | — | 1.000000 / 0.000000 / 0.000000 | 10 / 0 |

(The emu/numpy node counts differ from `model_baselines_notes.md` because `add_8` now runs on
the emulator — another agent's concurrent change to the add path, not this one; in the fixed
flash run the only NumPy node left is `gelu`.) The remaining sdpa error (rel-L2 0.0071) is
bf16 rounding of scores/probs/output in the kernel; the harness reference computed from the
*same* bf16 inputs agrees with the emulator at cos 0.999995 / rel-L2 0.0046.

## Evidence trail (harness = `out/flash/sdpa_harness.py`, reproduced in the appendix)

The harness reads Q/K/V back from `<run>/sdpa.in` (bf16 at `0x1000/0x3000/0x5000`, `scale`
f32 at `0x0C00`) and the kernel output from `<run>/sdpa_mem.out` (`0x7000`) with the same
little-endian word packing as `run_graph.py::_read_bf16`, then scores the output against
candidate expressions. Measured `scale = 0.176777 = 1/√32` (so scale is not the problem).

**Stage 1 — original kernel (`out/flash/repro`)**, |O| = 2.86:

| candidate | cos | rel-L2 |
|---|---|---|
| reference `softmax(QKᵀ·s)V` | −0.1056 | 1.156 |
| `mean(V)` broadcast (what zero K-weights + row-major V would give) | −0.1102 | 1.170 |
| unscaled / K not transposed / K or V row-reversed / column softmax / raw QKᵀ / Oᵀ | all ≤ 0.09 | ≥ 1.0 |
| **row-to-row spread of O** | **0.0** (all 32 rows identical) | |
| **O[0] vs `V.mean(axis=1)` (per-row mean of V)** | matches to 3 decimals (`0.180 0.098 0.025 0.138 …`) | |

Identical rows ⇒ scores independent of q ⇒ K weights are zero (SP1 read, defect 1). The
row equals the *per-row* mean of V, not the per-column mean, ⇒ with uniform probs the
PV stage returned `(1/32)·Σ_k V[j,k]`, i.e. `p · Vᵀ` (defect 2).

**Stage 2 — K read from SP0 (`out/flash/fix1`)**, rows now differ (spread 0.153):

| candidate | cos | rel-L2 | max_abs |
|---|---|---|---|
| **`softmax(QKᵀ·s) · Vᵀ`** | **0.999993** | **0.005146** | 0.0020 |
| reference `softmax(QKᵀ·s) · V` | −0.0849 | 1.165 | 0.544 |
| `softmax(QK·s)·Vᵀ`, `softmax(Q K[::-1]ᵀ·s)·Vᵀ`, `Pᵀ·Vᵀ`, `P·V[::-1]ᵀ` | 0.84 / 0.89 / 0.87 / −0.16 | | |

So after defect 1 the score + softmax stages are exactly right (K orientation, scale, row
softmax, reversed-load ordering all confirmed) and the only remaining error is the V
orientation in the PV multiply.

**Stage 3 — V^T stored (`out/flash/fix2`)**: emulator vs reference built from the DRAM
inputs: cos 0.999995 / rel-L2 0.004628 / max_abs 0.0039; vs the old wrong expression
`P·Vᵀ`: cos −0.085. `run_graph.py` reports the node at cos 0.999987 against the PyTorch
bf16 reference (its reference rounds slightly differently).

## Why the contract is what it is (and how it matches the working matmul kernel)

`functional_sim/src/functional_sim.py` (`lw.vi`): each load writes the vector into
`gemm_weights[:, num_weights]` while `num_weights < 32` (fill mode), otherwise shifts the
buffer right and inserts at column 0 (shift mode). `gemm.vv` then does
`EU.execute("gemm.vv", A=a(1×32), B=gemm_weights)` = plain `a @ G`, so `out[j] = Σ_k a[k]·G[k,j]`,
and `G[k, j]` is lane k of the vector sitting in column j. Lane j of the output is therefore the
dot product of the activation with the vector that ended up in column j. This is the same
"each SDMA row is a logical column of W" convention that `functional_sim/src/components/gemm.py`
documents for the tiled PE GEMM, which is why the (working) matmul path packs its weight tiles
transposed and why `K` (used as `Kᵀ` in `QKᵀ`) happens to be correct row-major while `V` must be
stored transposed. `scores[j] = q·K[j,:]` ⇒ column j must be K row j ⇒ K row-major.
`o[j] = p·V[:,j]` ⇒ column j must be V column j ⇒ DRAM tile must be `Vᵀ`.

The kernel's load loop (`idx = (N-1) - col`, col = 0..31) loads scratchpad rows 31, 30, …, 0.
In shift mode the last-loaded row 0 lands in column 0 and row c in column c — the identity
mapping the contract needs. In fill mode the same loop puts row 31−c in column c (a reversal).

## Caveat: `lw.vi` fill-vs-shift semantics and the pending upstream change

* Current sim (`f159f0d`): the first 32 `lw.vi` after reset fill columns 0..31 in load order;
  every later `lw.vi` shifts. `num_weights` is reset to 0 only by an SP1 DMA (`scpad.ld` with
  sid 1). This kernel never DMAs to SP1, so the counter is reset exactly once (at start).
* The kernel's 32-load warm-up (`load_weights(warm)` × 32) exists only to consume that single
  fill phase so that *every* real K/V load, including the first query row's, runs in shift mode.
* Measured without the warm-up (`out/flash/nowarm`, kernel otherwise fixed): sdpa cos 0.996482 /
  rel-L2 0.083908; per-row rel-L2 is **0.4505 for row 0 and ≤ 0.0080 for rows 1..31**, and row 0
  matches the prediction `softmax(q₀·K[::-1]ᵀ·s)·V` with rel-L2 0.0038 (fill mode reversed the K
  rows for the first 32 loads; by the time V is loaded the array is already full and shifting).
  End-to-end vit_micro cos 0.999885. This confirms the fill/shift model above and that the
  warm-up is load-bearing under the current sim.
* Upstream pending change (James's `investigate-weight-load-order`: `lw.vi` always shifts in at
  column 0): the reversed load loop then maps row c → column c from the very first load, which
  is exactly the configuration the warm-up currently forces. The 32 warm-up loads become
  harmless (they are fully displaced by the 32 real loads). **Both fixes (sid 0 for the K
  reads, Vᵀ in DRAM) are layout/addressing fixes that do not depend on the fill-vs-shift
  behaviour, so the kernel is correct under both the current sim and the pending semantics.**
  The only thing that would break it is a semantics where the *first* load lands in the
  *last* column (i.e. forward fill with no shift), in which case the loop's reversal would have
  to be dropped — not a planned change.
* If a future kernel variant ever DMAs into SP1 between weight loads, the counter reset would put
  the next 32 `lw.vi` back into fill mode (see `functional_sim.py` "num_weights set to 0 before
  SP1 DMA"), and the reversed loop would reverse that tile. Keep all flash traffic on SP0, or
  re-warm after any SP1 DMA.

## Files changed (uncommitted, this agent only)

* `atalla-graph/kernels/flash_sdpa_n32d32.c` — K weight loads read SP0 (`vector_load(sp, idx,
  N_M1, 0)`, was `1`); header comment now states the DRAM/weight contract. LF line endings kept.
* `atalla-graph/codegen/c_emitter.py::emit_sdpa` — stores `V^T` (`np.ascontiguousarray(vf[0].T)`)
  at `0x5000`; comment explains the contract. The NumPy-reference branch (default path) is untouched.
* `docs/flash_sdpa_investigation.md` — this file.
* Not touched: `graph/tile_planner.py::_plan_sdpa` (the `ATALLA_SDPA_FLASH` gate is unchanged and
  still restricted to `B=1, N=32, D=32`), `kernels/common.py`.

## Not done / follow-ups

* `ATALLA_SDPA_FLASH` still defaults to `0`. Flipping the default is a one-line change in
  `_plan_sdpa` but was deliberately left for the owner to decide (other agents' baselines in
  `model_baselines_notes.md` were measured with the NumPy path). With the fix, the flash path
  costs 78.6k dynamic packets vs 36.6k for vit_micro (numbers from `model_baselines_notes.md`'s
  flash trial, same kernel instruction stream) for a 0.000002 cos difference.
* The functional sim's `build_flash_attention.py` writes V row-major *and* passes a legacy
  7-operand `vreg.ld … 0, 1, row` "column" flag for the K loads that
  `build.py::expand_vreg_seven_operand_asm` silently drops. I did not run that golden; if it
  passes in the sim, its reference must differ from `softmax(QKᵀ)V` and should be re-checked
  against this finding. Likewise the kernel's comment pointed at
  `aihw-ppci-compiler/atalla_tests/kernels/flash_attention_n32_d32.c`, which does not exist in
  the compiler submodule at `351b5fab`.
* A causal / multi-head flash variant (needed for `gpt2_attn` fusion) still does not exist;
  see `gpt2_operator_coverage.md`.

## Appendix — harness (`atalla-graph/out/flash/sdpa_harness.py`)

```python
"""Compare the flash_sdpa emulator output (DRAM 0x7000) against candidate references.
Usage: python sdpa_harness.py <run_dir>   (expects <run_dir>/sdpa.in and sdpa_mem.out)
NOTE: after the fix the V tile in sdpa.in is V^T; use V = mat(win, V_A).T for the reference."""
import sys, struct
import numpy as np

N = D = 32
Q_A, K_A, V_A, O_A, S_A = 0x1000, 0x3000, 0x5000, 0x7000, 0x0C00

def load_words(path):
    words = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        a, d = [x.strip() for x in line.split(":")]
        d = d.replace(" ", "")
        if len(d) == 8:                       # 32-bit data words only (skip 160-bit instr lines)
            words[int(a, 16)] = int(d, 16)
    return words

def rd_bf16(words, ba):                       # same packing as run_graph.py::_read_bf16
    wa = ba & ~3
    w = words.get(wa)
    if w is not None:
        h = (w >> 16) & 0xFFFF if (ba & 2) else w & 0xFFFF
    else:
        h = words.get(ba, 0) & 0xFFFF
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]

def mat(words, base):
    return np.array([rd_bf16(words, base + 2 * i) for i in range(N * D)], dtype=np.float32).reshape(N, D)

def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def rel_l2(ref, x):
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))

def softmax(s, axis=-1):
    m = s.max(axis=axis, keepdims=True); e = np.exp(s - m)
    return e / e.sum(axis=axis, keepdims=True)

run = sys.argv[1]
win = load_words(f"{run}/sdpa.in"); wout = load_words(f"{run}/sdpa_mem.out")
Q, K, V = mat(win, Q_A), mat(win, K_A), mat(win, V_A)
scale = struct.unpack("<f", struct.pack("<I", win.get(S_A, 0)))[0]
O = mat(wout, O_A)
print(f"scale={scale:.6f}  |O|={np.linalg.norm(O):.4f}")
ref = softmax(Q @ K.T * scale) @ V
cands = {
    "ref softmax(QK^T*s)V": ref,
    "P @ V^T": softmax(Q @ K.T * scale) @ V.T,
    "mean(V) bcast (K weights=0)": np.repeat(V.mean(0, keepdims=True), N, 0),
    "unscaled softmax(QK^T)V": softmax(Q @ K.T) @ V,
    "K not transposed": softmax(Q @ K * scale) @ V,
    "K rows reversed": softmax(Q @ K[::-1].T * scale) @ V,
    "V rows reversed": softmax(Q @ K.T * scale) @ V[::-1],
    "column softmax": softmax(Q @ K.T * scale, axis=0) @ V,
    "raw QK^T*s (no softmax/V)": Q @ K.T * scale,
    "ref transposed": ref.T,
}
for name, c in cands.items():
    print(f"{name:32s} cos={cos(O, c):10.6f} rel_l2={rel_l2(c, O):10.6f} max_abs={np.abs(O - c).max():10.6f}")
print("max row-to-row spread of O:", float(np.abs(O - O[0]).max()))
```
