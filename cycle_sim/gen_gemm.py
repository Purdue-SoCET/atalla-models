#!/usr/bin/env python3
"""Generate standalone GEMM programs for atalla-sim from atalla-graph/kernels/gemm.py.

For each (M, N, K, schedule) writes into <out>/<sched>_<M>x<N>x<K>/:
  gemm.c        the AtallaC kernel exactly as gemm_c(M, N, K, schedule) emits it
  params.npy    u32 ADDR_TABLE words (same as c_emitter._write_gemm_params)   @ 60
  a.npy         A padded to (M, k_stride), float32 holding BF16 values        @ A_GMEM
  wt.npy        W.T padded to (N, k_stride) (c_emitter._write_gemm_rhs_weight) @ W_GMEM
  z.npy         one zero row (TILE)                                           @ Z_GMEM
  ref.npy       NumPy reference A @ W (fp32 accumulate of BF16 inputs), (M, N)
  args.txt      the atalla-sim run arguments (one per line)
DRAM layout is the one tests/kernels/test_gemm_schedule.py::_sim_child uses.

Run with the atalla-models venv (only numpy needed):
  ~/.venvs/atalla/bin/python cycle_sim/gen_gemm.py --out cycle_runs/programs
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KROOT = ROOT / "atalla-graph" / "kernels"


def load_gemm():
    name = "_atalla_emit_kernels"
    spec = importlib.util.spec_from_file_location(
        name, KROOT / "__init__.py", submodule_search_locations=[str(KROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import importlib as il
    return il.import_module(f"{name}.gemm"), il.import_module(f"{name}.common")


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def align_data(nbytes):  # c_emitter._align_data
    return int(math.ceil(nbytes / 0x1000) * 0x1000) + 0x1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="32x32x128,128x32x32")
    ap.add_argument("--schedules", default="baseline,double_buffer,weight_stationary")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    gemm, common = load_gemm()
    TILE, ADDR_TABLE = common.TILE, common.ADDR_TABLE
    out = Path(a.out)
    for shape in a.shapes.split(","):
        M, N, K = (int(v) for v in shape.split("x"))
        rng = np.random.default_rng(a.seed)
        A = bf16_round(rng.standard_normal((M, K)).astype(np.float32))
        W = bf16_round(rng.standard_normal((K, N)).astype(np.float32) * 0.2)
        ks = int(math.ceil(max(K, 1) / TILE) * TILE)
        A_GMEM = 0x1000
        W_GMEM = A_GMEM + align_data(M * ks * 2)
        C_GMEM = W_GMEM + align_data(N * ks * 2)
        Z_GMEM = C_GMEM + M * N * 2
        params = np.array([A_GMEM, W_GMEM, C_GMEM, M, N, ks, math.ceil(M / TILE),
                           math.ceil(N / TILE), math.ceil(K / TILE), TILE, Z_GMEM],
                          dtype=np.uint32)
        a_pad = np.zeros((M, ks), np.float32); a_pad[:, :K] = A
        wt = np.zeros((N, ks), np.float32); wt[:, :K] = W.T
        ref = (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float32)
        for sched in a.schedules.split(","):
            d = out / f"{sched}_{M}x{N}x{K}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "gemm.c").write_text(gemm.gemm_c(M, N, K, schedule=sched))
            np.save(d / "params.npy", params)
            np.save(d / "a.npy", a_pad)
            np.save(d / "wt.npy", wt)
            np.save(d / "z.npy", np.zeros(TILE, np.float32))
            np.save(d / "A.npy", A)
            np.save(d / "W.npy", W)
            np.save(d / "ref.npy", ref)
            args = [
                f"--input=params.npy@{ADDR_TABLE}:u32",
                f"--input=a.npy@{A_GMEM:#x}:bf16",
                f"--input=wt.npy@{W_GMEM:#x}:bf16",
                f"--input=z.npy@{Z_GMEM:#x}:bf16",
                f"--output={C_GMEM:#x}:{M}x{N}:bf16=C_cycle.npy",
            ]
            (d / "args.txt").write_text("\n".join(args) + "\n")
            (d / "layout.json").write_text(json.dumps(dict(
                M=M, N=N, K=K, schedule=sched, k_stride=ks, ADDR_TABLE=ADDR_TABLE,
                A_GMEM=A_GMEM, W_GMEM=W_GMEM, C_GMEM=C_GMEM, Z_GMEM=Z_GMEM,
                seed=a.seed, traffic_model=gemm.gemm_traffic(M, N, K, sched)), indent=1))
            print(d)


if __name__ == "__main__":
    main()
