#!/usr/bin/env python3
"""Small conv2d golden checks (same DRAM layout as ``emit_conv`` / im2col + GEMM).

Run from ``atalla-graph`` root:
  python scripts/debug_conv_golden.py --case all
  python scripts/debug_conv_golden.py --case cin2 --seed 0

Compares emulator output to ``torch.nn.functional.conv2d`` in BF16.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_FUNC_SIM = _ROOT.parent / "functional_sim"
for p in (_ROOT, _FUNC_SIM):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build import DRAMWriter  # noqa: E402
from build_alexnet_layer import im2col  # noqa: E402
from codegen.c_emitter import (  # noqa: E402
    _align_data,
    _gemm_k_stride,
    _padded_gemm_a,
    _write_gemm_params,
    _write_gemm_rhs_weight,
    _write_matrix,
    _write_zeros,
    compile_and_assemble,
    render_in_file,
)
from kernels.common import TILE  # noqa: E402
from kernels.gemm import gemm_c as _gemm_c  # noqa: E402
from run_graph import _layer_compare_metrics, _read_bf16, _run_emulator  # noqa: E402


def _ref_conv_bf16(
    x_nchw: np.ndarray,
    w_oihw: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    pad: int,
) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x_nchw, np.float32)).to(torch.bfloat16)
    wt = torch.from_numpy(np.asarray(w_oihw, np.float32)).to(torch.bfloat16)
    b = (
        torch.from_numpy(np.asarray(bias, np.float32)).to(torch.bfloat16)
        if bias is not None
        else None
    )
    y = F.conv2d(xt, wt, bias=b, stride=stride, padding=pad)
    return y.float().numpy()


def _run_conv(
    x_nchw: np.ndarray,
    w_oihw: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    pad: int,
    work_dir: str,
    tag: str,
) -> np.ndarray:
    """x: (1,Cin,H,W), w: (Cout,Cin,R,S), bias: (Cout,) or None."""
    x_nchw = np.asarray(x_nchw, dtype=np.float32)
    w_oihw = np.asarray(w_oihw, dtype=np.float32)
    _, Cin, H, W = x_nchw.shape
    Cout, Cin2, R, S = w_oihw.shape
    if Cin != Cin2:
        raise ValueError(f"Cin mismatch {Cin} vs {Cin2}")
    Ho = (H + 2 * pad - R) // stride + 1
    Wo = (W + 2 * pad - S) // stride + 1
    M = Ho * Wo
    N = Cout
    K = R * S * Cin
    input_nhwc = x_nchw.transpose(0, 2, 3, 1)
    A_mat = im2col(input_nhwc, 1, H, W, Cin, R, S, stride, pad)
    ks = _gemm_k_stride(K)
    A_dram = _padded_gemm_a(A_mat, K)
    weight_flat = (
        w_oihw.reshape(N, Cin, R, S).transpose(2, 3, 1, 0).reshape(K, N)
    )

    A_GMEM = 0x1000
    W_GMEM = A_GMEM + _align_data(M * ks * 2)
    C_GMEM = W_GMEM + _align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = DRAMWriter()
    _write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    _write_matrix(img, A_GMEM, A_dram, M, ks)
    _write_gemm_rhs_weight(img, W_GMEM, weight_flat)
    if bias is not None and bias.size == N:
        c_init = np.tile(np.asarray(bias, dtype=np.float32).reshape(1, N), (M, 1))
        _write_matrix(img, C_GMEM, c_init, M, N)
    else:
        _write_zeros(img, C_GMEM, M * N)
    _write_zeros(img, Z_GMEM, TILE)

    from codegen.c_emitter import LayerEmission  # noqa: E402

    em = LayerEmission()
    em.c_source = _gemm_c(M, N, K)
    em.dram = img
    em.output_addr = C_GMEM
    em.output_elements = M * N
    em.conv_post = {"Ho": Ho, "Wo": Wo, "C": N, "final_shape": (1, Cout, Ho, Wo)}

    os.makedirs(work_dir, exist_ok=True)
    compile_and_assemble(em, work_dir, tag)
    in_path = Path(work_dir) / f"{tag}.in"
    in_path.write_text(render_in_file(em))
    mem, _eu = _run_emulator(str(in_path), work_dir, tag)
    raw = _read_bf16(mem, C_GMEM, M * N).reshape(Ho, Wo, N)
    out = raw.transpose(2, 0, 1).reshape(1, Cout, Ho, Wo)
    return out


def _report(case: str, x: np.ndarray, w: np.ndarray, b: np.ndarray | None, verbose: bool, **kw) -> None:
    work = str(_ROOT / "out" / "debug_conv")
    tag = case.replace(" ", "_")
    emu = _run_conv(x, w, b, work_dir=work, tag=tag, **kw)
    ref = _ref_conv_bf16(x, w, b, kw["stride"], kw["pad"])
    m = _layer_compare_metrics(ref, emu)
    print(f"\n=== {case} ===")
    print(
        f"  cos={m['cos_sim']:.6f}  rel_l2={m['rel_l2_error']:.6f}  "
        f"relmax={m['rel_max_abs_error']:.6f}  max_abs={m['max_abs_error']:.6f}"
    )
    if verbose:
        print(f"  out shape ref={ref.shape} emu={emu.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=("3x3_s1", "cin2", "cout4", "bias", "all"),
        default="all",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    def run(name: str, x: np.ndarray, w: np.ndarray, b: np.ndarray | None, **kw) -> None:
        _report(name, x, w, b, verbose=False, **kw)

    if args.case in ("3x3_s1", "all"):
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32) * 0.05
        w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.05
        run("3x3 s1p1 (1->2 ch)", x, w, None, stride=1, pad=1)
    if args.case in ("cin2", "all"):
        x = rng.standard_normal((1, 2, 6, 6)).astype(np.float32) * 0.05
        w = rng.standard_normal((3, 2, 3, 3)).astype(np.float32) * 0.05
        run("Cin=2 Cout=3 3x3 s1p1", x, w, None, stride=1, pad=1)
    if args.case in ("cout4", "all"):
        x = rng.standard_normal((1, 1, 7, 7)).astype(np.float32) * 0.05
        w = rng.standard_normal((4, 1, 3, 3)).astype(np.float32) * 0.05
        run("Cout=4 3x3 s1p1", x, w, None, stride=1, pad=1)
    if args.case in ("bias", "all"):
        x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32) * 0.05
        w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.05
        b = rng.standard_normal((2,)).astype(np.float32) * 0.01
        run("bias 2 filters", x, w, b, stride=1, pad=1)

    print("\nDone. Artifacts under out/debug_conv/")
    print("Interpret: cos~1 & low rel_l2 => conv path matches PyTorch BF16 conv2d.")


if __name__ == "__main__":
    main()
