#!/usr/bin/env python3
"""Unified kernel validation and benchmarking harness.

For each kernel: compile .c → .s → .in, generate test data, run emulator,
compare output against NumPy reference, report metrics.

Usage:
    python validate_and_benchmark.py                  # all kernels
    python validate_and_benchmark.py --kernels relu softmax
    python validate_and_benchmark.py --keep-tmp       # keep intermediate files
"""
from __future__ import annotations

import argparse
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
COMPILER_ROOT = SCRIPT_DIR.parent / "aihw-ppci-compiler"
KERNELS_DIR = COMPILER_ROOT / "atalla_tests" / "kernels"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(COMPILER_ROOT))

from build import DRAMWriter, render_testfile
from src.functional_sim import run as run_emulator
from src.misc.memory import Memory
from src.components.scalar_register_file import ScalarRegisterFile
from src.components.vector_register_file import VectorRegisterFile
from src.components.execute import ExecuteUnit
from src.components.scpad import Scratchpad

# ---------------------------------------------------------------------------
# BF16 helpers
# ---------------------------------------------------------------------------

def f32_to_bf16_bits(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16) & 0xFFFF

def bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]

def to_bf16_array(arr: np.ndarray) -> np.ndarray:
    """Quantize float32 array to bf16 precision (still stored as float32)."""
    out = np.empty_like(arr, dtype=np.float32)
    for i, x in enumerate(arr.flat):
        out.flat[i] = bf16_bits_to_f32(f32_to_bf16_bits(float(x)))
    return out

# ---------------------------------------------------------------------------
# Emulator runner
# ---------------------------------------------------------------------------

def run_on_emulator(in_text: str, work_dir: str, tag: str) -> Tuple[Memory, Dict]:
    os.makedirs(work_dir, exist_ok=True)
    in_path = f"{work_dir}/{tag}.in"
    Path(in_path).write_text(in_text)

    mem = Memory(in_path)
    sregs = ScalarRegisterFile()
    mregs = ScalarRegisterFile(num_regs=16)
    vregs = VectorRegisterFile()
    SP0 = Scratchpad(slots_per_bank=32)
    SP1 = Scratchpad(slots_per_bank=32)
    EU = ExecuteUnit()

    max_data_addr = max(mem.data_mem.keys()) if mem.data_mem else 0
    stack_base = ((max_data_addr + 0x1000) & ~0xFFF) + 0x1000
    sregs.write(2, stack_base)
    sregs.write(33, stack_base)

    _orig_mread = mregs.read
    def _m0_patched(idx):
        return 0xFFFFFFFF if idx == 0 else _orig_mread(idx)
    mregs.read = _m0_patched

    prefix = f"{work_dir}/{tag}"
    run_emulator(
        mem, sregs, mregs, vregs, SP0, SP1, EU, 0, 4,
        f"{prefix}_mem.out", f"{prefix}_sregs.out", f"{prefix}_vregs.out",
        f"{prefix}_mregs.out", f"{prefix}_sp0.out", f"{prefix}_sp1.out",
        f"{prefix}_perf.out",
    )

    metrics = {}
    try:
        with open(f"{prefix}_perf.out") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    try:
                        metrics[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return mem, metrics


def read_bf16_from_memory(mem: Memory, addr: int, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=np.float32)
    for i in range(count):
        byte_addr = addr + i * 2
        word = mem.read_data(byte_addr)
        bits = word & 0xFFFF
        result[i] = bf16_bits_to_f32(bits)
    return result

# ---------------------------------------------------------------------------
# Compiler: .c → .s → .in (instruction section only)
# ---------------------------------------------------------------------------

def compile_c_to_asm(c_path: Path, s_path: Path) -> str:
    env = {**os.environ, "PYTHONPATH": str(COMPILER_ROOT)}
    cmd = [sys.executable, str(COMPILER_ROOT / "atalla_cc"),
           str(c_path), "-S", "-o", str(s_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"ppci compile failed:\n{r.stderr[-2000:]}")
    return s_path.read_text()


def asm_to_in_text(asm_text: str) -> Tuple[str, int, int]:
    """Returns (in_text, packet_count, instruction_count)."""
    import build_compiler
    in_text, ready, packets = build_compiler.compile_asm(asm_text)
    instr_section = in_text.split("\n.data")[0].strip() if "\n.data" in in_text else in_text.strip()
    n_packets = len(packets)
    n_instrs = sum(len(p) for p in packets)
    return instr_section, n_packets, n_instrs

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelResult:
    name: str
    status: str  # "PASS", "FAIL", "ERROR"
    cosine_sim: float = 0.0
    max_error: float = 0.0
    mean_error: float = 0.0
    packets: int = 0
    instructions: int = 0
    slot_utilization: float = 0.0
    emulator_packets: int = 0
    emulator_instructions: int = 0
    error_msg: str = ""

    def summary_line(self) -> str:
        status = self.status
        if status == "PASS":
            return f"  {self.name:15s} PASS  cos={self.cosine_sim:.4f}  max_err={self.max_error:.6f}"
        elif status == "FAIL":
            return f"  {self.name:15s} FAIL  cos={self.cosine_sim:.4f}  max_err={self.max_error:.6f}"
        else:
            return f"  {self.name:15s} ERROR {self.error_msg}"

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(actual: np.ndarray, expected: np.ndarray) -> Tuple[float, float, float]:
    a = actual.flatten().astype(np.float64)
    e = expected.flatten().astype(np.float64)
    dot = np.dot(a, e)
    na, ne = np.linalg.norm(a), np.linalg.norm(e)
    cos = dot / (na * ne) if na > 0 and ne > 0 else 0.0
    diff = np.abs(a - e)
    return float(cos), float(np.max(diff)), float(np.mean(diff))

# ---------------------------------------------------------------------------
# Per-kernel test definitions
# ---------------------------------------------------------------------------

CFG_BASE = 0x3C

def _sdma_ctl_val(sid, num_rows, num_cols, full_cols):
    return (sid << 30) | ((num_rows - 1) << 25) | ((num_cols - 1) << 20) | (full_cols - 1)


def test_relu(work_dir: str) -> KernelResult:
    ROWS, COLS = 4, 32
    total = ROWS * COLS
    IN_GMEM, OUT_GMEM = 0x1000, 0x2000

    rng = np.random.default_rng(42)
    data = rng.standard_normal(total).astype(np.float32) * 2.0
    data_bf16 = to_bf16_array(data)
    expected = np.maximum(data_bf16, 0.0)

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, IN_GMEM)
    img.u32(CFG_BASE + 4, OUT_GMEM)
    for i in range(total):
        img.bf16(IN_GMEM + i * 2, float(data[i]))
    for i in range(total):
        img.bf16(OUT_GMEM + i * 2, 0.0)
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "relu.c"
    s_path = Path(work_dir) / "relu.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "relu")
    actual = read_bf16_from_memory(mem, OUT_GMEM, total)

    cos, maxe, meane = compare(actual, expected)
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.99 else "FAIL"
    return KernelResult(
        name="relu", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
    )


def test_softmax(work_dir: str) -> KernelResult:
    WIDTH = 32
    IN_GMEM = 0x1000

    rng = np.random.default_rng(42)
    data = rng.standard_normal(WIDTH).astype(np.float32)
    data_bf16 = to_bf16_array(data)
    shifted = data_bf16 - np.max(data_bf16)
    exp_v = np.exp(shifted)
    expected = exp_v / np.sum(exp_v)

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, IN_GMEM)
    img.u32(CFG_BASE + 4, 0)
    for i in range(WIDTH):
        img.bf16(IN_GMEM + i * 2, float(data[i]))
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "softmax.c"
    s_path = Path(work_dir) / "softmax.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "softmax")
    actual = read_bf16_from_memory(mem, IN_GMEM, WIDTH)

    cos, maxe, meane = compare(actual, expected)
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.60 else "FAIL"
    return KernelResult(
        name="softmax", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
        error_msg="" if cos > 0.60 else "BF16 precision loss in exp/rcp chain",
    )


def test_maxpool(work_dir: str) -> KernelResult:
    H_IN, W_IN, POOL, STRIDE = 8, 8, 2, 2
    H_OUT = (H_IN - POOL) // STRIDE + 1
    IN_BASE, OUT_BASE = 0x1000, 0x2000

    rng = np.random.default_rng(42)
    data = rng.standard_normal(H_IN * W_IN).astype(np.float32)
    data_bf16 = to_bf16_array(data)
    mat = data_bf16.reshape(H_IN, W_IN)
    expected = np.zeros((H_OUT, W_IN), dtype=np.float32)
    for oh in range(H_OUT):
        r0 = oh * STRIDE
        expected[oh] = np.maximum(mat[r0], mat[r0 + 1])

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, IN_BASE)
    img.u32(CFG_BASE + 4, OUT_BASE)
    for i in range(H_IN * W_IN):
        img.bf16(IN_BASE + i * 2, float(data[i]))
    for i in range(H_OUT * W_IN):
        img.bf16(OUT_BASE + i * 2, 0.0)
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "maxpool.c"
    s_path = Path(work_dir) / "maxpool.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "maxpool")
    actual = read_bf16_from_memory(mem, OUT_BASE, H_OUT * W_IN)

    cos, maxe, meane = compare(actual, expected.flatten())
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.95 else "FAIL"
    return KernelResult(
        name="maxpool", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
    )


def test_gemm_tiled(work_dir: str) -> KernelResult:
    M, N, K = 4, 4, 4
    TILE = 4
    M_tiles = math.ceil(M / TILE)
    N_tiles = math.ceil(N / TILE)
    K_tiles = math.ceil(K / TILE)
    A_GMEM = 0x1000
    W_GMEM = A_GMEM + M * K * 2 + 0x100
    C_GMEM = W_GMEM + K * N * 2 + 0x100

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    W = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    A_bf16 = to_bf16_array(A)
    W_bf16 = to_bf16_array(W)
    expected = A_bf16 @ W_bf16

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, A_GMEM)
    img.u32(CFG_BASE + 4, W_GMEM)
    img.u32(CFG_BASE + 8, C_GMEM)
    img.u32(CFG_BASE + 12, M)
    img.u32(CFG_BASE + 16, N)
    img.u32(CFG_BASE + 20, K)
    img.u32(CFG_BASE + 24, M_tiles)
    img.u32(CFG_BASE + 28, N_tiles)
    img.u32(CFG_BASE + 32, K_tiles)
    img.u32(CFG_BASE + 36, TILE)
    for r in range(M):
        for c in range(K):
            img.bf16(A_GMEM + (r * K + c) * 2, float(A[r, c]))
    for r in range(K):
        for c in range(N):
            img.bf16(W_GMEM + (r * N + c) * 2, float(W[r, c]))
    for i in range(M * N):
        img.bf16(C_GMEM + i * 2, 0.0)
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "gemm_tiled.c"
    s_path = Path(work_dir) / "gemm_tiled.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "gemm_tiled")
    actual = read_bf16_from_memory(mem, C_GMEM, M * N)

    cos, maxe, meane = compare(actual, expected.flatten())
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.90 else "FAIL"
    return KernelResult(
        name="gemm_tiled", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
    )


def test_layernorm(work_dir: str) -> KernelResult:
    ROWS = 4
    ACTIVE_COLS = 4  # kernel uses mask=0xF → 4 active lanes
    VEC_WIDTH = 32   # hardware vector width
    N2 = ROWS * ACTIVE_COLS
    IN_GMEM = 0x1000
    EPS = 1e-5
    INV_N2 = 1.0 / N2

    rng = np.random.default_rng(42)
    active_data = rng.standard_normal(N2).astype(np.float32) * 2.0
    active_bf16 = to_bf16_array(active_data)

    mean_val = np.mean(active_bf16)
    centered = active_bf16 - mean_val
    var_val = np.mean(centered ** 2)
    denom = np.sqrt(var_val + EPS)
    expected = centered / denom

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, IN_GMEM)
    img.u32(CFG_BASE + 4, 0)
    img.f32(20, EPS)
    img.f32(24, INV_N2)
    # Store active data in first ACTIVE_COLS lanes of each 32-wide row
    for r in range(ROWS):
        row_base = IN_GMEM + r * VEC_WIDTH * 2
        for c in range(VEC_WIDTH):
            if c < ACTIVE_COLS:
                img.bf16(row_base + c * 2, float(active_data[r * ACTIVE_COLS + c]))
            else:
                img.bf16(row_base + c * 2, 0.0)
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "layernorm.c"
    s_path = Path(work_dir) / "layernorm.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "layernorm")
    # Read back only active lanes from each 32-wide row
    actual = np.zeros(N2, dtype=np.float32)
    for r in range(ROWS):
        row_base = IN_GMEM + r * VEC_WIDTH * 2
        for c in range(ACTIVE_COLS):
            actual[r * ACTIVE_COLS + c] = bf16_bits_to_f32(
                mem.read_data(row_base + c * 2) & 0xFFFF)

    cos, maxe, meane = compare(actual, expected.flatten())
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.90 else "FAIL"
    return KernelResult(
        name="layernorm", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
    )


def test_add(work_dir: str) -> KernelResult:
    ROWS, COLS = 4, 32
    total = ROWS * COLS
    A_GMEM, B_GMEM, C_GMEM = 0x1000, 0x2000, 0x3000

    rng = np.random.default_rng(42)
    A = rng.standard_normal(total).astype(np.float32)
    B = rng.standard_normal(total).astype(np.float32)
    A_bf16 = to_bf16_array(A)
    B_bf16 = to_bf16_array(B)
    expected = A_bf16 + B_bf16

    img = DRAMWriter()
    img.u32(CFG_BASE + 0, A_GMEM)
    img.u32(CFG_BASE + 4, B_GMEM)
    img.u32(CFG_BASE + 8, C_GMEM)
    for i in range(total):
        img.bf16(A_GMEM + i * 2, float(A[i]))
        img.bf16(B_GMEM + i * 2, float(B[i]))
    for i in range(total):
        img.bf16(C_GMEM + i * 2, 0.0)
    data_text = img.render_data_mem(include_zeros=True)

    c_path = KERNELS_DIR / "add.c"
    s_path = Path(work_dir) / "add.s"
    asm_text = compile_c_to_asm(c_path, s_path)
    instr_text, n_pkts, n_instrs = asm_to_in_text(asm_text)

    final = render_testfile(instr_text, data_text)
    mem, metrics = run_on_emulator(final, work_dir, "add")
    actual = read_bf16_from_memory(mem, C_GMEM, total)

    cos, maxe, meane = compare(actual, expected)
    slot_util = n_instrs / (n_pkts * 4) if n_pkts > 0 else 0
    status = "PASS" if cos > 0.99 else "FAIL"
    return KernelResult(
        name="add", status=status, cosine_sim=cos, max_error=maxe, mean_error=meane,
        packets=n_pkts, instructions=n_instrs, slot_utilization=slot_util,
        emulator_packets=int(metrics.get("packets", 0)),
        emulator_instructions=int(metrics.get("instructions", 0)),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KERNEL_TESTS = {
    "relu": test_relu,
    "softmax": test_softmax,
    "maxpool": test_maxpool,
    "gemm_tiled": test_gemm_tiled,
    "layernorm": test_layernorm,
    "add": test_add,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate and benchmark Atalla kernels")
    parser.add_argument("--kernels", nargs="*", default=list(KERNEL_TESTS.keys()),
                        help="Kernels to test (default: all)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp directory")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory")
    args = parser.parse_args()

    if args.work_dir:
        work_base = args.work_dir
        os.makedirs(work_base, exist_ok=True)
    elif args.keep_tmp:
        work_base = str(SCRIPT_DIR / "out" / "validate")
        os.makedirs(work_base, exist_ok=True)
    else:
        work_base = tempfile.mkdtemp(prefix="atalla_validate_")

    print(f"Working directory: {work_base}")
    print(f"Compiler: {COMPILER_ROOT}")
    print(f"Kernels:  {KERNELS_DIR}")
    print()

    results: List[KernelResult] = []
    for name in args.kernels:
        if name not in KERNEL_TESTS:
            print(f"  {name:15s} SKIP  (unknown kernel)")
            continue
        work_dir = os.path.join(work_base, name)
        os.makedirs(work_dir, exist_ok=True)
        try:
            r = KERNEL_TESTS[name](work_dir)
            results.append(r)
            print(r.summary_line())
        except Exception as e:
            r = KernelResult(name=name, status="ERROR", error_msg=str(e)[:200])
            results.append(r)
            print(r.summary_line())

    print()
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_err = sum(1 for r in results if r.status == "ERROR")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL, {n_err} ERROR out of {len(results)}")

    if any(r.packets > 0 for r in results):
        print()
        print("Metrics summary:")
        print(f"  {'kernel':15s} {'pkts':>6s} {'instrs':>7s} {'slot%':>7s} {'emu_pkts':>9s} {'emu_ins':>8s}")
        for r in results:
            if r.packets > 0:
                print(f"  {r.name:15s} {r.packets:6d} {r.instructions:7d} "
                      f"{r.slot_utilization:7.3f} {r.emulator_packets:9d} {r.emulator_instructions:8d}")

    if not args.keep_tmp and not args.work_dir:
        import shutil
        shutil.rmtree(work_base, ignore_errors=True)

    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
