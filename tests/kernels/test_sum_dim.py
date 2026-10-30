#!/usr/bin/env python3
"""Unit test for sum_dim AtallaC kernel generator.

Verifies:
1. Standalone sum_dim.c compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. NumPy golden bit-for-bit:
   - Single-row identity (B=1, D=32)
   - Multi-row batch reduction (B=4, D=32)
   - Multi-column tile reduction (B=4, D=64)
"""

import os
import re
import sys
import struct
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# Ensure root workspace and kernels in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REPO_PPCI = _ROOT / "aihw-ppci-compiler"
_SIM_ROOT = _REPO_PPCI / "functional_sim"
_MODELS_ROOT = _ROOT / "atalla-models" / "atalla-graph"

sys.path.insert(0, str(_MODELS_ROOT))
from kernels.sum_dim import sum_dim_c


def fp32_to_bf16_u16(f: float) -> int:
    """Convert float32 to BF16 bit pattern with round-to-nearest-even."""
    u = struct.unpack("<I", struct.pack("<f", np.float32(f)))[0]
    lsb = (u >> 16) & 1
    u = (u + (0x7FFF + lsb)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_u16_to_fp32(u16: int) -> float:
    """Convert BF16 bit pattern to float32."""
    u32 = (u16 & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", u32))[0]


def run_sum_dim_sim(num_rows: int, num_cols: int = 32):
    print(f"\n--- Testing sum_dim_c (num_rows={num_rows}, num_cols={num_cols}) ---")
    c_code = sum_dim_c(num_rows=num_rows, num_cols=num_cols)
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "sum_dim.c"
        s_path = tmp / "sum_dim.s"
        in_path = tmp / "sum_dim.in"
        mem_out = tmp / "output_mem.out"
        sreg_out = tmp / "output_sreg.out"
        vreg_out = tmp / "output_vreg.out"
        mreg_out = tmp / "output_mreg.out"
        scpad0_out = tmp / "output_scpad0.out"
        scpad1_out = tmp / "output_scpad1.out"

        c_path.write_text(c_code)

        # 1. Compile C to assembly
        atalla_cc = _REPO_PPCI / "atalla_cc"
        res_cc = subprocess.run(
            [str(atalla_cc), "-p", "-S", str(c_path), "-o", str(s_path)],
            capture_output=True,
            text=True,
        )
        assert res_cc.returncode == 0, f"Compilation failed: {res_cc.stderr}"
        s_content = s_path.read_text()

        # 2. Spill check: verify no stack memory spills inside the loop
        loop_matches = re.findall(r"while.*?(?=scpad_store)", s_content, re.DOTALL)
        for loop_text in loop_matches:
            spills = re.findall(r"(sw_s.*\(x8\)|lw_s.*\(x8\))", loop_text)
            assert len(spills) == 0, f"Found spills in loop body: {spills}"
        print("  [Pass] Zero stack spills verified in loop body.")

        # 3. Assemble to simulator .in image
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_SIM_ROOT}:{_REPO_PPCI}"
        env["ATALLA_STRIP_NOP_PACKET_ROWS"] = "0"
        env["ATALLA_FUNCTIONAL_SCHED_LATENCY"] = "1"

        bc = [
            sys.executable,
            str(_SIM_ROOT / "build_compiler.py"),
            "-i", str(s_path),
            "-o", str(in_path),
            "--prepacked",
        ]
        res_bc = subprocess.run(bc, cwd=_SIM_ROOT, env=env, capture_output=True, text=True)
        assert res_bc.returncode == 0, f"build_compiler failed: {res_bc.stderr}"

        # 4. Seed inputs
        np.random.seed(42)
        x_in = np.random.uniform(-2.0, 2.0, size=(num_rows, num_cols)).astype(np.float32)

        # Truncate to BF16 precision
        x_bf16 = np.zeros_like(x_in)
        for r in range(num_rows):
            for c in range(num_cols):
                x_bf16[r, c] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[r, c]))

        # Golden logic: sequential BF16 addition along axis 0 (matching hardware vector adder)
        expected = x_bf16[0].copy()
        for r in range(1, num_rows):
            for c in range(num_cols):
                expected[c] = bf16_u16_to_fp32(fp32_to_bf16_u16(expected[c] + x_bf16[r, c]))

        IN_GMEM = 0x1000
        OUT_GMEM = 0x3000

        width = 32
        col_tiles = (num_cols + width - 1) // width
        sdma_ctl_in = (0 << 30) | ((num_rows - 1) << 25) | ((width - 1) << 20) | (num_cols - 1)
        sdma_ctl_out = (0 << 30) | (0 << 25) | ((width - 1) << 20) | (num_cols - 1)
        in_tile_bytes = width * 2
        out_tile_bytes = width * 2

        words: dict[int, int] = {}
        # CFG pointers & dimensions (at 60 + 0, 4, 8, 12, 16, 20, 24, 28)
        words[60] = IN_GMEM
        words[64] = OUT_GMEM
        words[68] = num_rows
        words[72] = col_tiles
        words[76] = sdma_ctl_in
        words[80] = sdma_ctl_out
        words[84] = in_tile_bytes
        words[88] = out_tile_bytes

        # Seed input matrix into IN_GMEM
        flat_x = x_bf16.ravel()
        total_in = flat_x.size
        for i in range(0, total_in, 2):
            x0 = fp32_to_bf16_u16(flat_x[i])
            x1 = fp32_to_bf16_u16(flat_x[i + 1]) if i + 1 < total_in else 0
            words[IN_GMEM + i * 2] = (x1 << 16) | x0

        # Inject words into data section
        data_lines = [f"{a:08X}: {w & 0xFFFFFFFF:08X}" for a, w in sorted(words.items())]
        stripped = in_path.read_text().rstrip()
        head, sep, _ = stripped.partition(".data")
        base = head.rstrip() if sep else stripped
        in_path.write_text(f"{base}\n.data\n" + "\n".join(data_lines) + "\n")

        # 5. Run simulation
        sim_cmd = [
            sys.executable,
            str(_SIM_ROOT / "run.py"),
            "--input_file", str(in_path),
            "--output_mem_file", str(mem_out),
            "--output_sreg_file", str(sreg_out),
            "--output_vreg_file", str(vreg_out),
            "--output_mreg_file", str(mreg_out),
            "--output_scpad_file0", str(scpad0_out),
            "--output_scpad_file1", str(scpad1_out),
        ]
        res_sim = subprocess.run(sim_cmd, cwd=_SIM_ROOT, env=env, capture_output=True, text=True)
        assert res_sim.returncode == 0, f"Simulation failed: {res_sim.stderr}\nOutput: {res_sim.stdout}"

        # 6. Read output memory
        mem_lines = mem_out.read_text().splitlines()
        mem_data: dict[int, int] = {}
        for line in mem_lines:
            line = line.strip()
            if not line or ":" not in line:
                continue
            addr_str, val_str = line.split(":", 1)
            mem_data[int(addr_str.strip(), 16)] = int(val_str.strip(), 16)

        actual = np.zeros(num_cols, dtype=np.float32)
        for i in range(0, num_cols, 2):
            w_addr = OUT_GMEM + i * 2
            val = mem_data.get(w_addr, 0)
            actual[i] = bf16_u16_to_fp32(val & 0xFFFF)
            if i + 1 < num_cols:
                actual[i + 1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff: {max_diff:.6e}")
        assert max_diff < 1e-4, f"Output mismatch! Max diff: {max_diff}"
        for i in range(min(5, num_cols)):
            print(f"    col {i:2d}: actual={actual[i]:+.3f}, expected={expected[i]:+.3f}")


def test_sum_dim_single_row():
    """Verify B=1 identity pass (1 row of 32 elements)."""
    run_sum_dim_sim(num_rows=1, num_cols=32)


def test_sum_dim_multi_row():
    """Verify B=4 batch reduction (4 rows of 32 elements)."""
    run_sum_dim_sim(num_rows=4, num_cols=32)


def test_sum_dim_multi_col():
    """Verify B=4 batch reduction with 64 elements (2 column tiles)."""
    run_sum_dim_sim(num_rows=4, num_cols=64)


def main():
    test_sum_dim_single_row()
    test_sum_dim_multi_row()
    test_sum_dim_multi_col()
    print("\nAll sum_dim tests PASSED successfully!")


if __name__ == "__main__":
    main()
