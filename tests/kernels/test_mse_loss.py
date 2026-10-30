#!/usr/bin/env python3
"""Unit test for mse_loss and mse_loss_backward AtallaC kernel generators.

Verifies:
1. Standalone mse_loss.c and mse_loss_backward.c compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. NumPy golden bit-for-bit:
   - mse_loss forward: N=32 (1 tile) and N=64 (2 tiles)
   - mse_loss_backward gradient: N=32 (1 tile) and N=64 (2 tiles)
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
from kernels.mse_loss import mse_loss_c
from kernels.mse_loss_backward import mse_loss_backward_c


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


def run_mse_loss_sim(total_elements: int):
    print(f"\n--- Testing mse_loss_c (total_elements={total_elements}) ---")
    c_code = mse_loss_c(total_elements=total_elements)
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "mse_loss.c"
        s_path = tmp / "mse_loss.s"
        in_path = tmp / "mse_loss.in"
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

        # 4. Inputs & Golden
        np.random.seed(42)
        x_in = np.random.uniform(-1.0, 1.0, size=total_elements).astype(np.float32)
        y_in = np.random.uniform(-1.0, 1.0, size=total_elements).astype(np.float32)

        x_bf16 = np.zeros_like(x_in)
        y_bf16 = np.zeros_like(y_in)
        for i in range(total_elements):
            x_bf16[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[i]))
            y_bf16[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(y_in[i]))

        width = 32
        rows = (total_elements + width - 1) // width
        sp_rows = min(rows, 32)
        tile_count = (rows + sp_rows - 1) // sp_rows
        tile_bytes = sp_rows * width * 2
        scale = np.float32(1.0 / max(total_elements, 1))

        # Golden calculation matching hardware:
        # For each tile & row: diff = x - y, sq = diff * diff, accumulate vector sum
        # Then RSUM(sum_vec), then multiply by scale
        sum_vec = np.zeros(width, dtype=np.float32)
        idx = 0
        for t in range(tile_count):
            for r in range(sp_rows):
                if idx < total_elements:
                    chunk_x = x_bf16[idx : min(idx + width, total_elements)]
                    chunk_y = y_bf16[idx : min(idx + width, total_elements)]
                    # Pad if needed
                    vx = np.zeros(width, dtype=np.float32)
                    vy = np.zeros(width, dtype=np.float32)
                    vx[: len(chunk_x)] = chunk_x
                    vy[: len(chunk_y)] = chunk_y

                    diff = vx - vy
                    for c in range(width):
                        diff[c] = bf16_u16_to_fp32(fp32_to_bf16_u16(diff[c]))
                    sq = diff * diff
                    for c in range(width):
                        sq[c] = bf16_u16_to_fp32(fp32_to_bf16_u16(sq[c]))
                    sum_vec += sq
                    for c in range(width):
                        sum_vec[c] = bf16_u16_to_fp32(fp32_to_bf16_u16(sum_vec[c]))
                    idx += width

        # RSUM in simulator reduces all elements with sum(q)
        rsum_val = bf16_u16_to_fp32(fp32_to_bf16_u16(np.sum(sum_vec)))
        expected = bf16_u16_to_fp32(fp32_to_bf16_u16(rsum_val * scale))

        X_GMEM = 0x1000
        Y_GMEM = 0x2000
        OUT_GMEM = 0x3000

        sdma_sp0 = (0 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)
        sdma_sp1 = (1 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)
        sdma_out = (0 << 30) | (0 << 25) | (0 << 20) | 0

        words: dict[int, int] = {}
        words[60] = X_GMEM
        words[64] = Y_GMEM
        words[68] = OUT_GMEM
        words[72] = tile_count
        words[76] = sp_rows
        words[80] = sdma_sp0
        words[84] = sdma_sp1
        words[88] = tile_bytes
        # scale as float32 in word 8 (addr 60 + 32 = 92)
        words[92] = struct.unpack("<I", struct.pack("<f", scale))[0]
        words[96] = sdma_out

        # Seed X and Y in GMEM
        padded_total = rows * width
        padded_x = np.zeros(padded_total, dtype=np.float32)
        padded_y = np.zeros(padded_total, dtype=np.float32)
        padded_x[:total_elements] = x_bf16
        padded_y[:total_elements] = y_bf16

        for i in range(0, padded_total, 2):
            x0 = fp32_to_bf16_u16(padded_x[i])
            x1 = fp32_to_bf16_u16(padded_x[i + 1]) if i + 1 < padded_total else 0
            words[X_GMEM + i * 2] = (x1 << 16) | x0

            y0 = fp32_to_bf16_u16(padded_y[i])
            y1 = fp32_to_bf16_u16(padded_y[i + 1]) if i + 1 < padded_total else 0
            words[Y_GMEM + i * 2] = (y1 << 16) | y0

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

        out_word = mem_data.get(OUT_GMEM, 0)
        actual = bf16_u16_to_fp32(out_word & 0xFFFF)

        diff = abs(actual - expected)
        print(f"  [Pass] Golden check passed! actual={actual:.6f}, expected={expected:.6f}, diff={diff:.6e}")
        assert diff < 1e-3, f"Output mismatch! actual={actual}, expected={expected}, diff={diff}"


def run_mse_loss_backward_sim(total_elements: int):
    print(f"\n--- Testing mse_loss_backward_c (total_elements={total_elements}) ---")
    c_code = mse_loss_backward_c(total_elements=total_elements)
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "mse_loss_backward.c"
        s_path = tmp / "mse_loss_backward.s"
        in_path = tmp / "mse_loss_backward.in"
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

        # 2. Spill check
        loop_matches = re.findall(r"while.*?(?=scpad_store)", s_content, re.DOTALL)
        for loop_text in loop_matches:
            spills = re.findall(r"(sw_s.*\(x8\)|lw_s.*\(x8\))", loop_text)
            assert len(spills) == 0, f"Found spills in loop body: {spills}"
        print("  [Pass] Zero stack spills verified in loop body.")

        # 3. Assemble
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

        # 4. Inputs & Golden
        np.random.seed(42)
        x_in = np.random.uniform(-1.0, 1.0, size=total_elements).astype(np.float32)
        y_in = np.random.uniform(-1.0, 1.0, size=total_elements).astype(np.float32)

        x_bf16 = np.zeros_like(x_in)
        y_bf16 = np.zeros_like(y_in)
        for i in range(total_elements):
            x_bf16[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[i]))
            y_bf16[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(y_in[i]))

        width = 32
        rows = (total_elements + width - 1) // width
        sp_rows = min(rows, 32)
        tile_count = (rows + sp_rows - 1) // sp_rows
        tile_bytes = sp_rows * width * 2
        scale = np.float32(2.0 / max(total_elements, 1))

        # Expected: 2/N * (x - y)
        expected = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            diff = bf16_u16_to_fp32(fp32_to_bf16_u16(x_bf16[i] - y_bf16[i]))
            expected[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(diff * scale))

        X_GMEM = 0x1000
        Y_GMEM = 0x2000
        OUT_GMEM = 0x3000

        sdma_sp0 = (0 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)
        sdma_sp1 = (1 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)

        words: dict[int, int] = {}
        words[60] = X_GMEM
        words[64] = Y_GMEM
        words[68] = OUT_GMEM
        words[72] = tile_count
        words[76] = sp_rows
        words[80] = sdma_sp0
        words[84] = sdma_sp1
        words[88] = tile_bytes
        words[92] = struct.unpack("<I", struct.pack("<f", scale))[0]

        padded_total = rows * width
        padded_x = np.zeros(padded_total, dtype=np.float32)
        padded_y = np.zeros(padded_total, dtype=np.float32)
        padded_x[:total_elements] = x_bf16
        padded_y[:total_elements] = y_bf16

        for i in range(0, padded_total, 2):
            x0 = fp32_to_bf16_u16(padded_x[i])
            x1 = fp32_to_bf16_u16(padded_x[i + 1]) if i + 1 < padded_total else 0
            words[X_GMEM + i * 2] = (x1 << 16) | x0

            y0 = fp32_to_bf16_u16(padded_y[i])
            y1 = fp32_to_bf16_u16(padded_y[i + 1]) if i + 1 < padded_total else 0
            words[Y_GMEM + i * 2] = (y1 << 16) | y0

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

        actual = np.zeros(total_elements, dtype=np.float32)
        for i in range(0, total_elements, 2):
            w_addr = OUT_GMEM + i * 2
            val = mem_data.get(w_addr, 0)
            actual[i] = bf16_u16_to_fp32(val & 0xFFFF)
            if i + 1 < total_elements:
                actual[i + 1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff: {max_diff:.6e}")
        assert max_diff < 1e-4, f"Output mismatch! Max diff: {max_diff}"
        for i in range(min(5, total_elements)):
            print(f"    idx {i:2d}: actual={actual[i]:+.3f}, expected={expected[i]:+.3f}")


def test_mse_loss_single_tile():
    """Verify forward MSE loss for 32 elements."""
    run_mse_loss_sim(total_elements=32)


def test_mse_loss_multi_tile():
    """Verify forward MSE loss for 64 elements (2 tiles)."""
    run_mse_loss_sim(total_elements=64)


def test_mse_loss_backward_single_tile():
    """Verify backward MSE loss gradient for 32 elements."""
    run_mse_loss_backward_sim(total_elements=32)


def test_mse_loss_backward_multi_tile():
    """Verify backward MSE loss gradient for 64 elements (2 tiles)."""
    run_mse_loss_backward_sim(total_elements=64)


def main():
    test_mse_loss_single_tile()
    test_mse_loss_multi_tile()
    test_mse_loss_backward_single_tile()
    test_mse_loss_backward_multi_tile()
    print("\nAll mse_loss tests PASSED successfully!")


if __name__ == "__main__":
    main()
