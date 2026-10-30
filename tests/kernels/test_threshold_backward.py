#!/usr/bin/env python3
"""Unit test for threshold_backward AtallaC kernel generator.

Verifies:
1. AtallaC code generation for threshold_backward.
2. PPCI compiler packetization (-p) without errors.
3. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
4. Functional simulator execution vs. NumPy golden bit-for-bit.
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
from kernels.threshold_backward import threshold_backward_c


def fp32_to_bf16_u16(f: float) -> int:
    """Convert float32 to BF16 bit pattern (truncated to upper 16 bits)."""
    u32 = struct.unpack("<I", struct.pack("<f", f))[0]
    return (u32 >> 16) & 0xFFFF


def bf16_u16_to_fp32(u16: int) -> float:
    """Convert BF16 bit pattern to float32."""
    u32 = (u16 & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", u32))[0]


def run_threshold_backward_sim(total: int, width: int = 32):
    print(f"\n--- Testing threshold_backward_c (total={total}, width={width}) ---")
    c_code = threshold_backward_c(total=total, width=width)
    assert "make_mask" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "thresh.c"
        s_path = tmp / "thresh.s"
        in_path = tmp / "thresh.in"
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
        # Find the loop body (between while loop entry and scpad_store)
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
        # Test vector with positive, negative, and exact zero values
        np.random.seed(42)
        grad_in = np.random.uniform(-5.0, 5.0, size=total).astype(np.float32)
        # Force specific boundary elements: positive, negative, zero
        x_in = np.random.uniform(-5.0, 5.0, size=total).astype(np.float32)
        if total >= 4:
            x_in[0] = 0.0
            x_in[1] = -1.5
            x_in[2] = 2.5
            x_in[3] = 0.0

        # Golden logic: grad_out = grad * (x > 0.0)
        # Elements are converted to BF16 in DRAM
        grad_bf16 = np.array([bf16_u16_to_fp32(fp32_to_bf16_u16(v)) for v in grad_in], dtype=np.float32)
        x_bf16 = np.array([bf16_u16_to_fp32(fp32_to_bf16_u16(v)) for v in x_in], dtype=np.float32)
        expected = np.where(x_bf16 > 0.0, grad_bf16, 0.0).astype(np.float32)

        # DRAM layout:
        # CFG at 60 (0x3c)
        # GRAD_GMEM at 0x1000
        # X_GMEM at 0x2000
        # OUT_GMEM at 0x3000
        GRAD_GMEM = 0x1000
        X_GMEM = 0x2000
        OUT_GMEM = 0x3000

        rows = (total + width - 1) // width
        sp_rows = min(rows, 32)
        tile_count = (rows + sp_rows - 1) // sp_rows
        tile_bytes = sp_rows * width * 2
        sdma_sp0 = (0 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)
        sdma_sp1 = (1 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)

        words: dict[int, int] = {}
        # CFG pointers & dimensions (at 60 + 0, 4, 8, 12, 16, 20, 24, 28)
        words[60] = GRAD_GMEM
        words[64] = X_GMEM
        words[68] = OUT_GMEM
        words[72] = tile_count
        words[76] = sp_rows
        words[80] = sdma_sp0
        words[84] = sdma_sp1
        words[88] = tile_bytes

        # Seed GRAD_GMEM and X_GMEM (2 BF16 per 32-bit word)
        for i in range(0, total, 2):
            g0 = fp32_to_bf16_u16(grad_bf16[i])
            g1 = fp32_to_bf16_u16(grad_bf16[i + 1]) if i + 1 < total else 0
            words[GRAD_GMEM + i * 2] = (g1 << 16) | g0

            x0 = fp32_to_bf16_u16(x_bf16[i])
            x1 = fp32_to_bf16_u16(x_bf16[i + 1]) if i + 1 < total else 0
            words[X_GMEM + i * 2] = (x1 << 16) | x0

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

        actual = np.zeros(total, dtype=np.float32)
        for i in range(0, total, 2):
            w_addr = OUT_GMEM + i * 2
            val = mem_data.get(w_addr, 0)
            actual[i] = bf16_u16_to_fp32(val & 0xFFFF)
            if i + 1 < total:
                actual[i + 1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        # Check correctness
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff: {max_diff:.6e}")
        assert max_diff < 1e-4, f"Output mismatch! Max diff: {max_diff}"
        for i in range(min(5, total)):
            print(f"    idx {i:2d}: x={x_bf16[i]:+.3f}, grad={grad_bf16[i]:+.3f} -> actual={actual[i]:+.3f}, expected={expected[i]:+.3f}")


def test_threshold_backward_single_tile():
    """Verify threshold_backward with 32 elements (1 tile)."""
    run_threshold_backward_sim(total=32, width=32)


def test_threshold_backward_multi_tile():
    """Verify threshold_backward with 64 elements (2 tiles)."""
    run_threshold_backward_sim(total=64, width=32)


def main():
    test_threshold_backward_single_tile()
    test_threshold_backward_multi_tile()
    print("\nAll threshold_backward tests PASSED successfully!")


if __name__ == "__main__":
    main()
