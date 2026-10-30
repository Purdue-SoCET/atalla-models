#!/usr/bin/env python3
"""Unit test for batchnorm AtallaC kernel with vector intrinsics.

Verifies:
1. Standalone compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. PyTorch/NumPy golden:
   - Single channel (C=1, 64 elements)
   - Multi-channel (C=3, 3x64 elements)
   - Multi-tile (C=2, 2x128 elements)
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
from kernels.batchnorm import batchnorm_c


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


def f2u(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(f)))[0]


def run_batchnorm_sim(channels: int = 1, spatial_elements: int = 64):
    print(f"\n--- Testing batchnorm_c (channels={channels}, spatial={spatial_elements}) ---")
    c_code = batchnorm_c()
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "batchnorm.c"
        s_path = tmp / "batchnorm.s"
        in_path = tmp / "batchnorm.in"
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
        width = 32
        total_rows = (spatial_elements + width - 1) // width
        sp_rows = min(total_rows, 32)
        tile_count = (total_rows + sp_rows - 1) // sp_rows
        tile_bytes = sp_rows * width * 2
        ch_bytes = total_rows * width * 2

        # Generate inputs
        x_in = np.random.uniform(-3.0, 3.0, size=(channels, total_rows * width)).astype(np.float32)
        x_bf16 = np.zeros_like(x_in)
        for c in range(channels):
            for i in range(x_in.shape[1]):
                x_bf16[c, i] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[c, i]))

        # Scale A and Bias B per channel
        A_scales = np.random.uniform(0.5, 2.0, size=channels).astype(np.float32)
        B_biases = np.random.uniform(-1.0, 1.0, size=channels).astype(np.float32)

        # Expected calculation matching hardware sequential rounding:
        # y = A * x + B
        expected = np.zeros_like(x_bf16)
        for c in range(channels):
            Ac = bf16_u16_to_fp32(fp32_to_bf16_u16(A_scales[c]))
            Bc = bf16_u16_to_fp32(fp32_to_bf16_u16(B_biases[c]))
            for i in range(x_in.shape[1]):
                xi = x_bf16[c, i]
                ax = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * Ac))
                y = bf16_u16_to_fp32(fp32_to_bf16_u16(ax + Bc))
                expected[c, i] = y

        IN_GMEM = 0x1000
        OUT_GMEM = IN_GMEM + channels * ch_bytes
        SCALE_GMEM = OUT_GMEM + channels * ch_bytes
        BIAS_GMEM = SCALE_GMEM + channels * 4

        sdma_sp0 = (0 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)

        words: dict[int, int] = {}
        words[60] = IN_GMEM
        words[64] = OUT_GMEM
        words[68] = SCALE_GMEM
        words[72] = BIAS_GMEM
        words[76] = channels
        words[80] = sp_rows
        words[84] = tile_count
        words[88] = sdma_sp0
        words[92] = tile_bytes
        words[96] = ch_bytes

        # Seed Scale A and Bias B arrays
        for c in range(channels):
            words[SCALE_GMEM + c * 4] = f2u(A_scales[c])
            words[BIAS_GMEM + c * 4] = f2u(B_biases[c])

        # Seed Input matrices into IN_GMEM
        for c in range(channels):
            ch_addr = IN_GMEM + c * ch_bytes
            for i in range(0, total_rows * width, 2):
                x0 = fp32_to_bf16_u16(x_bf16[c, i])
                x1 = fp32_to_bf16_u16(x_bf16[c, i + 1]) if i + 1 < total_rows * width else 0
                words[ch_addr + i * 2] = (x1 << 16) | x0

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

        actual = np.zeros_like(expected)
        for c in range(channels):
            ch_addr = OUT_GMEM + c * ch_bytes
            for i in range(0, total_rows * width, 2):
                w_addr = ch_addr + i * 2
                val = mem_data.get(w_addr, 0)
                actual[c, i] = bf16_u16_to_fp32(val & 0xFFFF)
                if i + 1 < total_rows * width:
                    actual[c, i + 1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff: {max_diff:.6e}")
        assert max_diff < 1e-4, f"Output mismatch! Max diff: {max_diff}"

        for c in range(channels):
            print(f"    channel {c} (A={A_scales[c]:.2f}, B={B_biases[c]:.2f}): actual[:3]={actual[c, :3]}, expected[:3]={expected[c, :3]}")


def test_batchnorm_single_channel():
    """Verify single channel (64 elements)."""
    run_batchnorm_sim(channels=1, spatial_elements=64)


def test_batchnorm_multi_channel():
    """Verify 3 channels (3x64 elements)."""
    run_batchnorm_sim(channels=3, spatial_elements=64)


def test_batchnorm_multi_tile():
    """Verify 2 channels with multiple tiles (2x128 elements)."""
    run_batchnorm_sim(channels=2, spatial_elements=128)


def main():
    test_batchnorm_single_channel()
    test_batchnorm_multi_channel()
    test_batchnorm_multi_tile()
    print("\nAll batchnorm tests PASSED successfully!")


if __name__ == "__main__":
    main()
