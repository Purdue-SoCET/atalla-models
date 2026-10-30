#!/usr/bin/env python3
"""Unit test for upsample AtallaC kernel (2D nearest-neighbor 2x upsampling).

Verifies:
1. Standalone compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in the inner loop body and entire program (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. PyTorch F.interpolate(..., scale_factor=2.0, mode='nearest'):
   - Single channel (C=1, 4x8 -> 8x16)
   - Multi-channel (C=3, 4x8 -> 8x16)
   - Larger spatial (C=2, 8x16 -> 16x32)
"""

import os
import re
import sys
import struct
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure root workspace and kernels in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REPO_PPCI = _ROOT / "aihw-ppci-compiler"
_SIM_ROOT = _REPO_PPCI / "functional_sim"
_MODELS_ROOT = _ROOT / "atalla-models" / "atalla-graph"

sys.path.insert(0, str(_MODELS_ROOT))
from kernels.upsample import upsample_c


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


def run_upsample_sim(channels: int = 1, H_in: int = 4, W_in: int = 8):
    print(f"\n--- Testing upsample_c (channels={channels}, Hin={H_in}, Win={W_in}) ---")
    c_code = upsample_c()
    assert "atalla_halt" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "upsample.c"
        s_path = tmp / "upsample.s"
        in_path = tmp / "upsample.in"
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

        # 2. Spill check: inner loop body must have 0 stack spills
        loop_matches = re.findall(r"while.*?(?=atalla_halt)", s_content, re.DOTALL)
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
        total_rows = channels * H_in
        row_words = W_in // 2
        in_row_bytes = W_in * 2
        out_row_bytes = (2 * W_in) * 2

        # Generate inputs
        x_in = np.random.uniform(-3.0, 3.0, size=(channels, H_in, W_in)).astype(np.float32)
        x_bf16 = np.zeros_like(x_in)
        for c in range(channels):
            for h in range(H_in):
                for w in range(W_in):
                    x_bf16[c, h, w] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[c, h, w]))

        # PyTorch reference golden
        x_torch = torch.from_numpy(x_bf16).unsqueeze(0)  # (1, C, H, W)
        expected = F.interpolate(x_torch, scale_factor=2.0, mode="nearest").squeeze(0).numpy()

        IN_GMEM = 0x1000
        in_total_bytes = channels * H_in * in_row_bytes
        OUT_GMEM = IN_GMEM + in_total_bytes

        words: dict[int, int] = {}
        words[60] = IN_GMEM
        words[64] = OUT_GMEM
        words[68] = total_rows
        words[72] = row_words
        words[76] = in_row_bytes
        words[80] = out_row_bytes

        # Populate IN_GMEM
        for c in range(channels):
            for h in range(H_in):
                r_idx = c * H_in + h
                row_addr = IN_GMEM + r_idx * in_row_bytes
                for w in range(0, W_in, 2):
                    val0 = fp32_to_bf16_u16(x_bf16[c, h, w])
                    val1 = fp32_to_bf16_u16(x_bf16[c, h, w + 1])
                    words[row_addr + w * 2] = (val1 << 16) | val0

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

        H_out = 2 * H_in
        W_out = 2 * W_in
        actual = np.zeros((channels, H_out, W_out), dtype=np.float32)
        for c in range(channels):
            for h in range(H_out):
                r_idx = c * H_out + h
                row_addr = OUT_GMEM + r_idx * out_row_bytes
                for w in range(0, W_out, 2):
                    w_addr = row_addr + w * 2
                    val = mem_data.get(w_addr, 0)
                    actual[c, h, w] = bf16_u16_to_fp32(val & 0xFFFF)
                    actual[c, h, w + 1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff vs PyTorch: {max_diff:.6e}")
        assert max_diff == 0.0, f"Output mismatch! Max diff: {max_diff}"
        for c in range(channels):
            print(f"    channel {c}: actual[0, :4]={actual[c, 0, :4]}, expected[0, :4]={expected[c, 0, :4]}")


def test_upsample_single_channel():
    """Verify single channel (1x4x8 -> 1x8x16)."""
    run_upsample_sim(channels=1, H_in=4, W_in=8)


def test_upsample_multi_channel():
    """Verify 3 channels (3x4x8 -> 3x8x16)."""
    run_upsample_sim(channels=3, H_in=4, W_in=8)


def test_upsample_large_spatial():
    """Verify 2 channels with larger spatial size (2x8x16 -> 2x16x32)."""
    run_upsample_sim(channels=2, H_in=8, W_in=16)


def main():
    test_upsample_single_channel()
    test_upsample_multi_channel()
    test_upsample_large_spatial()
    print("\nAll upsample tests PASSED successfully!")


if __name__ == "__main__":
    main()
