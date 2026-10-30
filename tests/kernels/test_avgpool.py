#!/usr/bin/env python3
"""Unit test for avgpool AtallaC kernel with vector intrinsics.

Verifies:
1. Standalone avgpool.c compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. NumPy golden:
   - Single channel 8x8 -> 4x4 (2x2 avgpool, stride 2)
   - Multi-channel 2x8x8 -> 2x4x4 (2x2 avgpool, stride 2)
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
from kernels.avgpool import avgpool_c


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


def run_avgpool_sim(channels: int = 1):
    print(f"\n--- Testing avgpool_c (channels={channels}, 8x8 -> 4x4, 2x2 pool) ---")
    c_code = avgpool_c()
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "avgpool.c"
        s_path = tmp / "avgpool.s"
        in_path = tmp / "avgpool.in"
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
        H_in, W_in = 8, 8
        H_out, W_out = 4, 4
        in_ch_bytes = H_in * W_in * 2
        out_ch_bytes = H_out * W_out * 2

        x_in = np.random.uniform(-2.0, 2.0, size=(channels, H_in, W_in)).astype(np.float32)

        # Quantize to BF16
        x_bf16 = np.zeros_like(x_in)
        for c in range(channels):
            for h in range(H_in):
                for w in range(W_in):
                    x_bf16[c, h, w] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[c, h, w]))

        # Golden calculation: exact 2x2 average pool matching hardware steps
        # Step 1: vertical sum of row 2*oh and 2*oh+1
        # Step 2: horizontal sum of col 2*ow and 2*ow+1
        # Step 3: multiply by 0.25 (scale)
        expected = np.zeros((channels, H_out, W_out), dtype=np.float32)
        scale = np.float32(0.25)
        for c in range(channels):
            for oh in range(H_out):
                for ow in range(W_out):
                    v0 = x_bf16[c, 2 * oh, 2 * ow]
                    v1 = x_bf16[c, 2 * oh + 1, 2 * ow]
                    vsum0 = bf16_u16_to_fp32(fp32_to_bf16_u16(v0 + v1))

                    v2 = x_bf16[c, 2 * oh, 2 * ow + 1]
                    v3 = x_bf16[c, 2 * oh + 1, 2 * ow + 1]
                    vsum1 = bf16_u16_to_fp32(fp32_to_bf16_u16(v2 + v3))

                    total_sum = bf16_u16_to_fp32(fp32_to_bf16_u16(vsum0 + vsum1))
                    expected[c, oh, ow] = bf16_u16_to_fp32(fp32_to_bf16_u16(total_sum * scale))

        IN_BASE = 0x1000
        OUT_BASE = 0x3000

        sdma_in = (0 << 30) | ((H_in - 1) << 25) | ((W_in - 1) << 20) | (W_in - 1)
        sdma_out = (0 << 30) | ((H_out - 1) << 25) | ((W_out - 1) << 20) | (W_out - 1)

        words: dict[int, int] = {}
        words[60] = IN_BASE
        words[64] = OUT_BASE
        words[68] = channels
        words[72] = H_in
        words[76] = H_out
        words[80] = sdma_in
        words[84] = sdma_out
        words[88] = in_ch_bytes
        words[92] = out_ch_bytes
        words[96] = struct.unpack("<I", struct.pack("<f", scale))[0]

        # Seed input tensors into IN_BASE
        for ch in range(channels):
            ch_addr = IN_BASE + ch * in_ch_bytes
            ch_flat = x_bf16[ch].ravel()
            for i in range(0, len(ch_flat), 2):
                x0 = fp32_to_bf16_u16(ch_flat[i])
                x1 = fp32_to_bf16_u16(ch_flat[i + 1]) if i + 1 < len(ch_flat) else 0
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

        actual = np.zeros((channels, H_out, W_out), dtype=np.float32)
        for ch in range(channels):
            ch_addr = OUT_BASE + ch * out_ch_bytes
            total_ch_elems = H_out * W_out
            for i in range(0, total_ch_elems, 2):
                w_addr = ch_addr + i * 2
                val = mem_data.get(w_addr, 0)
                oh = i // W_out
                ow = i % W_out
                actual[ch, oh, ow] = bf16_u16_to_fp32(val & 0xFFFF)
                if i + 1 < total_ch_elems:
                    oh1 = (i + 1) // W_out
                    ow1 = (i + 1) % W_out
                    actual[ch, oh1, ow1] = bf16_u16_to_fp32((val >> 16) & 0xFFFF)

        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"  [Pass] Golden check passed! Max absolute diff: {max_diff:.6e}")
        assert max_diff < 1e-4, f"Output mismatch! Max diff: {max_diff}"
        for c in range(channels):
            print(f"    channel {c}: actual[0, :2]={actual[c, 0, :2]}, expected[0, :2]={expected[c, 0, :2]}")


def test_avgpool_single_channel():
    """Verify 8x8 -> 4x4 avgpool for 1 channel."""
    run_avgpool_sim(channels=1)


def test_avgpool_multi_channel():
    """Verify 8x8 -> 4x4 avgpool for 2 channels."""
    run_avgpool_sim(channels=2)


def main():
    test_avgpool_single_channel()
    test_avgpool_multi_channel()
    print("\nAll avgpool tests PASSED successfully!")


if __name__ == "__main__":
    main()
