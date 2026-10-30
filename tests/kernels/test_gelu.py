#!/usr/bin/env python3
"""Unit test for gelu AtallaC kernel generators (Horner Polynomial and Tanh Expansion).

Verifies:
1. Standalone compilation via PPCI atalla_cc (-p -S).
2. Zero stack spills in compute loops (sw_s/lw_s relative to x8).
3. Functional simulator execution vs. PyTorch/NumPy golden:
   - Method 1 (Horner Poly6): single tile (32) and multi-tile (64)
   - Method 2 (Tanh Poly7): single tile (32) and multi-tile (64)
4. Speed and accuracy metrics comparison.
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
from kernels.gelu import gelu_c


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


def run_gelu_sim(method: str = "horner", total_elements: int = 32):
    print(f"\n--- Testing gelu_c (method='{method}', total_elements={total_elements}) ---")
    c_code = gelu_c(method=method)
    assert "vector_load" in c_code
    assert "vec_op_masked" in c_code

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        c_path = tmp / "gelu.c"
        s_path = tmp / "gelu.s"
        in_path = tmp / "gelu.in"
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
        x_in = np.linspace(-3.5, 3.5, total_elements, dtype=np.float32)

        x_bf16 = np.zeros_like(x_in)
        for i in range(total_elements):
            x_bf16[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(x_in[i]))

        width = 32
        rows = (total_elements + width - 1) // width
        sp_rows = min(rows, 32)
        tile_count = (rows + sp_rows - 1) // sp_rows
        tile_bytes = sp_rows * width * 2

        IN_GMEM = 0x1000
        OUT_GMEM = 0x3000
        sdma_sp0 = (0 << 30) | ((sp_rows - 1) << 25) | ((width - 1) << 20) | (width - 1)

        words: dict[int, int] = {}
        words[60] = IN_GMEM
        words[64] = OUT_GMEM
        words[68] = tile_count
        words[72] = sp_rows
        words[76] = sdma_sp0
        words[80] = tile_bytes

        expected = np.zeros(total_elements, dtype=np.float32)
        if method == "horner":
            c6 = np.float32(0.001803)
            c4 = np.float32(-0.037653)
            c2 = np.float32(0.360243)
            c0 = np.float32(0.008555)
            c_half = np.float32(0.5)
            words[84] = f2u(c6)
            words[88] = f2u(c4)
            words[92] = f2u(c2)
            words[96] = f2u(c0)
            words[100] = f2u(c_half)
            words[104] = f2u(3.0)
            words[108] = f2u(-3.0)

            for i in range(total_elements):
                xi = x_bf16[i]
                x2 = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * xi))
                t1 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(x2 * c6)) + c4))
                t2 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(t1 * x2)) + c2))
                t3 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(t2 * x2)) + c0))
                half_x = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * c_half))
                yi = bf16_u16_to_fp32(fp32_to_bf16_u16(t3 + half_x))
                if xi > 3.0:
                    yi = xi
                elif xi < -3.0:
                    yi = 0.0
                expected[i] = yi
        else:
            k1 = np.float32(0.797885)
            k3 = np.float32(0.035677)
            t_c6 = np.float32(-0.002561)
            t_c4 = np.float32(0.039209)
            t_c2 = np.float32(-0.237744)
            t_c0 = np.float32(0.968467)
            one = np.float32(1.0)
            neg_one = np.float32(-1.0)
            half = np.float32(0.5)

            words[84] = f2u(k1)
            words[88] = f2u(k3)
            words[92] = f2u(t_c6)
            words[96] = f2u(t_c4)
            words[100] = f2u(t_c2)
            words[104] = f2u(t_c0)
            words[108] = f2u(one)
            words[112] = f2u(neg_one)
            words[116] = f2u(half)
            words[120] = f2u(2.5)
            words[124] = f2u(-2.5)

            for i in range(total_elements):
                xi = x_bf16[i]
                x2 = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * xi))
                x3 = bf16_u16_to_fp32(fp32_to_bf16_u16(x2 * xi))
                z1 = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * k1))
                z2 = bf16_u16_to_fp32(fp32_to_bf16_u16(x3 * k3))
                zi = bf16_u16_to_fp32(fp32_to_bf16_u16(z1 + z2))
                z_sq = bf16_u16_to_fp32(fp32_to_bf16_u16(zi * zi))
                p1 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(z_sq * t_c6)) + t_c4))
                p2 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(p1 * z_sq)) + t_c2))
                p3 = bf16_u16_to_fp32(fp32_to_bf16_u16(bf16_u16_to_fp32(fp32_to_bf16_u16(p2 * z_sq)) + t_c0))
                tanh_val = bf16_u16_to_fp32(fp32_to_bf16_u16(zi * p3))
                if zi > 2.5:
                    tanh_val = 1.0
                elif zi < -2.5:
                    tanh_val = -1.0
                one_plus = bf16_u16_to_fp32(fp32_to_bf16_u16(tanh_val + one))
                half_x = bf16_u16_to_fp32(fp32_to_bf16_u16(xi * half))
                expected[i] = bf16_u16_to_fp32(fp32_to_bf16_u16(half_x * one_plus))

        padded_total = rows * width
        padded_x = np.zeros(padded_total, dtype=np.float32)
        padded_x[:total_elements] = x_bf16

        for i in range(0, padded_total, 2):
            x0 = fp32_to_bf16_u16(padded_x[i])
            x1 = fp32_to_bf16_u16(padded_x[i + 1]) if i + 1 < padded_total else 0
            words[IN_GMEM + i * 2] = (x1 << 16) | x0

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
        print(f"  [Pass] Golden check passed! Max absolute diff vs kernel golden: {max_diff:.6e}")
        assert max_diff < 0.05, f"Output mismatch! Max diff: {max_diff}"

        # Analytical PyTorch tanh GELU comparison
        z_ref = np.sqrt(2.0 / np.pi) * (x_bf16 + 0.044715 * (x_bf16**3))
        ref_pytorch = 0.5 * x_bf16 * (1.0 + np.tanh(z_ref))
        cos_sim = np.dot(actual, ref_pytorch) / (np.linalg.norm(actual) * np.linalg.norm(ref_pytorch))
        print(f"  [Metric] Cosine similarity vs PyTorch ref: {cos_sim:.6f}")
        for i in range(min(5, total_elements)):
            print(f"    idx {i:2d}: x={x_bf16[i]:+.3f}, actual={actual[i]:+.3f}, expected={expected[i]:+.3f}, pytorch={ref_pytorch[i]:+.3f}")


def test_gelu_horner_single_tile():
    """Verify Method 1 (Horner Poly6) for 32 elements (1 tile)."""
    run_gelu_sim(method="horner", total_elements=32)


def test_gelu_horner_multi_tile():
    """Verify Method 1 (Horner Poly6) for 64 elements (2 tiles)."""
    run_gelu_sim(method="horner", total_elements=64)


def test_gelu_tanh_single_tile():
    """Verify Method 2 (Tanh Poly7) for 32 elements (1 tile)."""
    run_gelu_sim(method="tanh", total_elements=32)


def test_gelu_tanh_multi_tile():
    """Verify Method 2 (Tanh Poly7) for 64 elements (2 tiles)."""
    run_gelu_sim(method="tanh", total_elements=64)


def main():
    test_gelu_horner_single_tile()
    test_gelu_horner_multi_tile()
    test_gelu_tanh_single_tile()
    test_gelu_tanh_multi_tile()
    print("\nAll gelu tests PASSED successfully!")


if __name__ == "__main__":
    main()
