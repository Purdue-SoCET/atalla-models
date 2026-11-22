#!/usr/bin/env python3
"""Summarise atalla-sim GEMM runs: cycles, MAC use, DMA/DRAM stats, output checks.

usage: python cycle_sim/analyze.py <programs dir>   (numpy only)
Writes <run>/summary.json and <programs dir>/summary.md / summary.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def bf16_bits(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def main():
    root = Path(sys.argv[1])
    rows = []
    outs = {}
    for d in sorted(p for p in root.iterdir() if (p / "gemm.c").exists()):
        lay = json.loads((d / "layout.json").read_text())
        st = json.loads((d / "run_status.json").read_text())
        perf = json.loads((d / "perf.json").read_text())
        fs = json.loads((d / "fsim" / "perf.json").read_text())
        log = (d / "run.log").read_text()
        m = re.search(r"dependency check: (\d+) findings at (\d+) pcs", log)
        ref = np.load(d / "ref.npy")
        cy = np.load(d / "C_cycle.npy").astype(np.float32)
        fc = np.load(d / "fsim" / "C.npy").astype(np.float32)
        M, N, K, s = lay["M"], lay["N"], lay["K"], lay["schedule"]
        outs[(M, N, K, s)] = cy
        mem, w, iss = perf["memory"], perf["work"], perf["issue"]
        mast = mem["masters"]
        sp_bytes = sum(mast[k]["bytes"] for k in mast if k.startswith("scpad"))
        r = dict(
            run=d.name, shape=f"{M}x{N}x{K}", schedule=s, rc=st["rc"], wall_s=st["wall_s"],
            cycles=perf["cycles"],
            packets=iss["packets"], ops=iss["ops"], slot_util=iss["slot_utilization"],
            gemm_ops=perf["instruction_mix"].get("gemm", 0),
            lw_vi=perf["instruction_mix"].get("weight_load", 0),
            sdma_load=perf["instruction_mix"].get("sdma_load", 0),
            sdma_store=perf["instruction_mix"].get("sdma_store", 0),
            vreg_load=perf["instruction_mix"].get("vreg_load", 0),
            vreg_store=perf["instruction_mix"].get("vreg_store", 0),
            array_flops=w["array_flops"], array_mac_util=w["array_mac_utilization"],
            flops_per_cycle=w["flops_per_cycle"],
            dram_bytes=mem["dram_bytes"], dram_B_per_cycle=mem["dram_bytes_per_cycle"],
            dram_channel_busy=mem["dram_channel_busy"],
            dram_icache_bytes=mast["icache"]["bytes"], dram_dcache_bytes=mast["dcache"]["bytes"],
            dram_scpad_bytes=sp_bytes,
            dram_scpad0_bytes=mast["scpad0"]["bytes"], dram_scpad1_bytes=mast["scpad1"]["bytes"],
            scpad_wait_cycles=sum(mast[k]["wait_cycles"] for k in mast if k.startswith("scpad")),
            cycle_breakdown=perf["cycle_breakdown"],
            fsim_bytes_moved=fs.get("bytes_moved"), fsim_packets=fs.get("packets_executed"),
            model_bytes=lay["traffic_model"]["bytes_loaded"] + lay["traffic_model"]["bytes_written"],
            dep_findings=int(m.group(1)) if m else None,
            cos_cycle_vs_ref=cos(cy, ref), cos_cycle_vs_ref_colrev=cos(cy, ref[:, ::-1]),
            maxabs_cycle_vs_ref=float(np.abs(cy - ref).max()),
            cos_fsim_vs_ref=cos(fc, ref),
            cycle_eq_fsim_bits=bool(np.array_equal(bf16_bits(cy), bf16_bits(fc))),
            cycle_eq_fsim_colrev_bits=bool(np.array_equal(bf16_bits(cy), bf16_bits(fc[:, ::-1]))),
        )
        (d / "summary.json").write_text(json.dumps(r, indent=1))
        rows.append(r)
    for r in rows:
        M, N, K = (int(x) for x in r["shape"].split("x"))
        base = outs.get((M, N, K, "baseline"))
        r["cycle_eq_baseline_bits"] = bool(np.array_equal(
            bf16_bits(outs[(M, N, K, r["schedule"])]), bf16_bits(base))) if base is not None else None
        (root / r["run"] / "summary.json").write_text(json.dumps(r, indent=1))
    (root / "summary.json").write_text(json.dumps(rows, indent=1))
    hdr = ("| shape | schedule | rc | cycles | packets | gemm | lw.vi | sdma ld/st | array MAC util | "
           "DRAM B (scpad / icache / dcache) | DRAM B/cyc | fsim bytes_moved | dep findings | "
           "cos(atalla-sim, ref) | cos(fsim, ref) | atalla-sim == fsim bits | == baseline bits |")
    lines = [hdr, "|" + "---|" * hdr.count("|")[:-1] if False else "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        lines.append(
            f"| {r['shape']} | {r['schedule']} | {r['rc']} | {r['cycles']} | {r['packets']} | "
            f"{r['gemm_ops']} | {r['lw_vi']} | {r['sdma_load']}/{r['sdma_store']} | "
            f"{100 * r['array_mac_util']:.3f}% | {r['dram_scpad_bytes']} / {r['dram_icache_bytes']} / "
            f"{r['dram_dcache_bytes']} | {r['dram_B_per_cycle']:.3f} | {r['fsim_bytes_moved']:.0f} | "
            f"{r['dep_findings']} | {r['cos_cycle_vs_ref']:.6f} | {r['cos_fsim_vs_ref']:.6f} | "
            f"{r['cycle_eq_fsim_bits']} | {r['cycle_eq_baseline_bits']} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
