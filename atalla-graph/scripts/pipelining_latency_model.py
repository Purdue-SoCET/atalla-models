#!/usr/bin/env python3
"""Generalized Critical-Path Pipelining & Memory-Overlap Latency Model.

Evaluates serial execution vs double-buffered/pipelined overlap for ANY kernel
on the Atalla AI Accelerator architecture.

Reads performance metrics from:
- A single simulator `_perf.out` file
- A directory of `_perf.out` files (e.g. `out/graph/`)
- Or explicit CLI parameters (bytes_loaded, bytes_written, packets, tiles)

Uses the authoritative Atalla latency parameters:
- L_scpad = 520 cycles (512 DDR transfer + 8 AXI handshake) per 64-byte DRAM line
- Line size = 64 bytes (Atalla DRAM burst/scpad row width)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_L_SCPAD = 520.0
DEFAULT_LINE_BYTES = 64


@dataclass
class KernelPipeliningReport:
    name: str
    tiles: int
    bytes_loaded: int
    bytes_written: int
    compute_packets: int
    dma_lines_load: int
    dma_lines_store: int
    dma_load_cycles: float
    dma_store_cycles: float
    compute_cycles: float
    t_serial: float
    t_pipelined: float
    projected_speedup: float
    asymptotic_speedup: float
    bound_type: str
    dma_to_compute_ratio: float


def parse_perf_file(path: Path) -> Dict[str, float]:
    """Parse key-value metrics from an output_perf.out file."""
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                metrics[k] = float(v)
            except ValueError:
                pass
    return metrics


def evaluate_pipelining(
    name: str,
    bytes_loaded: int,
    bytes_written: int,
    compute_packets: int,
    tiles: int = 1,
    lat_scpad: float = DEFAULT_L_SCPAD,
    line_bytes: int = DEFAULT_LINE_BYTES,
) -> KernelPipeliningReport:
    """Compute analytical serial vs pipelined overlap for an arbitrary kernel."""
    k = max(1, int(tiles))

    # Total and per-tile DRAM lines (64 bytes each)
    total_load_lines = max(1, math.ceil(bytes_loaded / float(line_bytes))) if bytes_loaded > 0 else 0
    total_store_lines = max(1, math.ceil(bytes_written / float(line_bytes))) if bytes_written > 0 else 0

    lines_load_per_tile = total_load_lines / float(k)
    lines_store_per_tile = total_store_lines / float(k)

    # Cycles
    d_load_tile = lines_load_per_tile * lat_scpad
    d_store_tile = lines_store_per_tile * lat_scpad
    c_tile = float(compute_packets) / float(k)

    # Total serial time across all K tiles
    t_serial = float(k) * (d_load_tile + c_tile + d_store_tile)

    # Double-buffered pipelined time:
    # Tile 0: load + compute
    # Tiles 1..K-1: max(load, compute) (DMA prefetch overlaps compute)
    # Plus all stores (or serialized writeback)
    if k <= 1:
        t_pipelined = t_serial
        proj_speedup = 1.0
    else:
        overlap_phase = float(k - 1) * max(d_load_tile, c_tile)
        t_pipelined = (d_load_tile + c_tile) + overlap_phase + (float(k) * d_store_tile)
        proj_speedup = t_serial / t_pipelined if t_pipelined > 0 else 1.0

    # Asymptotic speedup (as K -> infinity) for input prefetch double-buffering
    # T_serial/tile = D_load + C + D_store
    # T_pipe/tile   = max(D_load, C) + D_store
    denom = max(d_load_tile, c_tile) + d_store_tile
    asymptotic_speedup = (d_load_tile + c_tile + d_store_tile) / denom if denom > 0 else 1.0

    ratio = d_load_tile / c_tile if c_tile > 0 else float("inf")
    if ratio > 1.2:
        bound = "Memory-Bound (DRAM dominant)"
    elif ratio < 0.8:
        bound = "Compute-Bound (VALU dominant)"
    else:
        bound = "Balanced (Ideal Pipelining)"

    return KernelPipeliningReport(
        name=name,
        tiles=k,
        bytes_loaded=bytes_loaded,
        bytes_written=bytes_written,
        compute_packets=compute_packets,
        dma_lines_load=total_load_lines,
        dma_lines_store=total_store_lines,
        dma_load_cycles=d_load_tile * k,
        dma_store_cycles=d_store_tile * k,
        compute_cycles=float(compute_packets),
        t_serial=t_serial,
        t_pipelined=t_pipelined,
        projected_speedup=proj_speedup,
        asymptotic_speedup=asymptotic_speedup,
        bound_type=bound,
        dma_to_compute_ratio=ratio,
    )


def print_reports(reports: List[KernelPipeliningReport]):
    """Format and print benchmark reports table."""
    print("\n" + "=" * 98)
    print("           ATALLA GENERALIZED PIPELINING & DMA-OVERLAP LATENCY MODEL")
    print("=" * 98)
    hdr = f"{'Kernel / Layer':<20} | {'Tiles':<5} | {'Compute':<8} | {'DMA Load':<9} | {'Serial Cyc':<11} | {'Pipe Cyc':<11} | {'Speedup':<8} | {'Regime'}"
    print(hdr)
    print("-" * 98)
    for r in reports:
        row = (
            f"{r.name[:20]:<20} | "
            f"{r.tiles:<5} | "
            f"{int(r.compute_cycles):<8} | "
            f"{int(r.dma_load_cycles):<9} | "
            f"{int(r.t_serial):<11} | "
            f"{int(r.t_pipelined):<11} | "
            f"{r.projected_speedup:<7.3f}x | "
            f"{r.bound_type.split()[0]}"
        )
        print(row)
    print("=" * 98 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generalized Atalla Pipelining & Memory Overlap Latency Model"
    )
    parser.add_argument("--perf", type=Path, help="Path to single output_perf.out file")
    parser.add_argument("--dir", type=Path, help="Directory containing *_perf.out files")
    parser.add_argument("--name", type=str, default="custom_kernel", help="Kernel name")
    parser.add_argument("--tiles", type=int, default=4, help="Number of tiles (K)")
    parser.add_argument("--bytes-loaded", type=int, default=None, help="Bytes loaded via DMA")
    parser.add_argument("--bytes-written", type=int, default=None, help="Bytes written via DMA")
    parser.add_argument("--compute-packets", type=int, default=None, help="Compute packets retired")
    parser.add_argument("--lat-scpad", type=float, default=DEFAULT_L_SCPAD, help="DRAM line latency (default 520)")
    parser.add_argument("--line-bytes", type=int, default=DEFAULT_LINE_BYTES, help="Line bytes (default 64)")

    args = parser.parse_args()
    reports: List[KernelPipeliningReport] = []

    if args.dir:
        for perf_path in sorted(args.dir.glob("*_perf.out")):
            m = parse_perf_file(perf_path)
            kname = perf_path.stem.replace("_perf", "")
            bytes_ld = int(m.get("bytes_loaded", 0))
            bytes_wr = int(m.get("bytes_written", 0) or m.get("bytes_stored", 0))
            packets = int(m.get("packets_executed", 0))
            if bytes_ld > 0 or packets > 0:
                rep = evaluate_pipelining(
                    name=kname,
                    bytes_loaded=bytes_ld,
                    bytes_written=bytes_wr,
                    compute_packets=packets,
                    tiles=args.tiles,
                    lat_scpad=args.lat_scpad,
                    line_bytes=args.line_bytes,
                )
                reports.append(rep)
    elif args.perf:
        m = parse_perf_file(args.perf)
        kname = args.name if args.name != "custom_kernel" else args.perf.stem.replace("_perf", "")
        bytes_ld = int(args.bytes_loaded if args.bytes_loaded is not None else m.get("bytes_loaded", 0))
        bytes_wr = int(args.bytes_written if args.bytes_written is not None else (m.get("bytes_written", 0) or m.get("bytes_stored", 0)))
        packets = int(args.compute_packets if args.compute_packets is not None else m.get("packets_executed", 0))
        rep = evaluate_pipelining(
            name=kname,
            bytes_loaded=bytes_ld,
            bytes_written=bytes_wr,
            compute_packets=packets,
            tiles=args.tiles,
            lat_scpad=args.lat_scpad,
            line_bytes=args.line_bytes,
        )
        reports.append(rep)
    else:
        # Default demo across representative kernels if no args given
        demo_cases = [
            ("add_2048 (4 tiles)", 4096, 2048, 2094, 4),
            ("matmul_32x32 (4 tiles)", 4480, 2048, 4200, 4),
            ("conv_pipelined (8 tiles)", 8192, 4096, 12500, 8),
            ("relu_2048 (4 tiles)", 2048, 2048, 1800, 4),
            ("batchnorm_2048 (4 tiles)", 4096, 2048, 2600, 4),
        ]
        for name, b_ld, b_wr, pk, k in demo_cases:
            rep = evaluate_pipelining(
                name=name,
                bytes_loaded=b_ld,
                bytes_written=b_wr,
                compute_packets=pk,
                tiles=k,
                lat_scpad=args.lat_scpad,
                line_bytes=args.line_bytes,
            )
            reports.append(rep)

    print_reports(reports)


if __name__ == "__main__":
    main()
