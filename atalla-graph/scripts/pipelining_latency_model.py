#!/usr/bin/env python3
"""Unvalidated analytical estimate for uniform tiles with input prefetch.

Assumes independent load/compute stages, serialized stores, and sufficient
buffers. No bandwidth, contention, alignment, or hardware feasibility model.
520 cycles per line is a teaching heuristic, not a hardware measurement.
The Python API's historical compute_packets argument is a cycle-budget proxy;
callers must provide its provenance. Byte totals assume packed aligned transfers
per tile unless explicit per-tile transfer sizes are supplied.
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
    load_transfer_bytes=None,
    store_transfer_bytes=None,
    pipeline_overhead_cycles: float = 0,
) -> KernelPipeliningReport:
    """Compute analytical serial vs pipelined overlap for an arbitrary kernel."""
    values = (bytes_loaded, bytes_written, compute_packets, lat_scpad,
              pipeline_overhead_cycles)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("byte counts, cycle budgets, and latency must be finite and nonnegative")
    if not isinstance(tiles, int) or tiles < 1:
        raise ValueError("tiles must be a positive integer")
    if not isinstance(line_bytes, int) or line_bytes < 1:
        raise ValueError("line_bytes must be a positive integer")
    k = tiles

    def lines_per_tile(total, transfers):
        if transfers is None:
            if total % k:
                raise ValueError("uniform tiles require byte totals divisible by tiles")
            return math.ceil((total // k) / line_bytes)
        if any(not isinstance(n, int) or n < 0 for n in transfers):
            raise ValueError("transfer sizes must be nonnegative integers")
        if sum(transfers) * k != total:
            raise ValueError("per-tile transfer sizes must match total bytes / tiles")
        return sum((n + line_bytes - 1) // line_bytes for n in transfers)

    lines_load_per_tile = lines_per_tile(bytes_loaded, load_transfer_bytes)
    lines_store_per_tile = lines_per_tile(bytes_written, store_transfer_bytes)
    total_load_lines = lines_load_per_tile * k
    total_store_lines = lines_store_per_tile * k

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
        t_pipelined = t_serial + pipeline_overhead_cycles
        proj_speedup = t_serial / t_pipelined if t_pipelined else 1.0
    else:
        overlap_phase = float(k - 1) * max(d_load_tile, c_tile)
        t_pipelined = (d_load_tile + c_tile) + overlap_phase + (float(k) * d_store_tile) + pipeline_overhead_cycles
        proj_speedup = t_serial / t_pipelined if t_pipelined > 0 else 1.0

    # Asymptotic speedup (as K -> infinity) for input prefetch double-buffering
    # T_serial/tile = D_load + C + D_store
    # T_pipe/tile   = max(D_load, C) + D_store
    denom = max(d_load_tile, c_tile) + d_store_tile
    asymptotic_speedup = (d_load_tile + c_tile + d_store_tile) / denom if denom > 0 else 1.0

    ratio = (d_load_tile + d_store_tile) / c_tile if c_tile > 0 else float("inf")
    if ratio > 1.2:
        bound = "Memory-Bound (DRAM dominant)"
    elif ratio < 0.8:
        bound = "Compute-Bound (assumed cycle budget dominant)"
    else:
        bound = "Balanced (assumed DMA/compute costs)"

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
    print("Unvalidated analytical estimate: uniform tiles; ideal input prefetch; serialized stores.")
    print("Sufficient buffers and independent load/compute assumed; no contention model.")
    print("520 cycles/line default is heuristic; predictions are not measured hardware cycles.")
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
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--perf", type=Path)
    source.add_argument("--dir", type=Path)
    parser.add_argument("--name", default="custom_kernel")
    parser.add_argument("--tiles", type=int, required=True, help="Explicit uniform tile count")
    parser.add_argument("--bytes-loaded", type=int, help="Total useful bytes over all tiles")
    parser.add_argument("--bytes-written", type=int, help="Total useful bytes over all tiles")
    cycles = parser.add_mutually_exclusive_group()
    cycles.add_argument("--compute-cycles", type=float, help="Assumed total compute cycle budget")
    cycles.add_argument("--compute-packets", type=int, help="Legacy total packet proxy; requires proxy opt-in")
    parser.add_argument("--use-packets-as-compute-proxy", action="store_true")
    parser.add_argument("--assume-packed-transfers", action="store_true",
                        help="Assume one contiguous aligned transfer per direction per tile")
    parser.add_argument("--load-transfer-bytes", action="append", type=int,
                        help="Aligned load transfer size per tile; repeat for separate transfers")
    parser.add_argument("--store-transfer-bytes", action="append", type=int,
                        help="Aligned store transfer size per tile; repeat for separate transfers")
    parser.add_argument("--pipeline-overhead-cycles", type=float, default=0,
                        help="Assumed extra total pipeline control/setup cycles (added serially)")
    parser.add_argument("--lat-scpad", type=float, default=DEFAULT_L_SCPAD)
    parser.add_argument("--line-bytes", type=int, default=DEFAULT_LINE_BYTES)
    args = parser.parse_args()
    try:
        if args.compute_cycles is None and not args.use_packets_as_compute_proxy:
            raise ValueError("supply --compute-cycles or explicitly --use-packets-as-compute-proxy")
        if args.compute_cycles is not None and args.use_packets_as_compute_proxy:
            raise ValueError("choose compute cycles or packet proxy, not both")
        if args.assume_packed_transfers and (args.load_transfer_bytes is not None or args.store_transfer_bytes is not None):
            raise ValueError("choose explicit transfers or --assume-packed-transfers")
        if args.dir:
            if not args.dir.is_dir():
                raise ValueError(f"directory does not exist: {args.dir}")
            paths = sorted(args.dir.glob("*_perf.out"))
            if not paths:
                raise ValueError("directory contains no *_perf.out files")
        else:
            paths = [args.perf]
        reports = []
        for path in paths:
            metrics = parse_perf_file(path) if path else {}
            def byte_total(option, key, transfers):
                value = option if option is not None else metrics.get(key)
                if value is None and key == "bytes_written":
                    value = metrics.get("bytes_stored")
                if value is None and transfers is not None:
                    value = sum(transfers) * args.tiles
                if value is None:
                    raise ValueError(f"missing {key}; supply a total or explicit per-tile transfers")
                if not math.isfinite(value) or value < 0 or int(value) != value:
                    raise ValueError(f"{key} must be a nonnegative integer")
                if value and transfers is None and not args.assume_packed_transfers:
                    raise ValueError(f"{key}: supply per-tile transfer sizes or --assume-packed-transfers")
                return int(value)
            loaded = byte_total(args.bytes_loaded, "bytes_loaded", args.load_transfer_bytes)
            written = byte_total(args.bytes_written, "bytes_written", args.store_transfer_bytes)
            compute = args.compute_cycles
            if compute is None:
                compute = args.compute_packets if args.compute_packets is not None else metrics.get("packets_executed")
                if compute is None:
                    raise ValueError("missing packet count for compute proxy")
            name = path.stem.removesuffix("_perf") if path and args.name == "custom_kernel" else args.name
            reports.append(evaluate_pipelining(
                name, loaded, written, compute, args.tiles, args.lat_scpad, args.line_bytes,
                args.load_transfer_bytes, args.store_transfer_bytes, args.pipeline_overhead_cycles))
        print("Compute input: " + ("total retired packets used as a cycle proxy (includes non-compute work)"
                                   if args.use_packets_as_compute_proxy else "user-supplied assumed cycle budget"))
        print(f"Transfer input: {'packed aligned per-tile assumption' if args.assume_packed_transfers else 'explicit aligned per-tile transfers'}; "
              f"latency={args.lat_scpad:g} cycles/line; line={args.line_bytes} bytes; "
              f"extra pipeline overhead={args.pipeline_overhead_cycles:g} cycles")
        print_reports(reports)
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
