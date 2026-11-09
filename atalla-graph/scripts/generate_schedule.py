#!/usr/bin/env python3
"""Generate a forward schedule using the same AtallaC kernels as validation.

The result contains kernel implementations and a single main(), rather than
calls to the placeholder kernels/kernels.c interface. See schedule.json for the
input/output addresses in dram.bin. No reference intermediates are embedded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from codegen.kernel_schedule import KernelSchedule, build_schedule


def emit(gm, placeholder_data=None) -> str:
    """Return complete AtallaC source; placeholders require explicit input data."""
    return build_schedule(gm, placeholder_data or {}).source


def write(gm, out_dir: str, placeholder_data: dict) -> KernelSchedule:
    schedule = build_schedule(gm, placeholder_data)
    schedule.write(out_dir)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="Serialized, normalized GraphModule with tile plans")
    parser.add_argument("--inputs", required=True,
                        help="torch.save dictionary of BF16 tensors keyed by placeholder name")
    parser.add_argument("--out-dir", default="out/schedule")
    args = parser.parse_args()
    graph = torch.load(args.graph, weights_only=False)
    inputs = torch.load(args.inputs, weights_only=True)
    write(graph, args.out_dir, inputs)
    print(f"Schedule written to {Path(args.out_dir) / 'graph_schedule.c'}")


if __name__ == "__main__":
    main()
