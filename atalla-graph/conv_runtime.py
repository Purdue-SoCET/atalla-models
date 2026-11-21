"""Execute a named convolution: one compiler-built ELF on both simulators.

    result = execute_conv(op, buffers, work_dir)            # functional sim
    result = execute_conv(op, buffers, work_dir, cycle=True) # + atalla-sim

The functional simulator runs the ``.in`` written from the ELF and supplies the
numerical result and dynamic instruction metrics; atalla-sim runs the ELF
itself (Ramulator DRAM by default) and supplies cycles and its PerfMonitor
report, plus its own copy of every buffer so the two can be compared.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegen.conv import emit_conv
from elf_runtime import build_elf, run_cycle, run_functional


@dataclass
class ConvResult:
    buffers: dict
    plan: object
    elf: Path
    metrics: dict
    cycle: object = None          # elf_runtime.CycleRun
    cycle_buffers: dict | None = None


def execute_conv(op, buffers, work_dir, tag='conv', cycle=False, dram='ramulator',
                 functional=True, limit=50_000_000):
    if not tag or Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('tag must be a single artifact filename component')
    packed = emit_conv(op, buffers)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    elf = build_elf(packed.source, packed.image, work_dir, tag)
    values, metrics = None, {}
    if functional:
        fs = run_functional(elf, work_dir, tag)
        values = {name: fs.read(addr, packed.buffer_sizes[name])
                  for name, addr in packed.buffer_addresses.items()}
        metrics = fs.metrics
    result = ConvResult(values, packed.plan, elf, metrics)
    if cycle:
        regions = {name: (addr, packed.buffer_sizes[name])
                   for name, addr in packed.buffer_addresses.items() if packed.buffer_sizes[name]}
        run = run_cycle(elf, regions, work_dir, tag, dram=dram, limit=limit)
        result.cycle = run
        result.cycle_buffers = {name: run.regions.get(name, b'') for name in packed.buffer_addresses}
    return result
