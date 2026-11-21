"""Build one kernel ELF, then execute it on both simulators.

    elf = build_elf(source, image, work_dir, tag)   # compiler: compile, packetize, link
    fs  = run_functional(elf, work_dir, tag)        # functional sim, .in from the ELF
    cs  = run_cycle(elf, regions, work_dir, tag)    # atalla-sim, the ELF itself

``run_functional`` returns the emulator memory and its dynamic metrics (packets,
retired instructions per class). ``run_cycle`` returns cycles, the PerfMonitor
summary and the DRAM bytes of each requested region after halt.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from codegen.elf_image import DataImage, link_elf

_SCRIPT = Path(__file__).resolve().parent / 'scripts' / 'atalla_sim_elf.py'


@dataclass
class FunctionalRun:
    memory: object
    metrics: dict

    def read(self, address: int, size: int) -> bytes:
        return bytes((self.memory.read_data((address + i) & -4) >> (8 * ((address + i) & 3))) & 255
                     for i in range(size))


@dataclass
class CycleRun:
    cycles: int
    regions: dict
    summary: str
    native: str | None
    perf_json: Path | None
    rtl_violations: int

    def read(self, name: str) -> bytes:
        return self.regions[name]


def build_elf(source: Path, image: DataImage, work_dir, tag: str, level: str = 'safe') -> Path:
    return link_elf(Path(source), image, Path(work_dir), tag, level)


def elf_to_in(elf: Path, in_path: Path) -> Path:
    subprocess.run([sys.executable, str(_SCRIPT), str(elf), '--to-in', str(in_path)],
                   check=True, capture_output=True, text=True)
    return in_path


def run_functional(elf: Path, work_dir, tag: str) -> FunctionalRun:
    from run_graph import _run_emulator
    work_dir = Path(work_dir)
    image = elf_to_in(Path(elf), work_dir / (tag + '.in'))
    memory, unit = _run_emulator(str(image), str(work_dir), tag)
    return FunctionalRun(memory, dict(unit.perf_metrics.metrics))


def run_cycle(elf: Path, regions: dict, work_dir, tag: str, dram: str = 'ramulator',
              limit: int = 50_000_000, timeout: float | None = None) -> CycleRun:
    """``regions`` maps a name to (byte address, byte count)."""
    work_dir = Path(work_dir)
    dump = work_dir / (tag + '_cycle.json')
    perf = work_dir / (tag + '_perf.json')
    cmd = [sys.executable, str(_SCRIPT), str(elf), '--dram', dram, '--limit', str(limit),
           '--perf-json', str(perf), '--dump', str(dump)]
    for name, (address, size) in regions.items():
        cmd += ['--region', f'{name}:{address:#x}:{size}']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'atalla-sim failed on {elf.name}:\n{proc.stderr[-3000:]}')
    out = json.loads(dump.read_text())
    return CycleRun(out['cycles'], {k: bytes.fromhex(v) for k, v in out['regions'].items()},
                    out['summary'], out['native'], perf, out['rtl_violations'])
