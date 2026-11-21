#!/usr/bin/env python3
"""Load a linked Atalla ELF with atalla-sim's loader, then run it or export it.

Runs in its own process so atalla-sim's packages never share an import path
with the functional simulator.

    atalla_sim_elf.py K.elf --to-in K.in
        Write the functional simulator's .in image from the ELF's packets and
        data words (bootstrap packets included), so both simulators execute
        the same program.

    atalla_sim_elf.py K.elf --dram ramulator --perf-json K_perf.json \\
            --dump out.json --region y:0x80040:128
        Run cycle-accurately; report cycles and the PerfMonitor summary, and
        dump each named byte region of DRAM after halt as hex.

ATALLA_SIM_PATH selects the atalla-sim checkout.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def sim_root() -> Path:
    here = Path(__file__).resolve()
    models = here.parents[2]
    cands = [os.environ.get('ATALLA_SIM_PATH'), models.parent / 'atalla-sim',
             models.parent / 'atalla-inference' / 'atalla-sim']
    for cand in cands:
        if cand and (Path(cand) / 'src' / 'atalla' / 'atalla_platform.py').exists():
            return Path(cand)
    sys.exit('atalla-sim checkout not found; set ATALLA_SIM_PATH')


def write_in(program, path: Path) -> None:
    with path.open('w') as stream:
        for address, packet in sorted(program.instr.items()):
            stream.write(f'{address:08X}: {packet:040X}\n')
        stream.write('.data\n')
        for address, word in sorted(program.data.items()):
            stream.write(f'{address:08X}: {word:08X}\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('elf', type=Path)
    ap.add_argument('--to-in', type=Path)
    ap.add_argument('--dram', choices=('fixed', 'ramulator'), default='ramulator')
    ap.add_argument('--dram-config', default='rtl')
    ap.add_argument('--dram-latency', type=int, default=6)
    ap.add_argument('--limit', type=int, default=50_000_000)
    ap.add_argument('--perf-json', type=Path)
    ap.add_argument('--dump', type=Path, help='JSON file for cycles, summary and regions')
    ap.add_argument('--region', action='append', default=[], metavar='NAME:ADDR:BYTES')
    args = ap.parse_args()

    root = sim_root()
    os.environ.setdefault('ATALLA_NO_GOLDEN', '1')
    sys.path.insert(0, str(root / 'src'))
    from scheduler.elf import load_elf
    program = load_elf(args.elf)
    if args.to_in:
        write_in(program, args.to_in)
        return

    from atalla.atalla_platform import build_atalla_platform, c_entry_sregs
    channel = None
    if args.dram == 'ramulator':
        from memory.ramulator import RamulatorDRAM
        channel = RamulatorDRAM(args.dram_config)
    plat = build_atalla_platform(program.instr, program.data, dram_latency=args.dram_latency,
                                 init_sregs=c_entry_sregs(program.data),
                                 describe_pc=program.describe, dram_channel=channel,
                                 strict=True)
    started = time.time()
    cycles = plat.run_until_done(limit=args.limit)
    wall = time.time() - started
    summary = plat.perf.summary()
    print(f'{args.elf.name}: {cycles} cycles ({wall:.1f} s)')
    print(summary)
    if args.perf_json:
        plat.perf.to_json(str(args.perf_json))
    if args.dump:
        regions = {}
        for spec in args.region:
            name, addr, size = spec.rsplit(':', 2)
            regions[name] = plat.dram.read(int(addr, 0), int(size, 0)).hex()
        from native import kernels as _nk
        native = _nk.NATIVE_ISA if _nk.HAVE_NATIVE else None
        violations = len(plat.core.decode2.violations)
        args.dump.write_text(json.dumps(dict(cycles=cycles, wall_s=wall, regions=regions,
                                             dram=args.dram, native=native,
                                             rtl_violations=violations, summary=summary)))


if __name__ == '__main__':
    main()
