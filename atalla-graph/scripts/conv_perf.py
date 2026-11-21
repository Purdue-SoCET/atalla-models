#!/usr/bin/env python3
"""Cycle and utilization report for the named linalg.conv_* kernels.

For every case, one compiler-built ELF runs on atalla-sim (Ramulator DDR4
timing, the RTL configuration) for cycles, unit busy time and DRAM traffic, and
its .in runs on the functional simulator for dynamic instruction metrics and
the numerical check against float64 (small-integer data, so bf16 is exact).

    python scripts/conv_perf.py --out /scratch/$USER/conv-perf [--only alexnet] [-j 8]

Writes results.json, results.md and fig_*.png into --out.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# AlexNetSmall at the inference run's scale (3x3, stride 1, padding 1 folded
# into the input) and one layer whose channels fill the array.
LAYERS = [
    ('conv1', 1, 3, 32, 32, 2),
    ('conv2', 1, 2, 16, 16, 6),
    ('conv3', 1, 6, 8, 8, 12),
    ('conv4', 1, 12, 8, 8, 8),
    ('conv5', 1, 8, 8, 8, 8),
    ('wide64', 1, 64, 8, 8, 64),
]
# Earlier end-to-end run (scalar im2col C kernels, Ramulator, conv + bias).
BASELINE = {'conv1': 4122577, 'conv2': 1240156, 'conv3': 568202, 'conv4': 775443,
            'conv5': 624133}
LAYOUTS = ['conv_2d_nchw_fchw', 'conv_2d_nhwc_fhwc', 'conv_2d_nhwc_hwcf']

CANON = {'I': ('N', 'C', 'S0', 'S1'), 'K': ('F', 'C', 'K0', 'K1'), 'O': ('N', 'F', 'O0', 'O1')}


def bf16(x):
    import torch
    return torch.tensor(np.asarray(x, np.float32)).bfloat16().view(torch.int16).numpy().astype('<i2').tobytes()


def unbf16(raw, shape):
    return (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)


def case_list(only):
    cases = []
    for label, n, c, h, w, f in LAYERS:
        if only and only not in label and not (only == 'alexnet' and label in BASELINE):
            continue
        for name in LAYOUTS:
            cases.append(dict(label=label, name=name, N=n, C=c, H=h, W=w, F=f, K=3))
    if not only or only == 'other':
        cases.append(dict(label='plain64', name='conv_2d', N=1, C=1, H=64, W=64, F=1, K=3))
        cases.append(dict(label='q-conv3', name='conv_2d_nhwc_hwcf_q', N=1, C=6, H=8, W=8, F=12, K=3))
    return cases


def run_case(case, out_root):
    from graph.conv import OPS
    from graph.layout import TensorView
    from graph.conv import ConvOp
    from conv_runtime import execute_conv
    import torch
    name = case['name']
    rank, i_ax, k_ax, o_ax = OPS[name][:4]
    rng = np.random.default_rng(0)
    N, C, H, W, F, K = (case[k] for k in 'NCHWFK')
    q = name.endswith('_q')
    if name == 'conv_2d':
        x = rng.integers(-2, 3, (H + K - 1, W + K - 1)).astype(np.float32)
        wt = rng.integers(-2, 3, (K, K)).astype(np.float32)
        y = rng.integers(-2, 3, (H, W)).astype(np.float32)
        ref = y + torch.nn.functional.conv2d(torch.tensor(x)[None, None].double(),
                                             torch.tensor(wt)[None, None].double())[0, 0].numpy()
        views = (x, wt, y)
        logical = (x, wt, y)
        perms = None
    else:
        x = rng.integers(-1, 2, (N, C, H + K - 1, W + K - 1)).astype(np.float32)
        wt = rng.integers(-1, 2, (F, C, K, K)).astype(np.float32)
        y = rng.integers(-1, 2, (N, F, H, W)).astype(np.float32)
        ref = y + torch.nn.functional.conv2d(torch.tensor(x).double(), torch.tensor(wt).double()).numpy()
        perms = [[CANON[k].index(t) for t in spec.split()] for k, spec in zip('IKO', (i_ax, k_ax, o_ax))]
        logical = tuple(np.ascontiguousarray(np.transpose(a, p)) for a, p in zip((x, wt, y), perms))
    dt, odt = ('i8', 'i32') if q else ('bf16', 'bf16')
    enc = (lambda a, t: a.astype({'i8': '<i1', 'i32': '<i4'}[t]).tobytes()) if q else (lambda a, t: bf16(a))
    zp = dict(input_zero_point=0, filter_zero_point=0) if q else {}
    attrs = {} if name == 'conv_2d' else dict(strides=(1, 1), dilations=(1, 1))
    op = ConvOp(name, TensorView('x', logical[0].shape, dt), TensorView('w', logical[1].shape, dt),
                TensorView('y', logical[2].shape, odt), **attrs, **zp)
    bufs = {'x': enc(logical[0], dt), 'w': enc(logical[1], dt), 'y': enc(logical[2], odt)}
    work = Path(out_root) / f"{case['label']}-{name}"
    r = execute_conv(op, bufs, work, tag='conv', cycle=True, dram='ramulator', limit=200_000_000)
    got = (np.frombuffer(r.buffers['y'], '<i4').astype(np.float32).reshape(logical[2].shape) if q
           else unbf16(r.buffers['y'], logical[2].shape))
    if perms:
        got = np.transpose(got, np.argsort(perms[2]))
    perf = json.loads(Path(r.cycle.perf_json).read_text())
    macs = r.plan.macs
    cycles = r.cycle.cycles
    return dict(case, path=r.plan.path, restaged=[x.dst_buffer if x.dst_buffer in ('input', 'filter', 'output')
                                                  else 'output-back' for x in r.plan.pre + r.plan.post],
                correct=bool(np.array_equal(got, ref)),
                sims_agree=r.cycle_buffers['y'] == r.buffers['y'],
                cycles=cycles, macs=macs, macs_per_cycle=macs / cycles,
                array_utilization=macs / (cycles * 1024),
                native=r.cycle.native, rtl_violations=r.cycle.rtl_violations,
                breakdown=perf.get('cycle_breakdown'), unit_busy={k: v.get('fraction') for k, v in
                                                                   perf.get('unit_busy', {}).items()},
                dram_bytes=perf.get('memory', {}).get('dram_bytes'),
                masters={k: v.get('bytes') for k, v in perf.get('memory', {}).get('masters', {}).items()},
                instruction_mix=perf.get('instruction_mix'),
                functional={k: r.metrics.get(k) for k in (
                    'packets_executed', 'packet_slot_utilization_executed_pct', 'dyn_retired_gemm_systolic',
                    'dyn_retired_sdma', 'dyn_retired_vector_alu', 'dyn_retired_scalar_alu',
                    'dyn_retired_scalar_mem', 'dyn_retired_branch_control', 'dyn_retired_vector_mem')},
                baseline_cycles=BASELINE.get(case['label']))


def report(results, out):
    out = Path(out)
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    lines = ['| layer | op | path | restaged | cycles | MAC/cycle | array util | DRAM KB | vs im2col baseline | ok |',
             '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['label']} | {r['name']} | error | | | | | | | {r['error'][:60]} |")
            continue
        base = r['baseline_cycles']
        lines.append('| {label} | {name} | {path} | {rs} | {cycles:,} | {mpc:.2f} | {util:.2%} | {kb:.0f} | {vs} | {ok} |'.format(
            rs=', '.join(r['restaged']) or '-', mpc=r['macs_per_cycle'], util=r['array_utilization'],
            kb=(r['dram_bytes'] or 0) / 1024, vs=f"{base / r['cycles']:.1f}x faster" if base else '-',
            ok='yes' if r['correct'] and r['sims_agree'] else 'NO', **r))
    (out / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    figures(results, out)


def figures(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    ok = [r for r in results if 'error' not in r]
    layers = [l[0] for l in LAYERS if l[0] in BASELINE and any(r['label'] == l[0] for r in ok)]
    if layers:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.2
        series = [('im2col baseline (scalar)', lambda l: BASELINE[l], '#9aa3ad')] + [
            (name.replace('conv_2d_', ''), (lambda l, n=name: next((r['cycles'] for r in ok
                                            if r['label'] == l and r['name'] == n), 0)), col)
            for name, col in zip(LAYOUTS, ('#c4622d', '#2f6fb3', '#23896a'))]
        for k, (lab, fn, col) in enumerate(series):
            ax.bar([i + (k - 1.5) * width for i in range(len(layers))], [fn(l) for l in layers],
                   width, label=lab, color=col)
        ax.set_yscale('log')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_ylabel('cycles (log)')
        ax.set_title('AlexNetSmall conv layers on atalla-sim (Ramulator)')
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out / 'fig_alexnet_cycles.png', dpi=130)
        plt.close(fig)
    pick = [r for r in ok if r['name'] in LAYOUTS]
    if pick:
        groups = ['issue', 'frontend', 'stall:hazard:sdma', 'stall', 'other']
        def bucket(r):
            b = r['breakdown'] or {}
            tot = sum(b.values()) or 1
            agg = dict.fromkeys(groups, 0)
            for key, val in b.items():
                if key == 'issue': agg['issue'] += val
                elif key.startswith('frontend'): agg['frontend'] += val
                elif key == 'stall:hazard:sdma': agg['stall:hazard:sdma'] += val
                elif key.startswith('stall'): agg['stall'] += val
                else: agg['other'] += val
            return [agg[g] / tot for g in groups]
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(pick) + 1.2))
        left = np.zeros(len(pick))
        cols = ['#2f6fb3', '#c4622d', '#e0a43a', '#9aa3ad', '#d9dde2']
        data = np.array([bucket(r) for r in pick])
        labels = [f"{r['label']} {r['name'].replace('conv_2d_', '')}" for r in pick]
        for j, g in enumerate(groups):
            ax.barh(labels, data[:, j], left=left, color=cols[j], label=g)
            left += data[:, j]
        ax.set_xlabel('fraction of cycles')
        ax.set_title('Where the cycles go')
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        fig.savefig(out / 'fig_cycle_breakdown.png', dpi=130)
        plt.close(fig)


def _guarded(args):
    case, out = args
    try:
        return run_case(case, out)
    except Exception as exc:                     # keep the sweep going; report it
        return dict(case, error=f'{type(exc).__name__}: {exc}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', help="substring of a layer label, 'alexnet' or 'other'")
    ap.add_argument('-j', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--report-only', action='store_true', help='redraw from results.json')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(json.loads((args.out / 'results.json').read_text()), args.out)
        return
    os.environ.setdefault('ATALLA_OBJ_CACHE', str(args.out / 'objcache'))
    cases = case_list(args.only)
    with ProcessPoolExecutor(args.j) as pool:
        results = list(pool.map(_guarded, [(c, args.out) for c in cases]))
    report(results, args.out)


if __name__ == '__main__':
    main()
