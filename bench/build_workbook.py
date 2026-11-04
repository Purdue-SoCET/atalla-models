#!/usr/bin/env python
"""Collect docs/baselines/*/summary.json into the Atalla benchmarking workbook.

Writes:
    output/Atalla_Benchmarking_Workbook.xlsx   (sheets: Baselines, Per-layer,
                                                Metrics, Provenance)
    docs/baselines/README.md                   (same Baselines table in Markdown)

Usage:
    python bench/build_workbook.py [--baselines DIR] [--xlsx PATH] [--readme PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = REPO_ROOT / "docs" / "baselines"
DEFAULT_XLSX = REPO_ROOT / "output" / "Atalla_Benchmarking_Workbook.xlsx"
DEFAULT_README = DEFAULT_BASELINES / "README.md"

# (column header, summary.json field, number format)
BASELINE_COLS = [
    ("Model / config", "model_config", None),
    ("Strict pass", "strict_pass", None),
    ("Coverage E", "coverage_emulated", "0"),
    ("Coverage N", "coverage_numpy", "0"),
    ("Coverage P", "coverage_passthrough", "0"),
    ("Final cosine", "final_cosine", "0.000000"),
    ("rel-L2", "rel_l2", "0.000E+00"),
    ("rel-max", "rel_max", "0.000E+00"),
    ("Load bytes", "bytes_loaded", "#,##0"),
    ("Write bytes", "bytes_written", "#,##0"),
    ("Load SP0", "bytes_loaded_sp0", "#,##0"),
    ("Load SP1", "bytes_loaded_sp1", "#,##0"),
    ("Store SP0", "bytes_stored_sp0", "#,##0"),
    ("Store SP1", "bytes_stored_sp1", "#,##0"),
    ("FLOPs", "flops", "#,##0"),
    ("FLOPs systolic", "flops_matmul", "#,##0"),
    ("FLOPs vector", "flops_vector", "#,##0"),
    ("FLOPs scalar", "flops_scalar", "#,##0"),
    ("AI (FLOP/B)", "arithmetic_intensity", "0.000"),
    ("Sched pkts", "sched_packets", "#,##0"),
    ("Dyn pkts", "dyn_packets", "#,##0"),
    ("Retired instrs", "retired_instructions", "#,##0"),
    ("Static fill", "static_fill", "0.0%"),
    ("Dynamic fill", "dynamic_fill", "0.0%"),
    ("Ret. branch", "dyn_retired_branch_control", "#,##0"),
    ("Ret. sdma", "dyn_retired_sdma", "#,##0"),
    ("Ret. scalar mem", "dyn_retired_scalar_mem", "#,##0"),
    ("Ret. scalar alu", "dyn_retired_scalar_alu", "#,##0"),
    ("Ret. vector mem", "dyn_retired_vector_mem", "#,##0"),
    ("Ret. vector alu", "dyn_retired_vector_alu", "#,##0"),
    ("Ret. gemm systolic", "dyn_retired_gemm_systolic", "#,##0"),
    ("Ret. move/convert", "dyn_retired_move_convert", "#,##0"),
    ("Emulated layers", "emulated_layer_count", "0"),
    ("Elapsed s (wall)", "elapsed_s_wall", "0.0"),
    ("Cycles", "cycles", None),
    ("Date (UTC)", "timestamp_utc", None),
    ("Seed", "seed", None),
    ("Scale", "scale", None),
    ("Validate inputs", "validate_inputs", None),
    ("Flags", "flags", None),
    ("atalla-models SHA", "prov_atalla_models_sha", None),
    ("compiler SHA", "prov_compiler_sha", None),
    ("sim SHA", "prov_functional_sim_sha", None),
    ("Run id", "run_id", None),
]

LAYER_COLS = [
    ("Run id", "run_id", None), ("Model", "model", None),
    ("Node", "node", None), ("Op", "op", None), ("Backend", "backend", None),
    ("Shape", "shape", None), ("Elems", "elems", "0"), ("Map kind", "map_kind", None),
    ("Cosine", "cos_sim", "0.000000"), ("rel-L2", "rel_l2", "0.000E+00"),
    ("rel-max", "rel_max", "0.000E+00"), ("Max abs err", "max_abs_error", "0.000E+00"),
    ("Sched pkts", "sched_packets", "#,##0"), ("Dyn pkts", "dyn_packets", "#,##0"),
    ("Retired instrs", "retired_instructions", "#,##0"), ("Static fill", "static_fill", "0.0%"),
    ("Load bytes", "bytes_loaded", "#,##0"), ("Write bytes", "bytes_written", "#,##0"),
    ("Load SP0", "bytes_loaded_sp0", "#,##0"), ("Load SP1", "bytes_loaded_sp1", "#,##0"),
    ("Store SP0", "bytes_stored_sp0", "#,##0"), ("Store SP1", "bytes_stored_sp1", "#,##0"),
    ("FLOPs", "flops", "#,##0"), ("FLOPs systolic", "flops_matmul", "#,##0"),
    ("FLOPs vector", "flops_vector", "#,##0"), ("FLOPs scalar", "flops_scalar", "#,##0"),
    ("AI (FLOP/B)", "arithmetic_intensity", "0.000"),
    ("Ret. branch", "dyn_retired_branch_control", "#,##0"),
    ("Ret. sdma", "dyn_retired_sdma", "#,##0"),
    ("Ret. scalar mem", "dyn_retired_scalar_mem", "#,##0"),
    ("Ret. scalar alu", "dyn_retired_scalar_alu", "#,##0"),
    ("Ret. vector mem", "dyn_retired_vector_mem", "#,##0"),
    ("Ret. vector alu", "dyn_retired_vector_alu", "#,##0"),
    ("Ret. gemm systolic", "dyn_retired_gemm_systolic", "#,##0"),
    ("Ret. move/convert", "dyn_retired_move_convert", "#,##0"),
]

# Metric definitions (mirrors docs/benchmark_metrics.md)
METRIC_DEFS = [
    # (metric, functional-sim value / source key, meaning, why it matters)
    ("Output accuracy: final cosine", "end_to_end.cos_sim -> final_cosine",
     "Cosine similarity between the final Atalla (emulated, chained) output and the PyTorch reference output.",
     "Single headline number for 'did the model run correctly on Atalla'. 1.0 = identical direction."),
    ("Output accuracy: rel-L2", "end_to_end.rel_l2_error -> rel_l2",
     "||emu - ref||_2 / ||ref||_2 on the final output.",
     "Scale-aware error; catches magnitude errors cosine hides."),
    ("Output accuracy: rel-max", "end_to_end.rel_max_abs_error -> rel_max",
     "max|emu - ref| / max|ref| on the final output.",
     "Worst-element error; catches single-element corruption (e.g. a bad tile)."),
    ("Per-node accuracy", "kernel_metrics[].cos_sim / rel_l2_error / rel_max_abs_error -> layers[]",
     "Same three metrics per emulated layer. In chained mode the input of layer i is the emulated output of layer i-1, so per-layer error is cumulative.",
     "Localises which kernel introduces error."),
    ("Target coverage E/N/P", "stats.emulated / stats.numpy / stats.passthrough -> coverage_emulated / coverage_numpy / coverage_passthrough",
     "Number of graph nodes executed on the Atalla functional sim (E), in the NumPy fallback (N), or passed through from the PyTorch reference (P).",
     "Says how much of the model actually ran on Atalla. A high cosine with low E means little."),
    ("Scheduled packets", "aggregate_metrics.total_sched_packets -> sched_packets",
     "Static packet count in the compiled program (one VLIW bundle per packet, up to 4 slots).",
     "Code size / static instruction-level parallelism proxy."),
    ("Dynamic packets", "aggregate_metrics.total_emu_packets -> dyn_packets",
     "Packets actually executed by the functional sim, including loop iterations.",
     "Closest available proxy for issue cycles (no stall model)."),
    ("Retired instructions", "aggregate_metrics.total_emu_instructions_retired -> retired_instructions",
     "Individual slot-instructions retired at runtime.",
     "Work actually done; retired/dyn_packets = dynamic slot fill."),
    ("Static fill", "aggregate_metrics.aggregate_static_slot_efficiency -> static_fill",
     "Filled slots / (4 x scheduled packets) in the static program.",
     "How well the scheduler packs the VLIW bundles."),
    ("Dynamic fill", "aggregate_metrics.aggregate_dynamic_slot_efficiency -> dynamic_fill",
     "Retired instructions / (4 x dynamic packets).",
     "Runtime bundle utilisation; low values mean the machine issues mostly 1-wide."),
    ("External traffic: bytes loaded / written", "aggregate_metrics.total_bytes_loaded / total_bytes_written -> bytes_loaded / bytes_written",
     "Bytes moved from/to external memory by DMA into/out of the scratchpads.",
     "DRAM bandwidth demand; denominator of arithmetic intensity."),
    ("SP0 / SP1 split", "aggregate_metrics.total_bytes_loaded_sp0/sp1, total_bytes_stored_sp0/sp1 -> bytes_loaded_sp0/sp1, bytes_stored_sp0/sp1",
     "Same bytes split by destination/source scratchpad.",
     "Shows scratchpad balance (e.g. weights land in SP1, activations in SP0)."),
    ("Compute work: FLOPs by class", "kernel_metrics[].flops_matmul / flops_vector / flops_scalar (summed) -> flops_matmul / flops_vector / flops_scalar; aggregate_metrics.total_flops -> flops",
     "Floating-point operations attributed to the systolic (GEMM), vector and scalar units, derived from the op geometry.",
     "Tells which execution unit a model stresses; basis for the per-class cost model."),
    ("Retired-by-class counts", "aggregate_metrics.total_dyn_retired_<class> -> dyn_retired_<class>",
     "Retired instructions bucketed as branch_control, sdma, scalar_mem, scalar_alu, vector_mem, vector_alu, gemm_systolic, move_convert.",
     "Instruction mix; the O_* terms in the future cost model."),
    ("Arithmetic intensity", "aggregate_metrics.aggregate_arithmetic_intensity -> arithmetic_intensity",
     "total_flops / (bytes_loaded + bytes_written).",
     "Roofline x-axis: memory-bound vs compute-bound."),
    ("Provenance", "config.json (model, scale, seed, flags, shapes, git SHAs, versions, host, timestamp)",
     "Everything needed to reproduce the row.",
     "Rows from different compiler/sim revisions are not comparable without it."),
    ("Timing", "cycles / stalls = null",
     "NOT MEASURED. The functional sim has no cycle/latency/stall model.",
     "Placeholder for when an RTL or cycle-accurate sim exists."),
]


def flatten(summary: dict) -> dict:
    s = dict(summary)
    prov = summary.get("provenance") or {}
    for k, v in prov.items():
        s[f"prov_{k}"] = v
    cfg = f"{summary.get('model')} / {summary.get('tag')}"
    if summary.get("scale") is not None:
        cfg += f" / scale={summary['scale']}"
    s["model_config"] = cfg
    for k in ("strict_fail_reasons", "sched_slot_histogram"):
        if isinstance(s.get(k), (list, dict)):
            s[k] = json.dumps(s[k])
    return s


def load_runs(baselines: Path) -> list[dict]:
    runs = []
    for sp in sorted(baselines.glob("*/summary.json")):
        try:
            summ = json.loads(sp.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {sp}: {e}", file=sys.stderr)
            continue
        cfg = {}
        cp = sp.parent / "config.json"
        if cp.exists():
            try:
                cfg = json.loads(cp.read_text())
            except Exception:  # noqa: BLE001
                pass
        summ.setdefault("run_id", sp.parent.name)
        runs.append({"summary": summ, "config": cfg, "dir": sp.parent})
    runs.sort(key=lambda r: r["summary"].get("timestamp_utc") or "")
    return runs


def _cell(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    return v


HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="DDDDDD")


def write_table(ws, cols, rows):
    ws.append([c[0] for c in cols])
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append([_cell(r.get(c[1])) for c in cols])
    for j, (_, _, fmt) in enumerate(cols, start=1):
        if fmt:
            for i in range(2, ws.max_row + 1):
                ws.cell(row=i, column=j).number_format = fmt
        ws.column_dimensions[get_column_letter(j)].width = max(10, min(40, len(cols[j - 1][0]) + 4))
    ws.freeze_panes = "B2"


def build(baselines: Path, xlsx: Path, readme: Path) -> int:
    runs = load_runs(baselines)
    wb = Workbook()

    # Baselines
    ws = wb.active
    ws.title = "Baselines"
    flat = [flatten(r["summary"]) for r in runs]
    write_table(ws, BASELINE_COLS, flat)

    # Per-layer
    ws = wb.create_sheet("Per-layer")
    lrows = []
    for r in runs:
        for lr in r["summary"].get("layers") or []:
            d = dict(lr)
            d["run_id"] = r["summary"].get("run_id")
            d["model"] = r["summary"].get("model")
            lrows.append(d)
    write_table(ws, LAYER_COLS, lrows)

    # Metrics
    ws = wb.create_sheet("Metrics")
    mcols = [("Metric", "m", None), ("Functional-sim value (metrics.json -> summary.json)", "k", None),
             ("Meaning", "d", None), ("Why it matters", "w", None)]
    write_table(ws, mcols, [{"m": m, "k": k, "d": d, "w": w} for m, k, d, w in METRIC_DEFS])
    for col, wdt in zip("ABCD", (34, 60, 70, 60)):
        ws.column_dimensions[col].width = wdt
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    # Provenance (full config per run, one column per run)
    ws = wb.create_sheet("Provenance")
    keys = ["run_id", "model", "model_class", "tag", "scale", "seed", "seed_effective", "extra_args",
            "mode", "validate_inputs", "command", "cwd", "returncode", "elapsed_s", "timestamp_utc",
            "hostname", "platform", "versions.python", "versions.torch", "versions.numpy",
            "git.atalla_models.sha", "git.atalla_models.branch", "git.atalla_models.dirty",
            "git.compiler.sha", "git.compiler.branch", "git.functional_sim.sha",
            "git.functional_sim.branch", "shapes", "notes"]

    def dig(d, dotted):
        cur = d
        for part in dotted.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    ws.append(["Field"] + [r["summary"].get("run_id") for r in runs])
    for c in ws[1]:
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
    for k in keys:
        ws.append([k] + [_cell(dig(r["config"], k)) for r in runs])
    ws.column_dimensions["A"].width = 30
    for j in range(2, len(runs) + 2):
        ws.column_dimensions[get_column_letter(j)].width = 50
    ws.freeze_panes = "B2"

    xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx)

    # README.md
    md = ["# Atalla benchmark baselines", "",
          f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} by "
          "`bench/build_workbook.py` from `docs/baselines/*/summary.json`. "
          "Workbook: `output/Atalla_Benchmarking_Workbook.xlsx`. "
          "Metric definitions: `docs/benchmark_metrics.md`.", "",
          "All numbers come from the functional simulator (counts only, no cycle timing). "
          "Each run directory holds `run.log`, `config.json` (provenance), `metrics.json`, "
          "`layers.csv`, `summary.json` and the compiled artifacts under `out/`.", ""]
    md_cols = [c for c in BASELINE_COLS if c[1] in (
        "model_config", "strict_pass", "coverage_emulated", "coverage_numpy", "coverage_passthrough",
        "final_cosine", "rel_l2", "rel_max", "bytes_loaded", "bytes_written", "flops",
        "arithmetic_intensity", "sched_packets", "dyn_packets", "retired_instructions",
        "static_fill", "dynamic_fill", "timestamp_utc", "seed", "scale",
        "prov_atalla_models_sha", "prov_compiler_sha", "prov_functional_sim_sha", "run_id")]

    def fmt(v, f):
        if v is None:
            return "n/a"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if f and f.endswith("%"):
                return f"{100 * v:.1f}%"
            if f and "E+" in f:
                return f"{v:.3e}"
            if f == "0.000000":
                return f"{v:.6f}"
            if f == "#,##0":
                return f"{v:,.0f}"
            return f"{v:.3f}"
        if isinstance(v, int) and f == "#,##0":
            return f"{v:,}"
        s = str(v)
        if isinstance(v, str) and len(s) == 40 and all(ch in "0123456789abcdef" for ch in s):
            return s[:10]
        if s.endswith("+00:00"):
            return s[:16].replace("T", " ")
        return s

    md.append("| " + " | ".join(c[0] for c in md_cols) + " |")
    md.append("|" + "|".join("---" for _ in md_cols) + "|")
    for r in flat:
        md.append("| " + " | ".join(fmt(r.get(c[1]), c[2]) for c in md_cols) + " |")
    md += ["", "## Runs", ""]
    for r in runs:
        s, c = r["summary"], r["config"]
        md.append(f"- `{s.get('run_id')}`: `{c.get('command', 'n/a')}` (rc={c.get('returncode')}, "
                  f"{c.get('elapsed_s')} s)" + (f" notes: {'; '.join(c.get('notes'))}" if c.get("notes") else ""))
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("\n".join(md) + "\n")
    print(f"{len(runs)} runs, {len(lrows)} layer rows -> {xlsx}\nREADME -> {readme}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    p.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    p.add_argument("--readme", default=str(DEFAULT_README))
    a = p.parse_args(argv)
    return build(Path(a.baselines), Path(a.xlsx), Path(a.readme))


if __name__ == "__main__":
    sys.exit(main())
