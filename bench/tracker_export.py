#!/usr/bin/env python
"""Export benchmark runs into Robert's "Atalla Benchmarking Tracker" workbook layout.

Template: bench/templates/Atalla_Benchmarking_Tracker.xlsx
(tabs Overview, Current Baseline, Metrics, Cost Model, Model Ops, Kernel Status,
Planned Models, "Run - <name>").

Two entry points (API contract section E):

    export_run(run_dir) -> Path            standalone workbook with one "Run - <tag>" sheet
                                           -> output/runs/<run_id>.xlsx
    rebuild_tracker(run_dirs) -> Path      template copy with Current Baseline + Cost Model
                                           data rows replaced, one "Run - <tag>" sheet per run,
                                           and a few Kernel Status / Model Ops cells updated
                                           with measured facts -> output/Atalla_Benchmarking_Tracker.xlsx

CLI:
    python bench/tracker_export.py --run docs/baselines/<run_dir>      # standalone run workbook
    python bench/tracker_export.py --rebuild                            # all runs in docs/baselines
    python bench/tracker_export.py --rebuild --run <dir>                # both

Everything is read from <run>/summary.json (written by bench/run_benchmark.py),
with <run>/metrics.json as a fallback for per-node rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "bench" / "templates" / "Atalla_Benchmarking_Tracker.xlsx"
BASELINES = REPO_ROOT / "docs" / "baselines"
OUTPUT_DIR = REPO_ROOT / "output"
TRACKER_XLSX = OUTPUT_DIR / "Atalla_Benchmarking_Tracker.xlsx"
RUNS_DIR = OUTPUT_DIR / "runs"

RUN_SHEET_SOURCE = "Run - Basic"     # style source for new run sheets
TEMPLATE_RUN_SHEETS = ("Run - Basic", "Run - AlexNetSmall", "Run - ViT", "Run - ViT Flash")

FILL_PASS = "FFDDEFE3"      # green, as in the template's Current Baseline rows
FILL_WARN = "FFFFF4CC"      # amber (NumPy fallback present)
FILL_FAIL = "FFFCE5E5"      # red (accuracy fail / run error)

MODEL_TITLES = {
    "basic": "Basic model",
    "alexnet_small": "AlexNetSmall",
    "layernorm_smoke": "LayerNorm smoke",
    "vit_micro": "ViT Micro",
    "gpt2_mlp": "GPT-2 MLP block",
    "gpt2_attn": "GPT-2 attention block",
    "gpt2_block": "GPT-2 block",
}

COS_PASS = 0.999


# --------------------------------------------------------------------------- #
# run loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> dict:
    """Return summary.json (+ layers from metrics.json if summary lacks them)."""
    run_dir = Path(run_dir)
    sp = run_dir / "summary.json"
    if not sp.exists():
        raise FileNotFoundError(f"{sp} not found (run not finished?)")
    s = json.loads(sp.read_text())
    s.setdefault("run_id", run_dir.name)
    if not s.get("layers"):
        mp = run_dir / "metrics.json"
        if mp.exists():
            try:
                from bench.run_benchmark import layer_rows  # noqa: WPS433
                s["layers"] = layer_rows(json.loads(mp.read_text()))
            except Exception:  # noqa: BLE001
                s["layers"] = []
    s["_run_dir"] = str(run_dir)
    return s


def list_run_dirs(baselines: Path = BASELINES) -> list[Path]:
    """All run dirs with a summary.json, oldest first (stable for sheet order)."""
    dirs = [p for p in sorted(baselines.iterdir()) if (p / "summary.json").exists()] if baselines.exists() else []
    dirs.sort(key=lambda p: (_stamp_of(p.name), p.name))
    return dirs


def _stamp_of(run_id: str) -> str:
    m = re.search(r"(\d{8}-\d{4})$", run_id)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# derived labels
# --------------------------------------------------------------------------- #
def _flags(s: dict) -> str:
    return (s.get("flags") or "").strip()


def run_label(s: dict) -> str:
    """'GPT-2 block — chained; ATALLA_GELU_KERNEL=tanh' style row label."""
    title = MODEL_TITLES.get(s.get("model"), s.get("model_class") or s.get("model") or "?")
    parts = []
    if s.get("scale") is not None and s.get("model") == "alexnet_small":
        parts.append(f"scale {s['scale']}")
    if s.get("tag"):
        parts.append(str(s["tag"]))
    fl = _flags(s)
    if fl:
        parts.append(fl)
    return f"{title} — {'; '.join(parts)}" if parts else title


def coverage_str(s: dict) -> str:
    return f"{s.get('coverage_emulated')} / {s.get('coverage_numpy')} / {s.get('coverage_passthrough')}"


def verdict(s: dict) -> str:
    """pass | warn | fail from measured values."""
    cos = s.get("final_cosine")
    if s.get("run_returncode") not in (0, None) or cos is None or cos < COS_PASS:
        return "fail"
    if s.get("strict_pass") is False:
        return "fail"
    if (s.get("coverage_numpy") or 0) > 0:
        return "warn"
    return "pass"


def run_note(s: dict) -> str:
    v = verdict(s)
    n = s.get("coverage_numpy") or 0
    if v == "fail":
        if s.get("run_returncode") not in (0, None):
            return f"Run failed (returncode {s.get('run_returncode')}). Do not use for comparison."
        return "Fails accuracy gate. Do not use for optimization comparison."
    if v == "warn":
        return f"Coverage/correctness baseline. {n} node{'s' if n != 1 else ''} use NumPy reference."
    strict = "Strict chained" if s.get("strict_pass") else "Chained"
    return f"{strict} baseline. Zero NumPy fallback."


def use_now(s: dict) -> str:
    v = verdict(s)
    return {"pass": "Clean baseline", "warn": "Coverage/correctness only",
            "fail": "Do not fit until accuracy passes"}[v]


def _fill(v: str) -> PatternFill:
    rgb = {"pass": FILL_PASS, "warn": FILL_WARN, "fail": FILL_FAIL}[v]
    return PatternFill(fill_type="solid", fgColor=rgb, bgColor=rgb)


def sheet_title_for(s: dict, existing: set[str]) -> str:
    """'Run - <model> <tag>', shortened to Excel's 31-char limit."""
    model = str(s.get("model") or "")
    tag = str(s.get("tag") or "run")
    base = f"Run - {model} {tag}".strip()
    if len(base) > 31:
        tag = tag.replace("-chained", "").replace("chained", "") or "run"
        base = f"Run - {model} {tag}".strip()
    base = re.sub(r"[\[\]:*?/\\]", "-", base)[:31]
    title, i = base, 2
    while title in existing:
        suffix = f" ({i})"
        title = base[: 31 - len(suffix)] + suffix
        i += 1
    return title


# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #
class _Style:
    """Immutable snapshot of a cell's style (cells share a mutable style array, so
    copy.copy(cell) is NOT a safe snapshot)."""

    def __init__(self, cell):
        self.has_style = cell.has_style
        self.font = copy.copy(cell.font)
        self.fill = copy.copy(cell.fill)
        self.border = copy.copy(cell.border)
        self.alignment = copy.copy(cell.alignment)
        self.number_format = cell.number_format
        self.protection = copy.copy(cell.protection)
        self.value = cell.value


def _copy_style(src, dst):
    if not isinstance(src, _Style):
        src = _Style(src)
    if src.has_style:
        dst.font = copy.copy(src.font)
        dst.fill = copy.copy(src.fill)
        dst.border = copy.copy(src.border)
        dst.alignment = copy.copy(src.alignment)
        dst.number_format = src.number_format
        dst.protection = copy.copy(src.protection)


def _num(v):
    """None for missing; ints for whole floats so '#,##0' renders cleanly."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    return v


def _shape(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


# --------------------------------------------------------------------------- #
# Run - <tag> sheet
# --------------------------------------------------------------------------- #
RUN_FIELDS = [  # (row, label, key) -> column A/B, rows 4..15
    (4, "Model", "model_class"),
    (5, "Input mode", "validate_inputs"),
    (6, "Nodes (total / emulated / NumPy / pass-through)", None),
    (7, "Final cosine", "final_cosine"),
    (8, "Final rel-L2", "rel_l2"),
    (9, "Final rel-max", "rel_max"),
    (10, "Run note", None),
    (11, "Elapsed time (s)", "elapsed_s_run_graph"),
    (12, "Run id", "run_id"),
    (13, "Seed", "seed"),
    (14, "Flags / env", None),
    (15, "Revision", None),
]
RUN_AGG = [  # (row, label, key) -> column D/E
    (4, "Scheduled packets", "sched_packets"),
    (5, "Dynamic packets", "dyn_packets"),
    (6, "Retired instructions", "retired_instructions"),
    (7, "Bytes loaded", "bytes_loaded"),
    (8, "Bytes written", "bytes_written"),
    (9, "Total traffic bytes", "bytes_total"),
    (10, "Total FLOPs", "flops"),
    (11, "Arithmetic intensity", "arithmetic_intensity"),
    (12, "Static slot fill", "static_fill"),
    (13, "Dynamic slot fill", "dynamic_fill"),
    (14, "Retired systolic", "dyn_retired_gemm_systolic"),
    (15, "Retired vector ALU", "dyn_retired_vector_alu"),
]
KERNEL_COLS = [  # header row 18, data from 19
    ("Name", "node", None), ("Op", "op", None), ("Backend", "backend", None),
    ("Shape", "shape", "shape"), ("Sched. pkts", "sched_packets", "#,##0"),
    ("Dyn. pkts", "dyn_packets", "#,##0"), ("Retired", "retired_instructions", "#,##0"),
    ("Load bytes", "bytes_loaded", "#,##0"), ("Write bytes", "bytes_written", "#,##0"),
    ("FLOPs", "flops", "#,##0"), ("Cosine", "cos_sim", "0.000000"),
    ("Rel-L2", "rel_l2", "0.000000"), ("Rel-max", "rel_max", "0.000000"),
]
KERNEL_HEADER_ROW = 18
KERNEL_FIRST_ROW = 19


def fill_run_sheet(ws, s: dict) -> None:
    """ws is a copy of the template's run sheet; overwrite it with run `s`."""
    label_style = _Style(ws["A11"])
    value_style = _Style(ws["B11"])
    agg_label_style = _Style(ws["D11"])
    agg_value_style = _Style(ws["E11"])
    data_styles = [_Style(ws.cell(row=KERNEL_FIRST_ROW, column=c + 1)) for c in range(len(KERNEL_COLS))]

    title = f"Run Detail — {s.get('model_class') or s.get('model')} ({s.get('tag')})"
    ws["A1"] = title

    prov = s.get("provenance") or {}
    rev = prov.get("atalla_models_sha")
    rev_s = (f"{rev[:8]} ({prov.get('atalla_models_branch')})" if rev else None)
    if prov.get("compiler_sha"):
        rev_s = f"{rev_s or ''}; compiler {prov['compiler_sha'][:8]}; sim {(prov.get('functional_sim_sha') or '')[:8]}"
    field_values = {
        6: f"{s.get('coverage_total_nodes')} / {s.get('coverage_emulated')} / {s.get('coverage_numpy')} / {s.get('coverage_passthrough')}",
        10: run_note(s),
        14: _flags(s) or "(none)",
        15: rev_s,
    }
    for row, label, key in RUN_FIELDS:
        a, b = ws.cell(row=row, column=1), ws.cell(row=row, column=2)
        a.value = label
        b.value = _num(field_values[row]) if row in field_values else _num(s.get(key))
        _copy_style(label_style, a)
        _copy_style(value_style, b)
        b.number_format = {7: "0.000000", 8: "0.000000", 9: "0.000000", 11: "0.000"}.get(row, "General")
    for row, label, key in RUN_AGG:
        d, e = ws.cell(row=row, column=4), ws.cell(row=row, column=5)
        d.value = label
        e.value = _num(s.get(key))
        _copy_style(agg_label_style, d)
        _copy_style(agg_value_style, e)
        e.number_format = {11: "0.000", 12: "0.00%", 13: "0.00%"}.get(row, "#,##0")

    # kernel table
    for c, (hdr, _, _) in enumerate(KERNEL_COLS, start=1):
        ws.cell(row=KERNEL_HEADER_ROW, column=c).value = hdr
    for r in range(KERNEL_FIRST_ROW, max(ws.max_row, KERNEL_FIRST_ROW) + 1):
        for c in range(1, len(KERNEL_COLS) + 1):
            ws.cell(row=r, column=c).value = None
    for i, layer in enumerate(s.get("layers") or []):
        r = KERNEL_FIRST_ROW + i
        for c, (_, key, nf) in enumerate(KERNEL_COLS, start=1):
            cell = ws.cell(row=r, column=c)
            v = layer.get(key)
            cell.value = _shape(v) if nf == "shape" else _num(v)
            _copy_style(data_styles[c - 1], cell)
            if nf and nf != "shape":
                cell.number_format = nf
        ws.row_dimensions[r].height = 15.75
    ws.freeze_panes = ws.freeze_panes or "E19"


# --------------------------------------------------------------------------- #
# Current Baseline / Cost Model tables
# --------------------------------------------------------------------------- #
BASELINE_HEADER_ROW = 4
BASELINE_FIRST_ROW = 5
BASELINE_NCOLS = 14


def fill_current_baseline(ws, runs: list[dict]) -> None:
    """Replace the data rows (5..) with one row per run; keep the note row below."""
    # locate the note row (merged A..N below the data) and capture its style/value
    note_row = None
    for rng in list(ws.merged_cells.ranges):
        if rng.min_col == 1 and rng.max_col == BASELINE_NCOLS and rng.min_row > BASELINE_HEADER_ROW:
            note_row = rng.min_row
            ws.unmerge_cells(str(rng))
    note_cell = _Style(ws.cell(row=note_row, column=1)) if note_row else None
    note_value = note_cell.value if note_cell else None
    row_style = [_Style(ws.cell(row=BASELINE_FIRST_ROW, column=c)) for c in range(1, BASELINE_NCOLS + 1)]
    note_height = ws.row_dimensions[note_row].height if note_row else None

    last = max(ws.max_row, BASELINE_FIRST_ROW)
    for r in range(BASELINE_FIRST_ROW, last + 1):
        for c in range(1, BASELINE_NCOLS + 1):
            cell = ws.cell(row=r, column=c)
            cell.value = None
            cell.fill = PatternFill(fill_type=None)
        ws.row_dimensions[r].height = None

    for i, s in enumerate(runs):
        r = BASELINE_FIRST_ROW + i
        vals = [run_label(s), coverage_str(s), s.get("final_cosine"), s.get("rel_l2"), s.get("rel_max"),
                s.get("bytes_loaded"), s.get("bytes_written"), s.get("flops"), s.get("arithmetic_intensity"),
                s.get("sched_packets"), s.get("dyn_packets"), s.get("retired_instructions"),
                s.get("static_fill"), s.get("dynamic_fill")]
        fill = _fill(verdict(s))
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c)
            cell.value = _num(v)
            _copy_style(row_style[c - 1], cell)
            cell.fill = copy.copy(fill)
    end = BASELINE_FIRST_ROW + len(runs)
    # clear leftover styled rows between data and note
    for r in range(end, end + 2):
        for c in range(1, BASELINE_NCOLS + 1):
            ws.cell(row=r, column=c).fill = PatternFill(fill_type=None)
            ws.cell(row=r, column=c).border = openpyxl.styles.Border()
    if note_cell is not None:
        nr = end + 1
        cell = ws.cell(row=nr, column=1)
        cell.value = note_value
        _copy_style(note_cell, cell)
        for c in range(2, BASELINE_NCOLS + 1):
            _copy_style(note_cell, ws.cell(row=nr, column=c))
        ws.merge_cells(start_row=nr, start_column=1, end_row=nr, end_column=BASELINE_NCOLS)
        if note_height:
            ws.row_dimensions[nr].height = note_height


COST_HEADER_ROW = 17
COST_FIRST_ROW = 18
COST_NCOLS = 9


def fill_cost_model(ws, runs: list[dict]) -> None:
    """Replace the 'Current feature values' rows (18..) with one row per run."""
    row_style = [_Style(ws.cell(row=COST_FIRST_ROW, column=c)) for c in range(1, COST_NCOLS + 1)]
    last = max(ws.max_row, COST_FIRST_ROW)
    for r in range(COST_FIRST_ROW, last + 1):
        for c in range(1, COST_NCOLS + 1):
            cell = ws.cell(row=r, column=c)
            cell.value = None
            cell.fill = PatternFill(fill_type=None)
            cell.border = openpyxl.styles.Border()
    for i, s in enumerate(runs):
        r = COST_FIRST_ROW + i
        vals = [run_label(s), s.get("bytes_total"), s.get("dyn_packets"), s.get("dyn_retired_gemm_systolic"),
                s.get("dyn_retired_vector_alu"), s.get("dyn_retired_scalar_alu"), s.get("dyn_retired_scalar_mem"),
                s.get("dyn_retired_branch_control"), use_now(s)]
        v = verdict(s)
        for c, val in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c)
            cell.value = _num(val)
            _copy_style(row_style[c - 1], cell)
            cell.fill = copy.copy(_fill(v)) if v == "fail" else PatternFill(fill_type=None)


# --------------------------------------------------------------------------- #
# Kernel Status / Model Ops: measured-fact updates only
# --------------------------------------------------------------------------- #
def _find_row(ws, col: int, text: str, extra: tuple[int, str] | None = None) -> int | None:
    for r in range(1, ws.max_row + 1):
        v = ws.cell(row=r, column=col).value
        if isinstance(v, str) and v.strip().lower() == text.lower():
            if extra is None:
                return r
            ev = ws.cell(row=r, column=extra[0]).value
            if isinstance(ev, str) and ev.strip().lower() == extra[1].lower():
                return r
    return None


def _evidence(s: dict) -> str:
    return (f"{s['run_id']}: final cos {s.get('final_cosine'):.6f}, rel-L2 {s.get('rel_l2'):.6f}; "
            f"E/N/P {coverage_str(s)}")


def _best(runs: list[dict], pred) -> dict | None:
    """Newest passing run matching pred (fall back to newest matching run)."""
    cands = [s for s in runs if pred(s)]
    if not cands:
        return None
    cands.sort(key=lambda s: _stamp_of(s.get("run_id", "")), reverse=True)
    for s in cands:
        if verdict(s) != "fail":
            return s
    return cands[0]


def update_status_sheets(wb, runs: list[dict]) -> list[str]:
    """Overwrite specific Kernel Status / Model Ops cells from measured runs. Returns a change log."""
    log: list[str] = []
    ks = wb["Kernel Status"] if "Kernel Status" in wb.sheetnames else None
    mo = wb["Model Ops"] if "Model Ops" in wb.sheetnames else None

    def put(ws, r, c, v, what):
        if ws is None or r is None:
            return
        old = ws.cell(row=r, column=c).value
        ws.cell(row=r, column=c).value = v
        log.append(f"{ws.title}!{get_column_letter(c)}{r} ({what}): {old!r} -> {v!r}")

    passes = lambda s: verdict(s) != "fail"  # noqa: E731

    # Flash SDPA (vit_micro + ATALLA_SDPA_FLASH=1)
    flash = _best(runs, lambda s: s.get("model") == "vit_micro" and "ATALLA_SDPA_FLASH=1" in _flags(s))
    if flash and ks is not None:
        r = _find_row(ks, 1, "Flash SDPA")
        put(ks, r, 4, "Verified" if passes(flash) else "Failing", "Status")
        put(ks, r, 5, "B=1, N=32, D=32; " + _evidence(flash), "Evidence")
    if flash and mo is not None:
        r = _find_row(mo, 3, "flash SDPA")
        put(mo, r, 5, "Passes" if passes(flash) else "Fails accuracy", "Current status")
        put(mo, r, 6, "B=1, N=32, D=32; " + _evidence(flash), "Evidence")

    # GELU on the emulator (ATALLA_GELU_KERNEL=horner|tanh)
    gelu = _best(runs, lambda s: "ATALLA_GELU_KERNEL=" in _flags(s))
    if gelu and ks is not None:
        r = _find_row(ks, 1, "GELU")
        put(ks, r, 3, "AtallaC + functional sim (ATALLA_GELU_KERNEL)", "Current backend")
        put(ks, r, 4, "Verified" if passes(gelu) else "Failing", "Status")
        m = re.search(r"ATALLA_GELU_KERNEL=(\w+)", _flags(gelu))
        put(ks, r, 5, f"{m.group(1) if m else 'kernel'} variant; " + _evidence(gelu), "Evidence")

    # Add: a run with add nodes and zero NumPy fallback
    add = _best(runs, lambda s: (s.get("coverage_numpy") == 0
                                 and any(l.get("op") == "add" and l.get("backend") == "emulator"
                                         for l in s.get("layers") or [])
                                 and s.get("model") != "basic"))
    if add and ks is not None:
        r = _find_row(ks, 1, "Add")
        put(ks, r, 4, "Verified" if passes(add) else "Failing", "Status")
        put(ks, r, 5, "no NumPy fallback; " + _evidence(add), "Evidence")

    # GPT-2 small block rows in Model Ops
    mlp = _best(runs, lambda s: s.get("model") in ("gpt2_mlp", "gpt2_block"))
    if mlp and mo is not None:
        r = _find_row(mo, 1, "GPT-2 small block", extra=(2, "1"))
        put(mo, r, 5, "Baseline passes" if passes(mlp) else "Fails accuracy", "Current status")
        put(mo, r, 6, "N=32, D=32, hidden 128; " + _evidence(mlp), "Evidence")
    attn = _best(runs, lambda s: s.get("model") in ("gpt2_attn", "gpt2_block"))
    if attn and mo is not None:
        r = _find_row(mo, 1, "GPT-2 small block", extra=(2, "2"))
        put(mo, r, 5, "Baseline passes" if passes(attn) else "Fails accuracy", "Current status")
        put(mo, r, 6, "N=32, D=32 causal block; " + _evidence(attn), "Evidence")
    return log


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _load_template():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"template missing: {TEMPLATE}")
    return openpyxl.load_workbook(TEMPLATE)


def export_run(run_dir: Path | str, out_path: Path | str | None = None) -> Path:
    """Standalone workbook with one 'Run - <tag>' sheet for this run."""
    run_dir = Path(run_dir)
    s = load_run(run_dir)
    wb = _load_template()
    src = wb[RUN_SHEET_SOURCE]
    ws = wb.copy_worksheet(src)
    ws.title = sheet_title_for(s, set(wb.sheetnames))
    fill_run_sheet(ws, s)
    for name in list(wb.sheetnames):
        if name != ws.title:
            del wb[name]
    out = Path(out_path) if out_path else RUNS_DIR / f"{s['run_id']}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def rebuild_tracker(run_dirs: list[Path | str] | None = None, out_path: Path | str | None = None) -> Path:
    """Template copy with all runs merged in (see module docstring)."""
    dirs = [Path(d) for d in run_dirs] if run_dirs is not None else list_run_dirs()
    runs = []
    for d in dirs:
        try:
            runs.append(load_run(d))
        except FileNotFoundError:
            continue
    runs.sort(key=lambda s: (_stamp_of(s.get("run_id", "")), s.get("run_id", "")))
    wb = _load_template()
    src = wb[RUN_SHEET_SOURCE]
    # style source must survive until all copies are made
    titles = set(n for n in wb.sheetnames if n not in TEMPLATE_RUN_SHEETS)
    new_sheets = []
    for s in runs:
        ws = wb.copy_worksheet(src)
        ws.title = sheet_title_for(s, titles)
        titles.add(ws.title)
        fill_run_sheet(ws, s)
        new_sheets.append(ws.title)
    for name in TEMPLATE_RUN_SHEETS:
        if name in wb.sheetnames:
            del wb[name]
    if "Current Baseline" in wb.sheetnames:
        fill_current_baseline(wb["Current Baseline"], runs)
    if "Cost Model" in wb.sheetnames:
        fill_cost_model(wb["Cost Model"], runs)
    log = update_status_sheets(wb, runs)
    out = Path(out_path) if out_path else TRACKER_XLSX
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    rebuild_tracker.last_log = log  # type: ignore[attr-defined]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="RUN_DIR",
                   help="export a standalone workbook for this run dir (repeatable)")
    p.add_argument("--rebuild", action="store_true", help="rebuild output/Atalla_Benchmarking_Tracker.xlsx from docs/baselines")
    p.add_argument("--baselines", default=str(BASELINES))
    p.add_argument("--out", default=None, help="output path override (single --run or --rebuild)")
    a = p.parse_args(argv)
    if not a.run and not a.rebuild:
        p.error("give --run RUN_DIR and/or --rebuild")
    for rd in a.run:
        out = export_run(rd, a.out if (a.out and not a.rebuild and len(a.run) == 1) else None)
        print(f"run workbook: {out}")
    if a.rebuild:
        out = rebuild_tracker(list_run_dirs(Path(a.baselines)), a.out if a.out else None)
        print(f"tracker: {out}")
        for line in getattr(rebuild_tracker, "last_log", []):
            print("  updated", line)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    sys.exit(main())
