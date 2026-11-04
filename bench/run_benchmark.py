#!/usr/bin/env python
"""Atalla benchmark harness (Sprint 2/4 deliverable).

Runs atalla-graph/run_graph.py in validate mode for one model, captures the
full log, records provenance (git SHAs of the worktree + submodules, library
versions, host, timestamp, elapsed time) and derives the Sprint-2 metrics
table columns from the metrics.json that run_graph.py emits.

Outputs (under --out-root/<model>__<tag>__<YYYYMMDD-HHMM>/):
    run.log        stdout+stderr of run_graph.py
    config.json    provenance + exact command
    metrics.json   raw run_graph.py --metrics-json output
    layers.csv     raw run_graph.py --layer-metrics-csv output
    summary.json   Sprint-2 columns + per-layer rows (null for missing keys)
    out/           run_graph.py --out-dir artifacts (graph, .c, .s, .in ...)

Example:
    python bench/run_benchmark.py --model basic --tag strict-chained \
        --extra-args="--strict-end"      # note the "=": argparse needs it when the value starts with "--"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "atalla-graph"
DEFAULT_OUT_ROOT = REPO_ROOT / "docs" / "baselines"
DEFAULT_PYTHON = sys.executable

SUBMODULES = {
    "atalla_models": REPO_ROOT,
    "compiler": REPO_ROOT / "aihw-ppci-compiler",
    "functional_sim": REPO_ROOT / "functional_sim",
}

# kernel_metrics[].backend value -> coverage bucket
BACKEND_BUCKET = {
    "emulator": "emulated",
    "emu": "emulated",
    "atalla": "emulated",
    "numpy": "numpy",
    "reference": "numpy",
    "passthrough": "passthrough",
    "pytorch": "passthrough",
    "torch": "passthrough",
    "ref": "passthrough",
}

RETIRED_CLASSES = [
    "branch_control", "sdma", "scalar_mem", "scalar_alu",
    "vector_mem", "vector_alu", "gemm_systolic", "move_convert",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=20)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_provenance() -> dict:
    prov = {}
    for name, path in SUBMODULES.items():
        prov[name] = {
            "path": str(path),
            "sha": _git(path, "rev-parse", "HEAD"),
            "short_sha": _git(path, "rev-parse", "--short", "HEAD"),
            "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git(path, "status", "--porcelain", "--untracked-files=no")),
        }
    return prov


def lib_versions() -> dict:
    v = {"python": platform.python_version(), "python_exe": sys.executable}
    for mod in ("torch", "numpy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", None)
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


def _get(d, *keys, default=None):
    """Nested, null-safe lookup."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_div(a, b):
    try:
        if a is None or b in (None, 0):
            return None
        return a / b
    except Exception:  # noqa: BLE001
        return None


def coverage(metrics: dict) -> dict:
    """Coverage = how many graph nodes ran on each backend."""
    stats = metrics.get("stats") or {}
    km = metrics.get("kernel_metrics") or []
    counted = {"emulated": 0, "numpy": 0, "passthrough": 0, "other": 0}
    seen_backends = {}
    for k in km:
        b = str(k.get("backend", "")).lower()
        seen_backends[b] = seen_backends.get(b, 0) + 1
        counted[BACKEND_BUCKET.get(b, "other")] += 1
    # run_graph.py's own stats block is authoritative (it also counts nodes
    # that never make it into kernel_metrics, e.g. passthrough nodes).
    return {
        "coverage_emulated": stats.get("emulated", counted["emulated"]),
        "coverage_numpy": stats.get("numpy", counted["numpy"]),
        "coverage_passthrough": stats.get("passthrough", counted["passthrough"]),
        "coverage_total_nodes": stats.get("total"),
        "coverage_source": "metrics.stats" if stats else "kernel_metrics.backend",
        "backend_values_seen": seen_backends,
    }


def layer_rows(metrics: dict) -> list[dict]:
    rows = []
    for k in metrics.get("kernel_metrics") or []:
        row = {
            "node": k.get("name"),
            "op": k.get("op"),
            "backend": k.get("backend"),
            "shape": k.get("shape"),
            "elems": k.get("elems"),
            "cos_sim": k.get("cos_sim"),
            "rel_l2": k.get("rel_l2_error"),
            "rel_max": k.get("rel_max_abs_error"),
            "max_abs_error": k.get("max_abs_error"),
            "sched_packets": k.get("sched_packets"),
            "dyn_packets": k.get("packets_executed", k.get("packets")),
            "retired_instructions": k.get("instructions_executed", k.get("instructions")),
            "static_fill": k.get("sched_slot_efficiency"),
            "bytes_loaded": k.get("bytes_loaded"),
            "bytes_written": k.get("bytes_written"),
            "bytes_loaded_sp0": k.get("bytes_loaded_sp0"),
            "bytes_loaded_sp1": k.get("bytes_loaded_sp1"),
            "bytes_stored_sp0": k.get("bytes_stored_sp0"),
            "bytes_stored_sp1": k.get("bytes_stored_sp1"),
            "flops": k.get("flops_total"),
            "flops_matmul": k.get("flops_matmul"),
            "flops_vector": k.get("flops_vector"),
            "flops_scalar": k.get("flops_scalar"),
            "arithmetic_intensity": k.get("arithmetic_intensity"),
            "map_kind": k.get("map_kind"),
        }
        for c in RETIRED_CLASSES:
            row[f"dyn_retired_{c}"] = k.get(f"dyn_retired_{c}")
        rows.append(row)
    return rows


def summarize(metrics: dict, config: dict) -> dict:
    agg = metrics.get("aggregate_metrics") or {}
    e2e = metrics.get("end_to_end") or {}
    dyn_pk = agg.get("total_emu_packets")
    retired = agg.get("total_emu_instructions_retired")
    s = {
        "run_id": config.get("run_id"),
        "model": config.get("model"),
        "model_class": metrics.get("model"),
        "tag": config.get("tag"),
        "scale": config.get("scale"),
        "seed": config.get("seed_effective"),
        "seed_requested": config.get("seed"),
        "validate_inputs": metrics.get("validate_inputs"),
        "flags": (config.get("extra_args") or "") + ("".join(f" {k}={v}" for k, v in (config.get("env") or {}).items())),
        "timestamp_utc": config.get("timestamp_utc"),
        "strict_pass": metrics.get("strict_pass"),
        "strict_fail_reasons": metrics.get("strict_fail_reasons"),
        "run_returncode": config.get("returncode"),
        # accuracy
        "final_cosine": e2e.get("cos_sim"),
        "rel_l2": e2e.get("rel_l2_error"),
        "rel_max": e2e.get("rel_max_abs_error"),
        "max_abs_error": e2e.get("max_abs_error"),
        # packet work
        "sched_packets": agg.get("total_sched_packets"),
        "sched_slots_filled": agg.get("total_sched_slots_filled"),
        "dyn_packets": dyn_pk,
        "retired_instructions": retired,
        "static_fill": agg.get("aggregate_static_slot_efficiency"),
        "dynamic_fill": agg.get("aggregate_dynamic_slot_efficiency"),
        "sched_slot_histogram": agg.get("aggregate_sched_slot_histogram"),
        # external traffic
        "bytes_loaded": agg.get("total_bytes_loaded"),
        "bytes_written": agg.get("total_bytes_written"),
        "bytes_loaded_sp0": agg.get("total_bytes_loaded_sp0"),
        "bytes_loaded_sp1": agg.get("total_bytes_loaded_sp1"),
        "bytes_stored_sp0": agg.get("total_bytes_stored_sp0"),
        "bytes_stored_sp1": agg.get("total_bytes_stored_sp1"),
        "bytes_total": agg.get("total_bytes_memory_traffic"),
        # compute work
        "flops": agg.get("total_flops"),
        "arithmetic_intensity": agg.get("aggregate_arithmetic_intensity"),
        "emulated_layer_count": agg.get("emulated_layer_count"),
        # timing (functional sim has no cycle model)
        "elapsed_s_run_graph": metrics.get("elapsed_s"),
        "elapsed_s_wall": config.get("elapsed_s"),
        "cycles": None,
        "stalls": None,
    }
    s.update(coverage(metrics))
    for c in RETIRED_CLASSES:
        s[f"dyn_retired_{c}"] = agg.get(f"total_dyn_retired_{c}")
    # compute-class FLOPs rolled up from per-layer rows
    km = metrics.get("kernel_metrics") or []
    for cls in ("matmul", "vector", "scalar"):
        vals = [k.get(f"flops_{cls}") for k in km if k.get(f"flops_{cls}") is not None]
        s[f"flops_{cls}"] = sum(vals) if vals else None
    s["flops_per_dyn_packet"] = _safe_div(s["flops"], dyn_pk)
    s["layers"] = layer_rows(metrics)
    s["provenance"] = {
        "atalla_models_sha": _get(config, "git", "atalla_models", "sha"),
        "atalla_models_branch": _get(config, "git", "atalla_models", "branch"),
        "compiler_sha": _get(config, "git", "compiler", "sha"),
        "functional_sim_sha": _get(config, "git", "functional_sim", "sha"),
        "python": _get(config, "versions", "python"),
        "torch": _get(config, "versions", "torch"),
        "numpy": _get(config, "versions", "numpy"),
        "hostname": config.get("hostname"),
    }
    return s


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_cmd(args, run_dir: Path) -> list[str]:
    cmd = [args.python, "run_graph.py", "--model", args.model, "--mode", "validate",
           "--out-dir", str(run_dir / "out"),
           "--metrics-json", str(run_dir / "metrics.json"),
           "--layer-metrics-csv", str(run_dir / "layers.csv")]
    if args.scale is not None:
        cmd += ["--scale", str(args.scale)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.quiet:
        cmd += ["--quiet"]
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="passed to run_graph.py --seed if it accepts it; "
                        "otherwise the run is retried without it and noted.")
    p.add_argument("--tag", default="default")
    p.add_argument("--extra-args", default="",
                   help="extra flags passed verbatim to run_graph.py, e.g. "
                        "'--strict-end --validate-inputs chained'")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="extra environment variable for run_graph.py (repeatable), "
                        "e.g. --env ATALLA_SDPA_FLASH=1; recorded in config.json")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--run-dir", default=None,
                   help="explicit run directory (its basename becomes run_id); "
                        "default: <out-root>/<model>__<tag>__<YYYYMMDD-HHMM>")
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--quiet", action="store_true", help="pass --quiet to run_graph.py")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    stamp = _dt.datetime.now(_dt.timezone.utc)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_id = run_dir.name
    else:
        run_id = f"{args.model}__{args.tag}__{stamp.strftime('%Y%m%d-%H%M')}"
        run_dir = Path(args.out_root) / run_id
    cmd = build_cmd(args, run_dir)
    extra_env = dict(kv.split("=", 1) for kv in args.env if "=" in kv)
    env = {**os.environ, **extra_env}

    if args.dry_run:
        print(f"cwd: {GRAPH_DIR}")
        if extra_env:
            print("env:", " ".join(f"{k}={v}" for k, v in extra_env.items()))
        print("cmd:", " ".join(shlex.quote(c) for c in cmd))
        print(f"run_dir: {run_dir}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    t0 = time.time()
    log_path = run_dir / "run.log"

    def _run(c):
        with open(log_path, "ab") as lf:
            prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in extra_env.items())
            lf.write(("$ " + prefix + " ".join(shlex.quote(x) for x in c) + "\n").encode())
            lf.flush()
            r = subprocess.run(c, cwd=str(GRAPH_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env)
        return r.returncode

    rc = _run(cmd)
    # --seed may not exist yet in run_graph.py: retry without it, record note.
    if rc != 0 and args.seed is not None:
        log_txt = log_path.read_text(errors="replace")
        if "--seed" in log_txt and "unrecognized arguments" in log_txt:
            notes.append("run_graph.py rejected --seed; re-ran without it "
                         "(seed in config is the REQUESTED seed, run_graph.py "
                         "used its internal default seed=42).")
            cmd = [c for i, c in enumerate(cmd)
                   if not (c == "--seed" or (i > 0 and cmd[i - 1] == "--seed"))]
            rc = _run(cmd)
    elapsed = time.time() - t0

    metrics = {}
    mp = run_dir / "metrics.json"
    if mp.exists():
        try:
            metrics = json.loads(mp.read_text())
        except Exception as e:  # noqa: BLE001
            notes.append(f"metrics.json unreadable: {e}")
    else:
        notes.append("metrics.json not produced (run failed?)")

    shapes = [{"node": k.get("name"), "op": k.get("op"), "shape": k.get("shape")}
              for k in metrics.get("kernel_metrics") or []]

    config = {
        "run_id": run_id,
        "model": args.model,
        "model_class": metrics.get("model"),
        "tag": args.tag,
        "scale": args.scale,
        "seed": args.seed,
        "seed_effective": args.seed if not any("rejected --seed" in n for n in notes) else 42,
        "extra_args": args.extra_args,
        "env": extra_env,
        "mode": "validate",
        "validate_inputs": metrics.get("validate_inputs"),
        "command": " ".join(shlex.quote(c) for c in cmd),
        "cwd": str(GRAPH_DIR),
        "returncode": rc,
        "elapsed_s": round(elapsed, 3),
        "timestamp_utc": stamp.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "versions": lib_versions(),
        "git": git_provenance(),
        "shapes": shapes,
        "notes": notes,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    summary = summarize(metrics, config)
    summary["notes"] = notes
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"run_dir: {run_dir}")
    print(f"returncode={rc} elapsed={elapsed:.1f}s strict_pass={summary['strict_pass']} "
          f"final_cosine={summary['final_cosine']} coverage E/N/P="
          f"{summary['coverage_emulated']}/{summary['coverage_numpy']}/{summary['coverage_passthrough']} "
          f"sched_pkts={summary['sched_packets']} dyn_pkts={summary['dyn_packets']} "
          f"retired={summary['retired_instructions']} bytes L/W={summary['bytes_loaded']}/{summary['bytes_written']} "
          f"flops={summary['flops']}")
    for n in notes:
        print("note:", n)
    return rc


if __name__ == "__main__":
    sys.exit(main())
