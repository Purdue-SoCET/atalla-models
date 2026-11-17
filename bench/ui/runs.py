"""Run manager for the benchmark workbench (API contract section D, POST /api/runs).

One run at a time. A run is `bench/run_benchmark.py` launched as a subprocess in its
own process group with cwd = repo root; the run dir is precomputed
(`<model>__<tag>__<YYYYMMDD-HHMM>` under docs/baselines, same scheme as run_benchmark.py)
and passed via `--run-dir` so the id is known before the process starts.

Progress: run_graph.py appends JSON lines to <run>/progress.jsonl (contract section A)
when it supports `--progress-jsonl`. Until that lands, the manager still works: it maps
global overrides onto the existing env knobs, and `events()` synthesises a `start` event
on launch and a `done`/`error` event from summary.json when the process exits.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = REPO_ROOT / "atalla-graph"
RUN_GRAPH = GRAPH_DIR / "run_graph.py"
BASELINES = REPO_ROOT / "docs" / "baselines"
HARNESS = REPO_ROOT / "bench" / "run_benchmark.py"
DEFAULT_PYTHON = os.environ.get("ATALLA_PYTHON", "/Users/robert.zhang/.venvs/atalla/bin/python")
if not Path(DEFAULT_PYTHON).exists():
    DEFAULT_PYTHON = sys.executable

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")

# contract B "global" keys -> run_graph env knobs (fallback path until --overrides exists)
GLOBAL_ENV = {
    "gelu_kernel": lambda v: {"ATALLA_GELU_KERNEL": v} if v in ("horner", "tanh") else {"ATALLA_GELU_KERNEL": ""},
    "sdpa": lambda v: {"ATALLA_SDPA_FLASH": "1" if v == "flash" else "0"},
    "add_numpy_threshold": lambda v: {},  # no env equivalent; honored by --overrides only
}


class RunError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def run_graph_supports(flag: str) -> bool:
    """True if atalla-graph/run_graph.py mentions the CLI flag (cheap source scan,
    re-read on every call so a newly landed feature is picked up without restart)."""
    try:
        return flag in RUN_GRAPH.read_text(errors="replace")
    except OSError:
        return False


def valid_run_id(run_id: str) -> bool:
    return bool(run_id) and bool(RUN_ID_RE.match(run_id)) and ".." not in run_id


def run_dir_for(run_id: str) -> Path:
    if not valid_run_id(run_id):
        raise RunError(400, f"invalid run id {run_id!r}")
    return BASELINES / run_id


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_progress(run_dir: Path) -> list[dict]:
    p = run_dir / "progress.jsonl"
    out: list[dict] = []
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def parse_started(run_id: str, summary: Optional[dict] = None, config: Optional[dict] = None) -> Optional[str]:
    for d in (summary, config):
        if d and d.get("timestamp_utc"):
            return d["timestamp_utc"]
    m = re.search(r"(\d{8})-(\d{4})$", run_id)
    if m:
        try:
            return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(
                tzinfo=_dt.timezone.utc).isoformat()
        except ValueError:
            return None
    return None


class LiveRun:
    def __init__(self, run_id: str, run_dir: Path, proc: subprocess.Popen, request: dict, cmd: list[str],
                 env_extra: dict, notes: list[str]):
        self.run_id = run_id
        self.run_dir = run_dir
        self.proc = proc
        self.request = request
        self.cmd = cmd
        self.env_extra = env_extra
        self.notes = notes
        self.started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.status = "running"
        self.returncode: Optional[int] = None
        self.finished_at: Optional[float] = None
        self.cancelled = False

    def poll(self) -> str:
        if self.status == "running":
            rc = self.proc.poll()
            if rc is not None:
                self.returncode = rc
                self.finished_at = time.time()
                summary = _read_json(self.run_dir / "summary.json")
                ok = (rc == 0 and summary is not None and summary.get("final_cosine") is not None
                      and summary.get("run_returncode") in (0, None))
                if any(e.get("event") == "error" for e in read_progress(self.run_dir)):
                    ok = False
                self.status = "done" if ok and not self.cancelled else "error"
        return self.status


class RunManager:
    def __init__(self, python: str = DEFAULT_PYTHON, baselines: Path = BASELINES):
        self.python = python
        self.baselines = Path(baselines)
        self._lock = threading.Lock()
        self.live: dict[str, LiveRun] = {}
        self.current: Optional[LiveRun] = None

    # ------------------------------------------------------------------ launch
    def start(self, model: str, scale: Optional[float] = None, seed: Optional[int] = 42,
              tag: str = "ui", strict: bool = False, overrides: Optional[dict] = None) -> str:
        with self._lock:
            if self.current is not None and self.current.poll() == "running":
                raise RunError(409, f"run {self.current.run_id} is still running")
            if not model or not re.match(r"^[a-z0-9_]+$", model):
                raise RunError(400, f"invalid model {model!r}")
            tag = TAG_RE.sub("-", (tag or "ui").strip()).strip("-") or "ui"
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M")
            run_id = f"{model}__{tag}__{stamp}"
            run_dir = self.baselines / run_id
            n = 2
            while run_dir.exists():  # same minute, same tag: suffix the tag
                run_id = f"{model}__{tag}-{n}__{stamp}"
                run_dir = self.baselines / run_id
                n += 1
            run_dir.mkdir(parents=True, exist_ok=False)

            overrides = overrides or {}
            notes: list[str] = []
            extra: list[str] = []
            env_extra: dict[str, str] = {}
            has_overrides = bool(overrides.get("global") or overrides.get("nodes"))
            if has_overrides:
                (run_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))
            if run_graph_supports("--progress-jsonl"):
                extra += ["--progress-jsonl", str(run_dir / "progress.jsonl")]
            else:
                notes.append("run_graph.py has no --progress-jsonl yet; events are synthesised from summary.json")
            if has_overrides:
                if run_graph_supports("--overrides"):
                    extra += ["--overrides", str(run_dir / "overrides.json")]
                else:
                    notes.append("run_graph.py has no --overrides yet; only global overrides applied via env knobs")
                    if overrides.get("nodes"):
                        notes.append("per-node overrides ignored (not supported by this run_graph.py)")
            # global knobs also go through env so they work on either code path
            for k, v in (overrides.get("global") or {}).items():
                fn = GLOBAL_ENV.get(k)
                if fn and v is not None:
                    env_extra.update(fn(v))
            if strict:
                extra.append("--strict-end")

            cmd = [self.python, str(HARNESS), "--model", model, "--tag", tag, "--run-dir", str(run_dir),
                   "--python", self.python]
            if scale is not None:
                cmd += ["--scale", str(scale)]
            if seed is not None:
                cmd += ["--seed", str(int(seed))]
            for k, v in env_extra.items():
                cmd += ["--env", f"{k}={v}"]
            if extra:
                cmd += ["--extra-args=" + " ".join(shlex.quote(x) for x in extra)]

            request = {"model": model, "scale": scale, "seed": seed, "tag": tag, "strict": strict,
                       "overrides": overrides, "run_id": run_id, "command": " ".join(shlex.quote(c) for c in cmd),
                       "notes": notes, "launched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
            (run_dir / "request.json").write_text(json.dumps(request, indent=2))
            launch_log = open(run_dir / "launcher.log", "ab")
            try:
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=launch_log, stderr=subprocess.STDOUT,
                                        env={**os.environ, **env_extra}, start_new_session=True)
            except OSError as e:
                launch_log.close()
                raise RunError(500, f"failed to launch: {e}")
            lr = LiveRun(run_id, run_dir, proc, request, cmd, env_extra, notes)
            self.live[run_id] = lr
            self.current = lr
            return run_id

    # ------------------------------------------------------------------ query
    def status_of(self, run_id: str) -> str:
        lr = self.live.get(run_id)
        if lr is not None:
            return lr.poll()
        run_dir = run_dir_for(run_id)
        summary = _read_json(run_dir / "summary.json")
        if summary is None:
            return "error" if run_dir.exists() else "missing"
        if summary.get("run_returncode") not in (0, None) or summary.get("final_cosine") is None:
            return "error"
        return "done"

    def wait_finished(self, run_id: str, timeout_s: float = 20.0) -> str:
        """Block until a live run's process has exited (or timeout); returns its status."""
        lr = self.live.get(run_id)
        if lr is None:
            return self.status_of(run_id)
        t0 = time.time()
        while lr.poll() == "running" and time.time() - t0 < timeout_s:
            time.sleep(0.1)
        return lr.poll()

    def is_busy(self) -> bool:
        return self.current is not None and self.current.poll() == "running"

    def list_runs(self) -> list[dict]:
        rows: dict[str, dict] = {}
        if self.baselines.exists():
            for d in self.baselines.iterdir():
                if not d.is_dir() or not valid_run_id(d.name):
                    continue
                summary = _read_json(d / "summary.json")
                config = _read_json(d / "config.json")
                if summary is None and d.name not in self.live:
                    if config is None and not (d / "request.json").exists():
                        continue  # not a run dir
                rows[d.name] = self._row(d.name, summary, config)
        for run_id, lr in self.live.items():
            summary = _read_json(lr.run_dir / "summary.json")
            row = self._row(run_id, summary, None)
            row["status"] = lr.poll()
            row["started"] = lr.started
            rows[run_id] = row
        out = list(rows.values())
        out.sort(key=lambda r: (r.get("started") or "", r["run_id"]), reverse=True)
        return out

    def _row(self, run_id: str, summary: Optional[dict], config: Optional[dict]) -> dict:
        s = summary or {}
        model, tag = s.get("model"), s.get("tag")
        if model is None:
            parts = run_id.split("__")
            model = parts[0]
            tag = parts[1] if len(parts) > 2 else tag
        status = self.status_of(run_id) if run_id not in self.live else self.live[run_id].poll()
        return {
            "run_id": run_id,
            "model": model,
            "model_class": s.get("model_class"),
            "tag": tag,
            "status": status,
            "started": parse_started(run_id, summary, config),
            "final_cosine": s.get("final_cosine"),
            "rel_l2": s.get("rel_l2"),
            "rel_max": s.get("rel_max"),
            "strict_pass": s.get("strict_pass"),
            "coverage": {"emulated": s.get("coverage_emulated"), "numpy": s.get("coverage_numpy"),
                         "passthrough": s.get("coverage_passthrough"), "total": s.get("coverage_total_nodes")},
            "sched_packets": s.get("sched_packets"),
            "dyn_packets": s.get("dyn_packets"),
            "retired_instructions": s.get("retired_instructions"),
            "bytes_total": s.get("bytes_total"),
            "flops": s.get("flops"),
            "elapsed_s": s.get("elapsed_s_wall"),
            "flags": s.get("flags"),
        }

    def detail(self, run_id: str) -> dict:
        run_dir = run_dir_for(run_id)
        if not run_dir.exists():
            raise RunError(404, f"run {run_id} not found")
        lr = self.live.get(run_id)
        progress = read_progress(run_dir)
        status = lr.poll() if lr else self.status_of(run_id)
        config = _read_json(run_dir / "config.json")
        if config is None:
            config = _read_json(run_dir / "request.json")
        return {
            "run_id": run_id,
            "status": status,
            "config": config,
            "summary": _read_json(run_dir / "summary.json"),
            "overrides": _read_json(run_dir / "overrides.json"),
            "progress": progress + self._synth_tail(run_id, progress, status),
            "notes": (lr.notes if lr else (config or {}).get("notes")) or [],
        }

    # ------------------------------------------------------------------ events
    def _synth_tail(self, run_id: str, progress: list[dict], status: str) -> list[dict]:
        """Synthesised done/error event when the run finished without emitting one."""
        if status == "running" or any(e.get("event") in ("done", "error") for e in progress):
            return []
        run_dir = run_dir_for(run_id)
        summary = _read_json(run_dir / "summary.json") or {}
        now = time.time()
        if status == "done":
            return [{"event": "done", "final_cosine": summary.get("final_cosine"), "rel_l2": summary.get("rel_l2"),
                     "rel_max": summary.get("rel_max"), "strict_pass": summary.get("strict_pass"),
                     "elapsed_s": summary.get("elapsed_s_run_graph"), "ts": now, "synthesized": True}]
        msg = "run failed"
        lr = self.live.get(run_id)
        if lr and lr.cancelled:
            msg = "cancelled"
        else:
            log = run_dir / "run.log"
            if log.exists():
                tail = [l for l in log.read_text(errors="replace").splitlines() if l.strip()][-8:]
                if tail:
                    msg = tail[-1][:500]
            if lr and lr.returncode is not None:
                msg = f"exit code {lr.returncode}: {msg}"
        return [{"event": "error", "message": msg, "node": None, "ts": now, "synthesized": True}]

    def events(self, run_id: str, poll_s: float = 0.2, max_wait_s: float = 3600.0) -> Iterator[dict]:
        """Replay progress.jsonl then tail it until done/error or the process exits."""
        run_dir = run_dir_for(run_id)
        if not run_dir.exists():
            raise RunError(404, f"run {run_id} not found")
        lr = self.live.get(run_id)
        path = run_dir / "progress.jsonl"
        pos = 0
        buf = ""
        seen_start = False
        t0 = time.time()
        synthesized_start = False
        while True:
            if path.exists():
                with open(path, "r", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    buf += chunk
                    lines = buf.split("\n")
                    buf = lines.pop()  # partial last line
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("event") == "start":
                            seen_start = True
                        if ev.get("event") in ("done", "error"):
                            # run_graph has exited; give run_benchmark.py time to write summary.json
                            # so that a client calling /export right after the stream closes succeeds
                            self.wait_finished(run_id, timeout_s=20.0)
                            ev = dict(ev)
                            ev.setdefault("status", lr.poll() if lr else self.status_of(run_id))
                            yield ev
                            return
                        yield ev
            status = lr.poll() if lr else self.status_of(run_id)
            if status != "running":
                # process finished: flush any remaining partial line, then synthesise a terminal event
                if buf.strip():
                    try:
                        ev = json.loads(buf)
                        buf = ""
                        yield ev
                        if ev.get("event") in ("done", "error"):
                            return
                    except ValueError:
                        pass
                for ev in self._synth_tail(run_id, read_progress(run_dir), status):
                    yield ev
                return
            if (not seen_start and not synthesized_start and lr is not None and not path.exists()
                    and time.time() - t0 > 2.0):
                synthesized_start = True
                yield {"event": "start", "model": lr.request.get("model"), "mode": "validate", "total_nodes": None,
                       "compute_nodes": None, "ts": time.time(), "synthesized": True,
                       "notes": lr.notes}
            if time.time() - t0 > max_wait_s:
                yield {"event": "error", "message": "event stream timed out", "ts": time.time(), "synthesized": True}
                return
            time.sleep(poll_s)

    # ------------------------------------------------------------------ cancel
    def cancel(self, run_id: str) -> bool:
        lr = self.live.get(run_id)
        if lr is None:
            raise RunError(404, f"run {run_id} is not a live run")
        if lr.poll() != "running":
            return False
        lr.cancelled = True
        try:
            os.killpg(os.getpgid(lr.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(25):
            if lr.proc.poll() is not None:
                break
            time.sleep(0.2)
        else:
            try:
                os.killpg(os.getpgid(lr.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        lr.poll()
        lr.status = "error"
        return True

    # ------------------------------------------------------------------ artifacts
    def artifact(self, run_id: str, node: str, kind: str) -> dict:
        run_dir = run_dir_for(run_id)
        if kind not in ("c", "s", "in"):
            raise RunError(400, f"kind must be c|s|in, got {kind!r}")
        if not node or not re.match(r"^[A-Za-z0-9_.-]+$", node):
            raise RunError(400, f"invalid node name {node!r}")
        out_dir = run_dir / "out"
        if not out_dir.exists():
            raise RunError(404, f"run {run_id} has no out/ directory")
        cands = [out_dir / f"{node}.{kind}"] + sorted(out_dir.rglob(f"{node}.{kind}"))
        for p in cands:
            if p.is_file():
                text = p.read_text(errors="replace")
                return {"node": node, "kind": kind, "path": str(p.relative_to(REPO_ROOT)), "text": text,
                        "bytes": p.stat().st_size}
        raise RunError(404, f"no {node}.{kind} under {out_dir.relative_to(REPO_ROOT)}")

    def list_artifacts(self, run_id: str) -> dict:
        run_dir = run_dir_for(run_id)
        out_dir = run_dir / "out"
        nodes: dict[str, list[str]] = {}
        if out_dir.exists():
            for p in out_dir.rglob("*"):
                if p.suffix in (".c", ".s", ".in") and p.is_file():
                    nodes.setdefault(p.stem, []).append(p.suffix[1:])
        return {"run_id": run_id, "nodes": {k: sorted(v) for k, v in sorted(nodes.items())}}


MANAGER = RunManager()
