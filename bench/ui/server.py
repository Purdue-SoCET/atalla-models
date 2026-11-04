"""Atalla benchmark workbench HTTP API (API contract section D).

Start:
    cd /Users/robert.zhang/Documents/GitHub/atalla-models-rz
    /Users/robert.zhang/.venvs/atalla/bin/python -m uvicorn bench.ui.server:app --port 8765
    # or: /Users/robert.zhang/.venvs/atalla/bin/python bench/ui/server.py [--port 8765] [--host 127.0.0.1]

Serves bench/ui/static at / and /static (placeholder page if index.html is missing).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from bench.ui.runs import MANAGER, RunError, run_dir_for  # noqa: E402

GRAPH_DIR = REPO_ROOT / "atalla-graph"
KERNELS_DIR = GRAPH_DIR / "kernels"
STATIC_DIR = REPO_ROOT / "bench" / "ui" / "static"
BASELINES = REPO_ROOT / "docs" / "baselines"

app = FastAPI(title="Atalla benchmark workbench", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# error shape: {"error": "..."}
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def _http_exc(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse({"error": detail}, status_code=exc.status_code)


@app.exception_handler(RunError)
async def _run_exc(_: Request, exc: RunError):
    return JSONResponse({"error": exc.message}, status_code=exc.status)


@app.exception_handler(Exception)
async def _any_exc(_: Request, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-4000:]}, status_code=500)


# --------------------------------------------------------------------------- #
# /api/models
# --------------------------------------------------------------------------- #
OPT_GELU = {"key": "gelu_kernel", "label": "GELU kernel", "choices": ["pytorch", "horner", "tanh"],
            "default": "pytorch", "env": "ATALLA_GELU_KERNEL"}
OPT_SDPA = {"key": "sdpa", "label": "SDPA path", "choices": ["numpy", "flash"], "default": "numpy",
            "env": "ATALLA_SDPA_FLASH"}
OPT_ADD = {"key": "add_numpy_threshold", "label": "Add → NumPy above N elems", "type": "int",
           "default": 1000000, "env": "ATALLA_FFN_ADD_NUMPY"}

MODEL_TABLE: dict[str, dict] = {
    "basic": {"title": "Basic", "description": "BasicModule: linear + relu + residual mul/add smoke model (dim 32). "
                                               "10 emulated nodes, ~7 s.",
              "defaults": {"scale": None, "seed": 42}, "options": [OPT_ADD]},
    "alexnet_small": {"title": "AlexNetSmall", "description": "Scaled-down AlexNet (conv/relu/maxpool stack + classifier). "
                                                             "--scale shrinks channels; 0.01 -> 21 emulated nodes.",
                      "defaults": {"scale": 0.01, "seed": 42}, "options": [OPT_ADD]},
    "layernorm_smoke": {"title": "LayerNorm smoke", "description": "LayerNorm-only smoke model (3 emulated nodes).",
                        "defaults": {"scale": None, "seed": 42}, "options": [OPT_ADD]},
    "vit_micro": {"title": "ViT Micro", "description": "Tiny ViT (B=1, N=32, D=32) with a fused AtallaSdpa leaf; "
                                                       "LayerNorm, Q/K/V matmul, SDPA (NumPy or flash kernel), GELU FFN.",
                  "defaults": {"scale": None, "seed": 42}, "options": [OPT_GELU, OPT_SDPA, OPT_ADD]},
    "gpt2_mlp": {"title": "GPT-2 MLP block", "description": "x + c_proj(gelu(c_fc(ln(x)))) on a 32x32 activation (hidden 128).",
                 "defaults": {"scale": None, "seed": 42}, "options": [OPT_GELU, OPT_ADD]},
    "gpt2_attn": {"title": "GPT-2 attention block", "description": "Causal self-attention block (N=32, D=32) with residual.",
                  "defaults": {"scale": None, "seed": 42}, "options": [OPT_GELU, OPT_ADD]},
    "gpt2_block": {"title": "GPT-2 block", "description": "Full GPT-2 block: LN -> causal attention -> residual -> LN -> MLP(GELU) "
                                                          "-> residual (N=32, D=32, hidden 128). 29 compute nodes, ~12 s.",
                   "defaults": {"scale": None, "seed": 42}, "options": [OPT_GELU, OPT_ADD]},
}


def run_graph_model_choices() -> list[str]:
    """Model names from run_graph.py's argparse `--model` choices (source scan; static fallback)."""
    try:
        src = (GRAPH_DIR / "run_graph.py").read_text(errors="replace")
        m = re.search(r'add_argument\(\s*"--model".*?choices\s*=\s*\[(.*?)\]', src, re.S)
        if m:
            names = re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))
            if names:
                return names
    except OSError:
        pass
    return list(MODEL_TABLE)


@app.get("/api/models")
def api_models():
    out = []
    for name in run_graph_model_choices():
        t = MODEL_TABLE.get(name, {"title": name, "description": "", "defaults": {"scale": None, "seed": 42},
                                   "options": [OPT_ADD]})
        out.append({"name": name, "title": t["title"], "description": t["description"],
                    "defaults": dict(t["defaults"]), "options": [dict(o) for o in t["options"]]})
    return out


# --------------------------------------------------------------------------- #
# /api/graph  (bench/ui/inspect.py from agent "core"; 503 until it exists)
# --------------------------------------------------------------------------- #
class GraphRequest(BaseModel):
    model: str
    scale: Optional[float] = None
    seed: Optional[int] = 42
    overrides: Optional[dict] = None


@app.post("/api/graph")
def api_graph(req: GraphRequest):
    if not (REPO_ROOT / "bench" / "ui" / "inspect.py").exists():
        raise HTTPException(503, "inspect not available yet")
    try:
        import importlib
        mod = importlib.import_module("bench.ui.inspect")
        mod = importlib.reload(mod)  # pick up edits without restarting the server
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"inspect not available yet: {type(e).__name__}: {e}")
    if req.model not in run_graph_model_choices():
        raise HTTPException(400, f"unknown model {req.model!r}")
    scale = req.scale if req.scale is not None else (MODEL_TABLE.get(req.model, {}).get("defaults", {}).get("scale"))
    if scale is None:
        scale = 0.01  # run_graph.py's --scale default
    try:
        return mod.inspect_graph(req.model, scale, req.seed if req.seed is not None else 42, req.overrides)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"inspect_graph failed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# /api/kernels
# --------------------------------------------------------------------------- #
# kernel file stem -> atalla ops it implements (from codegen/c_emitter.py emit_node dispatch)
KERNEL_OPS: dict[str, list[str]] = {
    "gemm": ["conv", "linear", "matmul"],
    "relu": ["relu"],
    "add": ["add"],
    "mul": ["mul"],
    "softmax": ["softmax"],
    "maxpool": ["maxpool"],
    "layernorm": ["layernorm"],
    "gelu": ["gelu"],
    "gelu_tanh": ["gelu"],
    "flash_sdpa_n32d32": ["atalla_sdpa"],
    "sum_dim": ["sum_dim"],
    "mse_loss": ["mse_loss"],
    "mse_loss_backward": ["mse_loss_backward"],
    "threshold_backward": ["threshold_backward"],
    "avgpool": ["adaptive_avg_pool"],
    "batchnorm": ["batchnorm"],
    "upsample": ["upsample"],
    "common": [],
    "kernels": [],
    "__init__": [],
}
KERNEL_NOTES: dict[str, str] = {
    "gemm": "Tiled GEMM generator used by conv (im2col), linear and matmul emitters.",
    "gelu": "gelu.py selects gelu.c (horner) or gelu_tanh.c (tanh) via ATALLA_GELU_KERNEL; default is the PyTorch reference.",
    "gelu_tanh": "7th-degree tanh expansion variant (ATALLA_GELU_KERNEL=tanh).",
    "flash_sdpa_n32d32": "Fused flash attention for B=1, N=32, D=32 (ATALLA_SDPA_FLASH=1); other shapes use the NumPy reference.",
    "common": "Shared constants: ADDR_TABLE=60, TILE=32, SDMA control-word packing.",
    "kernels": "Legacy prototype kernels / header for the standalone test suite.",
    "avgpool": "Static reference; adaptive_avg_pool runs on the NumPy path in validate.",
    "batchnorm": "Static reference kernel; not wired into emit_node yet.",
    "upsample": "Static reference kernel; not wired into emit_node yet.",
}
EMITTED_OPS = {"conv", "linear", "matmul", "relu", "add", "mul", "softmax", "maxpool", "layernorm", "gelu",
               "atalla_sdpa", "sum_dim", "mse_loss", "mse_loss_backward", "threshold_backward"}


def _kernel_entries() -> list[dict]:
    if not KERNELS_DIR.exists():
        return []
    out = []
    for p in sorted(KERNELS_DIR.iterdir()):
        if p.suffix == ".py":
            kind = "generator"
        elif p.suffix in (".c", ".h"):
            kind = "static_c"
        else:
            continue
        if p.stem == "__init__":
            continue
        name = p.stem if p.suffix != ".h" else p.stem + ".h"
        ops = KERNEL_OPS.get(p.stem, [])
        out.append({
            "name": name,
            "file": p.name,
            "path": str(p.relative_to(REPO_ROOT)),
            "kind": kind,
            "ops": ops,
            "wired": any(o in EMITTED_OPS for o in ops),
            "note": KERNEL_NOTES.get(p.stem),
            "bytes": p.stat().st_size,
        })
    return out


@app.get("/api/kernels")
def api_kernels():
    return _kernel_entries()


@app.get("/api/kernels/{name}")
def api_kernel(name: str):
    if not re.match(r"^[A-Za-z0-9_.]+$", name) or ".." in name:
        raise HTTPException(400, "bad kernel name")
    cands = [KERNELS_DIR / name, KERNELS_DIR / f"{name}.py", KERNELS_DIR / f"{name}.c"]
    for p in cands:
        if p.is_file() and p.suffix in (".py", ".c", ".h"):
            return {"name": p.stem if p.suffix != ".h" else p.stem + ".h", "path": str(p.relative_to(REPO_ROOT)),
                    "kind": "generator" if p.suffix == ".py" else "static_c",
                    "ops": KERNEL_OPS.get(p.stem, []), "text": p.read_text(errors="replace")}
    raise HTTPException(404, f"kernel {name!r} not found")


# --------------------------------------------------------------------------- #
# /api/runs
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    model: str
    scale: Optional[float] = None
    seed: Optional[int] = 42
    tag: str = "ui"
    strict: bool = False
    overrides: Optional[dict] = Field(default=None)


@app.post("/api/runs", status_code=201)
def api_runs_create(req: RunRequest):
    if req.model not in run_graph_model_choices():
        raise HTTPException(400, f"unknown model {req.model!r}")
    scale = req.scale
    if scale is None:
        scale = MODEL_TABLE.get(req.model, {}).get("defaults", {}).get("scale")
    run_id = MANAGER.start(model=req.model, scale=scale, seed=req.seed, tag=req.tag, strict=req.strict,
                           overrides=req.overrides)
    lr = MANAGER.live[run_id]
    return {"run_id": run_id, "status": "running", "command": lr.request["command"], "notes": lr.notes}


@app.get("/api/runs")
def api_runs_list():
    return MANAGER.list_runs()


@app.get("/api/runs/busy")
def api_runs_busy():
    cur = MANAGER.current
    return {"busy": MANAGER.is_busy(), "run_id": cur.run_id if cur and cur.poll() == "running" else None}


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str):
    return MANAGER.detail(run_id)


@app.get("/api/runs/{run_id}/events")
def api_run_events(run_id: str):
    run_dir = run_dir_for(run_id)
    if not run_dir.exists():
        raise HTTPException(404, f"run {run_id} not found")

    def gen():
        yield ": connected\n\n"
        try:
            for ev in MANAGER.events(run_id):
                yield f"data: {json.dumps(ev)}\n\n"
        except GeneratorExit:
            raise
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'event': 'error', 'message': f'{type(e).__name__}: {e}', 'synthesized': True})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/log")
def api_run_log(run_id: str, tail: Optional[int] = None):
    run_dir = run_dir_for(run_id)
    p = run_dir / "run.log"
    if not p.exists():
        p2 = run_dir / "launcher.log"
        if p2.exists():
            p = p2
        else:
            raise HTTPException(404, f"no run.log for {run_id}")
    text = p.read_text(errors="replace")
    if tail:
        text = "\n".join(text.splitlines()[-int(tail):])
    return PlainTextResponse(text)


@app.get("/api/runs/{run_id}/artifact")
def api_run_artifact(run_id: str, node: Optional[str] = None, kind: str = "c"):
    if node is None:
        return MANAGER.list_artifacts(run_id)
    return MANAGER.artifact(run_id, node, kind)


@app.post("/api/runs/{run_id}/cancel")
def api_run_cancel(run_id: str):
    killed = MANAGER.cancel(run_id)
    return {"ok": True, "killed": killed, "status": MANAGER.status_of(run_id)}


@app.post("/api/runs/{run_id}/export")
def api_run_export(run_id: str, rebuild: bool = True):
    run_dir = run_dir_for(run_id)
    if not (run_dir / "summary.json").exists() and run_id in MANAGER.live:
        MANAGER.wait_finished(run_id, timeout_s=20.0)  # harness may still be writing summary.json
    if not (run_dir / "summary.json").exists():
        raise HTTPException(409, f"run {run_id} has no summary.json yet (status {MANAGER.status_of(run_id)})")
    from bench import tracker_export
    run_xlsx = tracker_export.export_run(run_dir)
    out: dict[str, Any] = {"run_id": run_id, "run_xlsx": str(run_xlsx.relative_to(REPO_ROOT))}
    if rebuild:
        tracker = tracker_export.rebuild_tracker(tracker_export.list_run_dirs(BASELINES))
        out["tracker_xlsx"] = str(tracker.relative_to(REPO_ROOT))
        out["tracker_updates"] = getattr(tracker_export.rebuild_tracker, "last_log", [])
    return out


@app.get("/api/export/tracker")
def api_export_tracker_download():
    from bench import tracker_export
    p = tracker_export.TRACKER_XLSX
    if not p.exists():
        tracker_export.rebuild_tracker(tracker_export.list_run_dirs(BASELINES))
    return FileResponse(str(p), filename=p.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/runs/{run_id}/export/download")
def api_run_export_download(run_id: str):
    from bench import tracker_export
    run_dir = run_dir_for(run_id)
    if not (run_dir / "summary.json").exists():
        raise HTTPException(404, f"run {run_id} has no summary.json")
    p = tracker_export.export_run(run_dir)
    return FileResponse(str(p), filename=p.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/health")
def api_health():
    return {"ok": True, "repo": str(REPO_ROOT), "busy": MANAGER.is_busy(),
            "inspect_available": (REPO_ROOT / "bench" / "ui" / "inspect.py").exists(),
            "static_index": (STATIC_DIR / "index.html").exists()}


# --------------------------------------------------------------------------- #
# static UI
# --------------------------------------------------------------------------- #
PLACEHOLDER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Atalla workbench</title>
<style>body{background:#000;color:#e7e9ea;font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:48px}
a{color:#1d9bf0}code{background:#16181c;padding:2px 6px;border-radius:4px}h1{font-weight:800;font-size:22px}
li{margin:4px 0}</style></head><body>
<h1>Atalla benchmark workbench</h1>
<p><code>bench/ui/static/index.html</code> is not there yet; the API is up.</p>
<ul>
<li><a href="/api/health">/api/health</a></li>
<li><a href="/api/models">/api/models</a></li>
<li><a href="/api/kernels">/api/kernels</a></li>
<li><a href="/api/runs">/api/runs</a></li>
<li><a href="/docs">/docs</a> (OpenAPI)</li>
</ul></body></html>"""


@app.get("/", include_in_schema=False)
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx), headers={"Cache-Control": "no-cache"})
    return HTMLResponse(PLACEHOLDER)


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main(argv=None) -> int:
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args(argv)
    uvicorn.run("bench.ui.server:app", host=a.host, port=a.port, reload=a.reload, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
