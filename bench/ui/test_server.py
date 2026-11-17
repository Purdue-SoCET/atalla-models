"""pytest for bench/ui/server.py against the existing docs/baselines runs (no emulator run).

    /Users/robert.zhang/.venvs/atalla/bin/python -m pytest bench/ui/test_server.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from bench.ui import server  # noqa: E402

client = TestClient(server.app)
BASELINES = REPO_ROOT / "docs" / "baselines"


def _existing_runs() -> list[str]:
    return sorted(p.name for p in BASELINES.iterdir() if (p / "summary.json").exists()) if BASELINES.exists() else []


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    names = {m["name"] for m in models}
    assert {"basic", "gpt2_block", "vit_micro"} <= names
    for m in models:
        assert set(m) >= {"name", "title", "description", "defaults", "options"}
        assert "seed" in m["defaults"] and "scale" in m["defaults"]
        keys = [o["key"] for o in m["options"]]
        assert "add_numpy_threshold" in keys
        if m["name"] in ("vit_micro", "gpt2_block", "gpt2_mlp"):
            assert "gelu_kernel" in keys
        if m["name"] == "vit_micro":
            assert "sdpa" in keys
    alex = next(m for m in models if m["name"] == "alexnet_small")
    assert alex["defaults"]["scale"] == 0.01


def test_kernels():
    r = client.get("/api/kernels")
    assert r.status_code == 200
    ks = r.json()
    names = {k["name"] for k in ks}
    assert "gemm" in names and "flash_sdpa_n32d32" in names
    gemm = next(k for k in ks if k["name"] == "gemm")
    assert gemm["kind"] == "generator" and "matmul" in gemm["ops"]
    assert gemm["path"] == "atalla-graph/kernels/gemm.py"
    r = client.get("/api/kernels/gemm")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "gemm" and "def gemm_c" in body["text"]
    r = client.get("/api/kernels/flash_sdpa_n32d32")
    assert r.status_code == 200 and r.json()["kind"] == "static_c"
    r = client.get("/api/kernels/does_not_exist")
    assert r.status_code == 404 and "error" in r.json()


def test_runs_list():
    r = client.get("/api/runs")
    assert r.status_code == 200
    rows = r.json()
    existing = _existing_runs()
    ids = [x["run_id"] for x in rows]
    for e in existing:
        assert e in ids
    for row in rows:
        assert set(row) >= {"run_id", "model", "tag", "status", "started", "final_cosine", "coverage",
                            "dyn_packets", "bytes_total"}
        assert row["status"] in ("running", "done", "error")
        assert set(row["coverage"]) >= {"emulated", "numpy", "passthrough"}
    # newest first
    starts = [row["started"] or "" for row in rows]
    assert starts == sorted(starts, reverse=True)


@pytest.mark.skipif(not _existing_runs(), reason="no baseline runs")
def test_run_detail_log_artifact():
    run_id = next(r for r in _existing_runs() if r.startswith("basic__"))
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "done" and d["summary"]["final_cosine"] is not None
    assert d["config"]["model"] == "basic"
    assert isinstance(d["progress"], list)
    # a finished run without progress.jsonl still yields a terminal event
    assert d["progress"] and d["progress"][-1]["event"] in ("done", "error")

    r = client.get(f"/api/runs/{run_id}/log")
    assert r.status_code == 200 and "run_graph.py" in r.text

    r = client.get(f"/api/runs/{run_id}/artifact", params={"node": "matmul", "kind": "c"})
    assert r.status_code == 200
    a = r.json()
    assert a["node"] == "matmul" and a["kind"] == "c" and a["path"].endswith("out/matmul.c") and len(a["text"]) > 50
    r = client.get(f"/api/runs/{run_id}/artifact", params={"node": "matmul", "kind": "s"})
    assert r.status_code == 200
    r = client.get(f"/api/runs/{run_id}/artifact", params={"node": "nope", "kind": "c"})
    assert r.status_code == 404 and "error" in r.json()
    r = client.get(f"/api/runs/{run_id}/artifact")
    assert r.status_code == 200 and "matmul" in r.json()["nodes"]


@pytest.mark.skipif(not _existing_runs(), reason="no baseline runs")
def test_events_replay_for_finished_run():
    run_id = _existing_runs()[0]
    with client.stream("GET", f"/api/runs/{run_id}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "data: " in body
    last = [l for l in body.splitlines() if l.startswith("data: ")][-1]
    assert '"event": "done"' in last or '"event": "error"' in last


@pytest.mark.skipif(not _existing_runs(), reason="no baseline runs")
def test_export(tmp_path):
    import openpyxl
    run_id = next(r for r in _existing_runs() if r.startswith("basic__"))
    r = client.post(f"/api/runs/{run_id}/export")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_xlsx"] == f"output/runs/{run_id}.xlsx"
    assert body["tracker_xlsx"] == "output/Atalla_Benchmarking_Tracker.xlsx"
    run_wb = openpyxl.load_workbook(REPO_ROOT / body["run_xlsx"])
    assert len(run_wb.sheetnames) == 1 and run_wb.sheetnames[0].startswith("Run - ")
    ws = run_wb.active
    assert ws["A3"].value == "Field" and ws["A18"].value == "Name" and ws["M18"].value == "Rel-max"
    assert ws["A19"].value == "matmul" and ws["E19"].value is not None
    trk = openpyxl.load_workbook(REPO_ROOT / body["tracker_xlsx"])
    for tab in ("Overview", "Current Baseline", "Metrics", "Cost Model", "Model Ops", "Kernel Status", "Planned Models"):
        assert tab in trk.sheetnames
    cb = trk["Current Baseline"]
    assert cb["A4"].value == "Model / configuration" and cb["N4"].value == "Dynamic slot fill"
    labels = [cb.cell(row=i, column=1).value for i in range(5, 5 + len(_existing_runs()))]
    assert all(labels), labels
    assert any(n.startswith("Run - basic") for n in trk.sheetnames)
    cm = trk["Cost Model"]
    assert cm["A17"].value == "Run" and cm["I17"].value == "Use now" and cm["A18"].value


def test_run_validation_errors():
    r = client.post("/api/runs", json={"model": "not_a_model"})
    assert r.status_code == 400 and "error" in r.json()
    r = client.get("/api/runs/../../etc")
    assert r.status_code in (400, 404)
    r = client.get("/api/runs/does__not__exist")
    assert r.status_code == 404 and "error" in r.json()


def test_graph_endpoint_shape():
    r = client.post("/api/graph", json={"model": "basic", "seed": 42})
    if r.status_code == 503:
        assert r.json()["error"].startswith("inspect not available")
        return
    assert r.status_code == 200, r.text
    g = r.json()
    assert {"nodes", "edges", "counts"} <= set(g)
    assert g["nodes"] and all("backend" in n for n in g["nodes"])


def test_index_and_static():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
