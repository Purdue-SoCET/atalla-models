"""Progress stream (contract A), plan overrides (B) and graph inspection (C) tests.

Run (from repo root):
    /Users/robert.zhang/.venvs/atalla/bin/python -m pytest atalla-graph/tests -q

The ``basic`` and ``gpt2_block`` validate runs use the emulator (~5 s and ~12 s); their
results are cached per override set so several tests can share one run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

GRAPH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GRAPH_DIR.parent
sys.path.insert(0, str(GRAPH_DIR))

from run_graph import build_graph, load_model, run_validate, ProgressWriter, graph_decisions  # noqa: E402
from graph.overrides import PlanOverrides, resolve_backend  # noqa: E402


def _load_inspect_module():
    """bench/ui/inspect.py shadows stdlib ``inspect`` if imported by name: load it by path."""
    path = REPO_ROOT / "bench" / "ui" / "inspect.py"
    spec = importlib.util.spec_from_file_location("bench_ui_inspect", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_inspect = _load_inspect_module()
inspect_graph = _inspect.inspect_graph

_RUN_CACHE = {}
_TMP = Path(tempfile.mkdtemp(prefix="atalla-core-tests-"))


def _validate(model: str, overrides=None, seed: int = 42):
    """Real validate run (emulator) with a progress stream; cached per (model, overrides)."""
    key = (model, json.dumps(overrides, sort_keys=True))
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    torch.manual_seed(seed)
    np.random.seed(seed)
    mdl, x, _ = load_model(model, 0.01)
    gm = build_graph(mdl, x, verbose=False, overrides=overrides)
    out_dir = _TMP / f"{model}_{abs(hash(key)) % 10_000}"
    jsonl = out_dir / "progress.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    pw = ProgressWriter(str(jsonl))
    try:
        res = run_validate(gm, mdl, x, str(out_dir), verbose=False, progress=pw, model_name=model)
    finally:
        pw.close()
    events = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    _RUN_CACHE[key] = (res, events, gm)
    return _RUN_CACHE[key]


class ProgressJsonlTests(unittest.TestCase):
    def test_basic_progress_has_start_done_and_one_node_done_per_compute_node(self):
        res, events, gm = _validate("basic")
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        self.assertNotIn("error", kinds)
        start, done = events[0], events[-1]
        self.assertEqual(start["mode"], "validate")
        self.assertEqual(start["total_nodes"], len(list(gm.graph.nodes)))
        stats = res["stats"]
        compute = stats["emulated"] + stats["numpy"] + stats["skip"]
        self.assertEqual(start["compute_nodes"], compute)
        node_done = [e for e in events if e["event"] == "node_done"]
        self.assertEqual(len(node_done), stats["total"])  # every non-output node reports
        compute_done = [e for e in node_done if e["backend"] in ("emulator", "numpy", "skip")]
        self.assertEqual(len(compute_done), compute)
        # one node_done per compute node of the actual kernel_metrics, carrying the numbers
        km_names = [k["name"] for k in res["kernel_metrics"]]
        self.assertEqual([e["name"] for e in compute_done], km_names)
        emu = [e for e in compute_done if e["backend"] == "emulator"]
        self.assertTrue(emu)
        for e in emu:
            for key in ("cos", "rel_l2", "rel_max", "sched_packets", "packets", "instructions",
                        "bytes_loaded", "bytes_written", "flops", "elapsed_s"):
                self.assertIsNotNone(e[key], f"{e['name']} missing {key}")
        # passthrough / graph nodes carry null metrics, indices are monotonic
        idx = [e["index"] for e in node_done]
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(all(e["cos"] is None for e in node_done if e["backend"] == "passthrough"))
        self.assertAlmostEqual(done["final_cosine"], res["end_to_end"]["cos_sim"])
        self.assertEqual(done["stats"], stats)


class OverrideTests(unittest.TestCase):
    RELU_NUMPY = {"nodes": {"relu": {"backend": "numpy"}}}
    RELU1_SKIP = {"nodes": {"relu_1": {"backend": "skip"}}}

    def test_numpy_override_on_relu_in_inspect_and_validate(self):
        insp = inspect_graph("basic", 0.01, 42, self.RELU_NUMPY)
        node = next(n for n in insp["nodes"] if n["name"] == "relu")
        self.assertEqual(node["backend"], "numpy")
        self.assertTrue(node["forced"])
        self.assertIsNone(node["kernel"]["source"])
        self.assertEqual(insp["counts"]["numpy"], 1)

        res, events, _ = _validate("basic", self.RELU_NUMPY)
        km = next(k for k in res["kernel_metrics"] if k["name"] == "relu")
        self.assertEqual(km["backend"], "numpy")
        self.assertEqual(res["stats"]["numpy"], 1)
        self.assertGreater(km["cos_sim"], 0.999)  # chained torch eval of the node matches ref
        ev = next(e for e in events if e["event"] == "node_done" and e["name"] == "relu")
        self.assertEqual(ev["backend"], "numpy")
        self.assertIsNone(ev["sched_packets"])
        # the rest of the graph is unchanged and the model still validates
        self.assertGreater(res["end_to_end"]["cos_sim"], 0.999)

    def test_skip_override_on_shape_preserving_node(self):
        insp = inspect_graph("basic", 0.01, 42, self.RELU1_SKIP)
        node = next(n for n in insp["nodes"] if n["name"] == "relu_1")
        self.assertEqual(node["backend"], "skip")
        self.assertEqual(node["skip_source"], "add_1")
        self.assertEqual(insp["counts"]["skip"], 1)

        res, events, _ = _validate("basic", self.RELU1_SKIP)
        self.assertEqual(res["stats"]["skip"], 1)
        km = next(k for k in res["kernel_metrics"] if k["name"] == "relu_1")
        self.assertEqual(km["backend"], "skip")
        self.assertEqual(km["skip_source"], "add_1")
        self.assertNotIn("error", [e["event"] for e in events])
        ev = next(e for e in events if e["event"] == "node_done" and e["name"] == "relu_1")
        self.assertEqual(ev["backend"], "skip")
        # identity on a relu input is a real change: the skipped node no longer matches ref
        self.assertLess(km["cos_sim"], 0.999)

    def test_skip_with_shape_mismatch_falls_back_to_numpy_with_error(self):
        ov = {"nodes": {"matmul_6": {"backend": "skip"}}}  # (32,32) -> (32,128)
        insp = inspect_graph("gpt2_block", 0.01, 42, ov)
        node = next(n for n in insp["nodes"] if n["name"] == "matmul_6")
        self.assertEqual(node["backend"], "numpy")
        self.assertIn("shapes differ", node["error"])

    def test_emulator_override_rejected_without_kernel(self):
        ov = {"nodes": {"transpose": {"backend": "emulator"}}}
        insp = inspect_graph("gpt2_block", 0.01, 42, ov)
        node = next(n for n in insp["nodes"] if n["name"] == "transpose")
        self.assertEqual(node["backend"], "passthrough")
        self.assertIsNotNone(node["error"])

    def test_global_gelu_override_wins_over_env_and_does_not_leak(self):
        before = os.environ.get("ATALLA_GELU_KERNEL")
        os.environ["ATALLA_GELU_KERNEL"] = "horner"
        try:
            insp = inspect_graph("gpt2_block", 0.01, 42, {"global": {"gelu_kernel": "tanh"}})
            gelu = next(n for n in insp["nodes"] if n["name"] == "gelu")
            self.assertEqual(gelu["backend"], "emulator")
            self.assertEqual(gelu["kernel"]["variant"], "tanh")
            self.assertEqual(gelu["kernel"]["source"], "atalla-graph/kernels/gelu_tanh.c")
            self.assertEqual(os.environ.get("ATALLA_GELU_KERNEL"), "horner")  # restored
            # env still works when no override is given
            insp2 = inspect_graph("gpt2_block", 0.01, 42, None)
            gelu2 = next(n for n in insp2["nodes"] if n["name"] == "gelu")
            self.assertEqual(gelu2["kernel"]["variant"], "horner")
            self.assertEqual(gelu2["kernel"]["source"], "atalla-graph/kernels/gelu.c")
        finally:
            if before is None:
                os.environ.pop("ATALLA_GELU_KERNEL", None)
            else:
                os.environ["ATALLA_GELU_KERNEL"] = before
        insp3 = inspect_graph("gpt2_block", 0.01, 42, None)
        gelu3 = next(n for n in insp3["nodes"] if n["name"] == "gelu")
        self.assertEqual(gelu3["backend"], "numpy" if before is None else gelu3["backend"])

    def test_parse_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            PlanOverrides.from_dict({"nodes": {"relu": {"backend": "gpu"}}})
        with self.assertRaises(ValueError):
            PlanOverrides.from_dict({"global": {"gelu_kernel": "cubic"}})


class InspectTests(unittest.TestCase):
    def test_inspect_counts_match_run_validate_for_gpt2_block(self):
        insp = inspect_graph("gpt2_block", 0.01, 42, None)
        res, events, gm = _validate("gpt2_block")
        stats = res["stats"]
        c = insp["counts"]
        self.assertEqual(c["emulated"], stats["emulated"])
        self.assertEqual(c["numpy"], stats["numpy"])
        self.assertEqual(c["passthrough"], stats["passthrough"])
        self.assertEqual(c["skip"], stats["skip"])
        self.assertEqual(c["total"], len(list(gm.graph.nodes)))
        # per-node agreement with what actually ran
        ran = {k["name"]: k["backend"] for k in res["kernel_metrics"]}
        for n in insp["nodes"]:
            if n["name"] in ran:
                self.assertEqual(n["backend"], ran[n["name"]], n["name"])
        self.assertEqual(insp["model_class"], "GPT2Block")
        self.assertEqual(insp["input_shape"], [32, 32])

    def test_inspect_is_fast_and_complete(self):
        t0 = time.time()
        insp = inspect_graph("gpt2_block", 0.01, 42, None)
        self.assertLess(time.time() - t0, 3.0)
        names = {n["name"] for n in insp["nodes"]}
        for e in insp["edges"]:
            self.assertIn(e["source"], names)
            self.assertIn(e["target"], names)
        graph_nodes = [n for n in insp["nodes"] if n["backend"] == "graph"]
        self.assertTrue(all(n["fx_op"] in ("placeholder", "get_attr", "output") for n in graph_nodes))
        mm = next(n for n in insp["nodes"] if n["name"] == "matmul")
        self.assertEqual(mm["kernel"]["name"], "gemm")
        self.assertEqual(mm["kernel"]["source"], "atalla-graph/kernels/gemm.py")
        self.assertEqual(mm["kernel"]["tile"]["kernel_type"], "matmul")
        self.assertEqual(mm["kernel"]["tile"]["M"], 32)
        self.assertTrue(mm["kernel"]["dram_addr"].startswith("0x"))
        self.assertEqual(mm["phase"], "forward")
        self.assertTrue(mm["inputs"])
        self.assertIn("order", mm)

    def test_resolve_backend_is_the_single_decision_point(self):
        torch.manual_seed(42)
        mdl, x, _ = load_model("basic", 0.01)
        gm = build_graph(mdl, x, verbose=False)
        decisions = graph_decisions(gm)
        insp = inspect_graph("basic", 0.01, 42, None)
        for n in insp["nodes"]:
            self.assertEqual(n["backend"], decisions[n["name"]].backend)
            self.assertEqual(n["reason"], decisions[n["name"]].reason)


if __name__ == "__main__":
    unittest.main()
