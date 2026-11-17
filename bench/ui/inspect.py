"""Graph inspection for the benchmark workbench (bench/ui/API_CONTRACT.md section C).

``inspect_graph(model, scale, seed, overrides)`` builds the real FX graph (same
``load_model`` / ``build_graph`` as ``run_graph.py``, same seed) and labels every node with
the backend ``run_validate`` would use, via the ONE decision function
``graph.overrides.resolve_backend``. No emulator, no compiler: < 3 s for gpt2_block.

CLI:
    /Users/robert.zhang/.venvs/atalla/bin/python bench/ui/inspect.py --model gpt2_block --json out.json
    ... --overrides ov.json --scale 0.01 --seed 42

Import note: this file is named ``inspect.py``; import it by path
(``importlib.util.spec_from_file_location``) or as ``bench.ui.inspect`` with the repo root
on ``sys.path`` -- never put ``bench/ui`` itself first on ``sys.path`` or it shadows the
stdlib ``inspect`` module that torch imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Running as a script puts bench/ui first on sys.path -> stdlib ``inspect`` would resolve to
# this file. Drop it before anything imports torch.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE] or sys.path
if __name__ == "__main__" and "inspect" in sys.modules and getattr(sys.modules["inspect"], "__file__", "") == str(Path(__file__).resolve()):
    del sys.modules["inspect"]

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

REPO_ROOT = _HERE.parents[1]
GRAPH_DIR = REPO_ROOT / "atalla-graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_graph import build_graph, load_model, graph_decisions  # noqa: E402
from graph.fx_capture import get_node_shape  # noqa: E402
from graph.overrides import load_overrides, PlanOverrides  # noqa: E402
from codegen.c_emitter import kernel_source_for, has_emitter  # noqa: E402

MODELS = ["basic", "alexnet_small", "layernorm_smoke", "vit_micro", "gpt2_mlp", "gpt2_attn", "gpt2_block"]

_DTYPE_LABEL = {
    "torch.bfloat16": "bf16", "torch.float32": "f32", "torch.float16": "f16",
    "torch.int64": "i64", "torch.int32": "i32", "torch.bool": "bool",
}


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu().tolist()
    return str(v)


def _dtype_of(node) -> Optional[str]:
    tm = node.meta.get("tensor_meta")
    dt = getattr(tm, "dtype", None)
    if dt is None:
        return None
    return _DTYPE_LABEL.get(str(dt), str(dt).replace("torch.", ""))


def _target_label(node) -> str:
    t = node.target
    if node.op in ("placeholder", "get_attr", "call_module", "call_method"):
        return str(t)
    if node.op == "output":
        return "output"
    mod = getattr(t, "__module__", None) or ""
    name = getattr(t, "__name__", None) or getattr(t, "name", None) or str(t)
    if mod and not mod.startswith("builtins"):
        return f"{mod}.{name}"
    return str(name)


def _input_names(node) -> List[str]:
    return [n.name for n in node.all_input_nodes]


def _dram_hex(v) -> Optional[str]:
    return f"0x{int(v):X}" if isinstance(v, (int, np.integer)) else None


def inspect_graph(model: str, scale: float = 0.01, seed: int = 42,
                  overrides: Optional[Dict] = None) -> Dict:
    """Return the contract-C dict for *model* (see module docstring)."""
    t0 = time.time()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    mdl, example_input, _expected = load_model(model, scale if scale is not None else 0.01)
    model_class = type(mdl).__name__
    ov: PlanOverrides = load_overrides(overrides)
    gm = build_graph(mdl, example_input, verbose=False, overrides=ov)
    decisions = graph_decisions(gm, ov)

    nodes = []
    edges = []
    counts = {"emulated": 0, "numpy": 0, "passthrough": 0, "skip": 0, "graph": 0, "total": 0}
    for order, node in enumerate(gm.graph.nodes):
        dec = decisions[node.name]
        tc = node.meta.get("tile_config")
        op = node.meta.get("atalla_op")
        shape = get_node_shape(node)
        kernel = None
        if tc is not None:
            src = kernel_source_for(op, tc) if dec.backend == "emulator" else None
            tile = {"kernel_type": tc.kernel_type, **_jsonable(tc.params)}
            kernel = {
                "name": src["name"] if src else None,
                "source": src["source"] if src else None,
                "kind": src["kind"] if src else None,
                "variant": dec.variant,
                "tile": tile,
                "dram_addr": _dram_hex(node.meta.get("dram_addr")),
                "dram_bytes": node.meta.get("dram_bytes"),
                "params": _jsonable(tc.params),
                "has_emitter": has_emitter(op),
            }
        elif node.meta.get("dram_addr") is not None:
            kernel = {"name": None, "source": None, "kind": None, "variant": None, "tile": None,
                      "dram_addr": _dram_hex(node.meta.get("dram_addr")),
                      "dram_bytes": node.meta.get("dram_bytes"), "params": {},
                      "has_emitter": has_emitter(op)}
        entry = {
            "id": node.name,
            "name": node.name,
            "op": op if op is not None else node.op,
            "fx_op": node.op,
            "target": _target_label(node),
            "phase": node.meta.get("phase", "forward"),
            "shape": list(shape) if shape is not None else None,
            "dtype": _dtype_of(node),
            "backend": dec.backend,
            "kernel": kernel,
            "inputs": _input_names(node),
            "reason": dec.reason,
            "order": order,
            "forced": bool(dec.forced),
            "error": dec.error,
            "override": node.meta.get("override"),
        }
        if dec.backend == "skip":
            entry["skip_source"] = dec.skip_source
        if node.meta.get("parameter_name"):
            entry["parameter_name"] = node.meta["parameter_name"]
        nodes.append(entry)
        for inp in node.all_input_nodes:
            edges.append({"source": inp.name, "target": node.name})
        counts["total"] += 1
        bucket = {"emulator": "emulated"}.get(dec.backend, dec.backend)
        counts[bucket] = counts.get(bucket, 0) + 1

    return {
        "model": model,
        "model_class": model_class,
        "input_shape": list(example_input.shape),
        "seed": int(seed),
        "scale": scale,
        "nodes": nodes,
        "edges": edges,
        "counts": counts,
        "effective_overrides": {**ov.as_dict(), "env": gm.meta.get("override_env", {})},
        "pending_kernel_ops": list(gm.meta.get("pending_kernel_ops", [])),
        "elapsed_s": time.time() - t0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inspect the Atalla graph plan for a model (no emulator).")
    p.add_argument("--model", default="gpt2_block", choices=MODELS)
    p.add_argument("--scale", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overrides", default=None, help="overrides JSON path (contract B)")
    p.add_argument("--json", default=None, help="write the inspect dict here (else stdout)")
    args = p.parse_args(argv)
    out = inspect_graph(args.model, args.scale, args.seed, args.overrides)
    text = json.dumps(out, indent=2)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
        c = out["counts"]
        print(f"{args.model}: {c['total']} nodes -> emulated {c['emulated']} / numpy {c['numpy']} / "
              f"passthrough {c['passthrough']} / skip {c['skip']} / graph {c['graph']} "
              f"({out['elapsed_s']:.2f}s) -> {args.json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
