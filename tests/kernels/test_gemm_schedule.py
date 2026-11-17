#!/usr/bin/env python3
"""GEMM schedule variants (``ATALLA_GEMM_SCHEDULE=baseline|double_buffer``).

Checks, for a 64x64x64 GEMM (2x2x2 tiles) and a partial-tile 40x20x100 GEMM:

1. Both variants compile with ``atalla_cc -S`` (in-tree ppci).
2. The ``double_buffer`` K-loop body issues exactly two ``scpad_load`` calls per
   iteration (the A tile and the next weight tile), and the C tile is loaded /
   stored once per (mi, ni) outside the K loop.
3. Both variants run on the functional simulator through the same
   ``c_emitter`` pipeline ``run_graph.py`` uses, agree bit-for-bit, and match a
   NumPy BF16 reference (cosine >= 0.999).

Run (from repo root):
    /Users/robert.zhang/.venvs/atalla/bin/python -m pytest tests/kernels/test_gemm_schedule.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_PPCI as _REPO_PPCI, MODELS_ROOT as _MODELS_ROOT  # noqa: E402



def _load_graph_kernels():
    """atalla-graph/kernels under a private name: functional_sim has its own ``kernels``
    package (``kernels.utils.dataloader``) that ``codegen/c_emitter.py`` needs on
    the bare name.  Same trick as ``c_emitter._load_emit_kernels``."""
    import importlib
    import importlib.util
    name = "_atalla_emit_kernels"
    if name not in sys.modules:
        root = _MODELS_ROOT / "kernels"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.gemm")


_gemm = _load_graph_kernels()
SCHEDULES, gemm_c, gemm_schedule = _gemm.SCHEDULES, _gemm.gemm_c, _gemm.gemm_schedule

SHAPES = [(64, 64, 64), (40, 20, 100)]


def _compile(c_source: str, tmp: Path, tag: str) -> str:
    c_path = tmp / f"{tag}.c"
    s_path = tmp / f"{tag}.s"
    c_path.write_text(c_source)
    res = subprocess.run(
        [sys.executable, str(_REPO_PPCI / "atalla_cc"), "-S", str(c_path), "-o", str(s_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"atalla_cc -S failed for {tag}: {res.stderr[-2000:]}"
    return s_path.read_text()


def _k_loop_body(c_source: str) -> str:
    """Text of the ``while (ki < K_tiles) { ... }`` block (brace-balanced)."""
    start = c_source.index("while (ki < K_tiles) {")
    depth = 0
    for i in range(start, len(c_source)):
        if c_source[i] == "{":
            depth += 1
        elif c_source[i] == "}":
            depth -= 1
            if depth == 0:
                return c_source[start:i + 1]
    raise AssertionError("unbalanced K loop")


def test_env_selection(monkeypatch):
    monkeypatch.delenv("ATALLA_GEMM_SCHEDULE", raising=False)
    assert gemm_schedule() == "baseline"
    assert gemm_c(64, 64, 64) == gemm_c(64, 64, 64, schedule="baseline")
    monkeypatch.setenv("ATALLA_GEMM_SCHEDULE", "double_buffer")
    assert gemm_schedule() == "double_buffer"
    assert gemm_c(64, 64, 64) == gemm_c(64, 64, 64, schedule="double_buffer")
    monkeypatch.setenv("ATALLA_GEMM_SCHEDULE", "bogus")
    try:
        gemm_schedule()
    except ValueError:
        pass
    else:
        raise AssertionError("bogus schedule accepted")


def test_both_variants_compile_and_double_buffer_structure():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for M, N, K in SHAPES:
            for sched in SCHEDULES:
                src = gemm_c(M, N, K, schedule=sched)
                asm = _compile(src, tmp, f"{sched}_{M}x{N}x{K}")
                assert "gemm" in asm and "lw" in asm
                # No addi_s with an immediate in the signed-12-bit overflow band.
                big = [int(x) for x in re.findall(r"addi_s\s+x\d+,\s*x\d+,\s*(\d+)", asm) if int(x) >= 2048]
                assert not big, f"{sched} {M}x{N}x{K}: addi_s immediates >= 2048: {big}"

            db = gemm_c(M, N, K, schedule="double_buffer")
            body = _k_loop_body(db)
            # Two DMA loads per K iteration: the A tile and the next weight tile
            # (inside fetch_weight_tile, guarded by `if (ki < K_tiles)`).
            direct = body.count("scpad_load(")
            fetch_calls = body.count("fetch_weight_tile(")
            assert direct == 1 and fetch_calls == 1, (direct, fetch_calls, body)
            fetch_fn = db[db.index("void fetch_weight_tile"):db.index("void push_weights")]
            assert fetch_fn.count("scpad_load(") == (1 if min(32, N) >= 32 else 2)
            assert "scpad_store(" not in body, "C must be stored outside the K loop"
            # C tile load/store once per (mi, ni), outside the K loop.
            outer = db.replace(body, "")
            assert outer.count("scpad_load(sp_c") == 1 and outer.count("scpad_store(sp_c") == 1
            # Weight loop unrolled by 2.
            push_fn = db[db.index("void push_weights"):db.index("int main()")]
            assert push_fn.count("load_weights(") == 2 and "wi = wi + 2;" in push_fn

            base = gemm_c(M, N, K, schedule="baseline")
            bbody = _k_loop_body(base)
            assert bbody.count("scpad_load(") == 2 and bbody.count("scpad_store(") == 1
            assert "load_weight_tile(" in bbody


def _bf16_round(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def _sim_child(M: int, N: int, K: int, sched: str, a_path: str, w_path: str, out_dir: str) -> None:
    """Child process: emulate ``C = A @ W`` through the same ``c_emitter`` pipeline
    ``run_graph.py`` uses and write ``C.npy`` + ``perf.json`` to ``out_dir``.

    Runs in its own interpreter because ``run_graph``/``c_emitter`` need the bare
    ``kernels`` name for functional_sim's package while the other tests in this
    directory import atalla-graph's ``kernels`` under that name.
    """
    import json
    if str(_MODELS_ROOT) not in sys.path:
        sys.path.append(str(_MODELS_ROOT))  # after functional_sim so bare `kernels` stays the sim's
    import run_graph  # sets up functional_sim / compiler import paths
    from codegen import c_emitter as ce

    A = np.load(a_path)
    W = np.load(w_path)
    ks = ce._gemm_k_stride(K)
    A_GMEM = 0x1000
    W_GMEM = A_GMEM + ce._align_data(M * ks * 2)
    C_GMEM = W_GMEM + ce._align_data(N * ks * 2)
    Z_GMEM = C_GMEM + M * N * 2

    img = ce.DRAMWriter()
    ce._write_gemm_params(img, A_GMEM, W_GMEM, C_GMEM, M, N, K, Z_GMEM)
    ce._write_matrix(img, A_GMEM, ce._padded_gemm_a(A, K), M, ks)
    ce._write_gemm_rhs_weight(img, W_GMEM, W)
    ce._write_zeros(img, C_GMEM, M * N)
    ce._write_zeros(img, Z_GMEM, ce.TILE)

    em = ce.LayerEmission()
    em.c_source = gemm_c(M, N, K, schedule=sched)
    em.dram = img
    tag = f"gemm_{sched}_{M}x{N}x{K}"
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    ce.compile_and_assemble(em, str(work), tag)
    in_path = work / f"{tag}.in"
    in_path.write_text(ce.render_in_file(em))
    mem, EU = run_graph._run_emulator(str(in_path), str(work), tag)
    C = run_graph._read_bf16(mem, C_GMEM, M * N).reshape(M, N)
    np.save(work / "C.npy", C)
    perf = dict(EU.perf_metrics.metrics) if hasattr(EU, "perf_metrics") else {}
    (work / "perf.json").write_text(json.dumps({k: v for k, v in perf.items()
                                                if isinstance(v, (int, float))}))


def _run_gemm_on_sim(M: int, N: int, K: int, sched: str, A: np.ndarray, W: np.ndarray,
                     out_dir: Path) -> tuple[np.ndarray, dict]:
    """Emulate ``C = A @ W`` with the given schedule in a child process."""
    import json
    work = out_dir / f"{sched}_{M}x{N}x{K}"
    work.mkdir(parents=True, exist_ok=True)
    a_path, w_path = work / "A.npy", work / "W.npy"
    np.save(a_path, A)
    np.save(w_path, W)
    res = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--sim-child",
         str(M), str(N), str(K), sched, str(a_path), str(w_path), str(work)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"sim child failed ({sched} {M}x{N}x{K}):\n{res.stderr[-3000:]}"
    C = np.load(work / "C.npy")
    perf = json.loads((work / "perf.json").read_text())
    return C, perf


def test_schedules_agree_on_sim():
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as td:
        for M, N, K in SHAPES:
            A = _bf16_round(rng.standard_normal((M, K)).astype(np.float32))
            W = _bf16_round(rng.standard_normal((K, N)).astype(np.float32) * 0.2)
            ref = A @ W
            outs = {}
            perfs = {}
            for sched in SCHEDULES:
                outs[sched], perfs[sched] = _run_gemm_on_sim(M, N, K, sched, A, W, Path(td))
                c = outs[sched].flatten()
                r = ref.flatten()
                cos = float(np.dot(c, r) / (np.linalg.norm(c) * np.linalg.norm(r)))
                assert cos >= 0.999, f"{sched} {M}x{N}x{K}: cosine {cos}"
            np.testing.assert_array_equal(
                outs["baseline"].view(np.uint32), outs["double_buffer"].view(np.uint32),
                err_msg=f"{M}x{N}x{K}: schedules differ bit-wise",
            )
            # Data-movement accounting: fewer DMA bytes, never more.
            b0 = perfs["baseline"].get("bytes_loaded")
            b1 = perfs["double_buffer"].get("bytes_loaded")
            if b0 is not None and b1 is not None:
                assert b1 <= b0, (b0, b1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sim-child":
        _M, _N, _K = (int(x) for x in sys.argv[2:5])
        _sim_child(_M, _N, _K, sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    else:
        test_both_variants_compile_and_double_buffer_structure()
        test_schedules_agree_on_sim()
        print("ok")
