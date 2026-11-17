# Atalla benchmark workbench: API server + exports

FastAPI server (`bench/ui/server.py`), run manager (`bench/ui/runs.py`) and tracker export
(`bench/tracker_export.py`). Contract: `bench/ui/API_CONTRACT.md` (sections D and E).

## Start

```bash
cd /Users/robert.zhang/Documents/GitHub/atalla-models-rz
PY=/Users/robert.zhang/.venvs/atalla/bin/python
$PY -m uvicorn bench.ui.server:app --host 127.0.0.1 --port 8765
# or
$PY bench/ui/server.py --port 8765            # --host 0.0.0.0 to expose on the LAN, --reload for dev
```

Open http://127.0.0.1:8765/ (serves `bench/ui/static/index.html`; a placeholder page if it is missing)
and http://127.0.0.1:8765/docs for the OpenAPI UI. CORS is open for `localhost` / `127.0.0.1` on any port.
Dependencies: `fastapi`, `uvicorn`, `openpyxl` (+ `httpx` for the tests), all in the atalla venv.

## Endpoints

| Method, path | What |
|---|---|
| GET `/api/health` | `{ok, busy, inspect_available, static_index}` |
| GET `/api/models` | models from `run_graph.py --model` choices + titles/descriptions/defaults/options (`gelu_kernel`, `sdpa`, `add_numpy_threshold`; only the relevant ones per model) |
| POST `/api/graph` `{model, scale, seed, overrides}` | call graph / backend plan via `bench.ui.inspect.inspect_graph` (503 `{"error":"inspect not available yet"}` if the module is missing) |
| GET `/api/kernels` | `atalla-graph/kernels/*.py` (`generator`) and `*.c/*.h` (`static_c`) with the ops they implement |
| GET `/api/kernels/{name}` | `{name, path, kind, ops, text}` |
| POST `/api/runs` `{model, scale, seed, tag, strict, overrides}` | launch `bench/run_benchmark.py` as a subprocess; returns `{run_id, status, command, notes}`; **409** while another run is active |
| GET `/api/runs` | newest first: `docs/baselines/*/summary.json` merged with live runs (`status` running/done/error, `final_cosine`, `coverage{emulated,numpy,passthrough,total}`, `dyn_packets`, `bytes_total`, ...) |
| GET `/api/runs/busy` | `{busy, run_id}` |
| GET `/api/runs/{id}` | `{status, config, summary, overrides, progress:[events], notes}` |
| GET `/api/runs/{id}/events` | Server-Sent Events: replays `<run>/progress.jsonl`, then tails it (200 ms poll) until `done`/`error`; the stream closes only after `summary.json` is written, so `/export` can be called right away |
| GET `/api/runs/{id}/log?tail=N` | `run.log` as text/plain |
| GET `/api/runs/{id}/artifact?node=matmul&kind=c|s|in` | `{node, kind, path, text}` from `<run>/out/`; without `node`: `{nodes: {name: [kinds]}}` |
| POST `/api/runs/{id}/cancel` | SIGTERM (then SIGKILL) the run's process group; the run is marked `error` with a synthesised `cancelled` event |
| POST `/api/runs/{id}/export` | `{run_xlsx, tracker_xlsx, tracker_updates}` (see below); `?rebuild=false` skips the tracker rebuild |
| GET `/api/runs/{id}/export/download` | the standalone run workbook as a file download |
| GET `/api/export/tracker` | `output/Atalla_Benchmarking_Tracker.xlsx` as a file download |
| GET `/`, `/static/*` | the UI |

Errors are JSON `{"error": "..."}` with a 4xx/5xx status.

### How a run is launched

`POST /api/runs` precomputes the run id `<model>__<tag>__<YYYYMMDD-HHMM>` (the harness's own scheme),
creates `docs/baselines/<run_id>/`, writes `overrides.json` (only when non-empty) and `request.json`, then runs

```
$PY bench/run_benchmark.py --model M --tag T --run-dir docs/baselines/<run_id> --seed S [--scale X]
    [--env ATALLA_GELU_KERNEL=... --env ATALLA_SDPA_FLASH=...]
    --extra-args="--progress-jsonl <run>/progress.jsonl --overrides <run>/overrides.json [--strict-end]"
```

with `cwd` = repo root in a new process group (`--run-dir` is a small backward-compatible addition to
`run_benchmark.py`). Global overrides are also mapped onto the env knobs (`gelu_kernel` ->
`ATALLA_GELU_KERNEL`, `sdpa=flash` -> `ATALLA_SDPA_FLASH=1`) so they work on either code path. The
manager checks `run_graph.py` for `--progress-jsonl` / `--overrides` support at launch time and drops the
flags (with a note in the response) if they are absent; in that case `start` and `done`/`error` events are
synthesised from `summary.json` so the UI still gets a terminal event. Status: `done` when the process exits
0 and `summary.json` has a final cosine, otherwise `error`.

## Export files

`bench/tracker_export.py` uses the template `bench/templates/Atalla_Benchmarking_Tracker.xlsx` and keeps its
fonts, fills, borders, number formats, column widths and freeze panes.

```bash
$PY bench/tracker_export.py --run docs/baselines/<run_dir>     # -> output/runs/<run_id>.xlsx (one "Run - <model> <tag>" sheet)
$PY bench/tracker_export.py --rebuild                          # -> output/Atalla_Benchmarking_Tracker.xlsx from every run in docs/baselines
```

* `output/runs/<run_id>.xlsx`: one sheet in the template's run layout: Field/Value block (model, input mode,
  nodes total/E/N/P, final cosine/rel-L2/rel-max, run note, elapsed, run id, seed, flags/env, revision),
  Aggregate block (scheduled/dynamic packets, retired instrs, bytes loaded/written/total, FLOPs, AI,
  static/dynamic slot fill, retired systolic/vector ALU) and the "Kernel-level results from this run" table
  (Name, Op, Backend, Shape, Sched. pkts, Dyn. pkts, Retired, Load bytes, Write bytes, FLOPs, Cosine,
  Rel-L2, Rel-max), one row per `kernel_metrics` entry.
* `output/Atalla_Benchmarking_Tracker.xlsx`: the template with
  1. **Current Baseline** data rows replaced by one row per run (14 columns; green = passes with zero NumPy
     fallback, amber = passes with NumPy fallback, red = accuracy/run failure), note row kept below;
  2. **Cost Model** "Current feature values" rows replaced (Run, Traffic bytes, Dynamic pkts, Systolic ops,
     Vector ALU, Scalar ALU, Scalar mem, Branch/control, Use now);
  3. one `Run - <model> <tag>` sheet per run (the template's four example run sheets are replaced);
  4. Overview / Metrics / Model Ops / Kernel Status / Planned Models untouched except these measured-fact
     cells, written only when a matching run exists: Kernel Status "Flash SDPA" Status+Evidence (from the
     newest `vit_micro` run with `ATALLA_SDPA_FLASH=1`), "GELU" Current backend/Status/Evidence (newest run
     with `ATALLA_GELU_KERNEL=`), "Add" Status+Evidence (newest non-basic run with emulated adds and zero
     NumPy); Model Ops "flash SDPA" and both "GPT-2 small block" rows Current status+Evidence. The list of
     changed cells is printed by `--rebuild` and returned as `tracker_updates` by `/export`.

## Tests

```bash
$PY -m pytest bench/ui/test_server.py -q      # models, kernels, runs list/detail/log/artifact, SSE replay, export, validation
```
