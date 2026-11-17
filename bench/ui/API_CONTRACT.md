# Atalla Benchmark Workbench: API contract (v1, 2026-11-04)

Three parts are built in parallel against this document. Do not change field names without updating it.

## A. run_graph.py progress stream

`run_graph.py --progress-jsonl PATH` (also env `ATALLA_PROGRESS_JSONL`). Append one JSON object per line, flushed immediately:

```
{"event":"start","model":"gpt2_block","mode":"validate","total_nodes":47,"compute_nodes":29,"ts":1762266000.1}
{"event":"node_start","index":3,"name":"matmul","op":"matmul","backend":"emulator","shape":[32,32],"ts":...}
{"event":"node_done","index":3,"name":"matmul","op":"matmul","backend":"emulator","shape":[32,32],
   "cos":0.99999,"rel_l2":0.0012,"rel_max":0.004,"sched_packets":319,"packets":2148,"instructions":2453,
   "bytes_loaded":6208,"bytes_written":2048,"flops":74326,"elapsed_s":0.41,"ts":...}
{"event":"done","final_cosine":0.999995,"rel_l2":0.0029,"rel_max":0.0045,"strict_pass":true,"elapsed_s":11.8,"ts":...}
{"event":"error","message":"...","node":"add_7","ts":...}
```
`backend` ∈ `emulator | numpy | passthrough | skip`. `index` counts all FX nodes in execution order (0-based); passthrough nodes also emit node_start/node_done (with null metrics) so progress is monotonic. Numbers may be null when unavailable.

## B. Plan overrides

`run_graph.py --overrides PATH` (also env `ATALLA_PLAN_OVERRIDES`). JSON:
```
{"global": {"gelu_kernel": "pytorch|horner|tanh", "sdpa": "numpy|flash", "add_numpy_threshold": 1024},
 "nodes": {"<fx node name>": {"backend": "emulator|numpy|skip", "kernel": "<variant or null>"}}}
```
Semantics (applied in `tile_planner`/`run_validate`): `numpy` forces the reference path for that node; `emulator` forces the Atalla kernel (error event if no emitter exists); `skip` replaces the node with identity on its first tensor input (error if shapes differ). `kernel` picks a variant where one exists (gelu: horner|tanh; sdpa: flash|numpy). `global` maps to the existing env knobs `ATALLA_GELU_KERNEL`, `ATALLA_SDPA_FLASH`, `ATALLA_FFN_ADD_NUMPY` and must be honored even when those env vars are unset. The same override file is honored by `--mode schedule`.

## C. Graph inspection (Python, no emulator run)

`bench/ui/inspect.py::inspect_graph(model: str, scale: float, seed: int, overrides: dict | None) -> dict`
```
{"model":"gpt2_block","model_class":"GPT2Block","input_shape":[32,32],
 "nodes":[{"id":"matmul","name":"matmul","op":"matmul","target":"aten.mm.default","phase":"forward",
           "shape":[32,32],"dtype":"bf16","backend":"emulator",
           "kernel":{"name":"gemm","source":"atalla-graph/kernels/gemm.py","variant":null,
                     "tile":{"kernel_type":"matmul","M":32,"N":32,"K":32,"tile":32,...tc.params...},
                     "dram_addr":"0x5000","params":{...}},
           "inputs":["layer_norm"],"reason":"planned by _plan_matmul","order":3}],
 "edges":[{"source":"layer_norm","target":"matmul"}],
 "counts":{"emulated":21,"numpy":1,"passthrough":7,"skip":0,"total":47},
 "effective_overrides":{...}}
```
`backend` must match what `run_validate` would actually do for that node under the same overrides (reuse the same decision code; do not re-implement heuristics). Include placeholder/get_attr/output nodes with backend `"graph"` so the call graph is complete.

## D. HTTP API (FastAPI, `bench/ui/server.py`, default http://127.0.0.1:8765)

| Method, path | Body / params | Returns |
|---|---|---|
| GET `/api/models` | | `[{"name":"gpt2_block","title":"GPT-2 block","description":"...","defaults":{"scale":null,"seed":42},"options":[{"key":"gelu_kernel","label":"GELU kernel","choices":["pytorch","horner","tanh"],"default":"pytorch"},{"key":"sdpa","label":"SDPA path","choices":["numpy","flash"],"default":"numpy"},{"key":"add_numpy_threshold","label":"Add → NumPy above N elems","type":"int","default":1000000}]}]` (only options relevant to the model) |
| POST `/api/graph` | `{"model","scale","seed","overrides"}` | the inspect dict (C) |
| GET `/api/kernels` | | `[{"name":"gemm","path":"atalla-graph/kernels/gemm.py","kind":"generator|static_c","ops":["matmul","linear"]}]` |
| GET `/api/kernels/{name}` | | `{"name","path","text"}` |
| POST `/api/runs` | `{"model","scale","seed","tag","strict":bool,"overrides"}` | `{"run_id"}`; starts `bench/run_benchmark.py` as a subprocess with `--extra-args "--progress-jsonl <run>/progress.jsonl --overrides <run>/overrides.json [--strict-end]"`; at most one run at a time (409 if busy) |
| GET `/api/runs` | | `[{"run_id","model","tag","status":"running|done|error","started","final_cosine","coverage":{"emulated","numpy","passthrough"},"dyn_packets","bytes_total"}]` newest first, from `docs/baselines/*/summary.json` plus live runs |
| GET `/api/runs/{id}` | | `{"status","config":<config.json>,"summary":<summary.json or null>,"progress":[events so far]}` |
| GET `/api/runs/{id}/events` | | Server-Sent Events: each progress line as `data: {...}\n\n`; closes after `done`/`error`; replays existing lines first |
| GET `/api/runs/{id}/log` | | text/plain run.log |
| GET `/api/runs/{id}/artifact` | `?node=matmul&kind=c|s|in` | `{"node","kind","path","text"}` from `<run>/out/` |
| POST `/api/runs/{id}/cancel` | | `{"ok":true}` |
| POST `/api/runs/{id}/export` | | `{"run_xlsx":"output/runs/<run_id>.xlsx","tracker_xlsx":"output/Atalla_Benchmarking_Tracker.xlsx"}` via `bench/tracker_export.py` |
| GET `/` and `/static/*` | | the UI (`bench/ui/static/index.html`, `app.js`, `style.css`) |

Run ids are the `docs/baselines/` directory names (`<model>__<tag>__<YYYYMMDD-HHMM>`). Errors return JSON `{"error": "..."}` with 4xx/5xx.

## E. Tracker export (`bench/tracker_export.py`)

Template: `bench/templates/Atalla_Benchmarking_Tracker.xlsx` (Robert's sheet: tabs Overview, Current Baseline, Metrics, Cost Model, Model Ops, Kernel Status, Planned Models, `Run - <name>`). `export_run(run_dir) -> run_xlsx` writes a standalone workbook with one `Run - <tag>` sheet in the template's run layout (Field/Value block, Aggregate metric block, "Kernel-level results from this run" table with columns Name, Op, Backend, Shape, Sched. pkts, Dyn. pkts, Retired, Load bytes, Write bytes, FLOPs, Cosine, Rel-L2, Rel-max). `rebuild_tracker(run_dirs) -> tracker_xlsx` copies the template and (1) replaces the data rows of "Current Baseline" with one row per run in the template's 14 columns, (2) replaces the "Current feature values" rows of "Cost Model" (Run, Traffic bytes, Dynamic pkts, Systolic ops, Vector ALU, Scalar ALU, Scalar mem, Branch/control, Use now), (3) adds/replaces a `Run - <tag>` sheet per run, (4) leaves Overview/Metrics/Model Ops/Kernel Status/Planned Models untouched except where a cell is explicitly updated by code with a measured fact. Keep the template's fonts/column widths where openpyxl allows.
