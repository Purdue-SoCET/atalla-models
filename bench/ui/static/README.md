# Atalla Benchmark Workbench — UI (`bench/ui/static/`)

Single-page vanilla-JS front end for the benchmark API described in
`bench/ui/API_CONTRACT.md` (section D). No build step.

| file | role |
|---|---|
| `index.html` | shell: left nav rail, sticky top bar (title + primary Run button), centre column (max 860 px), right context column (360 px); inline SVG icon sprite; pinned CDN libraries |
| `style.css` | design tokens (`:root` vars, dark default, `data-theme="light"` override), layout, components, responsive rules |
| `app.js` | API client (real + mock fallback), hash router, the five views, Mermaid call graph (builder, pan/zoom viewport, copy/view source), Chart.js charts, modals/toasts |
| `mock.js` | contract-shaped offline data: gpt2_block graph (48 FX nodes incl. `output`, 29 compute), a streamed mock run, the 10 baseline rows from `docs/baselines/README.md`, the kernel list |

## Run it

With the FastAPI server (`bench/ui/server.py`, serves `/` and `/static/*`):

```
open http://127.0.0.1:8765/
open http://127.0.0.1:8765/?mock=1      # force mock data even when the API is up
```

Without the server (pure static):

```
/Users/robert.zhang/.venvs/atalla/bin/python -m http.server 8777 --directory bench/ui/static
open http://127.0.0.1:8777/index.html?mock=1
```

`index.html` picks the asset base at load time: `/static/` when the page is served at `/`
(FastAPI), `./` otherwise (http.server or `/static/index.html`).

## Mock mode

`app.js` uses `mock.js` when `?mock=1` is set, or automatically when any `/api` call fails with
a network error (the rail pill switches from "live API" to "mock data" and a toast says so).
The UI code path is identical in both modes: `mockApi` exposes the same methods as the real client
(`models, graph, kernels, kernel, createRun, runs, run, events, log, artifact, cancel, exportRun`).
The mock run streams `start → node_start/node_done × 29 → done` with `setTimeout`, using the real
per-node numbers from `docs/baselines/gpt2_block__chained__20261104-1459/layers.csv`.

## Views (hash routes)

| route | what it shows |
|---|---|
| `#/workbench` | model/seed/scale/options/strict/tag controls; `POST /api/graph` call graph (Mermaid `flowchart TD`, colour = backend, parameters hidden by default, "Show parameters" toggle, drag to pan / wheel or +−/fit to zoom, "Copy Mermaid" + "View source"); counts bar; override chips; right column = selected node (op/target/shape/dtype/phase, kernel + source link → modal from `GET /api/kernels/{name}`, tile params, dram_addr, reason) + override controls (auto/emulator/numpy/skip + variant); "Apply overrides" re-POSTs the graph |
| `#/run/<id>` | opened automatically after `POST /api/runs`; SSE feed from `/events` (newest on top), progress header, live cosine line + dynamic-packet bar charts; on `done` → summary tiles, "Export to tracker" (`POST /export`, shows paths), "Open log" (`/log` modal), click a feed card → artifact modal (`/artifact?kind=c|s|in`) |
| `#/runs` | sortable/filterable table of `GET /api/runs`; pick two with the switches → `#/compare/<a>/<b>` (grouped bars for bytes, dyn packets, FLOPs, retired instrs + overlaid per-layer cosine) |
| `#/kernels[/<name>]` | `GET /api/kernels` list; click → source viewer with line numbers |
| `#/metrics` | static copy of the 10 headline metric definitions from `docs/benchmark_metrics.md` |

Backend colours: emulator `#1d9bf0`, numpy `#ffd400`, passthrough `#71767b`, skip `#f4212e`,
graph nodes (placeholder/get_attr/output) dashed `#2f3336` outline. Cosine badges: ≥ 0.999 green,
≥ 0.99 yellow, else red.

## Libraries (pinned)

- mermaid 10.9.1 `mermaid.min.js` — cdnjs (11.x is not on cdnjs; checked with `curl -I`)
- Chart.js 4.4.1 `chart.umd.min.js` — cdnjs (4.4.4 does not exist on cdnjs; 4.4.1 is the newest 4.4.x there)
- Inter — Google Fonts

If the CDNs are unreachable the page still renders; the graph card shows a toast ("mermaid failed
to load") and charts are skipped.

## Call graph (Mermaid)

`buildMermaid()` in `app.js` turns the `/api/graph` result into `flowchart TD` text:

- nodes are declared in FX execution order (`order`), each followed by its incoming edges, so dagre ranks
  the chart the way the model executes (input at the top, `output` at the bottom, residual adds merge two branches);
- compute/passthrough nodes are `id["name<br/><small>op</small>"]:::backend`; `placeholder`/`output` are stadiums
  `id(["name"]):::boundary`; parameters (`get_attr`: weights, biases, mask buffers) are hidden by default and listed under
  **inputs** in the node panel. "Show parameters" adds them as dashed `id[/"name"/]:::param` nodes declared right before
  their consumer with a dotted `-.->` edge, so they sit beside the op instead of ranking at the top;
- ids are sanitized to `[A-Za-z0-9_]`, prefixed with `n_` when they start with a digit or collide with a flowchart keyword
  (`end`, `graph`, `x`, `o`, ...) and de-duplicated; `state.graphIds` maps them back to real node ids;
- `classDef` lines carry the backend colours as hex, resolved from the CSS tokens at render time (mermaid's style parser
  splits on commas, so `rgba()`/`color-mix()` values are converted first). The chart is re-rendered on theme toggle;
- mermaid has no orthogonal router, so after `mermaid.render` each `path.flowchart-link` (drawn with `curve: 'linear'`)
  is rewritten into vertical-first elbows (`orthogonalEdges()`);
- clicks are matched through the element id mermaid assigns (`flowchart-<id>-<n>`); the selected node gets a `.selected`
  class (thicker `--text` outline) and its edges `.hl`; pan/zoom is a CSS transform on `#graphCanvas` (fit-to-width on
  first render, "Fit" fits the whole chart);
- "Copy Mermaid" copies the same text to the clipboard and "View source" shows it, so the chart can be pasted inside a
  ```` ```mermaid ```` fence in docs.

## Responsive

- ≤ 1280 px: nav rail collapses to icons.
- ≤ 1180 px: right column stacks under the centre column.
- Verified at 1440×900 and 1100×850 (no horizontal scroll).

## Theme

Dark by default; the rail button toggles light/dark and stores the choice in `localStorage`
(`atalla.theme`, wrapped in try/catch).
