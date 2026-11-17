// app.js — Atalla Benchmark Workbench (vanilla ES module, no build step).
// Talks to bench/ui/server.py (API_CONTRACT.md section D). Falls back to mock.js
// when ?mock=1 is set or when the API is unreachable.
import { mockApi } from './mock.js';

/* ------------------------------------------------------------------ utils */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const icon = (id) => `<svg aria-hidden="true"><use href="#${id}"/></svg>`;
const fmtInt = (v) => (v == null || Number.isNaN(v)) ? '—' : Math.round(v).toLocaleString('en-US');
const fmtCos = (v) => v == null ? '—' : Number(v).toFixed(6);
const fmtSci = (v) => v == null ? '—' : Number(v).toExponential(3);
const fmtPct = (v) => v == null ? '—' : (v * 100).toFixed(1) + '%';
const fmtNum = (v, d = 3) => v == null ? '—' : Number(v).toFixed(d);
const fmtShape = (s) => Array.isArray(s) ? `[${s.join(', ')}]` : (s ?? '—');
const fmtDate = (iso) => { if (!iso) return '—'; const d = new Date(iso); return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }); };
const fmtDur = (s) => s == null ? '—' : (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`);
const cosClass = (c) => c == null ? '' : c >= 0.999 ? 'ok' : c >= 0.99 ? 'warn' : 'bad';
// Backend colours come from the CSS tokens (DESIGN_NOTES.md mapping) and are read at render time so a theme switch recolours everything.
const BACKEND_VAR = { emulator: '--accent', numpy: '--amber', passthrough: '--text-3', skip: '--red', graph: '--line' };
const cssVar = (name, cs = getComputedStyle(document.documentElement)) => cs.getPropertyValue(name).trim();
const backendColor = (b, cs) => cssVar(BACKEND_VAR[b] || '--line', cs);
const BACKEND_COLOR = new Proxy({}, { get: (_, b) => backendColor(b) });
const BACKEND_LABEL = { emulator: 'Emulated', numpy: 'NumPy', passthrough: 'Passthrough', skip: 'Skip', graph: 'Graph' };

function toast(msg, kind = '') {
  const t = el(`<div class="toast ${kind}" role="status">${esc(msg)}</div>`);
  $('#toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 200ms'; setTimeout(() => t.remove(), 220); }, kind === 'error' ? 6000 : 3200);
}

function modal(title, sub, bodyNode) {
  closeModal();
  const m = el(`<div class="modal-backdrop" role="dialog" aria-modal="true" aria-label="${esc(title)}">
    <div class="modal">
      <div class="modal-head"><div><h3>${esc(title)}</h3>${sub ? `<div class="sub mono">${esc(sub)}</div>` : ''}</div><span class="spacer"></span>
        <button class="btn icon ghost" data-close aria-label="Close">${icon('i-x')}</button></div>
      <div class="modal-body"></div>
    </div></div>`);
  m.querySelector('.modal-body').appendChild(bodyNode);
  m.addEventListener('click', (e) => { if (e.target === m || e.target.closest('[data-close]')) closeModal(); });
  $('#modalRoot').appendChild(m);
  m.querySelector('[data-close]').focus();
  return m;
}
function closeModal() { $('#modalRoot').innerHTML = ''; }
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

function codeView(text) {
  const lines = String(text ?? '').replace(/\r/g, '').split('\n');
  const c = el(`<div class="code"><pre></pre></div>`);
  const pre = c.querySelector('pre');
  pre.innerHTML = lines.map((l) => `<div class="ln"><span>${esc(l) || ' '}</span></div>`).join('');
  return c;
}
const skeleton = (tall = false) => el(`<div class="skeleton"><div class="bar w40"></div><div class="bar w80"></div><div class="bar w60"></div>${tall ? '<div class="bar tall"></div>' : ''}</div>`);
const emptyState = (title, body, iconId = 'i-inbox') => el(`<div class="empty">${icon(iconId)}<h3>${esc(title)}</h3><div>${esc(body)}</div></div>`);

/* ------------------------------------------------------------------ API */
const params = new URLSearchParams(location.search);
class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }
async function http(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json().catch(() => null) : await r.text();
  if (!r.ok) throw new ApiError((body && body.error) || (typeof body === 'string' && body.slice(0, 200)) || `${r.status} ${r.statusText}`, r.status);
  return body;
}
const realApi = {
  name: 'api',
  models: () => http('/api/models'),
  graph: (body) => http('/api/graph', { method: 'POST', body: JSON.stringify(body) }),
  kernels: () => http('/api/kernels'),
  kernel: (name) => http(`/api/kernels/${encodeURIComponent(name)}`),
  createRun: (body) => http('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  runs: () => http('/api/runs'),
  run: (id) => http(`/api/runs/${encodeURIComponent(id)}`),
  events: (id, onEvent) => {
    const es = new EventSource(`/api/runs/${encodeURIComponent(id)}/events`);
    let got = false;
    es.onmessage = (m) => { got = true; try { onEvent(JSON.parse(m.data)); } catch (e) { console.error('bad event', m.data); } };
    es.onerror = () => { if (es.readyState === EventSource.CLOSED && !got) onEvent({ event: 'error', message: 'event stream closed before any data' }); if (es.readyState === EventSource.CLOSED) es.close(); };
    return () => es.close();
  },
  log: (id) => http(`/api/runs/${encodeURIComponent(id)}/log`),
  artifact: (id, node, kind) => http(`/api/runs/${encodeURIComponent(id)}/artifact?node=${encodeURIComponent(node)}&kind=${encodeURIComponent(kind)}`),
  cancel: (id) => http(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  exportRun: (id) => http(`/api/runs/${encodeURIComponent(id)}/export`, { method: 'POST' }),
};
let api = params.get('mock') === '1' ? mockApi : realApi;
const isNetErr = (e) => e instanceof TypeError || /Failed to fetch|NetworkError|Load failed/i.test(String(e && e.message));
function setModePill() {
  const p = $('#modePill');
  p.classList.toggle('mock', api === mockApi);
  p.querySelector('.lbl').textContent = api === mockApi ? 'mock data' : 'live API';
  p.title = api === mockApi ? 'Mock mode: data from mock.js (?mock=1 or API unreachable)' : 'Connected to the FastAPI server';
}
async function call(fn, ...args) {
  try { return await api[fn](...args); }
  catch (e) {
    if (api === realApi && isNetErr(e)) { api = mockApi; setModePill(); toast('API unreachable: switched to mock data', 'warn'); return api[fn](...args); }
    throw e;
  }
}

/* ------------------------------------------------------------------ state */
const state = {
  models: [], model: null, seed: 42, scale: null, options: {}, strict: false, tag: '',
  overrides: { global: {}, nodes: {} },
  graph: null, graphLoading: false, selected: null,
  // mermaid call graph: diagram text, sanitized-id maps, rendered node elements, viewport transform, parameter toggle
  mermaidText: '', graphIds: null, graphEls: null, gz: null, showParams: false,
  runs: [], runsSort: { key: 'started', dir: -1 }, runsFilter: '', compare: [],
  run: null, // { id, status, events, summary, config, start, unsubscribe, charts, selected, export }
  kernels: [], kernel: null, kernelSrc: null,
  view: null,
};

/* ------------------------------------------------------------------ theme */
const prefersDark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : { matches: false };
const effectiveTheme = () => document.documentElement.getAttribute('data-theme') || (prefersDark.matches ? 'dark' : 'light');
function syncThemeButton() { const t = effectiveTheme(); $('#themeToggle').innerHTML = `${icon(t === 'dark' ? 'i-sun' : 'i-moon')}<span class="lbl">${t === 'dark' ? 'Light' : 'Dark'}</span>`; $('#themeToggle').title = `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`; }
function applyTheme(t) { if (t) { document.documentElement.setAttribute('data-theme', t); try { localStorage.setItem('atalla.theme', t); } catch (e) { /* ignore */ } } syncThemeButton(); refreshChartTheme(); }
$('#themeToggle').addEventListener('click', () => applyTheme(effectiveTheme() === 'dark' ? 'light' : 'dark'));
if (prefersDark.addEventListener) prefersDark.addEventListener('change', () => { if (!document.documentElement.getAttribute('data-theme')) applyTheme(null); });

function chartColors() {
  const cs = getComputedStyle(document.documentElement);
  return { text: cssVar('--text-2', cs), grid: cssVar('--line', cs), accent: cssVar('--accent', cs), success: cssVar('--green', cs), warning: cssVar('--amber', cs), danger: cssVar('--red', cs) };
}
function refreshChartTheme() {
  if (!window.Chart) return;
  const c = chartColors();
  Chart.defaults.color = c.text; Chart.defaults.borderColor = c.grid; Chart.defaults.font.family = getComputedStyle(document.body).fontFamily; Chart.defaults.font.size = 11;
  // Recolour live charts: datasets were built from the previous theme's tokens.
  Object.values(Chart.instances || {}).forEach((ch) => { recolourChart(ch, c); ch.update('none'); });
  // Graph colours are read from the CSS variables at draw time, so rebuild it on a theme change.
  if (state.view === 'workbench' && state.graph && $('#graphCanvas')) drawGraph();
}

// Datasets tag themselves with the token they were built from (`tok`) so a theme switch can re-read it.
function recolourChart(ch, c) {
  ch.data.datasets.forEach((ds) => {
    if (ds.tok && c[ds.tok]) { if (ds.borderColor) ds.borderColor = c[ds.tok]; ds.backgroundColor = c[ds.tok]; }
    if (ds.cosPoints && Array.isArray(ds.pointBackgroundColor)) ds.pointBackgroundColor = ds.data.map((v) => (v >= 0.999 ? c.success : v >= 0.99 ? c.warning : c.danger));
  });
  ch.options.color = c.text;
}

/* ------------------------------------------------------------------ router */
function route() {
  const hash = location.hash || '#/workbench';
  const [, path = 'workbench', ...rest] = hash.split('/');
  const view = path.split('?')[0];
  state.view = view;
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.view === (view === 'run' || view === 'compare' ? 'runs' : view)));
  closeModal();
  if (view === 'run' && rest[0]) return renderRun(decodeURIComponent(rest[0]));
  if (view === 'runs') return renderRuns();
  if (view === 'compare' && rest.length >= 2) return renderCompare(decodeURIComponent(rest[0]), decodeURIComponent(rest[1]));
  if (view === 'kernels') return renderKernels(rest[0] ? decodeURIComponent(rest[0]) : null);
  if (view === 'metrics') return renderMetrics();
  return renderWorkbench();
}
window.addEventListener('hashchange', route);
const go = (h) => { location.hash = h; };

function setHeader(title, sub, actionsHtml = '') { $('#title').textContent = title; $('#subtitle').textContent = sub || ''; $('#topActions').innerHTML = actionsHtml; }
function setRunButton() {
  const b = $('#runBtn');
  const running = state.run && state.run.status === 'running';
  b.innerHTML = running ? `${icon('i-stop')}Cancel` : `${icon('i-play')}Run`;
  b.classList.toggle('danger', !!running); b.classList.toggle('primary', !running);
  b.disabled = !state.model && !running;
}
$('#runBtn').addEventListener('click', async () => {
  if (state.run && state.run.status === 'running') { try { await call('cancel', state.run.id); toast('Cancel requested'); } catch (e) { toast(e.message, 'error'); } return; }
  launchRun();
});

/* ------------------------------------------------------------------ overrides helpers */
function currentModel() { return state.models.find((m) => m.name === state.model) || null; }
function buildOverrides() {
  const global = {};
  const m = currentModel();
  (m ? m.options : []).forEach((o) => { const v = state.options[o.key]; if (v != null && v !== '' && v !== o.default) global[o.key] = o.type === 'int' ? Number(v) : v; });
  const nodes = {};
  Object.entries(state.overrides.nodes).forEach(([k, v]) => { if (v && (v.backend || v.kernel)) nodes[k] = { backend: v.backend || 'emulator', kernel: v.kernel || null }; });
  return { global, nodes };
}
function runBody() {
  const m = currentModel();
  const body = { model: state.model, scale: m && m.defaults && m.defaults.scale != null ? Number(state.scale) : null, seed: Number(state.seed) || 42, tag: state.tag || 'ui', strict: !!state.strict, overrides: buildOverrides() };
  return body;
}

/* ================================================================== WORKBENCH */
async function ensureModels() {
  if (state.models.length) return;
  state.models = await call('models');
  if (!state.model && state.models.length) selectModel(state.models.find((m) => m.name === 'gpt2_block') ? 'gpt2_block' : state.models[0].name);
}
function selectModel(name) {
  state.model = name; const m = currentModel();
  state.seed = (m && m.defaults && m.defaults.seed) || 42;
  state.scale = m && m.defaults ? m.defaults.scale : null;
  state.options = {}; (m ? m.options : []).forEach((o) => { state.options[o.key] = o.default; });
  state.overrides = { global: {}, nodes: {} }; state.graph = null; state.selected = null;
  setRunButton();
}

async function renderWorkbench() {
  const content = $('#content'); content.innerHTML = ''; $('#right').innerHTML = '';
  setHeader('Workbench', 'Pick a model, inspect its call graph, map ops to Atalla kernels, then run.');
  content.appendChild(skeleton(true));
  try { await ensureModels(); } catch (e) { content.innerHTML = ''; content.appendChild(emptyState('Could not load models', e.message)); toast(e.message, 'error'); return; }
  if (state.view !== 'workbench') return;
  content.innerHTML = '';
  content.appendChild(controlsCard());
  content.appendChild(graphCard());
  renderNodePanel();
  setRunButton();
  if (state.graph) drawGraph(); else inspectGraph();
}

function controlsCard() {
  const m = currentModel();
  const card = el(`<div class="card"><div class="card-head"><h2>Model & options</h2><span class="spacer"></span><span class="muted small">${esc(m ? m.description : '')}</span></div><div class="form-grid" id="ctlGrid"></div><div class="btn-row mt16" id="ctlActions"></div></div>`);
  const grid = card.querySelector('#ctlGrid');
  const sel = el(`<div class="field"><label for="modelSel">Model</label><select id="modelSel">${state.models.map((x) => `<option value="${esc(x.name)}" ${x.name === state.model ? 'selected' : ''}>${esc(x.title || x.name)}</option>`).join('')}</select></div>`);
  sel.querySelector('select').addEventListener('change', (e) => { selectModel(e.target.value); renderWorkbench(); });
  grid.appendChild(sel);
  const seed = el(`<div class="field"><label for="seedIn">Seed</label><input id="seedIn" type="number" value="${esc(state.seed)}"></div>`);
  seed.querySelector('input').addEventListener('input', (e) => { state.seed = e.target.value; });
  grid.appendChild(seed);
  if (m && m.defaults && m.defaults.scale != null) {
    const sc = el(`<div class="field"><label for="scaleIn">Scale</label><input id="scaleIn" type="number" step="0.001" value="${esc(state.scale)}"></div>`);
    sc.querySelector('input').addEventListener('input', (e) => { state.scale = e.target.value; });
    grid.appendChild(sc);
  }
  (m ? m.options : []).forEach((o) => {
    let f;
    if (o.choices) {
      f = el(`<div class="field"><label for="opt_${esc(o.key)}">${esc(o.label)}</label><select id="opt_${esc(o.key)}">${o.choices.map((c) => `<option value="${esc(c)}" ${state.options[o.key] === c ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></div>`);
      f.querySelector('select').addEventListener('change', (e) => { state.options[o.key] = e.target.value; });
    } else {
      f = el(`<div class="field"><label for="opt_${esc(o.key)}">${esc(o.label)}</label><input id="opt_${esc(o.key)}" type="number" value="${esc(state.options[o.key])}"></div>`);
      f.querySelector('input').addEventListener('input', (e) => { state.options[o.key] = e.target.value; });
    }
    grid.appendChild(f);
  });
  const tag = el(`<div class="field"><label for="tagIn">Tag</label><input id="tagIn" placeholder="e.g. gelu-horner" value="${esc(state.tag)}"></div>`);
  tag.querySelector('input').addEventListener('input', (e) => { state.tag = e.target.value.trim(); });
  grid.appendChild(tag);
  const actions = card.querySelector('#ctlActions');
  const strict = el(`<label class="switch"><input type="checkbox" ${state.strict ? 'checked' : ''}> Strict end check (cos ≥ 0.999, rel-L2 ≤ 0.02, rel-max ≤ 0.05)</label>`);
  strict.querySelector('input').addEventListener('change', (e) => { state.strict = e.target.checked; });
  actions.appendChild(strict);
  actions.appendChild(el('<span class="spacer" style="flex:1"></span>'));
  const insp = el(`<button class="btn">${icon('i-graph')}Inspect graph</button>`); insp.addEventListener('click', () => inspectGraph());
  const run = el(`<button class="btn primary">${icon('i-play')}Run</button>`); run.addEventListener('click', () => launchRun());
  actions.appendChild(insp); actions.appendChild(run);
  return card;
}

function graphCard() {
  const card = el(`<div class="card" id="graphCard"><div class="card-head"><h2>Call graph <span class="count" id="graphCount"></span></h2><span class="spacer"></span>
      <button class="btn sm ghost" id="copyMm" title="Copy the Mermaid flowchart source to the clipboard">${icon('i-copy')}Copy Mermaid</button>
      <button class="btn sm ghost" id="srcMm" title="View the Mermaid flowchart source">${icon('i-code')}View source</button>
      <button class="btn sm" id="applyOv" disabled>Apply overrides</button></div>
    <div class="chip-list mb8" id="ovChips"></div>
    <div class="counts" id="counts"></div>
    <div class="graph-wrap"><div id="graphView" aria-label="Call graph: drag to pan, wheel to zoom, click a node to select it"><div id="graphCanvas"></div></div><div class="graph-tools">
      <button class="btn icon" data-g="in" aria-label="Zoom in" title="Zoom in">${icon('i-plus')}</button>
      <button class="btn icon" data-g="out" aria-label="Zoom out" title="Zoom out">${icon('i-minus')}</button>
      <button class="btn icon" data-g="fit" aria-label="Fit" title="Fit to view">${icon('i-fit')}</button></div>
      <div id="graphOverlay" class="hidden" style="position:absolute;inset:0;display:grid;place-items:center;background:color-mix(in srgb, var(--bg) 70%, transparent);z-index:4"></div>
    </div>
    <div class="legend"><span class="legend-items">${['emulator', 'numpy', 'passthrough', 'skip', 'graph'].map((b) => `<span><i class="sw" style="background:${b === 'graph' ? 'var(--fill)' : `var(${BACKEND_VAR[b]})`};border-color:${b === 'graph' ? 'var(--text-3)' : `var(${BACKEND_VAR[b]})`};${b === 'graph' ? 'border-style:dashed' : ''}"></i>${BACKEND_LABEL[b]}</span>`).join('')}<span><i class="sw param"></i>Parameter</span></span>
      <span class="spacer"></span><label class="switch small" title="Show weights, biases and mask buffers (get_attr nodes) as dashed nodes beside the op that consumes them"><input type="checkbox" id="showParams" ${state.showParams ? 'checked' : ''}> Show parameters</label></div></div>`);
  card.querySelector('.graph-tools').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b || !$('#graphCanvas svg')) return;
    if (b.dataset.g === 'fit') fitGraph('all');
    else { const v = $('#graphView'); zoomGraphAt(b.dataset.g === 'in' ? 1.25 : 0.8, v.clientWidth / 2, v.clientHeight / 2); }
  });
  card.querySelector('#applyOv').addEventListener('click', () => inspectGraph());
  card.querySelector('#copyMm').addEventListener('click', () => copyMermaid());
  card.querySelector('#srcMm').addEventListener('click', () => openMermaidSource());
  card.querySelector('#showParams').addEventListener('change', (e) => { state.showParams = e.target.checked; if (state.graph) drawGraph(); });
  initGraphViewport(card.querySelector('#graphView'), card.querySelector('#graphCanvas'));
  return card;
}

async function inspectGraph() {
  if (!state.model) return;
  state.graphLoading = true;
  const ov = $('#graphOverlay'); if (ov) { ov.classList.remove('hidden'); ov.innerHTML = '<div class="skeleton" style="width:60%"><div class="bar w80"></div><div class="bar w60"></div><div class="bar w40"></div></div>'; }
  try {
    const body = { model: state.model, scale: state.scale != null && state.scale !== '' ? Number(state.scale) : null, seed: Number(state.seed) || 42, overrides: buildOverrides() };
    const g = await call('graph', body);
    state.graph = g;
    if (state.selected && !g.nodes.find((n) => n.id === state.selected)) state.selected = null;
    if (state.view === 'workbench') { drawGraph(); renderNodePanel(); }
  } catch (e) { toast(`Inspect failed: ${e.message}`, 'error'); }
  finally { state.graphLoading = false; const o = $('#graphOverlay'); if (o) o.classList.add('hidden'); markOverridesDirty(false); }
}

function markOverridesDirty(dirty) { const b = $('#applyOv'); if (b) b.disabled = !dirty; }

/* ---------------------------------------------------------------- call graph (mermaid flowchart) */
// Ids that would collide with flowchart keywords or edge syntax (`x`/`o` next to `--` draw cross/circle edge ends).
const MERMAID_RESERVED = new Set(['end', 'graph', 'flowchart', 'subgraph', 'style', 'class', 'classdef', 'click', 'linkstyle', 'direction', 'default', 'o', 'x']);
const isParamNode = (n) => n.op === 'get_attr';
const isBoundaryNode = (n) => n.op === 'placeholder' || n.op === 'output';
const mmText = (s) => String(s ?? '').replace(/"/g, '#quot;').replace(/</g, '#lt;').replace(/>/g, '#gt;');

// Resolve a CSS colour expression to an opaque hex string (mermaid's classDef parser splits on commas, so rgba()/rgb() cannot be passed through).
function parseCssRgb(str) {
  // Chrome reports rgb()/rgba() with 0-255 channels, but color-mix() results as `color(srgb r g b / a)` with 0-1 floats.
  const m = (str.match(/-?[\d.]+(?:e-?\d+)?/g) || ['0', '0', '0']).map(Number);
  const srgb = /^color\(srgb/.test(str);
  const [r, g, b] = m.slice(0, 3).map((v) => (srgb ? v * 255 : v)); const a = m.length > 3 ? m[3] : 1;
  return [Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b)), a];
}
function cssHex(expr, bgRgb) {
  const d = document.createElement('span'); d.style.color = expr; document.body.appendChild(d);
  const m = parseCssRgb(getComputedStyle(d).color); d.remove();
  let [r, g, b] = m; const a = m[3];
  if (a < 1 && bgRgb) { r = a * r + (1 - a) * bgRgb[0]; g = a * g + (1 - a) * bgRgb[1]; b = a * b + (1 - a) * bgRgb[2]; }
  return '#' + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
}

// Theme + per-backend colours for one render, read from the CSS tokens (DESIGN_NOTES.md mapping) so a theme switch recolours the chart.
function graphPalette() {
  const cs = getComputedStyle(document.documentElement);
  const bgD = document.createElement('span'); bgD.style.color = cssVar('--bg', cs); document.body.appendChild(bgD);
  const bg = parseCssRgb(getComputedStyle(bgD).color).slice(0, 3); bgD.remove();
  const hx = (name) => cssHex(cssVar(name, cs), bg);
  const blend = (name, pct) => cssHex(`color-mix(in srgb, ${cssVar(name, cs)} ${pct}%, ${cssVar('--bg', cs)})`, bg);
  const dark = effectiveTheme() === 'dark';
  return {
    bg: hx('--bg'), fill: hx('--fill'), raised: hx('--raised'), text: hx('--text'), muted: hx('--text-2'), faint: blend('--text', 42),
    line: blend('--text', 22), edge: blend('--text', 34), accent: hx('--accent'), amber: hx('--amber'), red: hx('--red'),
    onPrimary: hx('--on-primary'), numpyText: dark ? hx('--on-primary') : '#ffffff', dark,
    font: getComputedStyle(document.body).fontFamily,
  };
}

// Build the `flowchart TD` text from the /api/graph result. Nodes are declared in FX execution order and their
// incoming edges follow immediately, so dagre sees the same order the model executes in. Parameters (get_attr)
// are left out unless showParams; when shown they are declared right before their consumer as dashed `[/name/]`
// nodes with a dotted edge, which keeps them beside the op instead of ranked at the top.
function buildMermaid(g, showParams, pal) {
  const nodes = g.nodes.slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const nameCount = {}; nodes.forEach((n) => { nameCount[n.name] = (nameCount[n.name] || 0) + 1; });
  const used = new Set(); const toMid = {}; const toReal = {};
  nodes.forEach((n) => {
    let s = String(n.id).replace(/[^A-Za-z0-9_]/g, '_');
    if (!s || /^[0-9]/.test(s) || MERMAID_RESERVED.has(s.toLowerCase())) s = 'n_' + s;
    let u = s, k = 2; while (used.has(u)) u = `${s}_${k++}`;
    used.add(u); toMid[n.id] = u; toReal[u] = n.id;
  });
  const title = (n) => mmText(nameCount[n.name] > 1 && n.name !== n.id ? `${n.name} (${n.id})` : n.name || n.id);
  const decl = (n) => {
    const id = toMid[n.id];
    if (isBoundaryNode(n)) return `${id}(["${title(n)}"]):::boundary`;
    if (isParamNode(n)) return `${id}[/"${title(n)}"/]:::param`;
    return `${id}["${title(n)}<br/><small>${mmText(n.op)}</small>"]:::${n.backend === 'graph' ? 'boundary' : n.backend}`;
  };
  const lines = ['flowchart TD'];
  const declared = new Set();
  nodes.forEach((n) => {
    if (isParamNode(n)) return; // declared beside their consumer (or hidden)
    lines.push(`  ${decl(n)}`); declared.add(n.id);
    (n.inputs || []).forEach((src) => {
      const s = byId.get(src); if (!s) return;
      if (isParamNode(s)) {
        if (!showParams) return;
        if (!declared.has(s.id)) { lines.push(`  ${decl(s)}`); declared.add(s.id); }
        lines.push(`  ${toMid[s.id]} -.-> ${toMid[n.id]}`);
      } else lines.push(`  ${toMid[s.id]} --> ${toMid[n.id]}`);
    });
  });
  lines.push('');
  lines.push(`  classDef emulator fill:${pal.accent},stroke:${pal.accent},color:#ffffff`);
  lines.push(`  classDef numpy fill:${pal.amber},stroke:${pal.amber},color:${pal.numpyText}`);
  lines.push(`  classDef passthrough fill:${pal.faint},stroke:${pal.faint},color:${pal.text}`);
  lines.push(`  classDef skip fill:${pal.red},stroke:${pal.red},color:#ffffff`);
  lines.push(`  classDef boundary fill:${pal.fill},stroke:${pal.muted},stroke-dasharray:4 3,color:${pal.text}`);
  lines.push(`  classDef param fill:${pal.fill},stroke:${pal.faint},stroke-dasharray:3 3,color:${pal.muted}`);
  return { text: lines.join('\n'), toMid, toReal };
}

let graphRenderSeq = 0;
async function drawGraph() {
  const g = state.graph; const canvas = $('#graphCanvas'); if (!g || !canvas) return;
  const c = g.counts || {};
  const params = g.nodes.filter(isParamNode).length;
  $('#graphCount').textContent = `${g.model_class || g.model} · ${c.total ?? g.nodes.length} nodes · input ${fmtShape(g.input_shape)}`;
  $('#counts').innerHTML = [['emulated', 'Emulated', 'emulator'], ['numpy', 'NumPy', 'numpy'], ['passthrough', 'Passthrough', 'passthrough'], ['skip', 'Skip', 'skip']].map(([k, l, b]) => `<span class="stat"><i class="sw" style="background:var(${BACKEND_VAR[b]})"></i>${l} <b>${c[k] ?? 0}</b></span>`).join('')
    + `<span class="stat">Graph nodes <b>${(c.total ?? g.nodes.length) - ((c.emulated || 0) + (c.numpy || 0) + (c.passthrough || 0) + (c.skip || 0))}</b></span>`
    + `<span class="stat" title="get_attr nodes (weights, biases, mask buffers); ${state.showParams ? 'shown as dashed nodes' : 'hidden; listed under inputs in the node panel'}">Parameters <b>${params}</b>${state.showParams ? '' : ' <span class="muted">hidden</span>'}</span>`;
  renderOverrideChips();
  if (!window.mermaid) { toast('mermaid failed to load from CDN', 'error'); return; }
  const pal = graphPalette();
  const built = buildMermaid(g, state.showParams, pal);
  state.mermaidText = built.text; state.graphIds = built;
  mermaid.initialize({
    startOnLoad: false, securityLevel: 'loose', theme: 'base', fontFamily: pal.font,
    themeVariables: { fontFamily: pal.font, fontSize: '12px', background: pal.fill, primaryColor: pal.raised, primaryTextColor: pal.text, primaryBorderColor: pal.line, lineColor: pal.edge, secondaryColor: pal.fill, tertiaryColor: pal.fill, textColor: pal.text, nodeBorder: pal.line, mainBkg: pal.raised, edgeLabelBackground: pal.raised, clusterBkg: pal.fill, clusterBorder: pal.line, titleColor: pal.text },
    flowchart: { curve: 'linear', nodeSpacing: 26, rankSpacing: 40, htmlLabels: true, useMaxWidth: false, padding: 10, diagramPadding: 12 },
  });
  const seq = ++graphRenderSeq;
  let out;
  try { out = await mermaid.render(`atallaGraph${seq}`, built.text); }
  catch (e) { console.error('mermaid render failed', e); toast(`Graph render failed: ${e.message || e}`, 'error'); return; }
  if (seq !== graphRenderSeq || !$('#graphCanvas')) return; // a newer render superseded this one or the view changed
  $('#graphCanvas').innerHTML = out.svg;
  const svg = $('#graphCanvas svg');
  svg.removeAttribute('style'); // mermaid sets max-width on the root svg; the wrapper owns sizing
  orthogonalEdges(svg);
  // Map rendered node groups back to real node ids: mermaid formats the element id as `flowchart-<id>-<n>`.
  const els = {};
  svg.querySelectorAll('g.node').forEach((ge) => { const m = /^flowchart-(.+)-\d+$/.exec(ge.id || ''); const real = m && built.toReal[m[1]]; if (real) { els[real] = ge; ge.dataset.node = real; ge.setAttribute('tabindex', '0'); ge.setAttribute('role', 'button'); } });
  state.graphEls = els;
  svg.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && e.target.closest) { const n = e.target.closest('g.node'); if (n && n.dataset.node) { e.preventDefault(); selectNode(n.dataset.node); } } });
  const key = `${g.model}|${state.showParams}|${svg.viewBox.baseVal.width}x${svg.viewBox.baseVal.height}`;
  if (!state.gz || state.gz.key !== key) { state.gz = { k: 1, x: 0, y: 0, key }; fitGraph('width'); } else applyGraphTransform();
  highlightGraphSelection();
}

// Mermaid has no orthogonal router (its d3 curves put the elbow at mid-x, which hugs the node edges in a TB chart).
// Rewrite each linear edge path into vertical-first elbows: down to the gap between ranks, across, down into the target.
function orthogonalEdges(svg) {
  svg.querySelectorAll('path.flowchart-link').forEach((path) => {
    const nums = (path.getAttribute('d') || '').match(/-?\d*\.?\d+(?:e-?\d+)?/g);
    if (!nums || nums.length < 4 || /[CQSA]/i.test((path.getAttribute('d') || '').replace(/^M/, '').replace(/L/g, ''))) return;
    const pts = []; for (let i = 0; i + 1 < nums.length; i += 2) pts.push({ x: +nums[i], y: +nums[i + 1] });
    // drop near-duplicate points and tiny x jogs between waypoints (dagre dummy nodes) so long edges do not staircase
    const q = [pts[0]];
    for (let i = 1; i < pts.length; i++) { const a = q[q.length - 1], b = pts[i]; if (Math.abs(a.x - b.x) < 1 && Math.abs(a.y - b.y) < 1) continue; q.push(i < pts.length - 1 && Math.abs(b.x - a.x) < 6 ? { x: a.x, y: b.y } : b); }
    if (q.length < 2) return;
    let d = `M${q[0].x},${q[0].y}`;
    for (let i = 1; i < q.length; i++) {
      const a = q[i - 1], b = q[i];
      if (Math.abs(a.x - b.x) < 0.5 || Math.abs(a.y - b.y) < 0.5) { d += ` L${b.x},${b.y}`; continue; }
      const ym = (a.y + b.y) / 2; d += ` L${a.x},${ym} L${b.x},${ym} L${b.x},${b.y}`;
    }
    path.setAttribute('d', d);
  });
}

function highlightGraphSelection() {
  const canvas = $('#graphCanvas'); if (!canvas || !state.graphIds) return;
  canvas.querySelectorAll('g.node.selected').forEach((n) => n.classList.remove('selected'));
  canvas.querySelectorAll('.flowchart-link.hl').forEach((p) => p.classList.remove('hl'));
  const mid = state.selected && state.graphIds.toMid[state.selected]; if (!mid) return;
  const ge = state.graphEls && state.graphEls[state.selected]; if (ge) ge.classList.add('selected');
  canvas.querySelectorAll(`.LS-${CSS.escape(mid)}, .LE-${CSS.escape(mid)}`).forEach((p) => p.classList.add('hl'));
}

function selectNode(id) {
  state.selected = id;
  highlightGraphSelection();
  renderNodePanel();
}

/* viewport: pan by dragging, zoom with the wheel or the +/−/fit buttons; a CSS transform on #graphCanvas */
function applyGraphTransform() { const gz = state.gz, c = $('#graphCanvas'); if (gz && c) c.style.transform = `translate(${gz.x}px, ${gz.y}px) scale(${gz.k})`; }
function zoomGraphAt(f, px, py) {
  const gz = state.gz; if (!gz) return;
  const k = Math.min(3, Math.max(0.15, gz.k * f)); const r = k / gz.k;
  gz.x = px - (px - gz.x) * r; gz.y = py - (py - gz.y) * r; gz.k = k; applyGraphTransform();
}
function fitGraph(mode = 'all') {
  const view = $('#graphView'), svg = $('#graphCanvas svg'), gz = state.gz; if (!view || !svg || !gz) return;
  const W = view.clientWidth, H = view.clientHeight, w = svg.viewBox.baseVal.width || svg.clientWidth, h = svg.viewBox.baseVal.height || svg.clientHeight, pad = 20;
  let k = (W - 2 * pad) / w; if (mode === 'all') k = Math.min(k, (H - 2 * pad) / h);
  k = Math.min(1.15, Math.max(0.15, k));
  gz.k = k; gz.x = (W - w * k) / 2; gz.y = mode === 'all' ? Math.max(pad, (H - h * k) / 2) : pad;
  applyGraphTransform();
}
function initGraphViewport(view, canvas) {
  let drag = null;
  view.addEventListener('pointerdown', (e) => { if (e.button !== 0) return; drag = { x: e.clientX, y: e.clientY, ox: state.gz ? state.gz.x : 0, oy: state.gz ? state.gz.y : 0, moved: false, target: e.target }; view.setPointerCapture(e.pointerId); });
  view.addEventListener('pointermove', (e) => {
    if (!drag || !state.gz) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; view.classList.add('dragging'); }
    if (drag.moved) { state.gz.x = drag.ox + dx; state.gz.y = drag.oy + dy; applyGraphTransform(); }
  });
  const end = () => { if (!drag) return null; const d = drag; drag = null; view.classList.remove('dragging'); return d; };
  view.addEventListener('pointerup', () => {
    const d = end(); if (!d || d.moved) return;
    const n = d.target && d.target.closest ? d.target.closest('g.node') : null;
    if (n && n.dataset.node) selectNode(n.dataset.node);
    else if (state.selected) { state.selected = null; highlightGraphSelection(); renderNodePanel(); }
  });
  view.addEventListener('pointercancel', end);
  view.addEventListener('wheel', (e) => { if (!state.gz) return; e.preventDefault(); const r = view.getBoundingClientRect(); zoomGraphAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  canvas.style.transformOrigin = '0 0';
}

/* Mermaid source: copy to clipboard / view in a modal so the same chart can be pasted into docs */
async function copyMermaid() {
  const text = state.mermaidText; if (!text) { toast('No graph yet', 'warn'); return; }
  try { await navigator.clipboard.writeText(text); toast('Mermaid source copied', 'ok'); return; } catch (e) { /* fall through to execCommand */ }
  try { const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); toast(ok ? 'Mermaid source copied' : 'Copy failed: select the text in View source', ok ? 'ok' : 'warn'); }
  catch (e) { toast('Copy failed: select the text in View source', 'warn'); }
}
function openMermaidSource() {
  const text = state.mermaidText; if (!text) { toast('No graph yet', 'warn'); return; }
  const g = state.graph || {};
  const body = el('<div></div>');
  const row = el(`<div class="btn-row mb8"><button class="btn sm">${icon('i-copy')}Copy</button><span class="muted small">Paste inside a <code>\`\`\`mermaid</code> fence in Markdown. Colours are the current theme's tokens.</span></div>`);
  row.querySelector('button').addEventListener('click', () => copyMermaid());
  body.appendChild(row); body.appendChild(codeView(text));
  modal('Mermaid source', `${g.model_class || g.model || ''} · ${text.split('\n').length} lines · parameters ${state.showParams ? 'shown' : 'hidden'}`, body);
}

function renderOverrideChips() {
  const box = $('#ovChips'); if (!box) return;
  const ov = buildOverrides();
  const chips = [];
  Object.entries(ov.global).forEach(([k, v]) => chips.push({ key: `g:${k}`, label: `${k} = ${v}`, cls: 'emulator' }));
  Object.entries(ov.nodes).forEach(([k, v]) => chips.push({ key: `n:${k}`, label: `${k} → ${v.backend}${v.kernel ? ` (${v.kernel})` : ''}`, cls: v.backend }));
  box.innerHTML = chips.length ? '' : '<span class="muted small">No overrides. Select a node to force a backend, or change model options.</span>';
  chips.forEach((c) => {
    const chip = el(`<span class="chip x ${esc(c.cls)}"><span class="dot"></span>${esc(c.label)}<button aria-label="Remove override">${icon('i-x')}</button></span>`);
    chip.querySelector('button').addEventListener('click', () => {
      if (c.key.startsWith('g:')) { const k = c.key.slice(2); const m = currentModel(); const o = m && m.options.find((x) => x.key === k); state.options[k] = o ? o.default : null; const f = $(`#opt_${CSS.escape(k)}`); if (f) f.value = state.options[k]; }
      else delete state.overrides.nodes[c.key.slice(2)];
      renderOverrideChips(); renderNodePanel(); markOverridesDirty(true);
    });
    box.appendChild(chip);
  });
}

function renderNodePanel() {
  const right = $('#right'); if (state.view !== 'workbench') return;
  right.innerHTML = '';
  const g = state.graph;
  if (!g) { right.appendChild(el('<div class="card"><h2>Node</h2></div>')).appendChild(skeleton()); return; }
  const n = g.nodes.find((x) => x.id === state.selected);
  if (!n) { const c = el('<div class="card"></div>'); c.appendChild(emptyState('Select a node', 'Click any node in the call graph to see its kernel mapping, tile config and override controls.', 'i-graph')); right.appendChild(c); return; }
  const k = n.kernel || {};
  const card = el(`<div class="card">
    <div class="card-head"><h2>${esc(n.name)}</h2><span class="spacer"></span><span class="chip ${esc(n.backend)}"><span class="dot"></span>${esc(BACKEND_LABEL[n.backend] || n.backend)}</span></div>
    <dl class="kv">
      <dt>op</dt><dd>${esc(n.op)}</dd>
      <dt>target</dt><dd class="mono">${esc(n.target || '—')}</dd>
      <dt>shape</dt><dd class="mono">${esc(fmtShape(n.shape))}</dd>
      <dt>dtype</dt><dd>${esc(n.dtype || '—')}</dd>
      <dt>phase</dt><dd>${esc(n.phase || '—')}</dd>
      <dt>order</dt><dd>${esc(n.order ?? '—')}</dd>
      <dt>inputs</dt><dd class="mono">${(n.inputs || []).map((i) => `<a href="#" data-node="${esc(i)}">${esc(i)}</a>`).join(', ') || '—'}</dd>
    </dl>
    <div class="section-title">Kernel mapping</div>
    ${n.kernel ? `<dl class="kv">
      <dt>kernel</dt><dd><b>${esc(k.name)}</b>${k.variant ? ` <span class="chip emulator">${esc(k.variant)}</span>` : ''}</dd>
      <dt>source</dt><dd><a href="#" data-kernel="${esc(k.name)}" class="mono">${esc(k.source || '—')}</a></dd>
      <dt>dram_addr</dt><dd class="mono">${esc(k.dram_addr || '—')}</dd>
    </dl>` : `<div class="muted small">${n.backend === 'passthrough' ? 'No kernel: this op is taken from the PyTorch reference (view/reshape).' : n.backend === 'graph' ? 'Graph node: parameter or boundary, not executed.' : n.backend === 'skip' ? 'Replaced by identity on its first tensor input.' : 'No Atalla kernel mapped.'}</div>`}
    ${k.tile ? `<div class="section-title">Tile config</div><dl class="kv" id="tileKv"></dl>` : ''}
    ${k.params && Object.keys(k.params).length ? `<div class="section-title">Params</div><dl class="kv" id="paramKv"></dl>` : ''}
    <div class="section-title">Reason</div><div class="small">${esc(n.reason || '—')}</div>
  </div>`);
  const kvFill = (sel, obj) => { const d = card.querySelector(sel); if (!d) return; d.innerHTML = Object.entries(obj).map(([a, b]) => `<dt>${esc(a)}</dt><dd class="mono">${esc(typeof b === 'object' ? JSON.stringify(b) : b)}</dd>`).join(''); };
  if (k.tile) kvFill('#tileKv', k.tile);
  if (k.params) kvFill('#paramKv', k.params);
  card.addEventListener('click', async (e) => {
    const a = e.target.closest('a[data-node]'); if (a) { e.preventDefault(); selectNode(a.dataset.node); return; }
    const kk = e.target.closest('a[data-kernel]'); if (kk) { e.preventDefault(); openKernelModal(kk.dataset.kernel); }
  });
  right.appendChild(card);

  if (n.backend !== 'graph') {
    const cur = state.overrides.nodes[n.id] || {};
    const variants = n.op === 'gelu' ? ['horner', 'tanh'] : /sdpa|attention/.test(n.op) ? ['flash', 'numpy'] : null;
    const ov = el(`<div class="card mt16"><h2>Override</h2>
      <div class="section-title">Backend</div>
      <div class="radio-row" id="bkRadio">${['auto', 'emulator', 'numpy', 'skip'].map((b) => `<label class="${(cur.backend || 'auto') === b ? 'on' : ''}"><input type="radio" name="bk" value="${b}" ${(cur.backend || 'auto') === b ? 'checked' : ''}>${b}</label>`).join('')}</div>
      ${variants ? `<div class="section-title">Kernel variant</div><div class="field"><select id="varSel"><option value="">default</option>${variants.map((v) => `<option value="${v}" ${cur.kernel === v ? 'selected' : ''}>${v}</option>`).join('')}</select></div>` : ''}
      <div class="btn-row mt16"><button class="btn primary sm" id="applyOv2">Apply overrides</button><button class="btn sm ghost" id="clearOv">Clear</button></div>
      <div class="muted small mt8">numpy forces the reference path; emulator forces the Atalla kernel (errors if no emitter); skip replaces the node with identity on its first tensor input.</div>
    </div>`);
    const stage = () => {
      const bk = ov.querySelector('input[name=bk]:checked').value;
      const kernel = ov.querySelector('#varSel') ? ov.querySelector('#varSel').value : '';
      if (bk === 'auto' && !kernel) delete state.overrides.nodes[n.id]; else state.overrides.nodes[n.id] = { backend: bk === 'auto' ? null : bk, kernel: kernel || null };
      ov.querySelectorAll('#bkRadio label').forEach((l) => l.classList.toggle('on', l.querySelector('input').checked));
      renderOverrideChips(); markOverridesDirty(true);
    };
    ov.querySelectorAll('input[name=bk]').forEach((r) => r.addEventListener('change', stage));
    if (ov.querySelector('#varSel')) ov.querySelector('#varSel').addEventListener('change', stage);
    ov.querySelector('#applyOv2').addEventListener('click', () => inspectGraph());
    ov.querySelector('#clearOv').addEventListener('click', () => { delete state.overrides.nodes[n.id]; renderOverrideChips(); renderNodePanel(); markOverridesDirty(true); });
    right.appendChild(ov);
  }
}

async function openKernelModal(name) {
  const body = el('<div></div>'); body.appendChild(skeleton(true));
  modal(`Kernel source: ${name}`, '', body);
  try { const k = await call('kernel', name); body.innerHTML = ''; const m = $('.modal-head .sub, .modal-head div'); if (m) { const s = $('.modal-head'); s.querySelector('div').innerHTML = `<h3>Kernel source: ${esc(k.name)}</h3><div class="sub mono">${esc(k.path)}</div>`; } body.appendChild(codeView(k.text)); }
  catch (e) { body.innerHTML = ''; body.appendChild(emptyState('Could not load kernel', e.message)); }
}

/* ================================================================== RUN */
async function launchRun() {
  if (!state.model) { toast('Pick a model first', 'warn'); return; }
  if (state.run && state.run.status === 'running') { toast('A run is already in progress', 'warn'); go(`#/run/${encodeURIComponent(state.run.id)}`); return; }
  const body = runBody();
  try {
    const r = await call('createRun', body);
    attachRun(r.run_id, { config: body, fresh: true });
    toast(`Run ${r.run_id} started`, 'ok');
    go(`#/run/${encodeURIComponent(r.run_id)}`);
  } catch (e) { toast(e.status === 409 ? 'Server busy: a run is already in progress' : `Run failed to start: ${e.message}`, 'error'); }
}

function attachRun(id, { config = null, fresh = false } = {}) {
  if (state.run && state.run.unsubscribe) state.run.unsubscribe();
  const run = { id, status: 'running', events: [], summary: null, config, start: null, total: null, maxIndex: -1, unsubscribe: null, charts: {}, selected: null, export: null, lastTs: null };
  state.run = run;
  run.unsubscribe = api.events(id, (ev) => onRunEvent(run, ev));
  setRunButton();
  return run;
}

function onRunEvent(run, ev) {
  run.events.push(ev);
  if (ev.event === 'start') { run.start = ev.ts || Date.now() / 1000; run.total = ev.total_nodes; run.computeTotal = ev.compute_nodes; }
  if (ev.event === 'node_done' || ev.event === 'node_start') { if (typeof ev.index === 'number') run.maxIndex = Math.max(run.maxIndex, ev.index); run.lastTs = ev.ts; }
  if (ev.event === 'done') { run.status = 'done'; run.doneEv = ev; if (run.unsubscribe) run.unsubscribe(); run.unsubscribe = null; finishRun(run); }
  if (ev.event === 'error') { run.status = 'error'; run.errorEv = ev; if (run.unsubscribe) run.unsubscribe(); run.unsubscribe = null; toast(`Run error: ${ev.message || 'unknown'}${ev.node ? ` at ${ev.node}` : ''}`, 'error'); }
  if (state.view === 'run' && state.run === run) updateRunView(run, ev);
  if (ev.event === 'done' || ev.event === 'error') setRunButton();
}

async function finishRun(run) {
  try { const r = await call('run', run.id); run.summary = r.summary; run.config = r.config || run.config; if (r.status) run.status = r.status; }
  catch (e) { /* summary not available yet: keep the done event numbers */ }
  if (state.view === 'run' && state.run === run) renderRunSummary(run);
  toast(`Run ${run.id} finished`, 'ok');
}

async function renderRun(id) {
  const content = $('#content'); content.innerHTML = ''; $('#right').innerHTML = '';
  setHeader(id, 'Run');
  let run = state.run && state.run.id === id ? state.run : null;
  if (!run) {
    content.appendChild(skeleton(true));
    try {
      const r = await call('run', id);
      if (state.view !== 'run') return;
      run = { id, status: r.status, events: r.progress || [], summary: r.summary, config: r.config, start: null, total: null, maxIndex: -1, unsubscribe: null, charts: {}, selected: null, export: null };
      const st = run.events.find((e) => e.event === 'start'); if (st) { run.start = st.ts; run.total = st.total_nodes; }
      run.events.forEach((e) => { if (typeof e.index === 'number') run.maxIndex = Math.max(run.maxIndex, e.index); });
      // Baseline runs have no progress.jsonl: synthesise node_done events from summary.layers so the same feed renders.
      if (!run.events.length && run.summary && Array.isArray(run.summary.layers)) {
        run.events = run.summary.layers.map((l, i) => ({ event: 'node_done', index: i, name: l.node, op: l.op, backend: l.backend, shape: l.shape, cos: l.cos_sim, rel_l2: l.rel_l2, rel_max: l.rel_max, sched_packets: l.sched_packets, packets: l.dyn_packets, instructions: l.retired_instructions, bytes_loaded: l.bytes_loaded, bytes_written: l.bytes_written, flops: l.flops, synthetic: true }));
        run.total = run.summary.coverage_total_nodes || run.events.length; run.maxIndex = run.events.length - 1;
      }
      if (r.status === 'running') { if (state.run && state.run.unsubscribe) state.run.unsubscribe(); state.run = run; run.unsubscribe = api.events(id, (ev) => { if (!run.events.includes(ev)) onRunEvent(run, ev); }); run.events = []; run.maxIndex = -1; }
    } catch (e) { content.innerHTML = ''; content.appendChild(emptyState('Run not found', e.message)); return; }
  }
  if (state.view !== 'run') return;
  content.innerHTML = '';
  const cfg = run.config || {};
  setHeader(cfg.model ? `${cfg.model}${cfg.tag ? ` · ${cfg.tag}` : ''}` : id, id);
  content.appendChild(el(`<div class="card" id="runHead">
    <div class="flex wrap"><span class="status-pill ${esc(run.status)}" id="runStatus"><span class="dot"></span>${esc(run.status)}</span>
      <span class="muted" id="runProg">—</span><span class="spacer" style="flex:1"></span><span class="muted mono small" id="runElapsed"></span></div>
    <div class="progress mt8 ${esc(run.status)}"><div id="runBar"></div></div>
    <div class="flex wrap mt8 small muted" id="runFacts"></div>
  </div>`));
  content.appendChild(el(`<div id="runSummary"></div>`));
  content.appendChild(el(`<div class="card"><h2>Live metrics</h2><div class="chart-grid"><div class="chart-box"><canvas id="chCos"></canvas></div><div class="chart-box"><canvas id="chPk"></canvas></div></div></div>`));
  content.appendChild(el(`<div class="card flush"><div class="card-head" style="padding:16px 16px 0"><h2>Node feed <span class="count" id="feedCount"></span></h2><span class="spacer"></span><span class="muted small">newest first · click a node for artifacts</span></div><div class="feed" id="feed"></div></div>`));
  $('#runFacts').innerHTML = [['model', cfg.model], ['tag', cfg.tag], ['seed', cfg.seed], ['scale', cfg.scale], ['strict', cfg.strict != null ? String(cfg.strict) : null], ['flags', cfg.flags]].filter(([, v]) => v != null && v !== '').map(([k, v]) => `<span>${esc(k)} <b>${esc(v)}</b></span>`).join('');
  buildRunCharts(run);
  const feed = $('#feed');
  const dones = run.events.filter((e) => e.event === 'node_done');
  if (!dones.length) feed.appendChild(emptyState(run.status === 'running' ? 'Waiting for the first node' : 'No node events', run.status === 'running' ? 'The pipeline is tracing and planning tiles.' : 'This run has no progress stream or per-layer rows.'));
  dones.forEach((e) => { feedCard(run, e, feed, false); pushChart(run, e); });
  run.charts.cos.update('none'); run.charts.pk.update('none');
  const live = run.events.slice().reverse().find((e) => e.event === 'node_start' && !run.events.find((d) => d.event === 'node_done' && d.index === e.index));
  if (live && run.status === 'running') feedCard(run, live, feed, true);
  updateProgress(run);
  if (run.status !== 'running') renderRunSummary(run);
  renderRunRight(run);
  if (run.status === 'running') startElapsedTimer(run);
}

let elapsedTimer = null;
function startElapsedTimer(run) {
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    if (state.view !== 'run' || state.run !== run || run.status !== 'running') { clearInterval(elapsedTimer); return; }
    const e = $('#runElapsed'); if (e && run.start) e.textContent = fmtDur(Date.now() / 1000 - run.start);
  }, 500);
}

function updateProgress(run) {
  const total = run.total || run.computeTotal || null;
  const n = run.maxIndex + 1;
  const pct = run.status === 'done' ? 100 : total ? Math.min(100, (n / total) * 100) : 0;
  const bar = $('#runBar'); if (bar) bar.style.width = `${pct}%`;
  const p = $('#runProg'); if (p) p.textContent = total ? `node ${run.status === 'done' ? total : n} / ${total}` : (run.events.length ? `${run.events.filter((e) => e.event === 'node_done').length} nodes` : 'starting…');
  const s = $('#runStatus'); if (s) { s.className = `status-pill ${run.status}`; s.innerHTML = `<span class="dot"></span>${esc(run.status)}`; }
  const pb = $('#runHead .progress'); if (pb) pb.className = `progress mt8 ${run.status}`;
  const fc = $('#feedCount'); if (fc) fc.textContent = `${run.events.filter((e) => e.event === 'node_done').length}`;
  const el2 = $('#runElapsed'); if (el2) { const d = run.doneEv; if (d && d.elapsed_s != null) el2.textContent = fmtDur(d.elapsed_s); else if (run.summary && run.summary.elapsed_s_wall != null) el2.textContent = fmtDur(run.summary.elapsed_s_wall); else if (run.start && run.lastTs) el2.textContent = fmtDur(run.lastTs - run.start); }
}

function feedCard(run, e, feed, live) {
  const old = feed.querySelector(`[data-idx="${e.index}"]`); if (old) old.remove();
  const emptyEl = feed.querySelector('.empty'); if (emptyEl) emptyEl.remove();
  const card = el(`<div class="feed-item ${live ? 'live' : ''}" data-idx="${esc(e.index)}" data-name="${esc(e.name)}" tabindex="0" role="button">
    <div class="rail-dot"><i></i></div>
    <div>
      <div class="head"><span class="name">${esc(e.name)}</span><span class="op">${esc(e.op)}</span><span class="chip ${esc(e.backend)}"><span class="dot"></span>${esc(BACKEND_LABEL[e.backend] || e.backend)}</span>
        ${live ? '<span class="chip emulator">running</span>' : e.cos != null ? `<span class="chip ${cosClass(e.cos)}">cos ${fmtCos(e.cos)}</span>` : ''}<span class="idx">#${esc(e.index)}</span></div>
      <div class="meta">${live ? `<span>shape <b>${esc(fmtShape(e.shape))}</b></span>` : [
        ['shape', fmtShape(e.shape)], ['rel-L2', e.rel_l2 != null ? fmtSci(e.rel_l2) : null], ['rel-max', e.rel_max != null ? fmtSci(e.rel_max) : null],
        ['sched', e.sched_packets != null ? fmtInt(e.sched_packets) : null], ['dyn pkts', e.packets != null ? fmtInt(e.packets) : null], ['retired', e.instructions != null ? fmtInt(e.instructions) : null],
        ['bytes L/W', e.bytes_loaded != null ? `${fmtInt(e.bytes_loaded)} / ${fmtInt(e.bytes_written)}` : null], ['FLOPs', e.flops != null ? fmtInt(e.flops) : null], ['t', e.elapsed_s != null ? `${Number(e.elapsed_s).toFixed(2)} s` : null],
      ].filter(([, v]) => v != null).map(([k, v]) => `<span>${esc(k)} <b>${esc(v)}</b></span>`).join('')}</div>
    </div></div>`);
  const open = () => { run.selected = e.name; feed.querySelectorAll('.feed-item').forEach((x) => x.classList.toggle('selected', x === card)); renderRunRight(run); openArtifacts(run, e); };
  card.addEventListener('click', open);
  card.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); } });
  feed.prepend(card);
}

function buildRunCharts(run) {
  if (!window.Chart) return;
  refreshChartTheme();
  const c = chartColors();
  const common = { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 60, minRotation: 45, autoSkip: true, font: { size: 10 } }, grid: { display: false } } } };
  run.charts.cos = new Chart($('#chCos'), { type: 'line', data: { labels: [], datasets: [{ label: 'cosine', tok: 'accent', cosPoints: true, data: [], borderColor: c.accent, backgroundColor: c.accent, tension: 0.25, pointRadius: 3, pointBackgroundColor: [] }] }, options: { ...common, plugins: { legend: { display: false }, title: { display: true, text: 'Per-node cosine', align: 'start', font: { weight: '600' } }, tooltip: { callbacks: { label: (t) => `cos ${fmtCos(t.parsed.y)}` } } }, scales: { ...common.scales, y: { suggestedMin: 0.999, max: 1, ticks: { callback: (v) => Number(v).toFixed(4) } } } } });
  run.charts.pk = new Chart($('#chPk'), { type: 'bar', data: { labels: [], datasets: [{ label: 'dynamic packets', tok: 'accent', data: [], backgroundColor: c.accent, borderRadius: 4 }] }, options: { ...common, plugins: { legend: { display: false }, title: { display: true, text: 'Per-node dynamic packets', align: 'start', font: { weight: '600' } } }, scales: { ...common.scales, y: { beginAtZero: true, ticks: { callback: (v) => fmtInt(v) } } } } });
}
function pushChart(run, e) {
  if (!run.charts.cos) return;
  if (e.cos != null) { const c = chartColors(); run.charts.cos.data.labels.push(e.name); run.charts.cos.data.datasets[0].data.push(e.cos); run.charts.cos.data.datasets[0].pointBackgroundColor.push(e.cos >= 0.999 ? c.success : e.cos >= 0.99 ? c.warning : c.danger); }
  if (e.packets != null) { run.charts.pk.data.labels.push(e.name); run.charts.pk.data.datasets[0].data.push(e.packets); }
}

function updateRunView(run, ev) {
  const feed = $('#feed'); if (!feed) return;
  if (ev.event === 'node_start') feedCard(run, ev, feed, true);
  if (ev.event === 'node_done') { feedCard(run, ev, feed, false); pushChart(run, ev); run.charts.cos.update('none'); run.charts.pk.update('none'); }
  if (ev.event === 'error') { feed.prepend(el(`<div class="feed-item"><div class="rail-dot"><i style="background:var(--red)"></i></div><div><div class="head"><span class="name" style="color:var(--red)">error</span>${ev.node ? `<span class="op">${esc(ev.node)}</span>` : ''}</div><div class="meta"><span>${esc(ev.message || '')}</span></div></div></div>`)); renderRunSummary(run); }
  updateProgress(run);
  renderRunRight(run);
}

function renderRunSummary(run) {
  const box = $('#runSummary'); if (!box) return;
  const s = run.summary || {};
  const d = run.doneEv || {};
  const cos = s.final_cosine ?? d.final_cosine, rl2 = s.rel_l2 ?? d.rel_l2, rmax = s.rel_max ?? d.rel_max, strict = s.strict_pass ?? d.strict_pass;
  if (run.status === 'error' && !run.summary) { box.innerHTML = `<div class="card"><h2>Run failed</h2><div class="small error-text">${esc((run.errorEv || {}).message || 'unknown error')}</div><div class="btn-row mt16"><button class="btn sm" id="openLog">${icon('i-doc')}Open log</button></div></div>`; $('#openLog').addEventListener('click', () => openLog(run)); return; }
  if (cos == null && !run.summary) { box.innerHTML = ''; return; }
  const tile = (k, v, sub = '', cls = '') => `<div class="tile ${cls}"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>${sub ? `<div class="s">${esc(sub)}</div>` : ''}</div>`;
  box.innerHTML = `<div class="card"><div class="card-head"><h2>Summary</h2><span class="spacer"></span>
      <button class="btn sm" id="openLog">${icon('i-doc')}Open log</button>
      <button class="btn sm primary" id="exportBtn">${icon('i-export')}Export to tracker</button></div>
    <div class="tiles">
      ${tile('Final cosine', fmtCos(cos), '', cosClass(cos))}${tile('rel-L2', fmtSci(rl2))}${tile('rel-max', fmtSci(rmax))}
      ${tile('Strict', strict == null ? 'n/a' : strict ? 'pass' : 'FAIL', strict == null ? 'run without --strict-end' : '', strict == null ? '' : strict ? 'ok' : 'bad')}
      ${tile('Coverage E / N / P', `${s.coverage_emulated ?? '—'} / ${s.coverage_numpy ?? '—'} / ${s.coverage_passthrough ?? '—'}`, s.coverage_total_nodes ? `of ${s.coverage_total_nodes} nodes` : '')}
      ${tile('Bytes L / W', `${fmtInt(s.bytes_loaded)} / ${fmtInt(s.bytes_written)}`, s.bytes_total != null ? `total ${fmtInt(s.bytes_total)}` : '')}
      ${tile('FLOPs', fmtInt(s.flops), s.flops_matmul != null ? `matmul ${fmtInt(s.flops_matmul)}` : '')}
      ${tile('Arith. intensity', fmtNum(s.arithmetic_intensity), 'FLOP / byte')}
      ${tile('Sched / dyn packets', `${fmtInt(s.sched_packets)} / ${fmtInt(s.dyn_packets)}`)}
      ${tile('Retired instrs', fmtInt(s.retired_instructions))}
      ${tile('Static / dyn fill', `${fmtPct(s.static_fill)} / ${fmtPct(s.dynamic_fill)}`)}
      ${tile('Elapsed', fmtDur(s.elapsed_s_wall ?? d.elapsed_s), 'host wall-clock')}
    </div>
    <div id="exportOut" class="mt16"></div></div>`;
  $('#openLog').addEventListener('click', () => openLog(run));
  $('#exportBtn').addEventListener('click', () => exportRun(run));
  if (run.export) showExport(run);
}
function showExport(run) {
  const o = $('#exportOut'); if (!o) return;
  // Download links use the server's FileResponse routes (bench/ui/server.py): per-run sheet at
  // GET /api/runs/{run_id}/export/download, the rebuilt tracker at GET /api/export/tracker. Plain text in mock mode.
  const link = (href, label) => (api === realApi ? `<a class="dl" href="${href}" download title="Download ${esc(label)}">${esc(label)}${icon('i-export')}</a>` : `<span class="mono">${esc(label)}</span>`);
  o.innerHTML = `<div class="section-title">Exported</div><dl class="kv"><dt>run sheet</dt><dd>${link(`/api/runs/${encodeURIComponent(run.id)}/export/download`, run.export.run_xlsx)}</dd><dt>tracker</dt><dd>${run.export.tracker_xlsx ? link('/api/export/tracker', run.export.tracker_xlsx) : '—'}</dd></dl>`;
}
async function exportRun(run) {
  const b = $('#exportBtn'); if (b) { b.disabled = true; b.textContent = 'Exporting…'; }
  try { run.export = await call('exportRun', run.id); showExport(run); toast('Exported to tracker', 'ok'); }
  catch (e) { toast(`Export failed: ${e.message}`, 'error'); }
  finally { if (b) { b.disabled = false; b.innerHTML = `${icon('i-export')}Export to tracker`; } }
}
async function openLog(run) {
  const body = el('<div></div>'); body.appendChild(skeleton(true)); modal('run.log', run.id, body);
  try { const t = await call('log', run.id); body.innerHTML = ''; body.appendChild(codeView(typeof t === 'string' ? t : JSON.stringify(t, null, 2))); }
  catch (e) { body.innerHTML = ''; body.appendChild(emptyState('Log unavailable', e.message)); }
}
async function openArtifacts(run, e) {
  if (e.backend === 'passthrough' || e.backend === 'graph' || e.backend === 'skip' || e.backend === 'numpy') { toast(`${e.name}: ${BACKEND_LABEL[e.backend] || e.backend} node has no compiled artifacts`, ''); return; }
  const body = el(`<div><div class="tabs">${['c', 's', 'in'].map((k, i) => `<button class="${i === 0 ? 'on' : ''}" data-k="${k}">${k === 'c' ? 'C (generated)' : k === 's' ? 'ASM (.s)' : 'Input (.in)'}</button>`).join('')}</div><div id="artBody"></div></div>`);
  modal(`Artifacts: ${e.name}`, `${e.op} · ${BACKEND_LABEL[e.backend] || e.backend}`, body);
  const cache = {};
  const show = async (k) => {
    body.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.k === k));
    const ab = body.querySelector('#artBody'); ab.innerHTML = ''; ab.appendChild(skeleton(true));
    try { cache[k] = cache[k] || await call('artifact', run.id, e.name, k); ab.innerHTML = `<div class="muted small mono mb8">${esc(cache[k].path || '')}</div>`; ab.appendChild(codeView(cache[k].text)); }
    catch (err) { ab.innerHTML = ''; ab.appendChild(emptyState('Artifact unavailable', err.message)); }
  };
  body.querySelector('.tabs').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (b) show(b.dataset.k); });
  show('c'); call('artifact', run.id, e.name, 's').then((r) => { cache.s = r; }).catch(() => {});
}
function renderRunRight(run) {
  const right = $('#right'); if (state.view !== 'run') return;
  const dones = run.events.filter((e) => e.event === 'node_done');
  const emu = dones.filter((e) => e.backend === 'emulator'), np = dones.filter((e) => e.backend === 'numpy'), pt = dones.filter((e) => e.backend === 'passthrough'), sk = dones.filter((e) => e.backend === 'skip');
  const worst = dones.filter((e) => e.cos != null).sort((a, b) => a.cos - b.cos).slice(0, 5);
  const sel = dones.find((e) => e.name === run.selected);
  right.innerHTML = `<div class="card"><h2>This run</h2>
    <div class="counts">${[['Emulated', emu.length, 'emulator'], ['NumPy', np.length, 'numpy'], ['Passthrough', pt.length, 'passthrough'], ['Skip', sk.length, 'skip']].map(([l, n, b]) => `<span class="stat"><i class="sw" style="background:var(${BACKEND_VAR[b]})"></i>${l} <b>${n}</b></span>`).join('')}</div>
    <dl class="kv"><dt>run id</dt><dd class="mono small">${esc(run.id)}</dd><dt>status</dt><dd>${esc(run.status)}</dd><dt>dyn packets so far</dt><dd>${fmtInt(emu.reduce((a, e) => a + (e.packets || 0), 0))}</dd><dt>bytes so far</dt><dd>${fmtInt(emu.reduce((a, e) => a + (e.bytes_loaded || 0) + (e.bytes_written || 0), 0))}</dd></dl>
    ${worst.length ? `<div class="section-title">Lowest cosine</div><div class="row-list">${worst.map((e) => `<div class="row-item" data-node="${esc(e.name)}" style="padding:8px 0"><span class="t">${esc(e.name)}</span><span class="d">${esc(e.op)}</span><span class="spacer"></span><span class="chip ${cosClass(e.cos)}">${fmtCos(e.cos)}</span></div>`).join('')}</div>` : ''}
  </div>
  ${sel ? `<div class="card mt16"><div class="card-head"><h2>${esc(sel.name)}</h2><span class="spacer"></span><span class="chip ${esc(sel.backend)}"><span class="dot"></span>${esc(BACKEND_LABEL[sel.backend] || sel.backend)}</span></div>
    <dl class="kv"><dt>op</dt><dd>${esc(sel.op)}</dd><dt>shape</dt><dd class="mono">${esc(fmtShape(sel.shape))}</dd><dt>cosine</dt><dd>${fmtCos(sel.cos)}</dd><dt>rel-L2</dt><dd>${fmtSci(sel.rel_l2)}</dd><dt>rel-max</dt><dd>${fmtSci(sel.rel_max)}</dd><dt>sched / dyn</dt><dd>${fmtInt(sel.sched_packets)} / ${fmtInt(sel.packets)}</dd><dt>retired</dt><dd>${fmtInt(sel.instructions)}</dd><dt>bytes L / W</dt><dd>${fmtInt(sel.bytes_loaded)} / ${fmtInt(sel.bytes_written)}</dd><dt>FLOPs</dt><dd>${fmtInt(sel.flops)}</dd></dl>
    <div class="btn-row mt16"><button class="btn sm" id="artBtn">${icon('i-doc')}View artifacts</button></div></div>` : ''}`;
  right.querySelectorAll('[data-node]').forEach((r) => r.addEventListener('click', () => { const e = dones.find((x) => x.name === r.dataset.node); if (e) { run.selected = e.name; renderRunRight(run); const fc = $(`#feed [data-name="${CSS.escape(e.name)}"]`); if (fc) { fc.scrollIntoView({ block: 'center', behavior: 'smooth' }); $('#feed').querySelectorAll('.feed-item').forEach((x) => x.classList.toggle('selected', x === fc)); } } }));
  const ab = right.querySelector('#artBtn'); if (ab && sel) ab.addEventListener('click', () => openArtifacts(run, sel));
}

/* ================================================================== RUNS */
async function renderRuns() {
  const content = $('#content'); content.innerHTML = ''; const right = $('#right'); right.innerHTML = '';
  setHeader('Runs', 'Baselines from docs/baselines plus live runs.');
  content.appendChild(skeleton(true));
  try { state.runs = await call('runs'); } catch (e) { content.innerHTML = ''; content.appendChild(emptyState('Could not load runs', e.message)); return; }
  if (state.view !== 'runs') return;
  drawRunsTable();
}
function drawRunsTable() {
  const content = $('#content'); content.innerHTML = '';
  const models = [...new Set(state.runs.map((r) => r.model))].sort();
  const cols = [['model', 'Run'], ['status', 'Status'], ['started', 'Started'], ['final_cosine', 'Cosine', 'num'], ['coverage', 'E / N / P', 'num'], ['dyn_packets', 'Dyn pkts', 'num'], ['bytes_total', 'Bytes', 'num']];
  const { key, dir } = state.runsSort;
  const val = (r, k) => k === 'coverage' ? (r.coverage ? r.coverage.emulated : null) : r[k];
  const rows = state.runs.filter((r) => !state.runsFilter || r.model === state.runsFilter).sort((a, b) => { const x = val(a, key), y = val(b, key); if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
  const card = el(`<div class="card flush"><div class="card-head" style="padding:16px 16px 8px"><h2>All runs <span class="count">${rows.length}</span></h2><span class="spacer"></span>
      <div class="field" style="width:200px"><select id="runsFilter"><option value="">All models</option>${models.map((m) => `<option value="${esc(m)}" ${state.runsFilter === m ? 'selected' : ''}>${esc(m)}</option>`).join('')}</select></div>
      <button class="btn sm" id="refreshRuns">Refresh</button></div>
    <div class="table-wrap"><table class="runs-table"><thead><tr>${cols.map(([k, l, c]) => `<th class="${c || ''}" data-k="${k}">${l}${key === k ? `<span class="arrow">${dir > 0 ? '▲' : '▼'}</span>` : ''}</th>`).join('')}<th title="Pick two runs to compare">Cmp</th></tr></thead><tbody>
      ${rows.length ? rows.map((r) => `<tr data-id="${esc(r.run_id)}" class="${state.compare.includes(r.run_id) ? 'picked' : ''}">
        <td><div><b>${esc(r.model)}</b> <span class="muted">· ${esc(r.tag)}</span></div><div class="rid mono muted" title="${esc(r.run_id)}">${esc(r.run_id)}</div></td><td><span class="status-pill ${esc(r.status)}"><span class="dot"></span>${esc(r.status)}</span></td><td>${esc(fmtDate(r.started))}</td>
        <td class="num"><span class="chip ${cosClass(r.final_cosine)}">${fmtCos(r.final_cosine)}</span></td><td class="num">${r.coverage ? `${r.coverage.emulated ?? '—'} / ${r.coverage.numpy ?? '—'} / ${r.coverage.passthrough ?? '—'}` : '—'}</td>
        <td class="num">${fmtInt(r.dyn_packets)}</td><td class="num">${fmtInt(r.bytes_total)}</td>
        <td><label class="switch" title="Pick for compare"><input type="checkbox" data-cmp="${esc(r.run_id)}" ${state.compare.includes(r.run_id) ? 'checked' : ''}></label></td></tr>`).join('') : `<tr><td colspan="9"><div class="empty">${icon('i-inbox')}<h3>No runs</h3><div>Start one from the Workbench.</div></div></td></tr>`}
    </tbody></table></div></div>`);
  card.querySelector('#runsFilter').addEventListener('change', (e) => { state.runsFilter = e.target.value; drawRunsTable(); });
  card.querySelector('#refreshRuns').addEventListener('click', () => renderRuns());
  card.querySelectorAll('th[data-k]').forEach((th) => th.addEventListener('click', () => { const k = th.dataset.k; state.runsSort = { key: k, dir: state.runsSort.key === k ? -state.runsSort.dir : (k === 'started' ? -1 : 1) }; drawRunsTable(); }));
  card.querySelectorAll('tbody tr[data-id]').forEach((tr) => tr.addEventListener('click', (e) => { if (e.target.closest('label')) return; go(`#/run/${encodeURIComponent(tr.dataset.id)}`); }));
  card.querySelectorAll('input[data-cmp]').forEach((cb) => cb.addEventListener('change', () => { const id = cb.dataset.cmp; if (cb.checked) { state.compare = [...state.compare.filter((x) => x !== id), id].slice(-2); } else state.compare = state.compare.filter((x) => x !== id); drawRunsTable(); }));
  content.appendChild(card);
  renderRunsRight();
}
function renderRunsRight() {
  const right = $('#right'); right.innerHTML = '';
  const c = el(`<div class="card"><h2>Compare</h2><div class="muted small">Pick two runs with the switches in the table.</div><div class="row-list mt8">${state.compare.map((id) => `<div class="row-item" style="padding:8px 0"><span class="t mono small">${esc(id)}</span></div>`).join('') || '<div class="muted small mt8">Nothing picked yet.</div>'}</div>
    <div class="btn-row mt16"><button class="btn primary" id="cmpBtn" ${state.compare.length === 2 ? '' : 'disabled'}>${icon('i-compare')}Compare</button><button class="btn ghost sm" id="cmpClear">Clear</button></div></div>`);
  c.querySelector('#cmpBtn').addEventListener('click', () => go(`#/compare/${encodeURIComponent(state.compare[0])}/${encodeURIComponent(state.compare[1])}`));
  c.querySelector('#cmpClear').addEventListener('click', () => { state.compare = []; drawRunsTable(); });
  right.appendChild(c);
}

async function renderCompare(a, b) {
  const content = $('#content'); content.innerHTML = ''; $('#right').innerHTML = '';
  setHeader('Compare', `${a} vs ${b}`);
  content.appendChild(skeleton(true));
  let A, B;
  try { [A, B] = await Promise.all([call('run', a), call('run', b)]); } catch (e) { content.innerHTML = ''; content.appendChild(emptyState('Could not load runs', e.message)); return; }
  if (state.view !== 'compare') return;
  content.innerHTML = '';
  const sa = A.summary || {}, sb = B.summary || {};
  const la = (A.config && A.config.tag) ? `${A.config.model} · ${A.config.tag}` : a, lb = (B.config && B.config.tag) ? `${B.config.model} · ${B.config.tag}` : b;
  const metrics = [['bytes_total', 'Bytes (L+W)'], ['dyn_packets', 'Dynamic packets'], ['flops', 'FLOPs'], ['retired_instructions', 'Retired instructions']];
  content.appendChild(el(`<div class="card"><h2>Aggregate</h2><div class="tiles">${[['Final cosine', fmtCos(sa.final_cosine), fmtCos(sb.final_cosine)], ['rel-L2', fmtSci(sa.rel_l2), fmtSci(sb.rel_l2)], ['Coverage E/N/P', `${sa.coverage_emulated ?? '—'}/${sa.coverage_numpy ?? '—'}/${sa.coverage_passthrough ?? '—'}`, `${sb.coverage_emulated ?? '—'}/${sb.coverage_numpy ?? '—'}/${sb.coverage_passthrough ?? '—'}`], ['Arith. intensity', fmtNum(sa.arithmetic_intensity), fmtNum(sb.arithmetic_intensity)]].map(([k, x, y]) => `<div class="tile"><div class="k">${k}</div><div class="v" style="font-size:16px">${x} <span class="muted">vs</span> ${y}</div></div>`).join('')}</div>
    <div class="legend mt8"><span><i class="sw" style="background:var(--accent)"></i>A: ${esc(la)}</span><span><i class="sw" style="background:var(--green)"></i>B: ${esc(lb)}</span></div></div>`));
  content.appendChild(el(`<div class="card"><h2>Work & traffic</h2><div class="chart-grid">${metrics.map(([k]) => `<div class="chart-box"><canvas id="cmp_${k}"></canvas></div>`).join('')}</div></div>`));
  content.appendChild(el(`<div class="card"><h2>Per-layer cosine</h2><div class="chart-box" style="height:280px"><canvas id="cmpCos"></canvas></div></div>`));
  refreshChartTheme(); const c = chartColors();
  metrics.forEach(([k, label]) => new Chart($(`#cmp_${k}`), { type: 'bar', data: { labels: [label], datasets: [{ label: 'A', tok: 'accent', data: [sa[k] ?? 0], backgroundColor: c.accent, borderRadius: 4 }, { label: 'B', tok: 'success', data: [sb[k] ?? 0], backgroundColor: c.success, borderRadius: 4 }] }, options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false }, title: { display: true, text: label, align: 'start', font: { weight: '600' } } }, scales: { y: { beginAtZero: true, ticks: { callback: (v) => fmtInt(v) } } } } }));
  const L = (s) => (s.layers || []).filter((l) => l.cos_sim != null);
  const labels = [...new Set([...L(sa), ...L(sb)].map((l) => l.node))];
  const series = (s) => labels.map((n) => { const l = L(s).find((x) => x.node === n); return l ? l.cos_sim : null; });
  new Chart($('#cmpCos'), { type: 'line', data: { labels, datasets: [{ label: la, tok: 'accent', data: series(sa), borderColor: c.accent, backgroundColor: c.accent, tension: 0.25, spanGaps: true }, { label: lb, tok: 'success', data: series(sb), borderColor: c.success, backgroundColor: c.success, tension: 0.25, spanGaps: true }] }, options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: 'bottom' } }, scales: { x: { ticks: { maxRotation: 60, minRotation: 45, font: { size: 10 } }, grid: { display: false } }, y: { suggestedMin: 0.999, max: 1, ticks: { callback: (v) => Number(v).toFixed(4) } } } } });
  if (!labels.length) content.appendChild(emptyState('No per-layer rows', 'Neither run has layers[] in its summary.'));
  $('#right').innerHTML = `<div class="card"><h2>Runs</h2><div class="row-list"><div class="row-item" style="padding:8px 0"><span class="chip emulator">A</span><a href="#/run/${encodeURIComponent(a)}" class="mono small">${esc(a)}</a></div><div class="row-item" style="padding:8px 0"><span class="chip ok">B</span><a href="#/run/${encodeURIComponent(b)}" class="mono small">${esc(b)}</a></div></div><div class="btn-row mt16"><a class="btn sm" href="#/runs">Back to runs</a></div></div>`;
}

/* ================================================================== KERNELS */
async function renderKernels(name) {
  const content = $('#content'); content.innerHTML = ''; const right = $('#right'); right.innerHTML = '';
  setHeader('Kernels', 'Atalla kernel emitters and the ops they cover.');
  if (!state.kernels.length) { content.appendChild(skeleton()); try { state.kernels = await call('kernels'); } catch (e) { content.innerHTML = ''; content.appendChild(emptyState('Could not load kernels', e.message)); return; } }
  if (state.view !== 'kernels') return;
  content.innerHTML = '';
  const list = el(`<div class="card flush"><div class="card-head" style="padding:16px 16px 0"><h2>Kernels <span class="count">${state.kernels.length}</span></h2></div><div class="row-list">${state.kernels.map((k) => `<div class="row-item ${k.name === name ? 'selected' : ''}" data-k="${esc(k.name)}"><div><div class="t">${esc(k.name)} <span class="chip ${k.kind === 'static_c' ? 'passthrough' : 'emulator'}">${esc(k.kind)}</span></div><div class="d mono">${esc(k.path)}</div></div><span class="spacer"></span><div class="chip-list">${(k.ops || []).map((o) => `<span class="chip">${esc(o)}</span>`).join('')}</div></div>`).join('')}</div></div>`);
  list.querySelectorAll('[data-k]').forEach((r) => r.addEventListener('click', () => go(`#/kernels/${encodeURIComponent(r.dataset.k)}`)));
  const src = el('<div class="card" id="kernelSrc"></div>');
  if (name) { content.appendChild(src); content.appendChild(list); } else { content.appendChild(list); content.appendChild(src); }
  if (!name) { src.appendChild(emptyState('Pick a kernel', 'Click a kernel to read its source with line numbers.', 'i-kernels')); right.appendChild(el('<div class="card"><h2>About</h2><div class="small muted">Generators are Python emitters that produce C for the in-tree pipeline (c_emitter → ppci -p packetized asm). static_c kernels are hand-written C compiled as-is.</div></div>')); return; }
  src.appendChild(skeleton(true));
  try {
    const k = await call('kernel', name);
    if (state.view !== 'kernels') return;
    src.innerHTML = `<div class="card-head"><h2>${esc(k.name)}</h2><span class="spacer"></span><span class="mono small muted">${esc(k.path)}</span></div>`;
    src.appendChild(codeView(k.text));
    const meta = state.kernels.find((x) => x.name === name) || {};
    right.innerHTML = `<div class="card"><h2>${esc(name)}</h2><dl class="kv"><dt>kind</dt><dd>${esc(meta.kind || '—')}</dd><dt>path</dt><dd class="mono">${esc(meta.path || k.path)}</dd><dt>ops</dt><dd>${(meta.ops || []).map((o) => `<span class="chip">${esc(o)}</span>`).join(' ') || '—'}</dd><dt>lines</dt><dd>${String(k.text || '').split('\n').length}</dd></dl></div>`;
  } catch (e) { src.innerHTML = ''; src.appendChild(emptyState('Could not load source', e.message)); }
}

/* ================================================================== METRICS */
const METRICS = [
  ['Final cosine', 'final_cosine', 'Cosine similarity of the final chained Atalla output vs the PyTorch reference output.', 'One headline number for "did the model run correctly on Atalla"; 1.0 = same direction.'],
  ['rel-L2', 'rel_l2', '‖emu − ref‖₂ / ‖ref‖₂ on the final output.', 'Scale-aware error; catches magnitude errors cosine hides.'],
  ['rel-max', 'rel_max', 'max|emu − ref| / max|ref| on the final output.', 'Worst single element; catches one bad tile that averages away.'],
  ['Strict pass', 'strict_pass', 'run_graph.py asserts cos ≥ 0.999, rel-L2 ≤ 0.02, rel-max ≤ 0.05 (needs --strict-end).', 'Pass/fail gate for CI-style regression.'],
  ['Coverage E / N / P', 'coverage_emulated / coverage_numpy / coverage_passthrough', 'Number of FX graph nodes executed on the Atalla functional sim (Emulated), in the NumPy fallback (N), or Passed through from the PyTorch reference.', 'Says how much of the model actually ran on Atalla. A perfect cosine with low E means little.'],
  ['Scheduled packets', 'sched_packets', 'Static packet (VLIW bundle, ≤ 4 slots) count in the compiled program, summed over emulated kernels.', 'Code size / static ILP proxy.'],
  ['Dynamic packets', 'dyn_packets', 'Packets actually issued by the functional sim (loops unrolled at run time).', 'Closest available proxy for issue cycles; the sim has no stall model.'],
  ['Retired instructions', 'retired_instructions', 'Slot-instructions retired at run time.', 'Work actually done; split by class (gemm_systolic, vector_alu, scalar_alu, ...) for the future cost model.'],
  ['Bytes loaded / written', 'bytes_loaded / bytes_written', 'Bytes DMA\'d from external memory into the scratchpads / written back.', 'DRAM bandwidth demand; denominator of arithmetic intensity.'],
  ['FLOPs / Arithmetic intensity', 'flops / arithmetic_intensity', 'FLOPs from op geometry (2·M·N·K for GEMM, elementwise counts otherwise); AI = flops / (bytes_loaded + bytes_written).', 'Numerator of arithmetic intensity; AI is the roofline x-axis.'],
];
function renderMetrics() {
  const content = $('#content'); content.innerHTML = ''; const right = $('#right'); right.innerHTML = '';
  setHeader('Metrics', 'What every column in a run means and why it is tracked.');
  content.appendChild(el(`<div class="card flush"><div class="card-head" style="padding:16px 16px 0"><h2>Metric definitions <span class="count">${METRICS.length}</span></h2></div><div class="table-wrap"><table class="metrics-table"><thead><tr><th style="cursor:default">Metric</th><th style="cursor:default">Field</th><th style="cursor:default">Meaning</th><th style="cursor:default">Why it matters</th></tr></thead><tbody>
    ${METRICS.map(([m, f, me, w]) => `<tr style="cursor:default"><td><b>${esc(m)}</b></td><td class="mono small wrap">${esc(f)}</td><td class="wrap">${esc(me)}</td><td class="wrap muted">${esc(w)}</td></tr>`).join('')}</tbody></table></div></div>`));
  right.innerHTML = `<div class="card"><h2>Read with care</h2><ul class="small muted" style="padding-left:18px;margin:0;display:flex;flex-direction:column;gap:8px">
    <li>Counts only, no timing: the functional simulator has no pipeline, latency or stall model. Dynamic packets are an issue-slot proxy, not cycles.</li>
    <li>Baseline rows are <b>chained</b> (emulated output of node i−1 feeds node i); chained and oracle rows are not comparable.</li>
    <li>FLOPs are analytic from op geometry and identical for every backend.</li>
    <li>Coverage is node-level: one passthrough node can hide most of the work. Read E/N/P with FLOPs.</li>
    <li>Only compare rows whose compiler SHA matches. Source: docs/benchmark_metrics.md.</li></ul></div>`;
}

/* ================================================================== boot */
applyTheme(null);
$('#origin').textContent = window.location.origin;
setModePill();
if (api === realApi) { fetch('/api/health').catch(() => {}); }
route();
