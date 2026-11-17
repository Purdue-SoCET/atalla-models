// mock.js — offline data shaped exactly like bench/ui/API_CONTRACT.md.
// Used when the page is opened with ?mock=1 or when the real API is unreachable.
// The gpt2_block graph (47 FX nodes, 29 compute nodes) and the per-node numbers
// come from docs/baselines/gpt2_block__chained__20261104-1459/{run.log,layers.csv}.

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
const OPT_GELU = { key: 'gelu_kernel', label: 'GELU kernel', choices: ['pytorch', 'horner', 'tanh'], default: 'pytorch' };
const OPT_SDPA = { key: 'sdpa', label: 'SDPA path', choices: ['numpy', 'flash'], default: 'numpy' };
const OPT_ADD = { key: 'add_numpy_threshold', label: 'Add → NumPy above N elems', type: 'int', default: 1000000 };

export const MODELS = [
  { name: 'basic', title: 'Basic MLP', description: 'Linear + ReLU smoke model (10 emulated nodes).', defaults: { scale: null, seed: 42 }, options: [OPT_ADD] },
  { name: 'alexnet_small', title: 'AlexNet (small)', description: 'Conv/pool/linear stack, scaled inputs.', defaults: { scale: 0.01, seed: 42 }, options: [OPT_ADD] },
  { name: 'layernorm_smoke', title: 'LayerNorm smoke', description: 'Single layernorm + add.', defaults: { scale: null, seed: 42 }, options: [] },
  { name: 'vit_micro', title: 'ViT micro', description: 'Patch embed + one encoder block with SDPA.', defaults: { scale: null, seed: 42 }, options: [OPT_GELU, OPT_SDPA, OPT_ADD] },
  { name: 'gpt2_mlp', title: 'GPT-2 MLP', description: 'LN + c_fc + GELU + c_proj.', defaults: { scale: null, seed: 42 }, options: [OPT_GELU, OPT_ADD] },
  { name: 'gpt2_attn', title: 'GPT-2 attention', description: 'LN + QKV + causal softmax + c_proj.', defaults: { scale: null, seed: 42 }, options: [OPT_SDPA, OPT_ADD] },
  { name: 'gpt2_block', title: 'GPT-2 block', description: 'Full transformer block: attention + MLP, residuals.', defaults: { scale: null, seed: 42 }, options: [OPT_GELU, OPT_SDPA, OPT_ADD] },
];

// ---------------------------------------------------------------------------
// gpt2_block graph (execution order from run.log)
// [name, fxop, target, shape, inputs]
// ---------------------------------------------------------------------------
const G = [
  ['x', 'placeholder', '-', [32, 32], []],
  ['attn_ln_1_weight', 'get_attr', '-', [32], []],
  ['attn_ln_1_bias', 'get_attr', '-', [32], []],
  ['layer_norm', 'layernorm', 'aten.layer_norm.default', [32, 32], ['x', 'attn_ln_1_weight', 'attn_ln_1_bias']],
  ['attn_q_proj_weight', 'get_attr', '-', [32, 32], []],
  ['transpose_1', 'transpose', 'aten.t.default', [32, 32], ['attn_q_proj_weight']],
  ['matmul_2', 'matmul', 'aten.mm.default', [32, 32], ['layer_norm', 'transpose_1']],
  ['attn_q_proj_bias', 'get_attr', '-', [32], []],
  ['add_3', 'add', 'aten.add.Tensor', [32, 32], ['matmul_2', 'attn_q_proj_bias']],
  ['attn_k_proj_weight', 'get_attr', '-', [32, 32], []],
  ['transpose_2', 'transpose', 'aten.t.default', [32, 32], ['attn_k_proj_weight']],
  ['matmul_3', 'matmul', 'aten.mm.default', [32, 32], ['layer_norm', 'transpose_2']],
  ['attn_k_proj_bias', 'get_attr', '-', [32], []],
  ['add_4', 'add', 'aten.add.Tensor', [32, 32], ['matmul_3', 'attn_k_proj_bias']],
  ['attn_v_proj_weight', 'get_attr', '-', [32, 32], []],
  ['transpose_3', 'transpose', 'aten.t.default', [32, 32], ['attn_v_proj_weight']],
  ['matmul_4', 'matmul', 'aten.mm.default', [32, 32], ['layer_norm', 'transpose_3']],
  ['attn_v_proj_bias', 'get_attr', '-', [32], []],
  ['add_5', 'add', 'aten.add.Tensor', [32, 32], ['matmul_4', 'attn_v_proj_bias']],
  ['transpose', 'transpose', 'aten.transpose.int', [32, 32], ['add_4']],
  ['matmul', 'matmul', 'aten.mm.default', [32, 32], ['add_3', 'transpose']],
  ['mul', 'mul', 'aten.mul.Tensor', [32, 32], ['matmul']],
  ['attn_causal_mask', 'get_attr', '-', [32, 32], []],
  ['add', 'add', 'aten.add.Tensor', [32, 32], ['mul', 'attn_causal_mask']],
  ['softmax', 'softmax', 'aten._softmax.default', [32, 32], ['add']],
  ['matmul_1', 'matmul', 'aten.mm.default', [32, 32], ['softmax', 'add_5']],
  ['attn_c_proj_weight', 'get_attr', '-', [32, 32], []],
  ['transpose_4', 'transpose', 'aten.t.default', [32, 32], ['attn_c_proj_weight']],
  ['matmul_5', 'matmul', 'aten.mm.default', [32, 32], ['matmul_1', 'transpose_4']],
  ['attn_c_proj_bias', 'get_attr', '-', [32], []],
  ['add_6', 'add', 'aten.add.Tensor', [32, 32], ['matmul_5', 'attn_c_proj_bias']],
  ['add_1', 'add', 'aten.add.Tensor', [32, 32], ['x', 'add_6']],
  ['mlp_ln_2_weight', 'get_attr', '-', [32], []],
  ['mlp_ln_2_bias', 'get_attr', '-', [32], []],
  ['layer_norm_1', 'layernorm', 'aten.layer_norm.default', [32, 32], ['add_1', 'mlp_ln_2_weight', 'mlp_ln_2_bias']],
  ['mlp_c_fc_weight', 'get_attr', '-', [128, 32], []],
  ['transpose_5', 'transpose', 'aten.t.default', [32, 128], ['mlp_c_fc_weight']],
  ['matmul_6', 'matmul', 'aten.mm.default', [32, 128], ['layer_norm_1', 'transpose_5']],
  ['mlp_c_fc_bias', 'get_attr', '-', [128], []],
  ['add_7', 'add', 'aten.add.Tensor', [32, 128], ['matmul_6', 'mlp_c_fc_bias']],
  ['gelu', 'gelu', 'aten.gelu.default', [32, 128], ['add_7']],
  ['mlp_c_proj_weight', 'get_attr', '-', [32, 128], []],
  ['transpose_6', 'transpose', 'aten.t.default', [128, 32], ['mlp_c_proj_weight']],
  ['matmul_7', 'matmul', 'aten.mm.default', [32, 32], ['gelu', 'transpose_6']],
  ['mlp_c_proj_bias', 'get_attr', '-', [32], []],
  ['add_8', 'add', 'aten.add.Tensor', [32, 32], ['matmul_7', 'mlp_c_proj_bias']],
  ['add_2', 'add', 'aten.add.Tensor', [32, 32], ['add_1', 'add_8']],
  ['output', 'output', '-', [32, 32], ['add_2']],
];

// Per-node measured metrics from layers.csv (gpt2_block / chained).
// name: [cos, rel_l2, rel_max, sched, dyn, retired, bytes_loaded, bytes_written, flops]
const LAYERS = {
  layer_norm: [0.999994, 0.003615, 0.005464, 259, 6831, 8571, 10240, 2048, 17751],
  matmul_2: [0.999992, 0.004387, 0.004630, 281, 1672, 2022, 6208, 2048, 67250],
  add_3: [0.999991, 0.004619, 0.009259, 104, 905, 1303, 4096, 2048, 2523],
  matmul_3: [0.999991, 0.004468, 0.004049, 281, 1672, 2022, 6208, 2048, 67250],
  add_4: [0.999991, 0.004526, 0.003953, 104, 905, 1303, 4096, 2048, 2523],
  matmul_4: [0.999992, 0.004109, 0.004274, 281, 1672, 2022, 6208, 2048, 67250],
  add_5: [0.999992, 0.004309, 0.007843, 104, 905, 1303, 4096, 2048, 2523],
  matmul: [0.999977, 0.007220, 0.009217, 281, 1672, 2022, 6208, 2048, 67250],
  mul: [0.999975, 0.007425, 0.006536, 110, 1004, 1305, 4096, 2048, 2523],
  add: [1.0, 0.0, 0.000001, 104, 905, 1303, 4096, 2048, 2523],
  softmax: [0.999993, 0.004291, 0.001953, 106, 2333, 3047, 2048, 2048, 9296],
  matmul_1: [0.999987, 0.005814, 0.005319, 281, 1672, 2022, 6208, 2048, 67250],
  matmul_5: [0.999985, 0.005965, 0.004132, 281, 1672, 2022, 6208, 2048, 67250],
  add_6: [0.999986, 0.005549, 0.008696, 104, 905, 1303, 4096, 2048, 2523],
  add_1: [0.999998, 0.001923, 0.005208, 104, 905, 1303, 4096, 2048, 2523],
  layer_norm_1: [0.999993, 0.004096, 0.005464, 259, 6831, 8571, 10240, 2048, 17751],
  matmul_6: [0.99999, 0.004780, 0.006250, 281, 6409, 7758, 24832, 8192, 268874],
  add_7: [0.999989, 0.004988, 0.007812, 104, 3503, 5062, 16384, 8192, 10029],
  gelu: [0.999984, 0.006040, 0.012552, null, null, null, null, null, null],
  matmul_7: [0.999977, 0.006973, 0.007812, 281, 6370, 7719, 24832, 8192, 268871],
  add_8: [0.999975, 0.007149, 0.007812, 104, 905, 1303, 4096, 2048, 2523],
  add_2: [0.999996, 0.002945, 0.004545, 104, 905, 1303, 4096, 2048, 2523],
};

const KERNEL_FOR_OP = {
  matmul: ['gemm', 'atalla-graph/kernels/gemm.py'],
  add: ['add', 'atalla-graph/kernels/add.py'],
  mul: ['mul', 'atalla-graph/kernels/mul.py'],
  layernorm: ['layernorm', 'atalla-graph/kernels/layernorm.py'],
  softmax: ['softmax', 'atalla-graph/kernels/softmax.py'],
  gelu: ['gelu', 'atalla-graph/kernels/gelu.py'],
};

function tileFor(op, shape, inputs) {
  if (op === 'matmul') {
    const M = shape[0], N = shape[1];
    const K = inputs[1] && inputs[1].endsWith('5') ? 32 : (N === 32 && inputs[0] === 'gelu' ? 128 : 32);
    return { kernel_type: 'matmul', M, N, K, tile: 32, M_tiles: M / 32, N_tiles: N / 32, K_tiles: K / 32, k_stride: K, reuse: 'A-tile sweeps N; W-tile sweeps M; K outer with accum in C' };
  }
  if (op === 'layernorm') return { kernel_type: 'layernorm', rows: shape[0], width: shape[1], D: shape[1], eps: 1e-5 };
  if (op === 'softmax') return { kernel_type: 'softmax', num_rows: shape[0], row_len: shape[1], axis: -1 };
  if (op === 'add' || op === 'mul') return { kernel_type: op, rows: shape[0], width: shape[1], elems: shape[0] * shape[1], inplace: true };
  if (op === 'gelu') return { kernel_type: 'gelu', elems: shape[0] * shape[1], approx: 'erf' };
  return null;
}

function baseBackend(op) {
  if (op === 'placeholder' || op === 'get_attr' || op === 'output') return 'graph';
  if (op === 'transpose') return 'passthrough';
  if (op === 'gelu') return 'numpy';
  return 'emulator';
}

function reasonFor(op, backend) {
  if (backend === 'graph') return op === 'get_attr' ? 'parameter (not executed)' : 'graph boundary';
  if (backend === 'passthrough') return 'view/reshape op: no data movement, taken from the reference';
  if (backend === 'numpy') return op === 'gelu' ? 'gelu_kernel=pytorch: no Atalla emitter selected (set gelu_kernel=horner|tanh)' : 'NumPy fallback';
  if (backend === 'skip') return 'override: replaced by identity on first tensor input';
  return `planned by _plan_${op}`;
}

export function mockGraph(body) {
  const ov = body.overrides || {};
  const g = ov.global || {};
  const nodeOv = ov.nodes || {};
  const geluVariant = g.gelu_kernel && g.gelu_kernel !== 'pytorch' ? g.gelu_kernel : null;
  const addThresh = typeof g.add_numpy_threshold === 'number' ? g.add_numpy_threshold : 1000000;

  const nodes = G.map((row, order) => {
    const [name, op, target, shape, inputs] = row;
    let backend = baseBackend(op);
    let variant = null;
    let reason = reasonFor(op, backend);
    if (op === 'gelu' && geluVariant) { backend = 'emulator'; variant = geluVariant; reason = `global gelu_kernel=${geluVariant}: Atalla GELU emitter (${geluVariant})`; }
    if (op === 'add' && shape[0] * shape[1] > addThresh) { backend = 'numpy'; reason = `elements ${shape[0] * shape[1]} > add_numpy_threshold ${addThresh}`; }
    const o = nodeOv[name];
    if (o && backend !== 'graph') {
      if (o.backend === 'numpy') { backend = 'numpy'; reason = 'override: forced NumPy reference path'; }
      else if (o.backend === 'skip') { backend = 'skip'; reason = reasonFor(op, 'skip'); }
      else if (o.backend === 'emulator') { backend = 'emulator'; reason = 'override: forced Atalla kernel'; }
      if (o.kernel) variant = o.kernel;
    }
    const km = KERNEL_FOR_OP[op];
    const kernel = km && backend !== 'graph' && backend !== 'passthrough' && backend !== 'skip' ? {
      name: km[0], source: km[1], variant,
      tile: tileFor(op, shape, inputs),
      dram_addr: backend === 'emulator' ? (op === 'layernorm' ? '0x3000' : op === 'softmax' ? '0x1000' : '0x5000') : null,
      params: { dtype: 'bf16', elems: shape.reduce((a, b) => a * b, 1) },
    } : null;
    const phase = order < 32 ? 'attention' : 'mlp';
    return { id: name, name, op, target, phase: backend === 'graph' ? 'graph' : phase, shape, dtype: 'bf16', backend, kernel, inputs, reason, order };
  });
  const edges = [];
  nodes.forEach(n => n.inputs.forEach(src => edges.push({ source: src, target: n.id })));
  // run_graph.py's stats block counts 47 nodes for gpt2_block (the output node is not counted).
  const counts = { emulated: 0, numpy: 0, passthrough: 0, skip: 0, total: nodes.filter(n => n.op !== 'output').length };
  nodes.forEach(n => { if (n.backend === 'emulator') counts.emulated++; else if (n.backend in counts) counts[n.backend]++; });
  return {
    model: body.model || 'gpt2_block', model_class: 'GPT2Block', input_shape: [32, 32],
    nodes, edges, counts, effective_overrides: { global: { gelu_kernel: g.gelu_kernel || 'pytorch', sdpa: g.sdpa || 'numpy', add_numpy_threshold: addThresh }, nodes: nodeOv },
  };
}

// ---------------------------------------------------------------------------
// Runs (10 baseline rows from docs/baselines/README.md, real numbers)
// ---------------------------------------------------------------------------
// [run_id, model, tag, strict, E, N, P, cos, relL2, relMax, load, write, flops, AI, sched, dyn, retired, sfill, dfill, scale, elapsed]
const BASE = [
  ['basic__strict-chained__20261104-1458', 'basic', 'strict-chained', true, 10, 0, 1, 1.0, 0.0, 0.0, 7488, 640, 7884, 0.970, 1593, 2941, 3423, 0.296, 0.291, null, 5.779],
  ['layernorm_smoke__strict-chained__20261104-1458', 'layernorm_smoke', 'strict-chained', true, 3, 0, 1, 0.999994, 3.867e-3, 6.061e-3, 2688, 192, 3019, 1.048, 644, 1100, 1294, 0.300, 0.294, null, 2.374],
  ['alexnet_small__scale0.01-chained__20261104-1458', 'alexnet_small', 'scale0.01-chained', null, 21, 0, 4, 0.999999, 1.287e-3, 2.392e-3, 97652, 9276, 3124918, 29.225, 3720, 101215, 119805, 0.294, 0.296, 0.01, 12.274],
  ['vit_micro__default-sdpa-chained__20261104-1458', 'vit_micro', 'default-sdpa-chained', null, 19, 2, 7, 0.999995, 3.127e-3, 3.289e-3, 121408, 45056, 668399, 4.015, 3525, 38427, 49390, 0.297, 0.321, null, 10.648],
  ['vit_micro__flash-trial-chained__20261104-1458', 'vit_micro', 'flash-trial-chained', null, 20, 1, 7, 0.999994, 3.341e-3, 4.934e-3, 254592, 47104, 833695, 2.763, 3801, 80397, 101131, 0.298, 0.314, null, 11.353],
  ['gpt2_mlp__chained__20261104-1459', 'gpt2_mlp', 'chained', null, 6, 1, 2, 0.999998, 2.013e-3, 3.731e-3, 84480, 30720, 570571, 4.953, 1133, 24923, 31716, 0.299, 0.318, null, 4.329],
  ['gpt2_mlp__gelu-emulator-chained__20261104-1459', 'gpt2_mlp', 'gelu-emulator-chained', null, 7, 0, 2, 0.999997, 2.581e-3, 3.731e-3, 92672, 38912, 719525, 5.468, 1447, 49073, 65358, 0.305, 0.333, null, 5.437],
  ['gpt2_attn__chained__20261104-1459', 'gpt2_attn', 'chained', null, 15, 0, 5, 0.999999, 1.658e-3, 5.000e-3, 78208, 30720, 448208, 4.115, 2785, 25630, 32873, 0.296, 0.321, null, 9.669],
  ['gpt2_block__chained__20261104-1459', 'gpt2_block', 'chained', null, 21, 1, 7, 0.999996, 2.945e-3, 4.545e-3, 162688, 61440, 1018779, 4.546, 3918, 50553, 64589, 0.297, 0.319, null, 11.881],
  ['gpt2_block__gelu-emulator-chained__20261104-1459', 'gpt2_block', 'gelu-emulator-chained', null, 22, 0, 7, 0.999995, 3.190e-3, 4.545e-3, 170880, 69632, 1167733, 4.855, 4232, 74703, 98231, 0.299, 0.329, null, 12.773],
];

function startedFrom(id) {
  const m = id.match(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})$/);
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:00+00:00` : null;
}

function summaryFrom(r) {
  const [run_id, model, tag, strict, E, N, P, cos, rl2, rmax, ld, wr, flops, ai, sched, dyn, ret, sf, df, scale, el] = r;
  const isBlock = model === 'gpt2_block';
  const layers = isBlock ? G.filter(x => LAYERS[x[0]]).map(x => {
    const L = LAYERS[x[0]];
    return { node: x[0], op: x[1], backend: x[0] === 'gelu' && tag === 'chained' ? 'numpy' : 'emulator', shape: x[3], cos_sim: L[0], rel_l2: L[1], rel_max: L[2], sched_packets: L[3], dyn_packets: L[4], retired_instructions: L[5], bytes_loaded: L[6], bytes_written: L[7], flops: L[8] };
  }) : [];
  return {
    run_id, model, tag, scale, seed: 42, validate_inputs: 'chained', flags: tag.includes('gelu') ? 'ATALLA_GELU_KERNEL=horner' : (tag.includes('flash') ? 'ATALLA_SDPA_FLASH=1' : ''),
    timestamp_utc: startedFrom(run_id), strict_pass: strict, strict_fail_reasons: [], run_returncode: 0,
    final_cosine: cos, rel_l2: rl2, rel_max: rmax, sched_packets: sched, dyn_packets: dyn, retired_instructions: ret,
    static_fill: sf, dynamic_fill: df, bytes_loaded: ld, bytes_written: wr, bytes_total: ld + wr, flops, arithmetic_intensity: ai,
    coverage_emulated: E, coverage_numpy: N, coverage_passthrough: P, coverage_total_nodes: isBlock ? 47 : null,
    elapsed_s_wall: el, layers,
  };
}

export const RUNS = BASE.map(r => ({ run_id: r[0], model: r[1], tag: r[2], status: 'done', started: startedFrom(r[0]), final_cosine: r[7], coverage: { emulated: r[4], numpy: r[5], passthrough: r[6] }, dyn_packets: r[15], bytes_total: r[10] + r[11] }));
const SUMMARIES = Object.fromEntries(BASE.map(r => [r[0], summaryFrom(r)]));

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
export const KERNELS = [
  { name: 'gemm', path: 'atalla-graph/kernels/gemm.py', kind: 'generator', ops: ['matmul', 'linear'] },
  { name: 'add', path: 'atalla-graph/kernels/add.py', kind: 'generator', ops: ['add'] },
  { name: 'mul', path: 'atalla-graph/kernels/mul.py', kind: 'generator', ops: ['mul'] },
  { name: 'layernorm', path: 'atalla-graph/kernels/layernorm.py', kind: 'generator', ops: ['layernorm'] },
  { name: 'softmax', path: 'atalla-graph/kernels/softmax.py', kind: 'generator', ops: ['softmax'] },
  { name: 'gelu', path: 'atalla-graph/kernels/gelu.py', kind: 'generator', ops: ['gelu'] },
  { name: 'gelu_tanh', path: 'atalla-graph/kernels/gelu_tanh.c', kind: 'static_c', ops: ['gelu'] },
  { name: 'flash_sdpa_n32d32', path: 'atalla-graph/kernels/flash_sdpa_n32d32.c', kind: 'static_c', ops: ['sdpa'] },
  { name: 'relu', path: 'atalla-graph/kernels/relu.py', kind: 'generator', ops: ['relu'] },
  { name: 'maxpool', path: 'atalla-graph/kernels/maxpool.py', kind: 'generator', ops: ['max_pool2d'] },
  { name: 'avgpool', path: 'atalla-graph/kernels/avgpool.py', kind: 'generator', ops: ['avg_pool2d', 'adaptive_avg_pool2d'] },
  { name: 'batchnorm', path: 'atalla-graph/kernels/batchnorm.py', kind: 'generator', ops: ['batch_norm'] },
  { name: 'sum_dim', path: 'atalla-graph/kernels/sum_dim.py', kind: 'generator', ops: ['sum'] },
  { name: 'mse_loss', path: 'atalla-graph/kernels/mse_loss.py', kind: 'generator', ops: ['mse_loss'] },
];

function kernelText(name) {
  return `# ${name}.py  (mock excerpt: the real API returns the full file)
"""Atalla ${name} kernel emitter.

Generates C for the in-tree pipeline (c_emitter -> ppci -p packetized asm),
then runs it on the functional simulator. Tile config comes from tile_planner.
"""
from .common import KernelSpec, emit_header, scpad_load, scpad_store


def plan(tc):
    # tc.params carries M/N/K/tile/etc.; nothing here guesses shapes.
    return KernelSpec(name="${name}", dram_addr=0x5000, tile=tc.params)


def emit(spec, out_path):
    with open(out_path, "w") as f:
        f.write(emit_header(spec))
        f.write("int main() {\\n")
        f.write("    int* cfg = (int*)60;\\n")
        f.write("    /* ... tile loop over cfg[] ... */\\n")
        f.write("    return 0;\\n}\\n")
`;
}

const ARTIFACT_C = `void load_weight_tile(int sp_w, int w_addr, int sdma_ctl_w) {
    int* cfg_tbl = (int*)60;
    int z_addr = *(cfg_tbl + 10);
    int tile_sz = *(cfg_tbl + 9);
    int sdma_ctl_z = 32505887;
    scpad_load(0, z_addr, sdma_ctl_z);
    vec zrow = vector_load(0, 0, 31, 0);
    scpad_load(sp_w, w_addr, sdma_ctl_w);
    int wn = 32;
    while (wn < tile_sz) {
        vector_store(zrow, sp_w, wn, 31, 1);
        wn = wn + 1;
    }
    int wi = 0;
    while (wi < tile_sz) {
        vec wvec = vector_load(sp_w, wi, 31, 1);
        load_weights(wvec);
        wi = wi + 1;
    }
}

int main() {
    int* cfg = (int*)60;
    int A_GMEM = *(cfg + 0);
    int W_GMEM = *(cfg + 1);
    int C_GMEM = *(cfg + 2);
    int gM = *(cfg + 3);
    int gN = *(cfg + 4);
    int gK = *(cfg + 5);
    int M_tiles = *(cfg + 6);
    /* ... (mock excerpt of out/<node>.c) ... */
    return 0;
}
`;

const ARTIFACT_S = `       .section data
       .section data
       .section code
       global load_weight_tile
       type load_weight_tile func
       .align 5
 load_weight_tile:
       addi_s x2, x2, -48
       sw_s x1, 4(x2)
       sw_s x8, 0(x2)
       sw_s x33, 8(x2)
       addi_s x8, x2, 8
       addi_s x2, x2, -16
       addi_s x33, x32, 0
       addi_s x32, x32, -128
 load_weight_tile_block0:
 load_weight_tile_block1:
       sw_s x12, 36(x8)
       sw_s x13, 32(x8)
       sw_s x14, 28(x8)
       ; ... (mock excerpt of out/<node>.s) ...
`;

// ---------------------------------------------------------------------------
// Mock API: same surface as the real client in app.js
// ---------------------------------------------------------------------------
const live = {}; // run_id -> { events:[], timers:[], listeners:Set, status }
const delay = (ms) => new Promise(r => setTimeout(r, ms));

function nowId(model, tag) {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${model}__${tag || 'ui'}__${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}-${p(d.getUTCHours())}${p(d.getUTCMinutes())}`;
}

function emit(run, ev) {
  run.events.push(ev);
  run.listeners.forEach(fn => { try { fn(ev); } catch (e) { console.error(e); } });
}

function startMockRun(id, body) {
  const graph = mockGraph({ model: body.model, overrides: body.overrides });
  const run = { events: [], listeners: new Set(), status: 'running', config: body, summary: null, graph };
  live[id] = run;
  const t0 = Date.now() / 1000;
  const compute = graph.nodes.filter(n => n.backend !== 'graph');
  emit(run, { event: 'start', model: body.model, mode: 'validate', total_nodes: graph.counts.total, compute_nodes: compute.length, ts: t0 });
  let t = 250;
  let agg = { sched: 0, dyn: 0, ret: 0, ld: 0, wr: 0, flops: 0 };
  let lastCos = 1;
  compute.forEach((n) => {
    const L = LAYERS[n.name];
    const isEmu = n.backend === 'emulator';
    const step = isEmu ? 380 : 90;
    setTimeout(() => emit(run, { event: 'node_start', index: n.order, name: n.name, op: n.op, backend: n.backend, shape: n.shape, ts: t0 + t / 1000 }), t);
    t += step;
    setTimeout(() => {
      const ev = { event: 'node_done', index: n.order, name: n.name, op: n.op, backend: n.backend, shape: n.shape, cos: null, rel_l2: null, rel_max: null, sched_packets: null, packets: null, instructions: null, bytes_loaded: null, bytes_written: null, flops: null, elapsed_s: step / 1000, ts: t0 + t / 1000 };
      if (L && n.backend !== 'passthrough' && n.backend !== 'skip') {
        ev.cos = L[0]; ev.rel_l2 = L[1]; ev.rel_max = L[2]; lastCos = L[0];
        if (isEmu) {
          const emu = L[3] != null ? L : [L[0], L[1], L[2], 310, 9100, 12800, 16384, 8192, 64570]; // gelu forced to emulator: numbers from gelu-emulator baseline delta
          ev.sched_packets = emu[3]; ev.packets = emu[4]; ev.instructions = emu[5]; ev.bytes_loaded = emu[6]; ev.bytes_written = emu[7]; ev.flops = emu[8];
          agg.sched += emu[3]; agg.dyn += emu[4]; agg.ret += emu[5]; agg.ld += emu[6]; agg.wr += emu[7]; agg.flops += emu[8];
        }
      }
      emit(run, ev);
    }, t);
    t += 40;
  });
  setTimeout(() => {
    const el = t / 1000;
    const c = graph.counts;
    const finalCos = c.skip > 0 ? 0.97 + Math.random() * 0.02 : (c.emulated >= 22 ? 0.999995 : 0.9999957);
    run.summary = {
      run_id: id, model: body.model, model_class: graph.model_class, tag: body.tag, scale: body.scale, seed: body.seed, validate_inputs: 'chained',
      flags: '', timestamp_utc: new Date(t0 * 1000).toISOString(), strict_pass: body.strict ? finalCos >= 0.999 : null, strict_fail_reasons: [], run_returncode: 0,
      final_cosine: finalCos, rel_l2: c.skip > 0 ? 0.21 : 0.0029, rel_max: c.skip > 0 ? 0.4 : 0.0045,
      sched_packets: agg.sched, dyn_packets: agg.dyn, retired_instructions: agg.ret, static_fill: 0.297, dynamic_fill: 0.319,
      bytes_loaded: agg.ld, bytes_written: agg.wr, bytes_total: agg.ld + agg.wr, flops: agg.flops, arithmetic_intensity: agg.flops / Math.max(1, agg.ld + agg.wr),
      coverage_emulated: c.emulated, coverage_numpy: c.numpy, coverage_passthrough: c.passthrough, coverage_total_nodes: c.total,
      elapsed_s_wall: el, layers: run.events.filter(e => e.event === 'node_done' && e.cos != null).map(e => ({ node: e.name, op: e.op, backend: e.backend, shape: e.shape, cos_sim: e.cos, rel_l2: e.rel_l2, rel_max: e.rel_max, sched_packets: e.sched_packets, dyn_packets: e.packets, retired_instructions: e.instructions, bytes_loaded: e.bytes_loaded, bytes_written: e.bytes_written, flops: e.flops })),
    };
    run.status = 'done';
    emit(run, { event: 'done', final_cosine: finalCos, rel_l2: run.summary.rel_l2, rel_max: run.summary.rel_max, strict_pass: run.summary.strict_pass, elapsed_s: el, ts: t0 + el });
  }, t + 300);
  return run;
}

export const mockApi = {
  name: 'mock',
  async models() { await delay(120); return MODELS; },
  async graph(body) { await delay(260); return mockGraph(body); },
  async kernels() { await delay(100); return KERNELS; },
  async kernel(name) { await delay(120); const k = KERNELS.find(k => k.name === name); if (!k) throw new Error(`unknown kernel ${name}`); return { name, path: k.path, text: kernelText(name) }; },
  async createRun(body) {
    await delay(200);
    if (Object.values(live).some(r => r.status === 'running')) { const e = new Error('a run is already in progress'); e.status = 409; throw e; }
    const id = nowId(body.model, body.tag);
    startMockRun(id, body);
    return { run_id: id };
  },
  async runs() {
    await delay(150);
    const extra = Object.entries(live).map(([id, r]) => ({ run_id: id, model: r.config.model, tag: r.config.tag, status: r.status, started: new Date(r.events[0].ts * 1000).toISOString(), final_cosine: r.summary ? r.summary.final_cosine : null, coverage: { emulated: r.graph.counts.emulated, numpy: r.graph.counts.numpy, passthrough: r.graph.counts.passthrough }, dyn_packets: r.summary ? r.summary.dyn_packets : null, bytes_total: r.summary ? r.summary.bytes_total : null }));
    return [...extra.reverse(), ...RUNS.slice().reverse()];
  },
  async run(id) {
    await delay(150);
    if (live[id]) { const r = live[id]; return { status: r.status, config: { ...r.config, run_id: id }, summary: r.summary, progress: r.events.slice() }; }
    const s = SUMMARIES[id]; if (!s) { const e = new Error('run not found'); e.status = 404; throw e; }
    return { status: 'done', config: { run_id: id, model: s.model, tag: s.tag, scale: s.scale, seed: s.seed, flags: s.flags, timestamp_utc: s.timestamp_utc }, summary: s, progress: [] };
  },
  events(id, onEvent) {
    const r = live[id];
    if (!r) { setTimeout(() => onEvent({ event: 'error', message: 'no live stream for a baseline run' }), 10); return () => {}; }
    r.events.forEach(e => onEvent(e));
    if (r.status !== 'running') return () => {};
    r.listeners.add(onEvent);
    return () => r.listeners.delete(onEvent);
  },
  async log(id) {
    await delay(120);
    const r = live[id];
    const lines = r ? r.events.map(e => e.event === 'node_done' ? `  [${e.name}] ${e.op} -> ${e.backend}${e.cos != null ? ` (cos=${e.cos.toFixed(6)}, rel_l2=${e.rel_l2.toFixed(6)})` : ''}` : `  ${JSON.stringify(e)}`) : ['  (baseline run.log is served by the real API)'];
    return `$ python run_graph.py --model ${(r ? r.config.model : id.split('__')[0])} --mode validate --seed 42 --progress-jsonl progress.jsonl\nFX trace + lower modules...\nShape propagation...\nNormalize ops + plan tiles...\n\n--- Graph (validate_inputs=chained) ---\n${lines.join('\n')}\n`;
  },
  async artifact(id, node, kind) {
    await delay(120);
    const text = kind === 'c' ? ARTIFACT_C : kind === 's' ? ARTIFACT_S : `# ${node}.in\n0x3000 1024 bf16\n...`;
    return { node, kind, path: `docs/baselines/${id}/out/${node}.${kind}`, text };
  },
  async cancel(id) { const r = live[id]; if (r) { r.status = 'error'; emit(r, { event: 'error', message: 'cancelled from the UI' }); } return { ok: true }; },
  async exportRun(id) { await delay(600); return { run_xlsx: `output/runs/${id}.xlsx`, tracker_xlsx: 'output/Atalla_Benchmarking_Tracker.xlsx' }; },
};
