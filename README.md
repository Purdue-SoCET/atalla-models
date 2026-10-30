# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → **atalla-functional-sim** emulator.

## Upstream sources

Two **Git submodules** plus an in-tree graph frontend (not a submodule):

| Path | Upstream | Branch |
|------|----------|--------|
| `aihw-ppci-compiler/` | [Purdue-SoCET/aihw-ppci-compiler](https://github.com/Purdue-SoCET/aihw-ppci-compiler) | `atalla-models` |
| `functional_sim/` | [Purdue-SoCET/atalla-functional-sim](https://github.com/Purdue-SoCET/atalla-functional-sim) | `main` |
| `atalla-graph/` | Based on [vihaanrc/atalla-graph](https://github.com/vihaanrc/atalla-graph); evolved in this repo | — |

Graph/compiler/emulator deltas are summarized in `pipeline_reference.md` §9.

## Clone and submodules

```bash
git clone --recurse-submodules git@github.com:Purdue-SoCET/atalla-models.git
cd atalla-models
git submodule update --init --recursive
```

You need **`functional_sim`** and **`aihw-ppci-compiler`** populated for validate mode. Nested submodules are not required for the graph path.


## Pipeline flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp + lower_linear_modules
    v
FX graph (normalized ops: conv, matmul, relu, softmax, maxpool, add, …)
    |  tile_planner.py
    v
Tiled FX graph
    |  c_emitter.py + kernels/*.py
    |  ├─ AtallaC source per node
    |  └─ DRAMWriter: config table + tensors as .data section
    v
AtallaC .c ─────────────────────────── DRAMWriter (data section)
    |  ppci atalla_cc -S                      |
    v                                         |
ppci .s                                       |
    |  functional_sim/build_compiler.compile_asm()
    |  (VLIW scheduling + encoding)           |
    v                                         |
instruction section ──── + ────── data section
                         |
                    render_testfile()
                         |
                         v
                 .in file (complete)
                         |
                    functional_sim (emulator)
                         |
                         v
                   layer output → activation_cache
```

## Directory layout

```
atalla-models/
    atalla-graph/
        graph/                 fx_capture.py, tile_planner.py (+ upstream export/lower/allocator)
        codegen/               c_emitter.py, dram_builder.py
        kernels/               AtallaC generators: gemm, relu, softmax, maxpool, add
        model/                 basic.py, alexnet.py, alexnet_small.py
        scripts/               generate_schedule.py
        run_graph.py           validate | schedule | both
    functional_sim/            Submodule (emulator + build.py + build_compiler + build_*.py)
    aihw-ppci-compiler/        Submodule (ppci / atalla_cc)
    pipeline_reference.md      Technical deep-dive
    CONTRIBUTING.md            How to add a kernel
```

## Usage

```bash
cd atalla-graph

python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate --scale 0.01


python run_graph.py --model basic --mode schedule --out-dir out/graph

python run_graph.py --model alexnet_small --mode validate --metrics-json metrics_alexnet.json
```

Models: `basic`, `alexnet_small`

## Validation snapshot (representative)

| Model | Emulated | NumPy | Passthrough | End-to-end cos sim |
|-------|----------|-------|-------------|-------------------|
| BasicModule (dim=32, depth=2) | 9 | 1 (`mul`) | 1 | ≈ **0.99999** |
| AlexNetSmall (`scale=0.01`) | 21 | 0 | 4 | ≈ **0.996** |


## Performance metrics (poster / roofline)


```bash
cd atalla-graph
python run_graph.py --model basic --mode validate --metrics-json metrics_basic.json
python run_graph.py --model alexnet_small --mode validate --metrics-json metrics_alexnet.json
```

Each emulated layer gets a row in `kernel_metrics`; the file also contains `aggregate_metrics` over **all** emulated layers. Tables below used **fixed seeds** (`torch`/`numpy` 42), **`scale=0.01`** for AlexNetSmall, current `main` submodules.

### Metrics from Akshath

| Metric | Meaning |
|--------|---------|
| **Static VLIW slot efficiency (scheduled)** | `sched_slots_filled / (sched_packets × 4)` from `build_compiler` packets. Measures how full each **4-slot** issue row is in the **static** `.in` program. |
| **Non-empty packet efficiency** | Same numerator; denominator uses only packets with **≥1** non-`nop` issue (excludes scheduler padding rows). Reported as `aggregate_static_slot_efficiency_nonempty`. |
| **Dynamic slot efficiency** | `instructions_retired / (emu_packets × 4)` from the functional model (counts what actually retired). Loops / control flow differ from static schedule. |
| **Bytes loaded / written** | BF16 traffic counted on **SDMA loads** (`scpad.ld`) and **stores** (`scpad.st`) in the emulator (2 bytes per element moved). Does not yet count every scalar/GMEM path. |
| **FLOPs (`flops_total`)** | Emulator counters: vector ALU + matmul contributions (approximate; use for relative comparisons). |
| **Arithmetic intensity** | `flops_total / (bytes_loaded + bytes_written)` after the layer run (“roofline-style” bytes moved for modeled DMA). **`arithmetic_intensity_loads`** uses **loads only** in the denominator. |
| **`sched_slot_histogram`** | Count of scheduled packets by **number of filled slots** (0–4). Explains padding-heavy schedules. |

### What these enable

- **Kernel comparison:** same op with handwritten `build_*.py` vs ppci output — compare static slot efficiency and bytes moved.
- **Tiling comparison:** baseline vs pipelined / different tile sizes — watch total bytes and intensity.
- **Per-layer insight:** e.g. large **conv** tiles should show high byte traffic; **matmul** depth shows in FLOPs vs bytes.

### Full model aggregates

| Model | Emul. layers | Static slot η (all pkts) | Static slot η (non-empty) | Dynamic slot η | Bytes L / W | Total bytes | FLOPs (cnt) | AI (FLOP/B) |
|-------|--------------|-------------------------|----------------------------|----------------|-------------|-------------|-------------|-------------|
| BasicModule | 9 | 0.037 | **0.269** | 0.063 | 7168 / 576 | 7744 | 1528 | 0.20 |
| AlexNetSmall (`scale=0.01`) | 21 | 0.043 | **0.271** | 0.097 | 94196 / 9276 | 103472 | 85559 | 0.83 |

### Per-layer samples (same runs)

| Node (model) | Op | Static slot η | Bytes load / store | FLOPs | AI (FLOP/B) |
|--------------|-----|---------------|--------------------|-------|-------------|
| `matmul` (Basic) | matmul | 0.042 | 2176 / 64 | 278 | 0.12 |
| `relu` (Basic) | relu | 0.051 | 64 / 64 | 159 | 1.24 |
| `add` (Basic) | add | 0.027 | 128 / 64 | 94 | 0.49 |
| `conv2d` (AlexNetSmall) | conv | 0.042 | 59072 / 2048 | 50504 | 0.83 |


## Design notes

- **Decoupled Architecture (Operation Without Kernels):** `atalla-models` operates independently of custom kernel compilation. Any op without a custom AtallaC kernel or marked `skip_emulator = True` automatically falls back to PyTorch/NumPy reference passthrough, allowing models to validate end-to-end without custom kernels.
- **Standalone C Goldens:** Kernels are developed as standalone `.c` files with vector intrinsics and 0 stack spills, verified independently against NumPy goldens on `atalla-functional-sim`, and designed to be ported directly to the native C++ MLIR compiler (`atalla-graph`).
- **Compiler path for compute ops.** Conv, Linear, Matmul, ReLU, Softmax, MaxPool, Add go through AtallaC → ppci → `build_compiler` → emulator. **`mul`** on the basic module may still use a NumPy fallback unless a kernel is added.
- **Passthrough ops** (flatten, transpose, dropout at inference): reshapes only — no `.in` file.
- **Graph owns `.data`.** `DRAMWriter` (ADDR_TABLE at `0x3C`) + `render_testfile()`; see `pipeline_reference.md` §6.2.
- **Per-layer emulator run.** Fresh emulator state per layer; `activation_cache` passes BF16-aligned activations.
- **Stack / DRAM.** Scalar `x2` placed above tensor data to avoid overlap with spills and globals.

See **`CONTRIBUTING.md`** to add a new op end-to-end.

## Kernel source files

Accelerator kernel bodies live in `atalla-graph/kernels/` as ordinary `.c`
files. Python wrappers read those files without substituting source text.
The graph emitter writes dimensions, strides, masks, and DMA control words
into configuration memory at address 60. Each C kernel reads its runtime
parameters from there. `kernels/common.py` defines the configuration layouts;
`codegen/c_emitter.py` writes integer and FP32 words with the correct types.

Vector load/store widths and scratchpad IDs are ISA immediates. Kernels use
fixed 32-lane instructions, runtime DMA dimensions, and masks where needed.
GEMM explicitly zeroes inactive weight lanes. Elementwise buffers are padded
to complete DMA tiles. MaxPool iterates its window in C; its existing
horizontal postprocessing remains in the graph runner. Existing unsupported
shape fallbacks (such as wide Softmax) are unchanged.

Run the source and numerical regression checks separately from the legacy
standalone-kernel suite, which uses a conflicting top-level `kernels` package:

```bash
../.venv/bin/python3 -m pytest tests -q
../.venv/bin/python3 -m pytest atalla-graph/tests/test_runtime_kernels.py atalla-graph/tests/test_chained_host_ops.py -q
```

## Analytical pipelining estimate

`atalla-graph/scripts/pipelining_latency_model.py` estimates ideal input
prefetch overlap for **uniform tiles**. Results are unvalidated estimates,
not hardware cycle measurements. The default 520 cycles per 64-byte line is
a heuristic. Compute cycles and tile count must be supplied explicitly;
simulator retired packets can instead be selected as an explicit proxy.

From this repository, using the workspace Python environment:

```bash
../.venv/bin/python3 atalla-graph/scripts/pipelining_latency_model.py \
  --name example --tiles 2 --bytes-loaded 128 --bytes-written 128 \
  --compute-cycles 20 --lat-scpad 10 --assume-packed-transfers

../.venv/bin/python3 atalla-graph/scripts/pipelining_latency_model.py \
  --perf out/graph/add_perf.out --tiles 1 \
  --use-packets-as-compute-proxy --assume-packed-transfers
```

Use `--dir out/graph` instead of `--perf` to analyze multiple files. All
files use the supplied uniform tile count and assumptions; this does not
infer geometry from a graph. CLI byte/cycle overrides apply to each file.

Byte and compute inputs are **totals over all tiles**. Packed mode assumes
one contiguous, aligned transfer in each direction per tile and rounds each
tile separately. Alternatively, repeat `--load-transfer-bytes N` and
`--store-transfer-bytes N` for independent aligned transfers **within each
tile**. Their sums times the tile count must match byte totals, if supplied.
For strided rows, specify each row separately. Unaligned transfers require
an external transaction model; useful-byte counters alone cannot recover
physical DRAM traffic.

For per-tile load cost L, compute budget C, store cost S, tile count K, and
extra total pipeline overhead H, the assumed schedule is:

```text
serial    = K * (L + C + S)
pipelined = L + C + (K - 1) * max(L, C) + K * S + H
```

Set `--pipeline-overhead-cycles H` to explore additional setup/control
costs. Stores are charged serially outside the overlap phase. The model
assumes sufficient buffers and independent load/compute resources; it does
not prove that the hardware or generated code supports this schedule.
Packet proxies include DMA/control/scalar work and are not isolated compute
measurements. Contention, bandwidth, dependencies, uneven tiles, and actual
buffer lifetimes are not modeled. Calibrate against timed RTL or hardware
before using these estimates as performance claims.

Run analytical/CLI regression tests with:

```bash
../.venv/bin/python3 -m pytest tests/test_pipelining_latency_model.py -q
```
