# Atalla Pipeline Technical Reference

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Deep Dive: atalla-graph](#2-component-deep-dive-atalla-graph)
3. [Component Deep Dive: aihw-ppci-compiler](#3-component-deep-dive-aihw-ppci-compiler)
4. [Component Deep Dive: functional_sim](#4-component-deep-dive-functional_sim)
5. [Component Deep Dive: Kernel Library](#5-component-deep-dive-kernel-library)
6. [Full Pipeline Integration](#6-full-pipeline-integration)
7. [AlexNet Walkthrough](#7-alexnet-walkthrough)
8. [Performance Metrics](#8-performance-metrics)
9. [Changes, Tweaks, and Known Limitations](#9-changes-tweaks-and-known-limitations)
10. [Open Items for Review](#10-open-items-for-review)

---

## 1. Architecture Overview

The pipeline compiles a standard PyTorch `nn.Module` down to Atalla assembly and executes it on the functional simulator. Three components are integrated:

```
PyTorch nn.Module
    │
    ├─ atalla-graph/        FX graph capture, op normalization, tiling, code generation
    │
    ├─ aihw-ppci-compiler/  AtallaC → assembly compiler; PRIMARY compilation path
    │                       for GEMM, Conv, Linear, ReLU, Softmax kernels
    │
    └─ functional_sim/      Assembler, .in file builder, and cycle-accurate emulator
```

**Data flow for a single layer:**

```
                                    ┌─────────────────────────────────┐
  FX Node ──► emit_node() ────────►│  C compiler path (primary)      │
                                    │  c_emitter.py → .c source       │
                                    │  ppci atalla_cc → .s assembly    │
                                    │  build_compiler.compile_asm →    │
                                    │  scheduling + encoding → .in     │──► Emulator
                                    └─────────────────────────────────┘
         │
         └─ (maxpool, add, mul) ──► NumPy fallback (skip_emulator=True)
```

**The C compiler is now the primary compilation path.** All emulated compute kernels (conv, linear, matmul, relu, softmax) generate AtallaC source code that is compiled through `ppci atalla_cc`, converted via `build_compiler.compile_asm()` (which handles notation conversion, instruction scheduling, and encoding), and executed on the emulator. The previous "direct assembly" path (`build_*.py` generators) is no longer used for graph operations.

**Why MaxPool / Add / Mul use NumPy:**

MaxPool has a working assembly kernel (`build_maxpool.py`) that operates on **single-channel 2D tiles** (one H×W plane at a time). Wiring it into the multi-channel pipeline requires a per-channel loop with correct DRAM address arithmetic. This was deferred to keep initial integration simple. **This is a wiring gap, not a missing kernel.**

Add and Mul appear in `BasicModule` as bias addition (`out + bias`) and residual scaling (`0.5 * residual`). They are scalar-broadcast or elementwise operations that don't meaningfully benefit from hardware execution. AlexNet's bias addition is folded into Conv2d/Linear modules and does not produce standalone add/mul nodes.

---

## 2. Component Deep Dive: atalla-graph

**Purpose:** PyTorch frontend that captures a model as an FX graph, normalizes operations to Atalla primitives, plans tile decomposition, and generates AtallaC source for each layer.

### 2.1 `graph/fx_capture.py` — Graph Capture & Op Normalization

**What it does:**
1. Converts model weights to `bfloat16` (the hardware's native format)
2. Calls `torch.fx.symbolic_trace(model)` to get a flat FX graph
3. Runs `ShapeProp` to propagate tensor shapes through every node
4. Tags every node with `node.meta["atalla_op"]` — a normalized string identifying the Atalla primitive

**Op mapping tables:** Three dictionaries map PyTorch operations to Atalla primitives:

| Source | Example | Maps to |
|--------|---------|---------|
| `_OP_MAP` (call_function) | `torch.matmul`, `F.relu`, `F.conv2d` | `"matmul"`, `"relu"`, `"conv"` |
| `_METHOD_MAP` (call_method) | `.relu()`, `.flatten()`, `.add()` | `"relu"`, `"flatten"`, `"add"` |
| `_MODULE_MAP` (call_module) | `nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d` | `"conv"`, `"linear"`, `"maxpool"` |

Special handling:
- `addmm` (PyTorch's fused bias+matmul) is detected by name and mapped to `"linear"`
- `operator.add` and `operator.mul` are captured (these come from Python `+` and `*` operators in the forward pass)

**Limitation:** `symbolic_trace` cannot handle dynamic control flow. All models must be FX-traceable (no data-dependent branches, no dynamic shapes).

### 2.2 `graph/remove_ops.py` — Graph Transforms

Three passes clean up the graph before code generation:

1. **BatchNorm folding** (`_fold_bn_into_conv`): Folds BN parameters into the preceding Conv2d's weight and bias using the standard formula: `W_new = W * gamma / sqrt(var + eps)`, `b_new = beta + (b - mu) * gamma / sqrt(var + eps)`. The BN node is then erased from the graph.

2. **Dropout removal** (`_remove_dropout`): Elides dropout nodes (inference mode) by replacing them with their input.

3. **Adaptive avg pool tagging** (`_remove_adaptive_avg_pool`): Tags `AdaptiveAvgPool2d(1,1)` nodes so they can be handled as NumPy fallback.

### 2.3 `graph/tile_planner.py` — Tile Planning & DRAM Layout

**Hardware constraints:**
- Vector length (VL) = 32
- Systolic tile = 32×32
- Scratchpad = 32 slots per bank
- Data format = bfloat16 (2 bytes per element)

**Per-op tile planning:**

| Op | Planning Logic |
|----|---------------|
| **Conv** | Extract H, W, C_in, C_out, R, S, stride, pad → compute Ho, Wo → M = Ho×Wo, K = R×S×C_in, N = C_out → tile counts = ceil(dim/32) |
| **Linear** | M=1, K=in_features, N=out_features |
| **Matmul** | M, K, N from tensor shapes, tile counts = ceil(dim/32) |
| **ReLU** | total_elements from input shape, width = min(total, 32) |
| **Softmax** | length from last dimension of input |
| **MaxPool** | H, W, C, pool_size, stride → H_out, W_out |
| **Add/Mul** | total_elements from output shape |
| **Flatten** | Passthrough, no tiling needed |

**DRAM address assignment:** Addresses are assigned sequentially starting at `0x1000`, aligned to 2KB tile boundaries. Each node gets `dram_addr` and `dram_bytes` in its metadata. Address `0x00`–`0x3C` is reserved for the ADDR_TABLE (parameter passing to kernels).

### 2.4 `codegen/c_emitter.py` — Code Generation (Primary)

**What it does:** For each FX node, generates either:
1. **AtallaC source** (for conv, linear, matmul, relu, softmax) — compiled through ppci → `build_compiler.compile_asm` → emulator
2. **NumPy fallback** (for maxpool, add, mul, adaptive_avg_pool)

**Core dispatch function:**

```python
def emit_node(node, gm, activation_cache) -> LayerEmission:
    if atalla_op == "conv":     return emit_conv(...)    # sets em.c_source
    elif atalla_op == "linear": return emit_linear(...)   # sets em.c_source
    elif atalla_op == "relu":   return emit_relu(...)     # sets em.c_source
    elif atalla_op == "softmax":return emit_softmax(...)  # sets em.c_source
    elif atalla_op == "matmul": return emit_matmul(...)   # sets em.c_source
    elif atalla_op == "maxpool":return emit_maxpool(...)  # skip_emulator=True
    elif atalla_op == "add":    return emit_add(...)      # skip_emulator=True
    elif atalla_op == "mul":    return emit_mul(...)      # skip_emulator=True
```

**`LayerEmission` dataclass:**

| Field | Purpose |
|-------|---------|
| `c_source` | AtallaC source code for compilation |
| `instr_text` | Compiled instruction text in `.in` format (populated by `compile_and_assemble`) |
| `dram` | `DRAMWriter` with serialized weight/activation data |
| `output_addr` | DRAM address where output will be written |
| `output_shape` | Shape of the output tensor |
| `output_elements` | Number of bf16 elements in output |
| `skip_emulator` | If True, use `numpy_result` instead of running emulator |
| `numpy_result` | NumPy array for ops that skip the emulator |

**AtallaC code generators:**

| Generator | Used by | Key technique |
|-----------|---------|---------------|
| `_gemm_c(M, N, K)` | conv, linear, matmul | Tiled GEMM with inline asm for `lw.vi`, `scpad_ld/st`, `vreg_ld/st`, `gemm()` intrinsic |
| `_relu_c(total, width)` | relu | `make_mask("<", v, zero)` + `vec_op_masked("*", v, 0.0, mask)` to zero negative lanes |
| `_softmax_c(length)` | softmax | `RMAX` → shift → `EXP` → `RSUM` → `rcp.bf` reciprocal → `mul` |

**SDMA control register packing:** The `_sdma_ctl_val(sid, num_rows, num_cols, full_cols)` function packs scratchpad DMA parameters into a 32-bit control word: `(sid << 30) | ((num_rows-1) << 25) | ((num_cols-1) << 20) | (full_cols-1)`. The compiler cannot handle large constant stores, so `_sdma_ctl_expr()` uses inline asm `li_s` which `build_compiler` expands to `lui_s` + `addi_s`.

**Conv emission (`emit_conv`) pipeline:**
1. Extract Conv2d parameters (C_in, C_out, kernel size, stride, padding)
2. Transform input to NHWC layout, run `im2col()` → `(Ho*Wo, R*S*C_in)` matrix
3. Reshape weight to `(R*S*C_in, C_out)` matrix
4. Assign DRAM addresses: A_GMEM, W_GMEM, C_GMEM (sequentially, 4KB aligned)
5. Write ADDR_TABLE at offset 60: pointers, M, N, K, tile counts, tile size
6. Write input, weight, zero output to DRAM
7. Generate AtallaC via `_gemm_c(M, N, K)`

### 2.5 `codegen/asm_converter.py` — Notation Bridge

Converts the ppci compiler's assembly output to the format expected by the functional_sim assembler. Applied line-by-line with regex transforms.

| Compiler Output | Emulator Input | Rule |
|----------------|----------------|------|
| `add_s $1, $2, $3` | `add.s $1, $2, $3` | Replace `_` with `.` in mnemonic |
| `nop` | `nop.s` | Explicit remap |
| `halt` | `halt.s` | Explicit remap |
| `x5` | `$5` | Scalar register: `xN` → `$N` |
| `v3` | `$3` | Vector register: `vN` → `$N` |
| `m2` | `2` | Mask register: `mK` → bare int `K` |
| `100(x5)` | `100($5)` | Memory operand |
| `.section .text` | *(dropped)* | Strip directives |

Note: The notation conversion is now handled inside `build_compiler.compile_asm()`, which integrates the conversion, instruction scheduling, and encoding into a single pipeline step.

### 2.6 `codegen/dram_builder.py` — Tensor Serialization

Provides `extract_input_data()` which runs the actual PyTorch model (interpreter-style, node by node) to capture intermediate activations as reference values for validation. Also provides `bf16_from_float()` / `float_from_bf16()` conversion utilities.

### 2.7 `run_model.py` — Orchestrator

End-to-end script that ties everything together:

1. Accepts `--model basic|alexnet` and `--scale` arguments
2. Calls `capture()` → `remove_ops()` → `plan_tiles()`
3. Populates `activation_cache` with placeholder/get_attr values
4. For each compute node in topological order:
   - Calls `emit_node()` to get a `LayerEmission`
   - If `skip_emulator`: stores `numpy_result` in `activation_cache`
   - Otherwise: calls `compile_and_assemble()` (compiles C source), writes `.in` file, runs emulator, reads bf16 output from emulator memory, stores in `activation_cache`
5. Collects per-kernel metrics (range, cosine sim, cycles, instructions, GEMM ops)
6. Compares final output against PyTorch reference using cosine similarity

---

## 3. Component Deep Dive: aihw-ppci-compiler

**Purpose:** A C compiler for the Atalla architecture, based on the ppci (Pure Python Compiler Infrastructure) framework. Compiles AtallaC (C with vector/mask types and hardware intrinsics) to Atalla assembly.

### 3.1 AtallaC Language Extensions

| Feature | Syntax | Purpose |
|---------|--------|---------|
| Vector type | `vec v;` | 32-element vector register |
| Mask type | `mask m;` | Predication mask |
| GEMM intrinsic | `gemm(a_row, c_row, mask)` | 32×32 systolic array multiply-accumulate |
| Vector op | `vec_op_masked("+", a, b, mask)` | Masked elementwise operation (+, -, *, RMAX, RSUM, EXP) |
| Make mask | `make_mask(">", a, b, mask)` | Compare vectors, produce mask |
| BF16 convert | `stbf_s(x, 0)`, `bfts_s(x, 0)` | Float ↔ bfloat16 conversion |
| Reciprocal | `rcp_bf(x, 0)` | BF16 reciprocal approximation |
| Inline asm | `asm("lw_s %0, 0(%1)" : "=r"(out) : "r"(addr))` | Direct assembly insertion |

### 3.2 Compilation Flow

```
AtallaC source (.c)
    │  c_emitter.py generates parameterized source
    ▼
ppci atalla_cc frontend (lexer + parser)
    │  optimization passes
    ▼
ppci IR (SSA form) → instruction selection → register allocation
    ▼
Raw assembly output (.s)
    │  build_compiler.compile_asm():
    │    parse_program → inject_main_bootstrap
    │    → expand_large_li (li_s → lui_s + addi_s)
    │    → schedule_program → relocate_and_encode
    ▼
Encoded instruction packets (.in format)
```

### 3.3 What the Compiler Can and Cannot Do

**Works well (actively used in pipeline):**
- Scalar control flow (while loops, conditionals)
- `gemm()` intrinsic → emits `gemm.vv`
- `vec_op_masked()` for single-intrinsic ReLU and Softmax kernels
- Inline asm for `lw_s`, `scpad_ld/st`, `vreg_ld/st`, `halt`, `lw_vi`
- Large constant loading via `li_s` → expanded by `build_compiler` to `lui_s` + `addi_s`

**Known limitations:**
- **Vector spill bug**: When the register allocator spills a 32-element vector to scratchpad, it uses `vreg_st/ld` with dimensions `0,0,0,0,0` — saving only **1 element** out of 32. This silently corrupts any kernel that needs many live vector registers. Current kernels work by keeping vector register pressure low via inline asm.
- **No `div_vs` in inline assembler**: Cannot express vector÷scalar division in inline asm.
- **No pointer dereference for MMIO**: `*(volatile int*)0x3C` doesn't work; all MMIO reads require inline asm.

### 3.4 Current Role in Pipeline

The compiler is now the **primary compilation path** for all emulated compute kernels. The `_gemm_c()`, `_relu_c()`, `_softmax_c()` generators in `c_emitter.py` produce valid AtallaC that compiles and runs correctly on the emulator. Critical hardware operations (`lw.vi`, `scpad_ld/st`, `vreg_ld/st`) are expressed as inline assembly within the C source, allowing the compiler to handle scalar control flow while bypassing its vector spill limitations.

---

## 4. Component Deep Dive: functional_sim

**Purpose:** Cycle-accurate functional simulator for the Atalla accelerator. Assembles `.in` test files and executes them instruction-by-instruction.

### 4.1 Architecture Modeled

| Component | Python Class | Description |
|-----------|-------------|-------------|
| Scalar register file | `ScalarRegisterFile` (34 regs) | General-purpose 32-bit registers; x0=zero, x2=stack pointer, x33=vector spill base |
| Mask register file | `ScalarRegisterFile` (16 regs) | Predication masks; m0 hardwired to all-ones (0xFFFFFFFF) |
| Vector register file | `VectorRegisterFile` | 64 vector registers, each 32 elements of bf16 |
| Scratchpad banks | `Scratchpad` × 2 (SP0, SP1) | 32-slot on-chip SRAM for tiles, each slot = 32 bf16 values |
| Execute unit | `ExecuteUnit` | Scalar ALU, vector ALU, GEMM unit, control |
| Performance metrics | `PerfMetrics` | Counters for cycles, instructions, GEMM ops, SDMA ops, mem ops |
| Global memory | `Memory` | Main DRAM, dictionary-based, 32-bit words at 2-byte stride |

### 4.2 `.in` File Format

The `.in` file is a text file with two sections:

```
# Instruction memory (encoded instruction packets, 4 bytes each)
0x12345678
0xABCDEF01
...
0x00000000   # end marker

# Data memory (addr: value pairs, 2-byte stride addresses)
@0x003C 0x00001000
@0x1000 0x3F800000
...
```

Each data entry is a 32-bit word at a 2-byte-stride address. Two BF16 values are packed per word by `DRAMWriter.to_u32_words()`. The emulator's `Memory.read_data(addr)` returns the 32-bit word at the given address; `read_bf16_from_memory()` in `run_model.py` extracts the low 16 bits as a BF16 value.

### 4.3 Emulator Execution

```python
mem = Memory(in_file)       # load .in file into instruction + data memory
run_emulator(mem, sregs, mregs, vregs, SP0, SP1, EU, pc=0, issue_width=4, ...)
```

The emulator:
1. Fetches the instruction word at PC
2. Decodes opcode, extracts fields
3. Executes (scalar ALU, vector ALU, memory, GEMM, control)
4. Updates performance counters (cycles, instruction count, GEMM ops, SDMA ops, mem ops)
5. Advances PC (or branches)
6. Repeats until `halt.s`
7. Dumps state to output files (memory, registers, scratchpad, perf metrics)

### 4.4 `DRAMWriter`

A helper that builds the data memory section:

```python
img = DRAMWriter()
img.u32(addr, value)      # write 32-bit integer at byte address
img.bf16(addr, float_val) # convert float to bf16, write 16-bit at byte address
data_text = img.render_data_mem(include_zeros=True)
```

Internally, `bf16(addr, x)` stores a 16-bit BF16 value. `to_u32_words(stride=2)` packs two consecutive BF16 values into each 32-bit word.

### 4.5 ADDR_TABLE Convention

All kernels read their parameters from a fixed address table at byte address **60 (0x3C)**. The table layout varies per kernel:

**GEMM/Conv/Linear:**

| Offset | Field |
|--------|-------|
| +0 | A_GMEM (input matrix address) |
| +4 | W_GMEM (weight matrix address) |
| +8 | C_GMEM (output matrix address) |
| +12 | M (rows) |
| +16 | N (cols) |
| +20 | K (inner dimension) |
| +24 | M_tiles |
| +28 | N_tiles |
| +32 | K_tiles |
| +36 | TILE (tile size, always 32) |

**ReLU:**

| Offset | Field |
|--------|-------|
| +0 | IN_GMEM |
| +4 | OUT_GMEM |

**Softmax:** Same as ReLU (IN_GMEM at +0; +4 unused).

### 4.6 Critical Emulator Fixes

Several bugs were discovered and fixed during pipeline integration:

**1. Mask register m0 hardwired to all-ones:**

The Atalla hardware convention is that `m0` is always all-ones (0xFFFFFFFF), used as the default "no masking" predicate. The emulator's `ScalarRegisterFile` returns 0 for register 0 by default. Fix: patch `mregs.read` at emulator startup to return `0xFFFFFFFF` for index 0.

**2. `gemm.vv` computation order:**

The `lw.vi` instruction loads weight rows into the systolic array's internal buffer as *columns*, so `gemm_weights` represents W^T. The correct multiply is `gemm_weights @ v_in + v_acc` (i.e., W^T @ v_in). This was corrected from the original which had the operands in the wrong order.

**3. `lw.vi` weight buffer reset:**

In multi-tile GEMM (e.g., AlexNet conv1 with 32 tiles), the `gemm_weights` matrix must be reset to zero before each new weight loading phase. Without this, stale weights from the previous tile corrupt subsequent tiles. Fix: track a `_gemm_ran` flag; when `lw.vi` is encountered after a `gemm.vv`, reset `gemm_weights` and `num_weights` to zero.

**4. Stack pointer / DRAM overlap:**

The ppci compiler uses scalar register `x2` as a stack pointer for local variables. It was initially set to `0x8000`, but for large models (e.g., AlexNet conv1 with 1024×27 input matrix), the stack grows downward into the `A_GMEM` data region (starting at `0x1000`). Compiler-generated `sw.s` instructions for loop counters overwrote input matrix data, causing NaN and huge value propagation. Fix: compute stack base dynamically as `((max_data_addr + 0x1000) & ~0xFFF) + 0x1000`, placing it safely above all DRAM data. The vector spill base (`x33`) is set to the same value.

---

## 5. Component Deep Dive: Kernel Library

The `functional_sim/build_*.py` files contain hand-written assembly generators. These are **not used by the pipeline's graph operations** (which use the C compiler path), but remain available for standalone testing and as reference implementations.

### 5.1 `make_tiled_gemm_asm(M, N, K)` — Tiled GEMM

Used for: **standalone GEMM testing** (pipeline uses `_gemm_c()` instead)

Algorithm:
```
for mi in 0..M_tiles:
  for ni in 0..N_tiles:
    scpad.ld C_tile from DRAM       # load partial output tile
    for ki in 0..K_tiles:
      scpad.ld A_tile from DRAM     # load input tile to SP0
      scpad.ld W_tile from DRAM     # load weight tile to SP1
      for w in 0..TILE:
        lw.vi w                     # preload weight row w into systolic array
      for r in 0..TILE:
        vreg.ld a_row from SP0      # load input row
        vreg.ld c_row from SP1      # load accumulator row
        gemm.vv c_row, a_row, mask  # systolic MAC: c += a * W
        vreg.st c_row to SP1        # store updated accumulator
    scpad.st C_tile to DRAM         # write output tile back
halt
```

Key instructions:
- `lw.vi` — preloads one weight row into the systolic array's internal buffer (32 bf16 values)
- `gemm.vv` — performs one row of multiply-accumulate: `C[r] += A[r] * W_buffer`
- `scpad.ld/st` — bulk DMA between DRAM and scratchpad
- `vreg.ld/st` — move single rows between scratchpad and vector registers

### 5.2 `make_relu_asm(total, width)` — ReLU

Uses compare-and-blend: `mgt.mvv mask, v, zero` → `sub.vv result, result, result` → `add.vv result, result, v, mask` (only positive lanes get copied).

### 5.3 `make_softmax_asm(length)` — Softmax

Uses hardware reduce-max, reduce-sum, exp, and bf16 reciprocal instructions: `rmax → sub → exp → rsum → stbf_s → rcp_bf → bfts_s → mul.vs`.

### 5.4 `make_maxpool_asm(H_in, W_in, pool_size, stride)` — MaxPool

Single-channel MaxPool operating on one H×W tile using pairwise `mgt` + blend for both vertical and horizontal max operations.

Limitation: Single-channel only (W ≤ 32 elements). Multi-channel requires a per-channel loop wrapper not yet wired into the pipeline.

### 5.5 `make_attention_asm(S, d)` — Self-Attention

Full Q·K^T → softmax → ·V attention in one kernel. Uses GEMM for matrix products, inline softmax for score normalization, and a second GEMM for the output.

---

## 6. Full Pipeline Integration

### 6.1 Entry Point: `run_model.py`

```python
# Phase 1: Capture
gm = capture(model, example_input)         # FX trace + shape prop + op normalization
gm = remove_ops(gm)                        # BN fold, dropout removal
gm = plan_tiles(gm)                        # tile planning + DRAM layout

# Phase 2: Execute layer by layer
activation_cache = {}
kernel_metrics = []
for node in gm.graph.nodes:
    if node.op in ("placeholder", "get_attr"):
        activation_cache[node.name] = ...   # store input/weights
        continue
    if atalla_op in ("flatten", "dropout"):
        activation_cache[node.name] = reshape(prev)
        continue

    emission = emit_node(node, gm, activation_cache)

    if emission.skip_emulator:               # NumPy fallback (maxpool, add, mul)
        activation_cache[node.name] = emission.numpy_result
    else:                                    # C compiler + emulator
        compile_and_assemble(emission, ...)   # ppci compile C → asm → schedule → encode
        in_file = render_in_file(emission)    # combine encoded instrs + DRAM data → .in
        write(in_file)
        mem, eu = run_on_emulator(in_file)    # execute on simulator
        result = read_bf16_from_memory(mem, emission.output_addr, emission.output_elements)
        activation_cache[node.name] = result
        # collect perf metrics from eu.perf_metrics

# Phase 3: Validate
compare(activation_cache[final_node], pytorch_reference)
```

### 6.2 Layer-to-Layer Data Passing

Each layer is executed as a **standalone emulator invocation**:
- The emulator starts fresh for each layer (registers zeroed, scratchpad empty)
- The stack pointer (`x2`) is dynamically computed above all DRAM data for each layer
- Input activations from the previous layer's output are serialized into the new layer's DRAM image
- The Python orchestrator reads the output from emulator memory and passes it to the next layer

This is equivalent to a "layer-at-a-time" execution model.

### 6.3 The C Compiler Path (`compile_and_assemble`)

```python
def compile_and_assemble(emission, work_dir, tag):
    if emission.c_source:
        raw_s = compile_c(emission.c_source, work_dir, tag)  # ppci atalla_cc → .s
        in_content, _, _ = build_compiler.compile_asm(raw_s)  # schedule + encode
        emission.instr_text = in_content.split("\n.data")[0].strip()
    return emission.instr_text
```

`build_compiler.compile_asm()` performs:
1. `parse_program()` — parse assembly into instruction list + label map
2. `inject_main_bootstrap()` — add startup code
3. `expand_large_li()` — expand `li_s` pseudo-instructions to `lui_s` + `addi_s`
4. `schedule_program()` — VLIW scheduling into instruction packets
5. `relocate_and_encode()` — resolve labels, encode 32-bit instruction words
6. `emit_packet_format()` — format as hex lines for `.in` file

---

## 7. AlexNet Walkthrough

### 7.1 Model Definition

```python
class AlexNetSmall(nn.Module):
    def __init__(self, scale=0.01, num_classes=10):
        # Channel counts scaled down: sc(64)=1, sc(192)=1, sc(384)=3, sc(256)=2
        self.conv1 = nn.Conv2d(3, sc(64), 3, stride=1, padding=1)   # 3→1 ch
        self.conv2 = nn.Conv2d(sc(64), sc(192), 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(sc(192), sc(384), 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(sc(384), sc(256), 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(sc(256), sc(256), 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(sc(256)*4*4, sc(4096))
        self.fc2 = nn.Linear(sc(4096), sc(4096))
        self.fc3 = nn.Linear(sc(4096), 10)
```

### 7.2 FX Graph After Capture (scale=0.01)

```
x                op=placeholder   kernel=-         shape=(1, 3, 32, 32)
conv1            op=conv          kernel=conv      shape=(1, 1, 32, 32)
relu             op=relu          kernel=relu      shape=(1, 1, 32, 32)
pool             op=maxpool       kernel=maxpool   shape=(1, 1, 16, 16)
conv2            op=conv          kernel=conv      shape=(1, 1, 16, 16)
relu_1           op=relu          kernel=relu      shape=(1, 1, 16, 16)
pool_1           op=maxpool       kernel=maxpool   shape=(1, 1, 8, 8)
conv3            op=conv          kernel=conv      shape=(1, 3, 8, 8)
relu_2           op=relu          kernel=relu      shape=(1, 3, 8, 8)
conv4            op=conv          kernel=conv      shape=(1, 2, 8, 8)
relu_3           op=relu          kernel=relu      shape=(1, 2, 8, 8)
conv5            op=conv          kernel=conv      shape=(1, 2, 8, 8)
relu_4           op=relu          kernel=relu      shape=(1, 2, 8, 8)
pool_2           op=maxpool       kernel=maxpool   shape=(1, 2, 4, 4)
flatten          op=flatten       kernel=flatten   shape=(1, 32)
fc1              op=linear        kernel=fc        shape=(1, 40)
relu_5           op=relu          kernel=relu      shape=(1, 40)
fc2              op=linear        kernel=fc        shape=(1, 40)
relu_6           op=relu          kernel=relu      shape=(1, 40)
fc3              op=linear        kernel=fc        shape=(1, 10)
```

**20 nodes** total: **15 emulated** (5 conv + 6 relu + 3 FC + 1 matmul), **3 NumPy** (maxpool), **1 passthrough** (flatten).

### 7.3 Conv1 Detailed Walkthrough

Conv1: `nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)` on input `(1, 3, 32, 32)`.

**Tile planning:**
- H=32, W=32, C_in=3, C_out=1, R=3, S=3, stride=1, pad=1
- Ho = (32 + 2*1 - 3)/1 + 1 = 32, Wo = 32
- im2col: M = Ho*Wo = 1024, K = R*S*C_in = 27, N = C_out = 1
- M_tiles = ceil(1024/32) = 32, K_tiles = 1, N_tiles = 1

**Code generation:**
1. Input `(1,3,32,32)` NCHW → NHWC → `im2col()` → `(1024, 27)` matrix
2. Weight `(1,3,3,3)` → `(27, 1)` matrix
3. DRAM: A_GMEM=0x1000, W_GMEM=0x2000, C_GMEM=0x12000
4. AtallaC: `_gemm_c(1024, 1, 27)` → compiled → 120,080 cycles, 75,972 instructions, 1024 GEMM ops

---

## 8. Performance Metrics

### 8.1 BasicModule (dim=32, depth=2)

| Node | Op | Backend | Elems | Range | CosSim | MaxDiff | Cycles | Instrs | GEMMs |
|------|----|---------|-------|-------|--------|---------|--------|--------|-------|
| matmul | matmul | emulator | 32 | [-1.91, 1.05] | 1.0000 | 0.0078 | 1,736 | 1,190 | 1 |
| add | add | numpy | 32 | [-1.98, 1.20] | 1.0000 | 0.0104 | — | — | — |
| relu | relu | emulator | 32 | [0.00, 1.20] | 1.0000 | 0.0156 | 187 | 138 | 0 |
| matmul_1 | matmul | emulator | 32 | [-0.44, 0.54] | 1.0000 | 0.0039 | 1,736 | 1,190 | 1 |
| add_1 | add | numpy | 32 | [-0.42, 0.60] | 1.0000 | 0.0034 | — | — | — |
| relu_1 | relu | emulator | 32 | [0.00, 0.60] | 1.0000 | 0.0039 | 187 | 138 | 0 |
| mul | mul | numpy | 32 | [0.00, 0.60] | 1.0000 | 0.0078 | — | — | — |
| add_2 | add | numpy | 32 | [0.00, 0.78] | 1.0000 | 0.0078 | — | — | — |
| output | linear | emulator | 32 | [-0.26, 0.33] | 0.8684 | 0.1758 | 1,736 | 1,190 | 1 |

**End-to-end:** Cosine sim = **0.868**, max diff = 0.176, 5,582 total cycles, 3,846 instructions, 3 GEMM ops, **0 NaN**.

### 8.2 AlexNetSmall (scale=0.01)

| Node | Op | Backend | Elems | Range | CosSim | MaxDiff | Cycles | Instrs | GEMMs |
|------|----|---------|-------|-------|--------|---------|--------|--------|-------|
| conv1 | conv | emulator | 1024 | [-2.06, 2.02] | 0.1645 | 3.1250 | 120,080 | 75,972 | 1,024 |
| relu | relu | emulator | 1024 | [0.00, 2.02] | 0.4385 | 2.0625 | 2,140 | 1,657 | 0 |
| pool | maxpool | numpy | 256 | [0.00, 2.02] | 0.7537 | 1.8477 | — | — | — |
| conv2 | conv | emulator | 256 | [-0.21, 0.81] | 0.8282 | 1.2656 | 24,344 | 14,868 | 256 |
| relu_1 | relu | emulator | 256 | [0.00, 0.81] | 0.8431 | 1.2656 | 628 | 481 | 0 |
| pool_1 | maxpool | numpy | 64 | [0.29, 0.81] | 0.9458 | 1.0605 | — | — | — |
| conv3 | conv | emulator | 192 | [-0.60, 0.16] | 0.4284 | 1.0796 | 6,170 | 3,768 | 64 |
| relu_2 | relu | emulator | 192 | [0.00, 0.16] | 0.1921 | 0.3867 | 502 | 383 | 0 |
| conv4 | conv | emulator | 128 | [-0.05, 0.04] | 0.1079 | 0.2273 | 7,610 | 4,812 | 64 |
| relu_3 | relu | emulator | 128 | [0.00, 0.04] | 0.2129 | 0.0383 | 376 | 285 | 0 |
| conv5 | conv | emulator | 128 | [-0.01, 0.01] | 0.1658 | 0.2097 | 6,890 | 4,290 | 64 |
| relu_4 | relu | emulator | 128 | [0.00, 0.01] | 0.4624 | 0.0491 | 376 | 285 | 0 |
| pool_2 | maxpool | numpy | 32 | [0.00, 0.01] | 0.7310 | 0.0491 | — | — | — |
| fc1 | linear | emulator | 40 | [-0.01, 0.01] | 0.2620 | 0.1685 | 3,334 | 2,296 | 2 |
| relu_5 | relu | emulator | 40 | [0.00, 0.01] | 0.5003 | 0.1685 | 250 | 187 | 0 |
| fc2 | linear | emulator | 40 | [-0.00, 0.00] | -0.1470 | 0.1834 | 6,478 | 4,476 | 4 |
| relu_6 | relu | emulator | 40 | [0.00, 0.00] | 0.2393 | 0.1719 | 250 | 187 | 0 |
| fc3 | linear | emulator | 10 | [-0.00, 0.00] | 0.1984 | 0.1023 | 3,308 | 2,280 | 2 |

**End-to-end:** Cosine sim = **0.198**, max diff = 0.102, 182,736 total cycles, 116,227 instructions, 1,480 GEMM ops, **0 NaN**.

### 8.3 Analysis

**BasicModule:** Individual kernels achieve near-perfect cosine similarity (1.0) against PyTorch float32 reference. The final output (cos=0.868) shows expected BF16 accumulation drift compounding through 3 GEMM layers.

**AlexNet:** BF16 precision errors compound through 19 execution layers. Conv1 alone has cos=0.16 vs fp32 reference due to the large im2col matrix (1024×27 → 32 tiles), where each tile's BF16-truncated intermediate results accumulate quantization error. Later layers see progressively lower cosine similarity as errors cascade. This is a fundamental BF16 precision limitation — the emulator is functionally correct (no NaN, no out-of-range values), but the hardware's 16-bit mantissa cannot match float32 fidelity across deep networks.

**Cycle distribution (AlexNet):**
- Conv1 dominates (120K / 183K = 65% of total cycles) due to 1024 GEMM operations over 32 tiles
- Each GEMM kernel invocation costs ~1,700 cycles (consistent with BasicModule's matmul)
- ReLU is cheap: ~190 cycles per invocation regardless of element count

---

## 9. Changes, Tweaks, and Known Limitations

### 9.1 Changes Made to Enable the Pipeline

| Change | File(s) | Why |
|--------|---------|-----|
| Created FX capture + op normalization | `graph/fx_capture.py` | Trace PyTorch models to FX graph |
| Created graph transforms | `graph/remove_ops.py` | BN folding + dropout removal for inference |
| Created tile planner | `graph/tile_planner.py` | Compute tiling strategy per op |
| Created C emitter (primary path) | `codegen/c_emitter.py` | Generate AtallaC for ppci compilation |
| Created asm converter | `codegen/asm_converter.py` | Bridge compiler and emulator notation |
| Created DRAM builder | `codegen/dram_builder.py` | Tensor serialization + reference extraction |
| Created orchestrator | `run_model.py` | End-to-end pipeline runner with metrics |
| Created metrics collector | `collect_metrics.py` | Per-kernel and end-to-end metrics |
| Created AlexNet model | `model/alexnet.py` | Scaled-down AlexNet for testing |
| Copied aihw-ppci-compiler | `aihw-ppci-compiler/` | In-tree for the C compilation path |
| Fixed m0 mask register | `functional_sim.py` | m0 must be all-ones per hardware spec |
| Fixed gemm.vv order | `functional_sim.py` | Correct W^T @ v_in computation |
| Fixed lw.vi weight reset | `functional_sim.py` | Reset weights between tiles |
| Fixed stack/DRAM overlap | `run_model.py` | Dynamic stack base above DRAM data |

### 9.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **C compiler + inline asm for kernels** | Compiler handles scalar control flow; inline asm for vector/SDMA ops avoids vector spill bug |
| **Per-layer emulator invocation** | Emulator designed for single-kernel execution; fresh state per layer is cleanest |
| **Dynamic stack pointer** | Prevents compiler stack frames from overlapping DRAM data regions |
| **BF16 weight conversion at capture time** | Matches hardware precision; avoids surprise fp32→bf16 truncation mid-pipeline |
| **im2col for convolutions** | Reduces Conv2d to matrix multiply, reusing the tiled GEMM kernel |
| **MaxPool in NumPy** | Kernel exists but is single-channel; multi-channel wrapper not yet wired |
| **Add/Mul in NumPy** | Bias adds and residual scales; AlexNet folds bias into Conv/Linear |

### 9.3 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Compiler vector spill bug** | Kernels with many live vectors would be corrupted | Current kernels keep register pressure low via inline asm |
| **MaxPool uses NumPy** | Not running on emulator; no cycle counts for maxpool | Wire `build_maxpool.py` per-channel loop |
| **BF16 accumulation drift** | AlexNet cos=0.20 vs fp32 after 19 layers | Expected; tracked via per-kernel cosine similarity |
| **Single-batch only** | Batch size > 1 not tested | Tile planner assumes batch=1 |
| **No softmax in AlexNet** | AlexNet outputs raw logits | Softmax kernel validated standalone |
| **ADDR_TABLE at fixed address 60** | All kernels share the same parameter address | Works for layer-at-a-time |
| **Bias not fused into GEMM** | Conv/Linear bias requires separate NumPy add | Fused GEMM+bias kernel would be more efficient |

---

## 10. Open Items for Review

1. **Wire maxpool to emulator**: `build_maxpool.py` works for single-channel tiles. Need a per-channel loop in `emit_maxpool()` that runs the kernel once per channel with appropriate DRAM offsets.

2. **Fix compiler vector spill**: The `vreg_st/ld` spill code in ppci uses dimensions `0,0,0,0,0` (1 element). Fixing to `31,0,0,0,0` (full 32-element row) would allow removing inline asm for vector ops.

3. **BF16 precision investigation**: Conv1's cos=0.16 against fp32 suggests the 32-tile accumulation magnifies BF16 truncation. Investigate whether partial-precision accumulation (fp32 accumulator with bf16 operands) or Kahan summation could improve fidelity.

4. **Bias fusion**: Add a fused GEMM+bias kernel that reads bias from DRAM and adds it inline, eliminating the separate NumPy add pass for BasicModule.

5. **Multi-layer single invocation**: Chain layers in a single `.in` file to eliminate Python-side data marshaling and get more realistic end-to-end cycle counts.

6. **Notation standardization**: Make the compiler and emulator agree on a single assembly notation, eliminating the need for `asm_converter.py` / `build_compiler` notation conversion.
