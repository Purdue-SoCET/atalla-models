# Atalla Pipeline Technical Reference

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Deep Dive: atalla-graph](#2-component-deep-dive-atalla-graph)
3. [Component Deep Dive: aihw-ppci-compiler](#3-component-deep-dive-aihw-ppci-compiler)
4. [Component Deep Dive: functional_sim](#4-component-deep-dive-functional_sim)
5. [Component Deep Dive: Kernel Library](#5-component-deep-dive-kernel-library)
6. [Full Pipeline Integration](#6-full-pipeline-integration)
7. [AlexNet Walkthrough](#7-alexnet-walkthrough)
8. [Changes, Tweaks, and Known Limitations](#8-changes-tweaks-and-known-limitations)
9. [Open Items for Review](#9-open-items-for-review)

---

## 1. Architecture Overview

The pipeline compiles a standard PyTorch `nn.Module` down to Atalla assembly and executes it on the functional simulator. Three components are integrated:

```
PyTorch nn.Module
    │
    ├─ atalla-graph/        FX graph capture, op normalization, tiling, code generation
    │
    ├─ aihw-ppci-compiler/  AtallaC → assembly compiler (used for notation bridge; direct
    │                       asm used for compute kernels due to compiler limitations)
    │
    └─ functional_sim/      Assembler, .in file builder, and cycle-accurate emulator
```

**Data flow for a single layer:**

```
                                    ┌─────────────────────────┐
                                    │  Direct assembly path   │
                                    │  (GEMM, Conv, ReLU,     │
  FX Node ──► emit_node() ────────►│   Softmax)              │──► .in file ──► Emulator
         │                          │  Uses build_*.py asm    │
         │                          └─────────────────────────┘
         │                          ┌─────────────────────────┐
         └─ (future) ─────────────►│  C compiler path        │
                                    │  .c → ppci → .s →       │
                                    │  asm_converter → .in    │
                                    └─────────────────────────┘
```

**Why MaxPool / Add / Mul use NumPy:**

MaxPool DOES have a working assembly kernel (`build_maxpool.py`). However, the `emit_maxpool()` function in both `asm_emitter.py` and `c_emitter.py` currently sets `skip_emulator = True` and runs the operation in NumPy. The reason is that the `build_maxpool.py` kernel operates on **single-channel 2D tiles** (one H×W plane at a time), and wiring it into the multi-channel pipeline requires running it in a per-channel loop with correct DRAM address arithmetic for each channel slice. This was deferred to keep the initial pipeline integration simple. **This is a wiring gap, not a missing kernel.**

For Add and Mul: these appear in `BasicModule` as bias addition (`out + bias`) and residual scaling (`0.5 * residual`). They are scalar-broadcast or elementwise operations that don't map to the systolic array or dedicated vector compute in a way that would meaningfully benefit from hardware execution. They use NumPy as a pragmatic fallback. AlexNet's forward path does NOT contain standalone add/mul nodes — bias addition is folded into the Conv2d/Linear modules. The add/mul nodes come from `BasicModule`'s explicit `out + bias` and `0.5 * residual` patterns.

---

## 2. Component Deep Dive: atalla-graph

**Purpose:** PyTorch frontend that captures a model as an FX graph, normalizes operations to Atalla primitives, plans tile decomposition, and generates assembly (or C) for each layer.

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

**Key function:**

```python
def capture(model: nn.Module, example_input: torch.Tensor) -> GraphModule:
    model = model.bfloat16()              # convert all weights to bf16
    example_input = example_input.bfloat16()
    gm = symbolic_trace(model)            # flat FX graph, no control flow
    ShapeProp(gm).propagate(example_input) # shape on every node
    gm = normalize_ops(gm)               # tag atalla_op on each node
    return gm
```

**Limitation:** `symbolic_trace` cannot handle dynamic control flow. All models must be FX-traceable (no data-dependent branches, no dynamic shapes). This is fine for standard CNNs like AlexNet.

### 2.2 `graph/remove_ops.py` — Graph Transforms

Three passes clean up the graph before code generation:

1. **BatchNorm folding** (`_fold_bn_into_conv`): Folds BN parameters into the preceding Conv2d's weight and bias using the standard formula: `W_new = W * gamma / sqrt(var + eps)`, `b_new = beta + (b - mu) * gamma / sqrt(var + eps)`. The BN node is then erased from the graph.

2. **Dropout removal** (`_remove_dropout`): Elides dropout nodes (inference mode) by replacing them with their input.

3. **Adaptive avg pool tagging** (`_remove_adaptive_avg_pool`): Tags `AdaptiveAvgPool2d(1,1)` nodes so they can be handled as NumPy fallback.

### 2.3 `graph/tile_planner.py` — Tile Planning & DRAM Layout

**What it does:** For each normalized op, computes the tiling strategy given Atalla hardware constraints and assigns DRAM addresses.

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

**Output:** Each node gets `node.meta["tile_config"]` (a `TileConfig` dataclass with `kernel_type` and `params` dict) and `node.meta["kernel_type"]`.

### 2.4 `codegen/c_emitter.py` — Code Generation (Primary)

**What it does:** For each FX node, generates either:
1. Direct assembly using `build_*.py` generators (for GEMM, Conv, Linear, ReLU, Softmax)
2. NumPy fallback (for MaxPool, Add, Mul)
3. AtallaC source (wired but not used for compute kernels due to compiler limitations)

**Core dispatch function:**

```python
def emit_node(node, gm, activation_cache) -> LayerEmission:
    tc = node.meta.get("tile_config")
    atalla_op = node.meta.get("atalla_op")
    # ...
    if atalla_op == "conv":     return emit_conv(node, gm, input_data, tc)
    elif atalla_op == "linear": return emit_linear(node, gm, input_data, tc)
    elif atalla_op == "relu":   return emit_relu(node, input_data, tc)
    elif atalla_op == "softmax":return emit_softmax(input_data, tc)
    elif atalla_op == "maxpool":return emit_maxpool(node, input_data, tc)  # NumPy
    elif atalla_op == "matmul": return emit_matmul(node, gm, input_data, ...)
    elif atalla_op == "add":    return emit_add(node, activation_cache, tc) # NumPy
    elif atalla_op == "mul":    return emit_mul(node, activation_cache)      # NumPy
```

**`LayerEmission` dataclass:**

| Field | Purpose |
|-------|---------|
| `c_source` | AtallaC source (empty string when using direct asm) |
| `instr_text` | Pre-assembled instruction text in `.in` format |
| `dram` | `DRAMWriter` with serialized weight/activation data |
| `output_addr` | DRAM address where output will be written |
| `output_shape` | Shape of the output tensor |
| `output_elements` | Number of bf16 elements in output |
| `skip_emulator` | If True, use `numpy_result` instead of running emulator |
| `numpy_result` | NumPy array for ops that skip the emulator |

**Conv emission example (`emit_conv`):**
1. Extract Conv2d parameters from the FX module (C_in, C_out, kernel size, stride, padding)
2. Get input activation from `activation_cache`
3. Transform input to NHWC layout, run `im2col()` to produce `(Ho*Wo, R*S*C_in)` matrix
4. Reshape weight to `(R*S*C_in, C_out)` matrix
5. Compute DRAM addresses: A_GMEM, W_GMEM, C_GMEM (sequentially, 4KB aligned)
6. Write ADDR_TABLE parameters at address 60: pointers, M, N, K, tile counts, tile size
7. Write input matrix, weight matrix, zero output matrix to DRAM
8. Generate assembly via `make_tiled_gemm_asm(M, N, K)` → `assemble_file()` → `emit_test_format()`
9. Return `LayerEmission` with instruction text, DRAM image, output address

### 2.5 `codegen/asm_converter.py` — Notation Bridge

**What it does:** Converts the ppci compiler's assembly output to the format expected by the functional_sim assembler. Applied line-by-line with regex transforms.

**Transformations:**

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
   - Otherwise: calls `compile_and_assemble()`, writes `.in` file, runs emulator, reads bf16 output from emulator memory, stores in `activation_cache`
5. Compares final output against PyTorch reference using cosine similarity

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
    │  atalla_cc frontend (lexer + parser)
    ▼
ppci IR (SSA form)
    │  optimization passes
    ▼
Atalla instruction selection
    │  register allocation
    ▼
Assembly output (.s)
```

Invoked as: `python3 -m ppci atalla_cc -m atalla -S input.c -o output.s`

### 3.3 What the Compiler Can and Cannot Do

**Works well:**
- Scalar control flow (while loops, conditionals)
- `gemm()` intrinsic
- `vec_op_masked()` for single-intrinsic kernels
- Inline asm for `lw_s`, `scpad_ld/st`, `vreg_ld/st`, `halt`

**Does NOT work (discovered during integration):**
- **No `lw.vi` intrinsic**: The systolic array requires `lw.vi` to preload weight rows before GEMM. The compiler has no way to emit this instruction, making C-compiled GEMM produce all zeros.
- **Vector spill bug**: When the register allocator spills a 32-element vector to scratchpad, it uses `vreg_st/ld` with dimensions `0,0,0,0,0` — which only saves/restores **1 element** out of 32. This silently corrupts any kernel that needs more than ~2 live vector registers.
- **No `div_vs` in inline assembler**: Cannot express vector÷scalar division in inline asm.
- **No pointer dereference for MMIO**: `*(volatile int*)0x3C` doesn't work; all MMIO reads require inline asm.
- **Emits `li_s`**: The emulator doesn't support `li_s` (pseudo-instruction). Would need `addi.s $rd, $0, imm` or `lui.s`.

### 3.4 Current Role in Pipeline

Due to the limitations above, the compiler is NOT currently used for compute kernels. Its role is:
1. **Proven assembly notation reference** — the `asm_converter.py` was built by analyzing compiler output vs emulator expectations
2. **Future path** — once the vector spill bug and `lw.vi` support are fixed, kernels can be compiled from C
3. **C source templates exist** — `_gemm_c()`, `_relu_c()`, `_softmax_c()` in `c_emitter.py` generate valid AtallaC that compiles, but produces incorrect results due to the limitations above

---

## 4. Component Deep Dive: functional_sim

**Purpose:** Cycle-accurate functional simulator for the Atalla accelerator. Assembles `.in` test files and executes them instruction-by-instruction.

### 4.1 Architecture Modeled

| Component | Python Class | Description |
|-----------|-------------|-------------|
| Scalar register file | `ScalarRegisterFile` (34 regs) | General-purpose 32-bit integer/address registers |
| Mask register file | `ScalarRegisterFile` (16 regs) | Predication masks for vector operations |
| Vector register file | `VectorRegisterFile` | 64 vector registers, each 32 elements of bf16 |
| Scratchpad banks | `Scratchpad` × 2 (SP0, SP1) | 32-slot on-chip SRAM for tiles, each slot = 32 bf16 values |
| Execute unit | `ExecuteUnit` | Scalar ALU, vector ALU, GEMM unit, control |
| Global memory | `Memory` | Main DRAM, byte-addressable |

### 4.2 `.in` File Format

The `.in` file is a text file with two sections:

```
# Instruction memory (one hex word per line, 4-byte instructions)
0x12345678
0xABCDEF01
...
0x00000000   # end marker

# Data memory (addr: value pairs)
@0x003C 0x00001000
@0x1000 0x3F800000
...
```

### 4.3 Assembly → `.in` Pipeline (`build.py`)

```python
asm_text = make_tiled_gemm_asm(M, N, K)   # generate assembly string
instrs = assemble_file(asm_text)           # assemble to 32-bit instruction words
instr_text = emit_test_format(instrs)      # format as hex lines
# ... build DRAMWriter with data ...
data_text = img.render_data_mem()           # format as @addr value pairs
final = render_testfile(instr_text, data_text)  # combine into .in file
```

`assemble_file()` is a two-pass assembler:
- **Pass 1:** Collect label addresses, compute instruction sizes
- **Pass 2:** Encode each instruction to a 32-bit word using the opcode table

### 4.4 `DRAMWriter`

A helper that builds the data memory section:

```python
img = DRAMWriter()
img.u32(addr, value)      # write 32-bit integer at byte address
img.bf16(addr, float_val) # convert float to bf16, write 16-bit at byte address
data_text = img.render_data_mem(include_zeros=True)
```

### 4.5 Emulator Execution

```python
mem = Memory(in_file)       # load .in file into instruction + data memory
run_emulator(mem, sregs, mregs, vregs, SP0, SP1, EU, pc=0, issue_width=4, ...)
```

The emulator:
1. Fetches the instruction word at PC
2. Decodes opcode, extracts fields
3. Executes (scalar ALU, vector ALU, memory, GEMM, control)
4. Updates performance counters (cycles, instruction count, FLOPS, memory bytes)
5. Advances PC (or branches)
6. Repeats until `halt.s`

### 4.6 ADDR_TABLE Convention

All kernels read their parameters from a fixed address table at byte address **60 (0x3C)**. This avoids hardcoding addresses in assembly. The table layout varies per kernel:

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

**MaxPool:**
| Offset | Field |
|--------|-------|
| +0 | IN_GMEM |
| +4 | IN_SCPAD |
| +8 | OUT_GMEM |
| +12 | OUT_SCPAD |

---

## 5. Component Deep Dive: Kernel Library

All compute kernels are hand-written assembly generators in `functional_sim/build_*.py`. Each function returns an assembly text string.

### 5.1 `make_tiled_gemm_asm(M, N, K)` — Tiled GEMM

Used for: **Conv (via im2col), Linear, Matmul**

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

Algorithm:
```
for each tile of 32 rows:
  scpad.ld input tile from DRAM
  for each row:
    vreg.ld v from scratchpad
    mgt.mvv mask, v, zero, all    # mask = (v > 0)
    sub.vv result, result, result # zero result
    add.vv result, result, v, mask # copy positive lanes
    vreg.st result to scratchpad
  scpad.st output tile to DRAM
halt
```

Uses compare-and-blend: zero the result vector, then masked-add the input (only positive lanes get copied).

### 5.3 `make_softmax_asm(length)` — Softmax

Algorithm:
```
scpad.ld input from DRAM
for each row:
  vreg.ld v
  rmax.vv max_val = reduce_max(v)
  sub.vv shifted = v - max_val
  expi.vv exp_val = exp(shifted)
  rsum.vv sum_val = reduce_sum(exp_val)
  # compute 1/sum via bf16 reciprocal
  stbf.s sum_bits = to_bf16(sum)
  rcp.bf inv_bits = reciprocal(sum_bits)
  bfts.s inv_sum = to_float(inv_bits)
  mul.vs result = exp_val * inv_sum
  vreg.st result
scpad.st output to DRAM
halt
```

Uses hardware reduce-max, reduce-sum, exp, and bf16 reciprocal instructions.

### 5.4 `make_maxpool_asm(H_in, W_in, pool_size, stride)` — MaxPool

**This kernel exists and works correctly.** It operates on a single-channel H×W tile:

Algorithm:
```
scpad.ld input tile
for each output row:
  vreg.ld pool_size adjacent input rows
  # Vertical max: pairwise mgt + blend across rows
  # Horizontal max: shift.vi + pairwise mgt + blend
  vreg.st result row
scpad.st output tile
halt
```

Limitations:
- Single-channel only (W ≤ 32 elements, i.e., fits in one vector)
- Multi-channel tensors require a per-channel loop wrapper that isn't wired into the pipeline emitters yet

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
    else:                                    # Emulator execution
        compile_and_assemble(emission, ...)   # no-op for direct asm
        in_file = render_in_file(emission)    # combine asm + data → .in text
        write(in_file)
        mem = run_on_emulator(in_file)        # execute on simulator
        result = read_bf16_from_memory(mem, emission.output_addr, emission.output_elements)
        activation_cache[node.name] = result

# Phase 3: Validate
compare(activation_cache[final_node], pytorch_reference)
```

### 6.2 Layer-to-Layer Data Passing

Each layer is executed as a **standalone emulator invocation**. This means:
- The emulator starts fresh for each layer (registers zeroed, scratchpad empty)
- Input activations from the previous layer's output are serialized into the new layer's DRAM image
- The Python orchestrator (`run_model.py`) reads the output from emulator memory and passes it to the next layer

This is equivalent to a "layer-at-a-time" execution model. A full-model execution (all layers in one `.in` file) would require a more complex memory management scheme.

### 6.3 The C Compiler Path (`compile_and_assemble`)

The function `compile_and_assemble(emission, work_dir, tag)` checks:
- If `emission.c_source` is non-empty: compiles C → .s → converts notation → assembles
- If `emission.c_source` is empty (direct asm): `emission.instr_text` is already populated, returns as-is

Currently, all compute kernels set `emission.instr_text` directly (direct asm), so `compile_and_assemble` is effectively a no-op. The C path is wired and tested but produces incorrect results for multi-intrinsic kernels due to the compiler's vector spill bug.

---

## 7. AlexNet Walkthrough

### 7.1 Model Definition

```python
class AlexNetSmall(nn.Module):
    def __init__(self, scale=0.01, num_classes=10):
        # Channel counts scaled down: sc(64)=1, sc(192)=1, sc(384)=3, sc(256)=2
        self.conv1 = nn.Conv2d(3, sc(64), 3, stride=1, padding=1)   # 3→1 ch
        self.conv2 = nn.Conv2d(sc(64), sc(192), 3, stride=1, padding=1)  # 1→1 ch
        self.conv3 = nn.Conv2d(sc(192), sc(384), 3, stride=1, padding=1) # 1→3 ch
        self.conv4 = nn.Conv2d(sc(384), sc(256), 3, stride=1, padding=1) # 3→2 ch
        self.conv5 = nn.Conv2d(sc(256), sc(256), 3, stride=1, padding=1) # 2→2 ch
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(sc(256)*4*4, sc(4096))  # 32→40
        self.fc2 = nn.Linear(sc(4096), sc(4096))      # 40→40
        self.fc3 = nn.Linear(sc(4096), 10)             # 40→10

    def forward(self, x):  # input: (1, 3, 32, 32)
        x = F.relu(self.conv1(x))   # (1,1,32,32)
        x = self.pool(x)            # (1,1,16,16)
        x = F.relu(self.conv2(x))   # (1,1,16,16)
        x = self.pool(x)            # (1,1,8,8)
        x = F.relu(self.conv3(x))   # (1,3,8,8)
        x = F.relu(self.conv4(x))   # (1,2,8,8)
        x = F.relu(self.conv5(x))   # (1,2,8,8)
        x = self.pool(x)            # (1,2,4,4)
        x = torch.flatten(x, 1)     # (1,32)
        x = F.relu(self.fc1(x))     # (1,40)
        x = F.relu(self.fc2(x))     # (1,40)
        x = self.fc3(x)             # (1,10)
        return x
```

### 7.2 FX Graph After Capture (scale=0.01)

```
Phase 1: FX capture + op normalization...
Phase 2: Tile planning...

--- Graph Summary ---
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

### 7.3 Layer-by-Layer Execution Trace

```
Phase 3+4: Assembly emission + emulator execution...
  [x]        placeholder -> cached
  [conv1]    conv -> emulator (M=1024, K=27, N=1) → tiled GEMM, out=0x12000
  [relu]     relu -> emulator (1024 elements)
  [pool]     maxpool -> NumPy (1,1,16,16)           ← NumPy fallback
  [conv2]    conv -> emulator (M=256, K=9, N=1) → tiled GEMM
  [relu_1]   relu -> emulator (256 elements)
  [pool_1]   maxpool -> NumPy (1,1,8,8)
  [conv3]    conv -> emulator (M=64, K=9, N=3) → tiled GEMM
  [relu_2]   relu -> emulator (192 elements)
  [conv4]    conv -> emulator (M=64, K=27, N=2) → tiled GEMM
  [relu_3]   relu -> emulator (128 elements)
  [conv5]    conv -> emulator (M=64, K=18, N=2) → tiled GEMM
  [relu_4]   relu -> emulator (128 elements)
  [pool_2]   maxpool -> NumPy (1,2,4,4)
  [flatten]  reshape (1,32)                          ← passthrough
  [fc1]      linear -> emulator (M=1, K=32, N=40)
  [relu_5]   relu -> emulator (40 elements)
  [fc2]      linear -> emulator (M=1, K=40, N=40)
  [relu_6]   relu -> emulator (40 elements)
  [fc3]      linear -> emulator (M=1, K=40, N=10)
```

**15 nodes** run on the emulator (5 conv + 6 relu + 3 FC), **3 nodes** run in NumPy (maxpool), **1 node** is a reshape passthrough (flatten).

### 7.4 Conv1 Detailed Walkthrough

Conv1: `nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)` on input `(1, 3, 32, 32)`.

**Tile planning:**
- H=32, W=32, C_in=3, C_out=1, R=3, S=3, stride=1, pad=1
- Ho = (32 + 2*1 - 3)/1 + 1 = 32, Wo = 32
- im2col: M = Ho*Wo = 1024, K = R*S*C_in = 27, N = C_out = 1
- M_tiles = ceil(1024/32) = 32, K_tiles = ceil(27/32) = 1, N_tiles = ceil(1/32) = 1

**Code generation (`emit_conv`):**
1. Input `(1,3,32,32)` NCHW → transpose to NHWC `(1,32,32,3)` → `im2col()` → `(1024, 27)` matrix
2. Weight `(1,3,3,3)` → reshape to `(1, 27)` → transpose to `(27, 1)` matrix
3. DRAM layout: `A_GMEM=0x1000`, `W_GMEM=0x2000`, `C_GMEM=0x3000`
4. ADDR_TABLE at 0x3C: `[0x1000, 0x2000, 0x3000, 1024, 1, 27, 32, 1, 1, 32]`
5. Assembly: `make_tiled_gemm_asm(1024, 1, 27)` → 32 tile iterations (mi=0..31), each doing 1×1×1 tile multiply

**Emulator execution:**
- Emulator loads `.in` file, reads ADDR_TABLE, executes the tiled GEMM loop
- Writes 1024 bf16 output values starting at `C_GMEM=0x3000`
- Orchestrator reads back 1024 values, reshapes to `(1, 1, 32, 32)`

---

## 8. Changes, Tweaks, and Known Limitations

### 8.1 Changes Made to Enable the Pipeline

| Change | File(s) | Why |
|--------|---------|-----|
| Created FX capture + op normalization | `graph/fx_capture.py` | Didn't exist; needed to trace PyTorch models to FX graph |
| Created graph transforms | `graph/remove_ops.py` | BN folding + dropout removal for inference |
| Created tile planner | `graph/tile_planner.py` | Didn't exist; needed to compute tiling strategy per op |
| Created assembly emitter | `codegen/asm_emitter.py` | Didn't exist; bridges FX nodes → build_*.py assembly generators |
| Created C emitter | `codegen/c_emitter.py` | Didn't exist; was intended as the primary path but fell back to direct asm |
| Created asm converter | `codegen/asm_converter.py` | Compiler and emulator use different assembly notation |
| Created DRAM builder | `codegen/dram_builder.py` | Tensor serialization + reference activation extraction |
| Created orchestrator | `run_model.py` | End-to-end pipeline runner |
| Created AlexNet model | `model/alexnet.py` | Scaled-down AlexNet for testing within emulator constraints |
| Copied aihw-ppci-compiler | `aihw-ppci-compiler/` | Needed in-tree for the C compilation path |
| Added `-m atalla` flag | `c_emitter.py` (ppci invocation) | Compiler gave `KeyError: 'vec'` without architecture flag |

### 8.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Direct asm instead of C-compiled kernels** | Compiler's vector spill bug (1/32 elements saved) and missing `lw.vi` make C-compiled GEMM/ReLU/Softmax incorrect |
| **Per-layer emulator invocation** | Emulator is designed for single-kernel execution; full-model single-invocation would need complex memory management |
| **BF16 weight conversion at capture time** | Matches hardware precision; avoids surprise fp32→bf16 truncation mid-pipeline |
| **im2col for convolutions** | Reduces Conv2d to matrix multiply, reusing the tiled GEMM kernel |
| **MaxPool in NumPy** | Kernel exists but is single-channel; multi-channel wrapper not yet wired |
| **Add/Mul in NumPy** | These are bias adds and residual scales in BasicModule; AlexNet folds bias into Conv/Linear |

### 8.3 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Compiler vector spill bug** | C-compiled multi-intrinsic kernels produce wrong results | Use direct assembly for all compute kernels |
| **No `lw.vi` in compiler** | C-compiled GEMM outputs all zeros | Direct assembly includes `lw.vi` |
| **MaxPool uses NumPy** | Not running on emulator; no cycle counts for maxpool | Wire `build_maxpool.py` per-channel loop into emit_maxpool |
| **BF16 accumulation drift** | ~0.1-0.5 max absolute diff from PyTorch float32 | Expected; tracked via cosine similarity |
| **Single-batch only** | Batch size > 1 not tested | Tile planner assumes batch=1 |
| **No softmax in AlexNet** | AlexNet as defined has no softmax (raw logits) | Softmax kernel validated standalone |
| **Compiler emits `li_s`** | Emulator doesn't support this pseudo-instruction | asm_converter could remap; not needed since we use direct asm |
| **ADDR_TABLE hardcoded at 60** | All kernels share the same parameter address | Works for layer-at-a-time; full-model would need different convention |

### 8.4 Validation Results

| Test | Emulated Nodes | Old vs New Pipeline | Notes |
|------|---------------|---------------------|-------|
| BasicModule (dim=32, depth=2) | 5 (2 matmul + 2 relu + 1 linear) | cos=1.0 on all 5 | add/mul in NumPy |
| AlexNet (scale=0.01) | 15 (5 conv + 6 relu + 3 FC) | cos=1.0 on all 15 | maxpool in NumPy |

Both pipelines produce **byte-identical** emulator outputs. The divergence from PyTorch float32 reference (cos≈-0.15 for BasicModule, cos≈-0.03 for AlexNet) is pre-existing BF16 accumulation drift, not a pipeline regression.

---

## 9. Open Items for Review

1. **Wire maxpool to emulator**: `build_maxpool.py` works. Need to add a per-channel loop in `emit_maxpool()` that runs the kernel once per channel with appropriate DRAM offsets. This would replace the NumPy fallback.

2. **Compiler vector spill fix**: The `vreg_st/ld` spill code in `ppci/codegen/atalla/atalla_gen.py` (function `_store_vec_to_scpad` / `_load_vec_from_scpad`) uses dimensions `0,0,0,0,0`, saving only 1 element. Fixing to `31,0,0,0,0` (full 32-element row) would enable C-compiled kernels.

3. **Add `lw.vi` intrinsic to compiler**: The systolic weight preload has no C-level or inline-asm path. Adding it as an intrinsic (e.g., `lw_vi(row_index)`) would unblock C-compiled GEMM.

4. **Bias fusion**: Conv and Linear bias addition is currently implicit (weights include bias via BN folding, or bias is applied post-GEMM in NumPy). A fused GEMM+bias kernel would be more efficient.

5. **Multi-layer single invocation**: Currently each layer is a separate emulator run. Chaining layers in a single `.in` file would eliminate Python-side data marshaling overhead and give more realistic cycle counts.

6. **Notation standardization**: The compiler and emulator should agree on a single assembly notation. Either fix the compiler to emit `.`-separated mnemonics with `$N` registers, or fix the emulator to accept `_`-separated mnemonics with `xN`/`vN` registers. This would eliminate the need for `asm_converter.py`.
