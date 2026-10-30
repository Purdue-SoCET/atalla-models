# Contributing: how to add a new kernel

This guide walks through adding a compute kernel to the Atalla pipeline: standalone checks first, then AtallaC reference, then parameterized generator, then graph wiring.

## Prerequisites

- Clone **with submodules** (`functional_sim`, `aihw-ppci-compiler`). See `README.md`.
- Python env with **PyTorch**, **NumPy**.
- From repo root, emulator code lives in **`functional_sim/`** (submodule); graph code in **`atalla-graph/`**.

## Architecture (four layers)

```
Layer 1: functional_sim/build_*.py     — hand-written asm + DRAMWriter; run on emulator
Layer 2: aihw-ppci-compiler/atalla_tests/kernels/*.c — fixed-shape reference AtallaC
Layer 3: atalla-graph/kernels/*.py   — parameterized AtallaC strings for arbitrary shapes
Layer 4: atalla-graph/codegen/c_emitter.py — emit_* + DRAMWriter + graph plumbing
```

Work bottom-up: validate Layer 1 on the emulator, then promote.

## Step 0: Study `relu`

| Layer | File | Role |
|-------|------|------|
| 1 | `functional_sim/build_relu.py` | Small BF16 tile, mask+ReLU, `render_testfile` |
| 2 | `aihw-ppci-compiler/atalla_tests/kernels/relu.c` | Same idea in AtallaC |
| 3 | `atalla-graph/kernels/relu.py` | `relu_c(total, width)` generator |
| 4 | `c_emitter.py` → `emit_relu()` | DRAM + `relu_c()` + output addr |

Also read **`functional_sim/ASSEMBLY_SYNTAX.md`** for current **`scpad.ld` / `scpad.st`** (three scalar operands + packed metadata) and **`vreg.ld` / `vreg.st`** (five operands: vd/vs, base, row reg, `num_cols`, `sid`).

## Step 1: Ground-truth asm (`functional_sim/build_<op>.py`)

Hand-write a minimal kernel and DRAM image using **`build.py`**: `DRAMWriter`, `assemble_file` (or scheduled path via `build_compiler.compile_asm` for multi-slot packets), `emit_test_format`, `render_testfile`.

**Do not copy decade-old 5-operand `scpad.ld` text** from informal examples. Follow **`build_gemms.py`**, **`build_relu.py`**, or **`emit_sdma_metadata_asm()`** in `build.py` for SDMA setup, and **`ASSEMBLY_SYNTAX.md`** for mnemonics.

Sketch:

```python

import numpy as np
from build import DRAMWriter, assemble_file, emit_test_format, render_testfile

# 1) asm: use li_s / scpad.ld rs1, rs2, rs3 / vreg.ld vd, rs1, rs2, ncols, sid / …
# 2) DRAMWriter: ADDR_TABLE at 0x3C + bf16 payload
# 3) instrs = assemble_file(asm)
# 4) final = render_testfile(emit_test_format(instrs), img.render_data_mem())
```

Run:

```bash
cd functional_sim
python build_myop.py -o tests/myop.in
python run.py --input_file tests/myop.in
```

Compare emulator `.out` regions against NumPy.

## Step 2: Reference C (`aihw-ppci-compiler/atalla_tests/kernels/<op>.c`)

Use intrinsics consistent with **`atalla-graph/kernels/*.py`** and the compiler README.

| Intrinsic | Role |
|-----------|------|
| `scpad_load(sp, gmem, sdma_ctl)` | DMA DRAM → scratchpad (third arg is packed metadata, often via `li_s`) |
| `scpad_store(sp, gmem, sdma_ctl)` | Scratchpad → DRAM |
| `vector_load(sp_row, row_idx, num_cols_m1, sid)` | Load one row into a `vec` |
| `vector_store(v, sp_row, row_idx, num_cols_m1, sid)` | Store `vec` to scratchpad row |
| `vec_op_masked(...)`, `make_mask(...)` | Masked element ops |
| `load_weights(v)` | `lw.vi` path into systolic |
| `gemm(...)` | MAC into accumulator vector |
| `sqrt(x)` | Scalar BF16 sqrt (where supported) |

**ADDR_TABLE** base is **`0x3C` (decimal 60)** — same constant as `ADDR_TABLE` in `atalla-graph/kernels/common.py`.

**Packed SDMA control:** compute `sdma_ctl_val` in **`atalla-graph/kernels/common.py`**, write it into configuration memory, and load it in C. Do not generate C statements in Python.

Compile:

```bash
cd aihw-ppci-compiler
./atalla_cc -S atalla_tests/kernels/myop.c
```

Schedule / encode the `.s`:

```bash
cd ../functional_sim
python -c "
import build_compiler as bc
s = open('../aihw-ppci-compiler/atalla_tests/kernels/myop.s').read()
text, _, _ = bc.compile_asm(s)
print(text[:800])
"
```

## Step 3: Source wrapper (`atalla-graph/kernels/<op>.py`)

Keep the complete algorithm in `kernels/<op>.c`. Python reads it unchanged:

```python
from .source import read_kernel


def myop_c(total: int, width: int = 32) -> str:
    return read_kernel("myop.c")
```

Define runtime configuration offsets in `kernels/common.py`, then write the
values in the emitter using `_write_runtime_params`. Use integer words for
counts, addresses, masks, strides, and SDMA control; use FP32 words for float
parameters. C reads the same offsets relative to address 60. Existing wrapper
arguments can remain for API compatibility; they do not specialize source.

Vector load/store `num_cols_m1` and scratchpad IDs must be compile-time
constants. Prefer fixed 32-lane instructions plus runtime DMA geometry and
lane masks. Ensure inactive lanes are initialized before a reduction or GEMM
can consume them. Pad the final DMA tile and allocate buffers accordingly.

Register in **`atalla-graph/kernels/__init__.py`** if you use a central export.

## Step 4: Graph emitter (`codegen/c_emitter.py`)

Add **`emit_myop()`** building `LayerEmission` (`c_source`, `dram`, `output_addr`, shapes, flags). Dispatch from **`emit_node()`**. Map FX op → **`atalla_op`** string in **`graph/fx_capture.py`** (`_OP_MAP` / `_METHOD_MAP` / `_MODULE_MAP`).

## Step 5: Validate

```bash
cd atalla-graph
python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate

python run_graph.py --model basic --mode validate --metrics-json metrics_basic.json
```

Expect **`emulator`** (not `NumPy`) for your op; check per-node cosine / `max_diff` in verbose logs.
<!--

## Common issues

| Problem | What to check |
|---------|----------------|
| `Unknown mnemonic for scheduling` | `build_compiler.py` aliases / spelling vs `ASSEMBLY_SYNTAX.md` |
| Immediate / offset overflow | Load the parameter from a u32 configuration word |
| `Tree … not covered` / stack errors | Reduce live `vec` variables; compiler frame limits |
| `cos` ≈ 0 or garbage | BF16 layout, ADDR_TABLE, SDMA metadata, GEMM mask (`mv.stm`), spill vs GMEM in emulator |
| Op stuck as NumPy | `skip_emulator` / missing `emit_*` dispatch / unsupported `atalla_op` |
-->

## File quick reference

| Piece | Location |
|-------|----------|
| Emulator loop | `functional_sim/src/functional_sim.py` |
| Scratchpad / GMEM vector access | `functional_sim/src/components/scpad_ls.py` |
| Scheduler + encoder | `functional_sim/build_compiler.py` |
| DRAM + single-issue asm | `functional_sim/build.py` |
| Opcodes | `functional_sim/src/misc/opcode_table.py` |
| Compiler backend | `aihw-ppci-compiler/ppci/arch/atalla/` |
| AtallaC frontend | `aihw-ppci-compiler/ppci/lang/atalla_c/` |
| Deep dive | `pipeline_reference.md` |
