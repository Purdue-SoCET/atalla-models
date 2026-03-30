# atalla-models

End-to-end pipeline: PyTorch model -> AtallaC -> Atalla assembly -> functional simulator.

## Setup

`atalla-models` and `aihw-ppci-compiler` must be cloned as sibling directories:

```
workspace/
├── atalla-models/
└── aihw-ppci-compiler/
```

Use the `atalla_arch_emul_robert` branch of `aihw-ppci-compiler` for updated ISA fixes
(sac operand, MTS/STM token layout, rcp.bf, build pipeline updates).

Override the compiler path if your layout differs:

```bash
export ATALLA_COMPILER_PATH=/path/to/aihw-ppci-compiler
```

## Pipeline Flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp
    v
FX Graph
    |  tile_planner.py
    v
Tiled FX Graph
    |  c_emitter.py -- AtallaC per node + DRAMWriter tensor data
    v
AtallaC .c
    |  ppci atalla_cc -S
    v
ppci .s
    |  build_compiler.compile_asm() -- notation conversion + scheduling + encoding
    v
.in test file (instruction packets + .data tensor payloads)
    |  functional_sim run.py
    v
Emulator output
```

## Directory Layout

```
atalla-models/
    atalla-graph/
        graph/         fx_capture.py, tile_planner.py, remove_ops.py
        codegen/       c_emitter.py (primary), asm_emitter.py (reference)
        model/         basic.py, alexnet.py
        run_model.py
    functional_sim/
        build.py       DRAMWriter, render_testfile
        build_*.py     kernel generators
        run.py
        tests/

# sibling repo (not inside atalla-models):
aihw-ppci-compiler/
    ppci/arch/atalla/  compiler backend
    emulator/          build_compiler.py: scheduler + encoder
    atalla_cc/         AtallaC frontend
    atalla_tests/      reference C programs
```

## Usage

```bash
cd atalla-graph
python run_model.py --model basic
python run_model.py --model alexnet --scale 0.01
```

## Key Design Decisions

**C compiler path for all compute ops.** ReLU, Softmax, GEMM, Conv, and Linear generate
AtallaC via `c_emitter.py`, compiled through ppci to `.s`, then scheduled and encoded by
`build_compiler.compile_asm()` from `aihw-ppci-compiler/emulator/`. Non-compute ops
(MaxPool, elementwise add/mul, adaptive avg pool) fall back to NumPy.

**`build_compiler.compile_asm()` replaces the legacy `asm_converter.py`.** Handles notation
conversion (underscore to dot mnemonics, symbolic to $N registers), hazard scheduling, and
binary encoding in one pass. Supports updated ISA instruction formats (register-based
scpad_ld/st, 5-arg vreg_ld/st, sac on VV ops).

**DRAMWriter data section.** `c_emitter.py` writes tensor data (weights, inputs, outputs)
to a `DRAMWriter` keyed by byte address. `render_in_file()` merges instruction packets with
the `.data` section into a single `.in` file.

**Per-layer execution.** Each layer is a standalone emulator invocation. Activations pass
between layers via the Python orchestrator.

## Compiler Status

| Issue | Status |
|-------|--------|
| Notation mismatch (mnemonics, registers) | Fixed via build_compiler.compile_asm() |
| sac operand missing from VV instructions | Fixed in atalla_arch_emul_robert |
| MTS/STM token bit layout swapped | Fixed in atalla_arch_emul_robert |
| halt/nop opcode mismatch | Fixed in atalla_arch_emul_robert |
| Vector spill stores only 1/32 elements | Fixed: ty="VEC" on AtallaVectorRegister |
| rcp.bf not recognized | Fixed in atalla_arch_emul_robert |
| vreg_ld/st 7-arg format in C templates | Fixed in c_emitter.py (now 5-arg) |
| scpad_ld/st 5-arg format in C templates | Fixed in c_emitter.py (now 3-register) |
| No lw.vi intrinsic for systolic weight preload | Pending |
| Compiler only colors v1/v2 vector registers | Pending |
