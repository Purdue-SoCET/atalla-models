# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → functional simulator.

## Upstream Sources

This repo vendors and extends three upstream components:

| Component | Upstream | Branch |
|-----------|----------|--------|
| `aihw-ppci-compiler/` | [Purdue-SoCET/aihw-ppci-compiler](https://github.com/Purdue-SoCET/aihw-ppci-compiler/tree/atalla-arch) | `atalla-arch` |
| `functional_sim/` | [Purdue-SoCET/atalla](https://github.com/Purdue-SoCET/atalla/tree/functional_sim/functional_sim) | `functional_sim` |
| `atalla-graph/` | [vihaanrc/atalla-graph](https://github.com/vihaanrc/atalla-graph) | `temp` (now `main`) |

Our changes are documented in `PIPELINE_TECHNICAL_REFERENCE.md` §9.

## Pipeline Flow

```
PyTorch nn.Module
    |  torch.fx.symbolic_trace + ShapeProp + lower_linear_modules
    v
FX Graph (normalized ops: conv, matmul, relu, softmax, maxpool, add, ...)
    |  tile_planner.py
    v
Tiled FX Graph
    |  c_emitter.py + kernels/*.py
    |  ├─ AtallaC source per node
    |  └─ DRAMWriter: config table + input/weight tensors as .data section
    v
AtallaC .c ─────────────────────────── DRAMWriter (data section)
    |  ppci atalla_cc -S                      |
    v                                         |
ppci .s                                       |
    |  build_compiler.compile_asm()           |
    |  (VLIW scheduling + encoding)           |
    v                                         |
instruction section ──── + ────── data section
                         |
                    render_testfile()
                         |
                         v
                 .in file (complete)
                         |
                    functional_sim
                         |
                         v
                   emulator output
```

## Directory Layout

```
atalla-models/
    atalla-graph/                      Graph frontend + kernel codegen
        graph/                         Vihaan's: export_fx, lower_modules, memoryallocator
                                       Ours: fx_capture.py, tile_planner.py
        codegen/                       c_emitter.py, dram_builder.py
        kernels/                       AtallaC generators: gemm, relu, softmax, maxpool, add
        model/                         basic.py, alexnet.py (Vihaan's), alexnet_small.py (ours)
        scripts/                       generate_schedule.py (Vihaan's C schedule emitter)
        run_graph.py                   Unified entry point: validate or schedule mode
    functional_sim/                    Emulator + encoding toolchain
        src/                           functional_sim.py, components/, misc/
        build.py                       DRAMWriter, render_testfile, assembler
        build_compiler.py              VLIW scheduler + encoder
        _asm_encoding.py               Instruction encoding library
        build_*.py                     Standalone kernel builders (team reference)
        run.py                         Standalone emulator entry
    aihw-ppci-compiler/                Vendored compiler
        ppci/arch/atalla/              Backend
        ppci/lang/atalla_c/            AtallaC frontend
        atalla_tests/kernels/          Reference .c kernels
    PIPELINE_TECHNICAL_REFERENCE.md    Full technical documentation
    CONTRIBUTING.md                    How to add a new kernel (step-by-step)
```

## Usage

```bash
cd atalla-graph
python run_graph.py --model basic --mode validate
python run_graph.py --model alexnet_small --mode validate
python run_graph.py --model basic --mode schedule --out-dir out/basic
```

## Validation Results

| Model | Emulated | NumPy | Passthrough | Final cos sim |
|-------|----------|-------|-------------|---------------|
| BasicModule (dim=32, depth=2) | 9 | 1 (mul) | 1 | 0.9999 |
| AlexNetSmall (scale=0.01) | 21 | 0 | 4 | 0.969 |

AlexNet runs **fully on-chip** with zero NumPy fallbacks. BasicModule has 1 remaining
NumPy op (`mul` for residual scaling). Cosine degradation is expected BF16 drift.

## Key Design Decisions

- **C compiler path for all compute ops.** Conv, Linear, Matmul, ReLU, Softmax, MaxPool, Add all generate AtallaC via `kernels/*.py`, compile through ppci, schedule via `build_compiler`, and run on the emulator.
- **Passthrough ops need no kernel.** Flatten, transpose, and dropout are pure reshapes — the orchestrator just reshapes the numpy array, no `.in` file generated.
- **atalla-graph owns the .data section.** Each `emit_<op>()` creates a `DRAMWriter` with config (ADDR_TABLE at 0x3C) + tensors. Combined with instruction section via `render_testfile()`. See `PIPELINE_TECHNICAL_REFERENCE.md` §6.2.
- **Per-layer emulator invocation.** Each layer gets a fresh emulator. Activations pass between layers via `activation_cache`.
- **Stack pointer set dynamically.** `x2` placed above all DRAM data to prevent stack/tensor overlap.

See `CONTRIBUTING.md` for how to add a new kernel step-by-step.
