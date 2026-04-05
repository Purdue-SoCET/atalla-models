# atalla-models

End-to-end pipeline: PyTorch model → AtallaC → Atalla assembly → functional simulator.

## Upstream Sources

This repo pulls two upstream components as **Git submodules** and vendors the graph frontend in-tree:

| Path | Upstream | Branch |
|------|----------|--------|
| `aihw-ppci-compiler/` | [Purdue-SoCET/aihw-ppci-compiler](https://github.com/Purdue-SoCET/aihw-ppci-compiler) | `atalla-models` |
| `functional_sim/` | [Purdue-SoCET/atalla-functional-sim](https://github.com/Purdue-SoCET/atalla-functional-sim) | `main` |
| `atalla-graph/` | [vihaanrc/atalla-graph](https://github.com/vihaanrc/atalla-graph) | `temp` (now `main`) |

Our changes are documented in `pipeline_reference.md` §9.

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
                                       Added: fx_capture.py, tile_planner.py
        codegen/                       c_emitter.py, dram_builder.py
        kernels/                       AtallaC generators: gemm, relu, softmax, maxpool, add
        model/                         basic.py, alexnet.py (Vihaan's), alexnet_small.py (added)
        scripts/                       generate_schedule.py (Vihaan's)
        run_graph.py                   unified entry point: validate or schedule mode (--validate, --schedule)
    functional_sim/                    Submodule: emulator + encoding toolchain
        src/                           functional_sim.py
        build.py                       DRAMWriter, render_testfile, assembler
        build_compiler.py              VLIW scheduler + encoder
        build_*.py                     Standalone kernel builders (team reference)
        run.py                         Standalone emulator entry
    aihw-ppci-compiler/                Compiler submodule
        ppci/arch/atalla/              Backend
        ppci/lang/atalla_c/            AtallaC frontend
        atalla_tests/kernels/          Reference .c kernels
    pipeline_reference.md    Full technical documentation
    CONTRIBUTING.md                    How to add a new kernel (step-by-step)
```

## Usage

Clone with submodules (required for the compiler and emulator):

```bash
git clone --recurse-submodules <URL>
# or, if already cloned:
git submodule update --init --recursive
```

Then:

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

alexnet runs fully on chip.

## Key Design Decisions

- **C compiler path for all compute ops.** Conv, Linear, Matmul, ReLU, Softmax, MaxPool, Add all generate AtallaC via `kernels/*.py`, compile through ppci, schedule via `build_compiler`, and run on the emulator.
- **Passthrough ops need no kernel.** Flatten, transpose, and dropout are pure reshapes — the orchestrator just reshapes the numpy array, no `.in` file generated.
- **atalla-graph owns the .data section.** Each `emit_<op>()` creates a `DRAMWriter` with config (ADDR_TABLE at 0x3C) + tensors. Combined with instruction section via `render_testfile()`. See `pipeline_reference.md` §6.2.
- **Per-layer emulator invocation.** Each layer gets a fresh emulator. Activations pass between layers via `activation_cache`.
- **Stack pointer set dynamically.** `x2` placed above all DRAM data to prevent stack/tensor overlap.

See `CONTRIBUTING.md` for how to add a new kernel step-by-step.
