# Environment setup (verified recipe)

Verified on macOS (Darwin 25) on 2026-11-04 with the `robert` branch. Every command below was run in that
session unless marked *not measured*.

## 1. Clone with submodules (ITaP URLs, ssh)

The compiler and simulator moved to Purdue's GitHub Enterprise in Oct 2026. `atalla-models` itself stays on
github.com. You need a Purdue login (and an ssh key registered there) for the submodules.

```bash
git clone git@github.com:Purdue-SoCET/atalla-models.git
cd atalla-models
git checkout robert            # or inference-dev
git submodule update --init    # NOT --remote, NOT --recursive (see gotchas)
git submodule status
#  351b5fabf76fa3352ebc2e9571e9793d53c7cbe6 aihw-ppci-compiler (351b5fab)
#  f159f0db15e48847e344cd8ada2106527278db94 functional_sim (heads/main-15-gf159f0d)
```

`.gitmodules` on this branch:

| Submodule | URL | Branch |
|---|---|---|
| `aihw-ppci-compiler` | `org-726@github.itap.purdue.edu:SoCETOrganization/aihw-ppci-compiler.git` | `atalla-arch` |
| `functional_sim` | `org-726@github.itap.purdue.edu:SoCETOrganization/atalla-functional-sim.git` | `main` |

If ssh to `github.itap.purdue.edu` is refused, add your key at the ITaP GitHub (Settings -> SSH keys) and test
with `ssh -T git@github.itap.purdue.edu`. The `org-726@` user in the URL is what the org's deploy settings
hand out; plain `git@` also works for a personal key.

## 2. Python 3.11 virtualenv

Homebrew's `python@3.14` is on `PATH` on this machine and **does not work** (`torch` has no 3.14 wheel, and
`atalla_cc` is `#!/usr/bin/env python3`, so it would pick 3.14 and fail with `No module named ppci`).
Use a dedicated 3.11 venv:

```bash
python3.11 -m venv ~/.venvs/atalla
~/.venvs/atalla/bin/python -m pip install --upgrade pip
~/.venvs/atalla/bin/python -m pip install torch numpy pytest pydot openpyxl
~/.venvs/atalla/bin/python -m pip install -e aihw-ppci-compiler     # editable ppci; never -r requirements.txt
```

Verified versions in `/Users/robert.zhang/.venvs/atalla/bin/python` (3.11.6): torch 2.15.0, numpy 2.4.6,
pytest 9.2.0, pydot 4.0.1, openpyxl 3.1.5, ppci 0.5.9 (editable from `aihw-ppci-compiler/`).

Do **not** `pip install -r aihw-ppci-compiler/requirements.txt`: it pins old tooling that conflicts with the
venv and is not needed to run `atalla_cc`.

Convenience: `export PY=~/.venvs/atalla/bin/python`, or `source ~/.venvs/atalla/bin/activate` so that bare
`python3` (and therefore `atalla_cc`) resolves to the venv.

## 3. The three canonical commands

All from `atalla-graph/`; outputs go under `out/<name>/` (gitignored). Numbers are from this session.

```bash
cd atalla-graph
PY=~/.venvs/atalla/bin/python

# (a) validate: per-node emulator run, chained inputs, final output vs PyTorch
$PY run_graph.py --model basic --mode validate --out-dir out/graph --metrics-json out/graph/m.json
#   -> 10 emulated layers, final cos = 1.000000, ~7 s wall

# (b) schedule: static DRAM allocation + sequential C schedule (no emulator)
$PY run_graph.py --model basic --mode schedule --out-dir out/sched
#   -> out/sched/graph_schedule.c, out/sched/dram.bin

# (c) training: forward + MSE loss + backward graph, gradients validated on the emulator
$PY run_graph.py --model basic --mode validate --training --out-dir out/train
#   -> 51 nodes: 24 emulated, 0 numpy, 18 passthrough; prediction cos = 1.000000;
#      loss emu 0.80859 vs ref 0.81361 (rel_l2 0.0062); 6 gradients compared, worst cos = 1.000000
```

Useful flags: `--quiet`, `--validate-inputs {chained,oracle}`, `--strict-end`, `--layer-metrics-csv PATH`,
`--export-kernel-bundle DIR`, `--all-gradient-losses`, `--scale` (AlexNetSmall channel scale), and `--seed`
(being added by the models work; check `run_graph.py --help`).

Other models (`alexnet_small`, `layernorm_smoke`, `vit_micro`) are being re-enabled in `load_model` on this
branch; `--model` rejects a name that is not registered with `ValueError: Unknown model`.

The benchmark harness wraps (a) with provenance and writes `docs/baselines/<model>__<config>__<stamp>/`
and the workbook: `bench/run_benchmark.py`, `bench/build_workbook.py` (schema in `docs/benchmark_metrics.md`).

## 4. Tests

| Suite | Command (from repo root) | Needs | Measured |
|---|---|---|---|
| graph capture (CPU only) | `$PY -m pytest atalla-graph/tests -q` | torch | **6 passed (+2 subtests)** in 1.8 s; a model missing from `load_model` is skipped, not failed |
| standalone kernels | `$PY -m pytest tests/kernels -q` or `./run_kernel_tests.sh [avgpool]` | compiler + sim submodules | **21 passed** in 14.7 s |
| simulator unit tests | `cd functional_sim && PYTHONPATH=$PWD/../aihw-ppci-compiler $PY -m pytest tests/unit -q` | compiler on `PYTHONPATH` | **58 passed** in 1.0 s |
| compiler's own suite | `cd aihw-ppci-compiler && $PY -m pytest test -q` | - | not measured |

`tests/kernels` resolves the toolchain through `tests/kernels/_paths.py`: `ATALLA_COMPILER_PATH` and
`ATALLA_SIM_PATH` if set, else the in-repo submodules. It runs `atalla_cc -p -S` (compiler packetizer),
checks for stack spills in the loop body, assembles with `functional_sim/build_compiler.py --prepacked`,
runs `functional_sim/run.py`, and diffs against a BF16 NumPy golden.

## 5. Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `ATALLA_SIM_PATH` | `atalla-graph/codegen/c_emitter.py`, `tests/kernels/_paths.py` | functional_sim checkout (default: `<repo>/functional_sim`) |
| `ATALLA_COMPILER_PATH` | `tests/kernels/_paths.py` | aihw-ppci-compiler checkout (default: `<repo>/aihw-ppci-compiler`) |
| `PYTHONPATH=<compiler>` | `functional_sim/tests/unit` | the sim's tests import `ppci` |
| `ATALLA_STRIP_NOP_PACKET_ROWS`, `ATALLA_FUNCTIONAL_SCHED_LATENCY` | `functional_sim/build_compiler.py` | set to `0` / `1` by `tests/kernels` for deterministic packing |
| `FUNCTIONAL_SIM_GEMM_TRACE_FILE` | `functional_sim/src/functional_sim.py` | append GEMM traces as JSON lines (debugging) |

## 6. Gotchas

* **Never `git submodule update --remote`.** It moves the pins to whatever the ITaP branch heads are; the
  pipeline is validated against the pins in `git submodule status`. Use `--init` only.
* **Nested `aihw-ppci-compiler/functional_sim/`** exists (the compiler repo carries its own sim reference)
  but this repo uses the top-level `functional_sim/`. `_paths.py` falls back to the nested copy only if the
  top-level one is missing.
* **`gh` needs a Purdue login for ITaP.** `gh auth status` here shows github.com only; for ITaP use
  `gh auth login --hostname github.itap.purdue.edu` or plain `git` over ssh. The github.com `main` branch
  of `atalla-models` still points its sim pin at `ef513d0b`, which is dangling after the move; use
  `robert` / `inference-dev`.
* **`atalla_cc` and PATH.** The script uses `#!/usr/bin/env python3`; outside an activated venv call it as
  `$PY aihw-ppci-compiler/atalla_cc ...`. The graph path avoids the script entirely
  (`python -m ppci atalla_cc -m atalla -S`).
* **Weight-load order (pending upstream).** `investigate-weight-load-order` on the compiler/sim will flip
  `lw.vi` to last-to-first. When those pins move, reverse the weight loop in `atalla-graph/kernels/gemm.py`
  or every GEMM-backed layer (conv/linear/matmul) will fail validation.
* **Never commit `out/`, `*.in`, `.DS_Store`.** `.gitignore` covers them.
* `timeout(1)` is not on macOS by default; use `gtimeout` from coreutils if you want to bound a run.
