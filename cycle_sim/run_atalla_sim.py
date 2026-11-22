#!/usr/bin/env python3
"""`atalla-sim run`, but compiling C at the compiler's default optimisation level.

atalla-sim's toolchain.compile_c calls `atalla_cc --machine atalla -O 2 -p -S`.
At -O2 the GEMM kernels are miscompiled: main keeps its loop counters (ki, ...)
in x9..x15 across `jal load_weight_tile` / `fetch_weight_tile`, and the callee
uses those registers without saving them, so every loop exits after one trip.
atalla-models compiles the same C with `python -m ppci atalla_cc -m atalla -S`
(no -O, codegen/c_emitter.py::compile_c), which spills locals to the frame.
This wrapper drops `-O 2` (keeps `-p`, the packetizer atalla-sim needs) and
otherwise runs tools/run_kernel.py unchanged.

Usage: <atalla-sim venv>/bin/python run_atalla_sim.py <run_kernel args...>
Env ATALLA_SIM_ROOT: the atalla-sim checkout (default ../../atalla-sim-itap).
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "ATALLA_SIM_ROOT",
    Path(__file__).resolve().parents[2] / "atalla-sim-itap"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from atalla import toolchain  # noqa: E402

_orig_run = toolchain._run


def _run_no_O2(cmd, *a, **kw):
    cmd = list(cmd)
    for i in range(len(cmd) - 1):
        if cmd[i] == "-O" and cmd[i + 1] == "2":
            del cmd[i:i + 2]
            break
    return _orig_run(cmd, *a, **kw)


toolchain._run = _run_no_O2

import run_kernel  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_kernel.main(sys.argv[1:]))
