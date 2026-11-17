"""Shared path resolution for the standalone kernel tests.

Resolution order (first hit wins):

* ``ATALLA_COMPILER_PATH`` -> aihw-ppci-compiler checkout (must contain ``atalla_cc``)
* ``ATALLA_SIM_PATH``      -> functional_sim checkout (must contain ``build_compiler.py`` / ``run.py``)
* otherwise the in-repo submodules: ``<repo>/aihw-ppci-compiler`` and
  ``<repo>/functional_sim`` (falling back to the nested
  ``<repo>/aihw-ppci-compiler/functional_sim`` if the top-level one is absent).

``<repo>`` is the atalla-models checkout that contains this ``tests/`` directory.
The generators under test are imported from ``<repo>/atalla-graph/kernels``.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str):
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else None


def compiler_root() -> Path:
    p = _env_path("ATALLA_COMPILER_PATH")
    if p is None:
        p = REPO_ROOT / "aihw-ppci-compiler"
    if not (p / "atalla_cc").exists():
        raise FileNotFoundError(
            f"atalla_cc not found under {p}; set ATALLA_COMPILER_PATH or run "
            "`git submodule update --init` in the atalla-models checkout"
        )
    return p


def sim_root() -> Path:
    p = _env_path("ATALLA_SIM_PATH")
    if p is None:
        p = REPO_ROOT / "functional_sim"
        if not (p / "build_compiler.py").exists():
            p = compiler_root() / "functional_sim"
    if not (p / "build_compiler.py").exists():
        raise FileNotFoundError(
            f"functional_sim (build_compiler.py) not found under {p}; set ATALLA_SIM_PATH"
        )
    return p


def models_root() -> Path:
    return REPO_ROOT / "atalla-graph"


REPO_PPCI = compiler_root()
SIM_ROOT = sim_root()
MODELS_ROOT = models_root()
