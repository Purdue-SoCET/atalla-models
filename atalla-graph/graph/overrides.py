"""Plan overrides (API_CONTRACT.md section B) and the ONE backend decision function.

An override file is JSON::

    {"global": {"gelu_kernel": "pytorch|horner|tanh", "sdpa": "numpy|flash",
                "add_numpy_threshold": 1024},
     "nodes": {"<fx node name>": {"backend": "emulator|numpy|skip", "kernel": "<variant>|null"}}}

``global`` keys map onto the existing env knobs (``ATALLA_GELU_KERNEL``,
``ATALLA_SDPA_FLASH``, ``ATALLA_FFN_ADD_NUMPY`` / ``ATALLA_FFN_ADD_NUMPY_THRESHOLD``):
:func:`apply_global_env` writes them into ``os.environ`` *before* ``plan_tiles`` so the
planner and the emitters see the same values (the override file wins over the env).

Per-node overrides are applied to the planned ``TileConfig`` by :func:`apply_node_overrides`
(after ``plan_tiles``) and then turned into a backend label by :func:`resolve_backend`, which
is the single decision point shared by ``run_graph.run_validate`` and ``bench/ui/inspect.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from torch.fx import Node

from .fx_capture import get_node_shape

BACKENDS = ("emulator", "numpy", "skip")
GELU_KERNELS = ("pytorch", "horner", "tanh")
SDPA_KERNELS = ("numpy", "flash")

# atalla_op -> {override kernel name: tile_config params to set}
_KERNEL_VARIANTS = {
    "gelu": {
        "pytorch": {"method": "torch", "use_numpy": True},
        "torch": {"method": "torch", "use_numpy": True},
        "numpy": {"method": "torch", "use_numpy": True},
        "horner": {"method": "horner", "use_numpy": False},
        "tanh": {"method": "tanh", "use_numpy": False},
    },
    "atalla_sdpa": {
        "flash": {"use_flash": True, "use_numpy": False},
        "numpy": {"use_flash": False, "use_numpy": True},
    },
}

# Ops whose emitter has a NumPy path that the planner can switch on/off via tile params.
_NUMPY_SWITCHABLE = frozenset({"add", "gelu", "atalla_sdpa"})


def gelu_method_from_env() -> str:
    """Resolve ``ATALLA_GELU_KERNEL`` to ``torch | horner | tanh`` (same table as c_emitter)."""
    v = os.environ.get("ATALLA_GELU_KERNEL", "").strip().lower()
    if v in ("", "0", "torch", "numpy", "off", "pytorch"):
        return "torch"
    if v in ("1", "horner"):
        return "horner"
    if v == "tanh":
        return "tanh"
    raise ValueError(f"ATALLA_GELU_KERNEL={v!r}: expected '', 'horner' or 'tanh'")


def add_numpy_threshold_from_env() -> Optional[int]:
    """Element count above which ``add`` runs on NumPy, or ``None`` for never.

    ``ATALLA_FFN_ADD_NUMPY=1`` keeps its historical meaning (NumPy above 1024 elements);
    ``ATALLA_FFN_ADD_NUMPY_THRESHOLD=N`` (set by the ``add_numpy_threshold`` override) wins.
    """
    thr = os.environ.get("ATALLA_FFN_ADD_NUMPY_THRESHOLD", "").strip()
    if thr:
        try:
            return int(thr)
        except ValueError:
            pass
    if os.environ.get("ATALLA_FFN_ADD_NUMPY", "0") == "1":
        return 1024
    return None


@dataclass
class NodeOverride:
    backend: Optional[str] = None  # emulator | numpy | skip
    kernel: Optional[str] = None   # gelu: horner|tanh|pytorch ; sdpa: flash|numpy

    def as_dict(self) -> Dict:
        return {"backend": self.backend, "kernel": self.kernel}


@dataclass
class PlanOverrides:
    gelu_kernel: Optional[str] = None
    sdpa: Optional[str] = None
    add_numpy_threshold: Optional[int] = None
    nodes: Dict[str, NodeOverride] = field(default_factory=dict)
    source: Optional[str] = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Optional[Dict], source: Optional[str] = None) -> "PlanOverrides":
        d = d or {}
        g = d.get("global") or {}
        ov = cls(source=source)
        gk = g.get("gelu_kernel")
        if gk is not None:
            gk = str(gk).strip().lower()
            if gk == "torch":
                gk = "pytorch"
            if gk not in GELU_KERNELS:
                raise ValueError(f"overrides.global.gelu_kernel={gk!r}: expected one of {GELU_KERNELS}")
            ov.gelu_kernel = gk
        sd = g.get("sdpa")
        if sd is not None:
            sd = str(sd).strip().lower()
            if sd not in SDPA_KERNELS:
                raise ValueError(f"overrides.global.sdpa={sd!r}: expected one of {SDPA_KERNELS}")
            ov.sdpa = sd
        thr = g.get("add_numpy_threshold")
        if thr is not None:
            ov.add_numpy_threshold = int(thr)
        for name, spec in (d.get("nodes") or {}).items():
            spec = spec or {}
            be = spec.get("backend")
            if be is not None:
                be = str(be).strip().lower()
                if be not in BACKENDS:
                    raise ValueError(f"overrides.nodes[{name!r}].backend={be!r}: expected one of {BACKENDS}")
            kv = spec.get("kernel")
            if kv is not None:
                kv = str(kv).strip().lower() or None
            if be is None and kv is None:
                continue
            ov.nodes[str(name)] = NodeOverride(backend=be, kernel=kv)
        return ov

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PlanOverrides":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text()), source=str(p))

    def as_dict(self) -> Dict:
        """JSON view of the parsed overrides (``effective_overrides`` in inspect)."""
        return {
            "global": {
                "gelu_kernel": self.gelu_kernel,
                "sdpa": self.sdpa,
                "add_numpy_threshold": self.add_numpy_threshold,
            },
            "nodes": {k: v.as_dict() for k, v in self.nodes.items()},
            "source": self.source,
        }

    def is_empty(self) -> bool:
        return (self.gelu_kernel is None and self.sdpa is None
                and self.add_numpy_threshold is None and not self.nodes)


def load_overrides(spec: Union[None, str, Path, Dict, PlanOverrides],
                   env_var: str = "ATALLA_PLAN_OVERRIDES") -> PlanOverrides:
    """Return a :class:`PlanOverrides` from a path / dict / object, else from ``env_var``."""
    if isinstance(spec, PlanOverrides):
        return spec
    if isinstance(spec, dict):
        return PlanOverrides.from_dict(spec)
    if spec:
        return PlanOverrides.load(spec)
    env_path = os.environ.get(env_var, "").strip()
    if env_path:
        return PlanOverrides.load(env_path)
    return PlanOverrides()


# -- global knobs -> env (before plan_tiles) --------------------------------

ENV_KNOBS = ("ATALLA_GELU_KERNEL", "ATALLA_SDPA_FLASH", "ATALLA_FFN_ADD_NUMPY",
             "ATALLA_FFN_ADD_NUMPY_THRESHOLD")


def snapshot_env() -> Dict[str, Optional[str]]:
    return {k: os.environ.get(k) for k in ENV_KNOBS}


def restore_env(snap: Dict[str, Optional[str]]) -> None:
    """Undo :func:`apply_global_env` (the planner has recorded its decisions in tile params)."""
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def apply_global_env(ov: Optional[PlanOverrides]) -> Dict[str, str]:
    """Write the ``global`` overrides into ``os.environ`` (override file wins over env)."""
    applied: Dict[str, str] = {}
    if ov is None:
        return applied
    if ov.gelu_kernel is not None:
        val = "" if ov.gelu_kernel == "pytorch" else ov.gelu_kernel
        os.environ["ATALLA_GELU_KERNEL"] = val
        applied["ATALLA_GELU_KERNEL"] = val
    if ov.sdpa is not None:
        val = "1" if ov.sdpa == "flash" else "0"
        os.environ["ATALLA_SDPA_FLASH"] = val
        applied["ATALLA_SDPA_FLASH"] = val
    if ov.add_numpy_threshold is not None:
        os.environ["ATALLA_FFN_ADD_NUMPY"] = "1"
        os.environ["ATALLA_FFN_ADD_NUMPY_THRESHOLD"] = str(int(ov.add_numpy_threshold))
        applied["ATALLA_FFN_ADD_NUMPY"] = "1"
        applied["ATALLA_FFN_ADD_NUMPY_THRESHOLD"] = str(int(ov.add_numpy_threshold))
    return applied


# -- per-node overrides -> tile params (after plan_tiles) -------------------

def _first_tensor_input(node: Node) -> Optional[Node]:
    for a in node.args:
        if isinstance(a, Node) and get_node_shape(a) is not None:
            return a
    for a in node.args:
        if isinstance(a, Node):
            return a
    return None


def apply_node_overrides(gm, ov: Optional[PlanOverrides]) -> None:
    """Push per-node ``backend`` / ``kernel`` requests into ``node.meta['tile_config'].params``.

    Only switchable paths are rewritten here (gelu method, sdpa flash, add numpy); every
    other consequence of an override is decided by :func:`resolve_backend`.
    """
    if ov is None or not ov.nodes:
        return
    for node in gm.graph.nodes:
        spec = ov.nodes.get(node.name)
        if spec is None:
            continue
        node.meta["override"] = spec.as_dict()
        tc = node.meta.get("tile_config")
        if tc is None:
            continue
        op = node.meta.get("atalla_op")
        p = tc.params
        variants = _KERNEL_VARIANTS.get(op, {})
        if spec.kernel is not None and spec.kernel in variants:
            p.update(variants[spec.kernel])
        if spec.backend == "numpy" and op in _NUMPY_SWITCHABLE:
            if op == "gelu":
                p.update(variants["pytorch"])
            elif op == "atalla_sdpa":
                p.update(variants["numpy"])
            else:
                p["use_numpy"] = True
        elif spec.backend == "emulator" and op in _NUMPY_SWITCHABLE:
            if op == "gelu":
                if p.get("method", "torch") == "torch":
                    p.update(variants[spec.kernel or "horner"] if spec.kernel in variants
                             else variants["horner"])
                p["use_numpy"] = False
            elif op == "atalla_sdpa":
                if _sdpa_flash_possible(p):
                    p.update(variants["flash"])
                # else: leave use_flash False -> resolve_backend reports the error.
            else:
                p["use_numpy"] = False


def _sdpa_flash_possible(p: Dict) -> bool:
    return int(p.get("B", 0)) == 1 and int(p.get("N", 0)) == 32 and int(p.get("D", 0)) == 32


# -- THE decision -----------------------------------------------------------

PASSTHROUGH_OPS = ("flatten", "dropout", "transpose", None)


@dataclass
class BackendDecision:
    backend: str                      # graph | passthrough | emulator | numpy | skip
    reason: str
    variant: Optional[str] = None     # kernel variant label (gelu method, sdpa flash, ...)
    forced: bool = False              # set by a per-node override
    error: Optional[str] = None       # override could not be honoured (fallback applied)
    skip_source: Optional[str] = None # for backend == skip: the input node name


def kernel_variant(op: Optional[str], tc) -> Optional[str]:
    """Human label of the kernel variant the planner picked (``None`` when there is none)."""
    if tc is None:
        return None
    p = tc.params
    if op == "gelu":
        m = p.get("method") or gelu_method_from_env()
        return "pytorch" if m == "torch" else m
    if op == "atalla_sdpa":
        return "flash" if p.get("use_flash") else "numpy"
    return None


def resolve_backend(node: Node, tc, has_emitter: bool,
                    overrides: Optional[PlanOverrides] = None) -> BackendDecision:
    """Decide how ``run_validate`` executes ``node``. Used verbatim by ``inspect_graph``.

    ``tc`` is ``node.meta['tile_config']`` (may be ``None``); ``has_emitter`` says whether
    ``codegen.c_emitter.emit_node`` has an ``emit_<op>`` for ``node.meta['atalla_op']``.
    """
    op = node.meta.get("atalla_op")
    if node.op == "placeholder":
        return BackendDecision("graph", "graph input (placeholder)")
    if node.op == "get_attr":
        return BackendDecision("graph", "parameter / buffer (get_attr)")
    if node.op == "output":
        return BackendDecision("graph", "graph output")

    variant = kernel_variant(op, tc)
    spec = overrides.nodes.get(node.name) if overrides is not None else None
    pending = tc is not None and bool(tc.params.get("pending"))

    if spec is not None and spec.backend == "skip":
        src = _first_tensor_input(node)
        in_shape = get_node_shape(src) if src is not None else None
        out_shape = get_node_shape(node)
        if src is None:
            return BackendDecision("numpy", "override skip: node has no tensor input; NumPy reference",
                                   variant, True, "skip requested but node has no tensor input")
        if in_shape is not None and out_shape is not None and tuple(in_shape) != tuple(out_shape):
            return BackendDecision(
                "numpy", f"override skip rejected: {src.name} {list(in_shape)} != output {list(out_shape)}; NumPy reference",
                variant, True,
                f"skip requested but shapes differ ({src.name} {list(in_shape)} vs {list(out_shape)})")
        return BackendDecision("skip", f"override: identity on {src.name}", variant, True,
                               skip_source=src.name)

    if spec is not None and spec.backend == "numpy":
        return BackendDecision("numpy", "override: forced NumPy reference", variant, True)

    if spec is not None and spec.backend == "emulator":
        if op in PASSTHROUGH_OPS:
            return BackendDecision("passthrough", f"override emulator ignored: {op or 'unknown'} is a layout op",
                                   variant, True, f"emulator requested for layout op {op!r}")
        if tc is None or pending or not has_emitter:
            why = ("no tile plan" if tc is None else "kernel pending (no planner)" if pending
                   else f"no emit_{op} in c_emitter")
            return BackendDecision("numpy", f"override emulator rejected: {why}; NumPy reference",
                                   variant, True, f"emulator requested but {why}")
        if tc.params.get("use_numpy"):
            why = _numpy_reason(op, tc)
            return BackendDecision("numpy", f"override emulator rejected: {why}; NumPy reference",
                                   variant, True, f"emulator requested but {why}")
        return BackendDecision("emulator", f"override: forced emulator ({_planned_by(op)})", variant, True)

    # -- planner defaults (mirror run_validate's control flow) ---------------
    if op in PASSTHROUGH_OPS:
        return BackendDecision("passthrough", f"layout op {op or 'unknown'}: reshaped in activation cache", variant)
    if tc is None:
        return BackendDecision("passthrough", f"no tile planner for {op!r}: PyTorch reference passthrough", variant)
    if pending:
        return BackendDecision("passthrough", f"kernel pending for {op!r}: PyTorch reference passthrough", variant)
    if not has_emitter:
        return BackendDecision("passthrough", f"no emit_{op} in c_emitter: PyTorch reference passthrough", variant)
    if tc.params.get("use_numpy"):
        forced_kernel = spec is not None and spec.kernel is not None
        return BackendDecision("numpy", _numpy_reason(op, tc), variant, forced_kernel)
    forced_kernel = spec is not None and spec.kernel is not None
    return BackendDecision("emulator", _planned_by(op) + (f" [{variant}]" if variant else ""),
                           variant, forced_kernel)


def _planned_by(op: Optional[str]) -> str:
    table = {
        "conv": "_plan_conv", "linear": "_plan_linear", "relu": "_plan_relu", "maxpool": "_plan_maxpool",
        "softmax": "_plan_softmax", "add": "_plan_add", "matmul": "_plan_matmul", "mul": "_plan_mul",
        "atalla_sdpa": "_plan_sdpa", "layernorm": "_plan_layernorm", "gelu": "_plan_gelu",
        "threshold_backward": "_plan_threshold_backward", "sum_dim": "_plan_sum_dim",
        "mse_loss": "_plan_mse_loss", "mse_loss_backward": "_plan_mse_loss_backward",
    }
    return f"planned by {table.get(op, 'tile_planner')}"


def _numpy_reason(op: Optional[str], tc) -> str:
    p = tc.params
    if op == "add":
        thr = p.get("numpy_threshold")
        return f"add with {p.get('total_elements')} elems > threshold {thr} (ATALLA_FFN_ADD_NUMPY)"
    if op == "gelu":
        return "GELU on PyTorch reference (ATALLA_GELU_KERNEL unset; pick horner|tanh for the kernel)"
    if op == "atalla_sdpa":
        return (f"fused attention B={p.get('B')} N={p.get('N')} D={p.get('D')}: flash kernel only for "
                f"B=1,N=32,D=32 with ATALLA_SDPA_FLASH=1")
    if op == "softmax":
        return f"softmax row_len {p.get('row_len')} > 32: PyTorch reference"
    if op == "layernorm":
        return f"layernorm D={p.get('D')} not a multiple of 32: PyTorch reference"
    if op == "maxpool":
        return f"maxpool W={p.get('W')} > 32: NumPy reference"
    if op == "adaptive_avg_pool":
        return "adaptive_avg_pool: PyTorch reference (no kernel)"
    return p.get("numpy_reason", "planner marked use_numpy")
