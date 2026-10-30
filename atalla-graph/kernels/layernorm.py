"""Standalone runtime-configured layernorm kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def layernorm_c(M: int, D: int, eps: float) -> str:
    if D % 32 != 0:
        raise ValueError(f"layernorm_c requires D % 32 == 0, got D={D}")
    return read_kernel("layernorm.c")
