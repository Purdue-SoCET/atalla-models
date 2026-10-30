"""Standalone runtime-configured mul kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def mul_c(total: int, width: int = 32) -> str:
    return read_kernel("mul.c")
