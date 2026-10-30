"""Standalone runtime-configured relu kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def relu_c(total: int, width: int) -> str:
    return read_kernel("relu.c")
