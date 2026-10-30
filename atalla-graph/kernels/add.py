"""Standalone runtime-configured add kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def add_c(total: int, width: int = 32) -> str:
    return read_kernel("add.c")
