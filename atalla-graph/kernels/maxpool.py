"""Standalone runtime-configured maxpool kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def maxpool_c(H: int, W: int, C: int, pool: int, stride: int) -> str:
    return read_kernel("maxpool.c")
