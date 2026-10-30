"""Standalone runtime-configured gemm kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def gemm_c(M: int, N: int, K: int) -> str:
    return read_kernel("gemm.c")
