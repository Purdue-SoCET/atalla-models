"""Standalone runtime-configured softmax kernel; shapes are supplied by the emitter."""
from .source import read_kernel


def softmax_c(length: int) -> str:
    return read_kernel("softmax.c")


def softmax_c_batched(num_rows: int, row_len: int) -> str:
    if row_len > 32:
        raise ValueError("softmax_c_batched requires row_len <= 32")
    return read_kernel("softmax.c")
