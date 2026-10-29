"""Shared constants and helpers for AtallaC kernel generators."""

import math

ADDR_TABLE = 60
TILE = 32


def sdma_ctl_val(sid: int, num_rows: int, num_cols: int, full_cols: int) -> int:
    """Pack scratchpad DMA control register value."""
    return (sid << 30) | ((num_rows - 1) << 25) | ((num_cols - 1) << 20) | (full_cols - 1)


def sdma_ctl_expr(name: str, sid: int, num_rows: int, num_cols: int, full_cols: int) -> str:
    """Emit C statement that defines sdma_ctl."""
    val = sdma_ctl_val(sid, num_rows, num_cols, full_cols)
    return f"    int {name} = {val};\n"
