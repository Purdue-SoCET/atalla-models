"""Shared constants and helpers for AtallaC kernel generators."""

import math

ADDR_TABLE = 60
TILE = 32


def sdma_ctl_val(sid: int, num_rows: int, num_cols: int, full_cols: int) -> int:
    """Pack scratchpad DMA control register value."""
    return (sid << 30) | ((num_rows - 1) << 25) | ((num_cols - 1) << 20) | (full_cols - 1)


def sdma_ctl_expr(name: str, sid: int, num_rows: int, num_cols: int, full_cols: int) -> str:
    """Emit C statement that loads a pre-computed sdma_ctl via inline asm li_s.

    The compiler's instruction selector can't handle large constant stores,
    so we use inline asm to let build_compiler expand li_s -> lui_s+addi_s.
    """
    val = sdma_ctl_val(sid, num_rows, num_cols, full_cols)
    return f'    int {name};\n    asm("li_s %0, {val}" : "=r"({name}));\n'
