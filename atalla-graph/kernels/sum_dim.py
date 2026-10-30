"""AtallaC sum_dim kernel: batch reduction sum across rows."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "sum_dim.c"


def sum_dim_c(num_rows: int = 1, num_cols: int = 32) -> str:
    """Return the static AtallaC sum_dim kernel source code."""
    return _C_PATH.read_text()
