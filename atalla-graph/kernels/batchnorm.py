"""AtallaC batchnorm kernel: 2D Channel Batch Normalization (Fused Scale & Bias)."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "batchnorm.c"


def batchnorm_c(**kwargs) -> str:
    """Return the static AtallaC batchnorm kernel source code."""
    return _C_PATH.read_text()
