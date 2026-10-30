"""AtallaC threshold_backward kernel: ReLU backward gradient grad_in = grad_out * (x > 0)."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "threshold_backward.c"


def threshold_backward_c(total: int = 0, width: int = 32) -> str:
    """Return the static AtallaC kernel source code."""
    return _C_PATH.read_text()
