"""AtallaC mse_loss_backward kernel: gradient calculation 2/N * (x - y)."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "mse_loss_backward.c"


def mse_loss_backward_c(total: int = 32, width: int = 32, total_elements: int = None, **kwargs) -> str:
    """Return the static AtallaC mse_loss_backward kernel source code."""
    return _C_PATH.read_text()
