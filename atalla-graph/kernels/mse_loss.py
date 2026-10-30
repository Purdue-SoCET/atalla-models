"""AtallaC mse_loss kernel: forward MSE loss calculation."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "mse_loss.c"


def mse_loss_c(total: int = 32, width: int = 32, total_elements: int = None, **kwargs) -> str:
    """Return the static AtallaC mse_loss kernel source code."""
    return _C_PATH.read_text()
