"""AtallaC gelu kernel generators: fast polynomial approximation and tanh approximation."""

from pathlib import Path

_C_DIR = Path(__file__).parent
_C_HORNER = _C_DIR / "gelu.c"
_C_TANH = _C_DIR / "gelu_tanh.c"


def gelu_c(method: str = "horner", **kwargs) -> str:
    """Return the static AtallaC gelu kernel source code.

    Methods:
      - 'horner': Fast 6th-degree minimax Horner polynomial (2x faster, 0.018 max error, ~12 instrs)
      - 'tanh': Tanh 7th-degree expansion (0.016 max error, ~22 instrs)
    """
    if method.lower() == "tanh":
        return _C_TANH.read_text()
    return _C_HORNER.read_text()
