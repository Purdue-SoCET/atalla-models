"""Ordinary-C GELU sources: legacy polynomials and stable piecewise evaluation."""

from pathlib import Path

_C_DIR = Path(__file__).parent
_C_HORNER = _C_DIR / "gelu.c"
_C_TANH = _C_DIR / "gelu_tanh.c"


def gelu_c(method: str = "horner", **kwargs) -> str:
    """Return the static AtallaC gelu kernel source code.

    Methods:
      - 'horner': Legacy 6th-degree Horner polynomial
      - 'tanh': Legacy tanh 7th-degree expansion
      - 'piecewise': Centered cubic intervals with stable negative-tail evaluation
    """
    if method.lower() == "piecewise":
        return (_C_DIR / "gelu_piecewise.c").read_text()
    if method.lower() == "tanh":
        return _C_TANH.read_text()
    return _C_HORNER.read_text()
