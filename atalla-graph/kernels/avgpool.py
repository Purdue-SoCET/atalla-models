"""AtallaC avgpool kernel: 2D Average Pooling with vector intrinsics."""

from pathlib import Path

_C_PATH = Path(__file__).parent / "avgpool.c"


def avgpool_c(**kwargs) -> str:
    """Return the static AtallaC avgpool kernel source code."""
    return _C_PATH.read_text()
