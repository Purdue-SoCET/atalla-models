"""Read standalone AtallaC sources relative to this module."""
from pathlib import Path


def read_kernel(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text()
