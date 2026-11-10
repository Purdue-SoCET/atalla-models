

from typing import Tuple

import torch

#static, 
TILE_SHAPE = (32, 32)
GROUP_SIZE = 4
GROUP_COUNT = 256
RETAINED_COUNT = 512


def _group_k_axis(tile: torch.Tensor) -> torch.Tensor:
    return tile.transpose(0, 1).contiguous().reshape(GROUP_COUNT, GROUP_SIZE)


def _ungroup_k_axis(groups: torch.Tensor) -> torch.Tensor:
    """Inverse of the provisional grouping adapter, returning logical B[K,N]."""
    return groups.reshape(TILE_SHAPE).transpose(0, 1).contiguous()


def _bf16_cpu(tensor: torch.Tensor, shape: tuple, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be a torch.bfloat16 tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor.detach().cpu().contiguous()


def _encode_groups(groups: torch.Tensor) -> Tuple[torch.Tensor, bytes]:
    bits = groups.view(torch.int16)
    if torch.any(bits == -32768):  # BF16 negative zero: 0x8000.
        raise ValueError("Negative zero cannot be preserved at omitted positions")
    values = torch.empty((GROUP_COUNT, 2), dtype=torch.int16)
    metadata = bytearray(GROUP_COUNT)
    for i, group in enumerate(bits):
        positions = [p for p, word in enumerate(group.tolist()) if word & 0x7FFF]
        if len(positions) != 2:
            raise ValueError(
                f"Group {i} must contain exactly two nonzero BF16 values; "
                f"found {len(positions)}"
            )
        p0, p1 = positions
        values[i] = group[positions]
        metadata[i] = p0 | (p1 << 2)
    return values.reshape(RETAINED_COUNT).view(torch.bfloat16), bytes(metadata)


def _decode_groups(weights: torch.Tensor, metadata: bytes) -> torch.Tensor:
    """Decode values/positions without knowing the tile's grouping dimension."""
    if not isinstance(metadata, bytes):
        raise TypeError("metadata must be bytes")
    if len(metadata) != GROUP_COUNT:
        raise ValueError(f"metadata must contain {GROUP_COUNT} bytes")
    values = weights.view(torch.int16).reshape(GROUP_COUNT, 2)
    if torch.any((values & 0x7FFF) == 0):
        raise ValueError("Packed weights must contain exactly two nonzeros per group")
    groups = torch.zeros((GROUP_COUNT, GROUP_SIZE), dtype=torch.int16)
    for i, code in enumerate(metadata):
        p0, p1 = code & 3, (code >> 2) & 3
        if code & 0xF0 or p0 >= p1:
            raise ValueError(f"Invalid position metadata for group {i}: 0x{code:02x}")
        groups[i, p0] = values[i, 0]
        groups[i, p1] = values[i, 1]
    return groups.view(torch.bfloat16)


def pack_2_4(tile: torch.Tensor) -> Tuple[torch.Tensor, bytes]:

    return _encode_groups(_group_k_axis(_bf16_cpu(tile, TILE_SHAPE, "tile")))


def decode_2_4(weights: torch.Tensor, metadata: bytes) -> torch.Tensor:
    """Reconstruct the CPU BF16 tile from only packed weights and metadata."""
    weights = _bf16_cpu(weights, (RETAINED_COUNT,), "weights")
    return _ungroup_k_axis(_decode_groups(weights, metadata))
