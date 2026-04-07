import math
import operator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node


TILE_HEIGHT = 32
TILE_WIDTH = 32
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * 2  # 2 KB tiles, bf16 2 bytes each

VIEW_FUNCTIONS = {operator.getitem}
VIEW_METHODS = {"reshape", "flatten", "permute", "transpose"}


def align(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


@dataclass
class _Block:
    start: int # byte address
    size: int # in bytes


def _insert_free_block(free_blocks: List[_Block], block: _Block) -> None:
    """Insert + merge adjacent free blocks."""
    free_blocks.append(block)
    free_blocks.sort(key=lambda b: b.start)
    merged: List[_Block] = []
    for cur in free_blocks:
        if not merged:
            merged.append(cur)
            continue
        prev = merged[-1]
        if prev.start + prev.size == cur.start:
            prev.size += cur.size
        else:
            merged.append(cur)
    free_blocks[:] = merged


def _alloc_from_free(free_blocks: List[_Block], size: int) -> Optional[int]:
    """Best-fit free-list allocation."""
    best_idx = -1
    best_size = None
    for idx, blk in enumerate(free_blocks):
        if blk.size < size:
            continue
        if best_size is None or blk.size < best_size:
            best_idx = idx
            best_size = blk.size
    if best_idx < 0:
        return None
    blk = free_blocks[best_idx]
    start = blk.start
    if blk.size == size:
        free_blocks.pop(best_idx)
    else:
        blk.start += size
        blk.size -= size
    return start


def _view_source(node: Node) -> Optional[Node]:
    if node.op == "call_function" and node.target in VIEW_FUNCTIONS:
        arg = node.args[0]
        if isinstance(arg, Node):
            return arg
    if node.op == "call_method":
        method = node.target.strip("'")
        if method in VIEW_METHODS:
            arg = node.args[0]
            if isinstance(arg, Node):
                return arg
    return None


def tensor_nbytes(node: Node) -> int:
    tensor_meta = node.meta.get("tensor_meta")

    if tensor_meta.dtype != torch.bfloat16:
        raise ValueError(
            f"dtype {tensor_meta.dtype} for node {node.name} is not torch.bfloat16")

    outer = 1
    shape = tensor_meta.shape
    if len(shape) == 1:
        height = 1
        width = int(shape[0])
    else:
        for dim in shape[:-2]:
            outer *= int(dim)
        height = int(shape[-2])
        width = int(shape[-1])

    tiles_h = math.ceil(height / TILE_HEIGHT)
    tiles_w = math.ceil(width / TILE_WIDTH)
    tiles_per_plane = tiles_h * tiles_w
    total_tiles = max(1, outer) * tiles_per_plane

    return total_tiles * TILE_BYTES


def tensor_for_node(
    node: Node, gm: GraphModule, placeholder_data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if node.op == "placeholder":
        if node.target not in placeholder_data:
            raise ValueError(f"Missing placeholder data for {node.target}")
        return placeholder_data[node.target]

    if node.op == "get_attr":
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part) #must be a tensor
        return attr

    return None

def tensor_bytes(tensor: torch.Tensor, allocation_size: int) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim == 0:
        raise ValueError("Scalar tensor")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim > 2:
        outer = 1
        for dim in tensor.shape[:-2]:
            outer *= int(dim)
        height = tensor.shape[-2]
        width = tensor.shape[-1]
        tensor = tensor.view(outer, height, width)
    else:
        tensor = tensor.unsqueeze(0)
        height = tensor.shape[-2]
        width = tensor.shape[-1]

    tiles = []
    for matrix in tensor:
        for row in range(0, height, TILE_HEIGHT):
            for col in range(0, width, TILE_WIDTH):
                tile = torch.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=tensor.dtype)
                h_chunk = min(TILE_HEIGHT, height - row)
                w_chunk = min(TILE_WIDTH, width - col)
                tile[:h_chunk, :w_chunk] = matrix[row:row + h_chunk, col:col + w_chunk]
                tiles.append(tile)

    raw = b"".join(
        tile.view(torch.uint16).numpy().astype(np.uint16).tobytes(order="C")
        for tile in tiles
    )
    if len(raw) > allocation_size:
        raise ValueError("Tile payload larger than allocation size")
    return raw + bytes(allocation_size - len(raw))


def _write_binary_payload(file, current_size: int, start_addr: int, payload: bytes) -> int:
    if start_addr < current_size:
        raise ValueError("start_addr rewound; allocations must be monotonic")
    gap = start_addr - current_size
    if gap:
        file.write(bytes(gap))
        current_size += gap
    file.write(payload)
    return current_size + len(payload)


def assign_address(node: Node, next_addr: int) -> Tuple[int, int, int]:
    bytes_needed = tensor_nbytes(node)
    aligned_addr = align(next_addr, TILE_BYTES)
    allocation_size = align(bytes_needed, TILE_BYTES)

    node.meta["dram_addr"] = f"0x{aligned_addr:08x}"
    node.meta["bytes"] = allocation_size

    return aligned_addr, allocation_size, aligned_addr + allocation_size


def allocate_memory(gm: GraphModule, bin_path: str, placeholder_data: Optional[Dict[str, torch.Tensor]] = None) -> GraphModule:
    placeholder_data = placeholder_data or {}

    nodes = list(gm.graph.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    #view ops (e.g: transposes) use their source allocation
    owner_by_node: Dict[Node, Node] = {}
    for node in nodes:
        view_src = _view_source(node)
        if view_src is not None and view_src in owner_by_node:
            owner_by_node[node] = owner_by_node[view_src]
        else:
            owner_by_node[node] = node

    # node index where value is used last
    last_use: Dict[Node, int] = {}
    for node in nodes:
        owner = owner_by_node[node]
        last_use.setdefault(owner, node_index[node])
        for user in node.users.keys():
            last_use[owner] = max(last_use[owner], node_index[user])

    # Allocate with liveness-based reuse
    free_blocks: List[_Block] = []
    next_addr = 0
    owner_addr: Dict[Node, int] = {}
    owner_size: Dict[Node, int] = {}
    peak_end = 0

    for idx, node in enumerate(nodes):
        if node.op == "output":
            continue

        # Reclaim blocks
        for owner, dead_idx in last_use.items():
            if dead_idx == idx - 1 and owner in owner_addr:
                _insert_free_block(
                    free_blocks,
                    _Block(start=owner_addr[owner], size=owner_size[owner]),
                )

        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        if tensor_meta.dtype != torch.bfloat16:
            # Treat non-bf16 tensors (e.g., attention bias indices) as compile-time constants.
            continue

        owner = owner_by_node[node]
        if owner in owner_addr:
            node.meta["dram_addr"] = f"0x{owner_addr[owner]:08x}"
            node.meta["bytes"] = owner_size[owner]
            continue

        size = align(tensor_nbytes(owner), TILE_BYTES)
        preinit = tensor_for_node(owner, gm, placeholder_data) is not None

        # Preinitialized tensors = input arguments/get_attr
        addr = None
        if not preinit:
            addr = _alloc_from_free(free_blocks, size)
        if addr is None:
            addr = align(next_addr, TILE_BYTES)
            next_addr = addr + size

        owner_addr[owner] = addr
        owner_size[owner] = size
        peak_end = max(peak_end, addr + size)

        node.meta["dram_addr"] = f"0x{addr:08x}"
        node.meta["bytes"] = size

    # Build DRAM image
    dram_image = bytearray(peak_end)
    for node in nodes:
        if node.op == "output":
            continue
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None or tensor_meta.dtype != torch.bfloat16:
            continue
        tensor_value = tensor_for_node(node, gm, placeholder_data)
        if tensor_value is None:
            continue
        addr = int(str(node.meta["dram_addr"]), 16)
        size = int(node.meta["bytes"])
        payload = tensor_bytes(tensor_value, size)
        dram_image[addr:addr + size] = payload

    with open(bin_path, "wb") as dram_file:
        dram_file.write(dram_image)

    gm.graph.lint()
    gm.recompile()

    return gm


def fake_allocate_memory(gm: GraphModule) -> GraphModule:
    """Simulate memory assignment without writing a binary file."""
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if "tensor_meta" not in node.meta:
            continue
        node.meta["dram_addr"] = "0x00000000"
        # Setting bytes to zero skips the check in emit()
        node.meta["bytes"] = 0

    gm.graph.lint()
    gm.recompile()
    return gm
