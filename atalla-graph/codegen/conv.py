"""Pack a convolution plan into an ELF data image for kernels/conv_*.c.

Configuration words (int32, at ``ELF_CONFIG``; mirrored by kernels/conv_engine.h):

     0 path           0 none, 1 scalar, 2 array, 3 lanes
     1 dtypes         input, filter, output codes (0 bf16, 1 i8, 2 i16, 3 i32)
     4 zero points    input, filter
     6 bases          input, filter, output (absolute byte addresses)
     9 extents        N G F C O0 O1 O2 K0 K1 K2
    19 steps          3, then 22 dilations 3
    25 input strides  N G S0 S1 S2 C         (bytes)
    31 filter strides G F K0 K1 K2 C
    37 output strides N G O0 O1 O2 F
    43 reduction      count, then 4 axis codes in Linalg order (0 C, 1 K0, 2 K1, 3 K2)
    48 constants      zero tile, identity tile (2 KB each)
    50 restages       pre count, post count, then 18 words each (pre first)
   124 snapshots      count, then (src, dst, bytes) x 2
   131 pad bases      SP0, SP1 scratchpad base addresses (0; read at run time so
                      each pad keeps one SDMA hold register)
   133 fold           K2 folded into C (1 = none): pixel rows move in residue classes
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codegen.elf_image import DataImage, KERNEL_DIR
from graph.conv import DTYPE_CODE, PATH_CODE, ConvOp, lower_conv

RESTAGE_BASE = 52
RESTAGE_WORDS = 18
SNAPSHOT_BASE = 124
_RED = {'C': 0, 'K0': 1, 'K1': 2, 'K2': 3}


@dataclass(frozen=True)
class ConvImage:
    image: DataImage
    plan: object
    source: Path
    buffer_addresses: dict
    buffer_sizes: dict


def _identity_tile() -> bytes:
    return (np.eye(32, dtype=np.float32).view(np.uint32) >> 16).astype('<u2').tobytes()


def emit_conv(op: ConvOp, buffers) -> ConvImage:
    plan = lower_conv(op, buffers)
    image = DataImage()
    addr, size = {}, {}
    for name in sorted(buffers):
        data = bytes(buffers[name])
        addr[name], size[name] = image.place(data), len(data)

    # Aliased operands read from a device-made snapshot of their initial bytes.
    snapshots = []
    views = {'input': plan.input, 'filter': plan.filter}
    out_buffer = plan.output[0]
    snap_of = {}
    for key in ('input', 'filter'):
        buf, off, strides = views[key]
        if buf == out_buffer and buf in addr:
            if buf not in snap_of:
                snap_of[buf] = image.reserve(max(4, (size[buf] + 3) & -4))
                snapshots.append((addr[buf], snap_of[buf], (size[buf] + 3) & -4))
            views[key] = ('@' + buf, off, strides)

    for name, nbytes in plan.workspaces.items():
        addr['ws:' + name] = image.reserve(nbytes)
    zero = image.reserve(2048)
    ident = image.place(_identity_tile())

    def resolve(buf, off):
        if buf.startswith('@'):
            return snap_of[buf[1:]] + off
        return (addr['ws:' + buf] if buf in plan.workspaces else addr[buf]) + off

    e = plan.extents
    cfg = [0] * (SNAPSHOT_BASE + 10)
    cfg[0] = PATH_CODE[plan.path]
    cfg[1:4] = [DTYPE_CODE[d] for d in plan.dtypes]
    cfg[4:6] = plan.zero_points
    cfg[6:9] = [resolve(*views['input'][:2]), resolve(*views['filter'][:2]),
                resolve(*plan.output[:2])]
    cfg[9:19] = [e[a] for a in ('N', 'G', 'F', 'C', 'O0', 'O1', 'O2', 'K0', 'K1', 'K2')]
    cfg[19:22] = plan.steps
    cfg[22:25] = plan.dilations
    cfg[25:31] = views['input'][2]
    cfg[31:37] = views['filter'][2]
    cfg[37:43] = plan.output[2]
    red = [_RED[a] for a in plan.reduction]
    cfg[43] = len(red)
    cfg[44:48] = red + [0] * (4 - len(red))
    cfg[48:50] = [zero, ident]
    cfg[50:52] = [len(plan.pre), len(plan.post)]
    for k, r in enumerate(plan.pre + plan.post):
        base = RESTAGE_BASE + RESTAGE_WORDS * k
        cfg[base:base + RESTAGE_WORDS] = [resolve(r.src_buffer, r.src_offset),
                                          resolve(r.dst_buffer, r.dst_offset),
                                          *r.batch, *r.src_batch, *r.dst_batch,
                                          r.rows, r.cols, r.src_row, r.dst_row]
    if RESTAGE_BASE + RESTAGE_WORDS * (len(plan.pre) + len(plan.post)) > SNAPSHOT_BASE:
        raise OverflowError('too many restages for the configuration block')
    cfg[SNAPSHOT_BASE] = len(snapshots)
    for k, s in enumerate(snapshots):
        cfg[SNAPSHOT_BASE + 1 + 3 * k:SNAPSHOT_BASE + 4 + 3 * k] = s
    cfg[SNAPSHOT_BASE + 9] = plan.fold
    image.config = cfg
    source = KERNEL_DIR / (plan.name + '.c')
    return ConvImage(image, plan, source, {k: v for k, v in addr.items() if not k.startswith('ws:')},
                     size)
