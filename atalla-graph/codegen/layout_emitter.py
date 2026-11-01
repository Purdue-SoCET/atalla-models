"""Prepare runtime config and raw buffers for the ordinary C movement kernel."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from graph.layout import INT_MAX, LayoutOp, LayoutPlan, lower_layout
from codegen.c_emitter import DRAMWriter, LayerEmission


# Keep room for the scalar stack initialized by the shared simulator runner.
MAX_DATA_ADDRESS = INT_MAX - 0x4000
CONFIG_ADDRESS = 60


@dataclass(frozen=True)
class LayoutEmission:
    emission: LayerEmission
    plan: LayoutPlan
    buffer_addresses: dict
    buffer_sizes: dict


def _align4(value):
    return (value + 3) & -4


def emit_layout(op: LayoutOp, buffers: Mapping[str, bytes]) -> LayoutEmission:
    """Serialize only; all gather/scatter and alias snapshotting execute in C."""
    plan = lower_layout(op, buffers)
    if any(not isinstance(name, str) or not name for name in buffers):
        raise ValueError('buffer allocation names must be nonempty strings')
    if any(not isinstance(data, (bytes, bytearray)) for data in buffers.values()):
        raise TypeError('buffer allocations must contain bytes or bytearray')
    rank = len(plan.destination.shape)
    cursor = _align4(max(0x1000, CONFIG_ADDRESS + 4 * (6 + 3 * rank)))
    addresses, sizes = {}, {}
    for name in sorted(buffers):
        addresses[name] = cursor
        sizes[name] = len(buffers[name])
        cursor += max(4, _align4(sizes[name]))
        if cursor > MAX_DATA_ADDRESS:
            raise OverflowError('layout buffers exceed signed 32-bit address budget')
    snapshot = cursor
    snapshot_size = max(4, _align4(plan.elements * plan.element_bytes))
    if snapshot + snapshot_size > MAX_DATA_ADDRESS:
        raise OverflowError('layout snapshot exceeds signed 32-bit address budget')

    config = [addresses[plan.source.buffer] + plan.source.offset_bytes,
              addresses[plan.destination.buffer] + plan.destination.offset_bytes,
              rank, plan.elements, plan.element_bytes, snapshot]
    config.extend(plan.destination.shape)
    config.extend(plan.source_strides_bytes)
    config.extend(plan.destination.strides_bytes)
    image = DRAMWriter()
    for i, value in enumerate(config):
        image.u32(CONFIG_ADDRESS + 4 * i, value & 0xFFFFFFFF)
    for name, data in buffers.items():
        image.write_bytes(addresses[name], data)
    # Reserve scratch in the image, so shared runner stack placement cannot alias it.
    image.write_zeros(snapshot, snapshot_size)
    emission = LayerEmission()
    emission.c_source = (Path(__file__).resolve().parents[1] / 'kernels' / 'layout.c').read_text()
    emission.dram = image
    emission.output_addr = addresses[plan.destination.buffer] + plan.destination.offset_bytes
    emission.output_shape = plan.destination.shape
    emission.output_elements = plan.elements
    emission.layer_metrics = {'map_kind': op.name, 'execution_kind': 'scalar_byte_movement'}
    return LayoutEmission(emission, plan, addresses, sizes)
