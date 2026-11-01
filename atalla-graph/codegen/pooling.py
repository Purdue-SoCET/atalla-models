"""Pack geometry and allocations for the standalone ordinary-C pooling kernel."""
from dataclasses import dataclass
from pathlib import Path

from codegen.c_emitter import DRAMWriter, LayerEmission
from codegen.layout_emitter import MAX_DATA_ADDRESS, CONFIG_ADDRESS
from graph.pooling import lower_pooling


@dataclass(frozen=True)
class PoolingEmission:
    emission: LayerEmission
    buffer_addresses: dict
    buffer_sizes: dict


def emit_pooling(op, buffers):
    plan = lower_pooling(op, buffers)
    if any(not isinstance(k, str) or not k for k in buffers):
        raise ValueError('buffer allocation names must be nonempty strings')
    if any(not isinstance(data, (bytes, bytearray)) for data in buffers.values()):
        raise TypeError('buffer allocations must contain bytes or bytearray')
    kernel_name = op.name.removeprefix('linalg.')
    rank = len(op.destination.shape)
    spatial_rank = rank - 2
    config_words = 11 + 3 * rank + 3 * spatial_rank
    cursor = (max(0x1000, CONFIG_ADDRESS + config_words * 4) + 3) & -4
    addresses, sizes = {}, {}
    for name in sorted(buffers):
        addresses[name] = cursor
        sizes[name] = len(buffers[name])
        cursor += max(4, (sizes[name] + 3) & -4)
        if cursor > MAX_DATA_ADDRESS:
            raise OverflowError('pooling buffers exceed address budget')
    snapshot, snapshot_bytes = 0, 0
    if op.source.buffer == op.destination.buffer:
        snapshot = cursor
        snapshot_bytes = max(4, (sizes[op.source.buffer] + 3) & -4)
        cursor += snapshot_bytes
        if cursor > MAX_DATA_ADDRESS:
            raise OverflowError('pooling source snapshot exceeds address budget')
    vector_stage = 0
    if (kernel_name in ('pooling_ncw_max', 'pooling_ncw_sum',
                        'pooling_nchw_max', 'pooling_nchw_sum')
            and plan.dtype == 0 and plan.strides[-1] == 2):
        vector_stage = cursor
        cursor += 64
        if cursor > MAX_DATA_ADDRESS:
            raise OverflowError('pooling vector staging exceeds address budget')
    if cursor + 4 > MAX_DATA_ADDRESS:
        raise OverflowError('pooling image marker exceeds address budget')
    image = DRAMWriter()
    config = [addresses[op.source.buffer], addresses[op.destination.buffer],
              op.source.offset_bytes, op.destination.offset_bytes, plan.elements,
              plan.kernel_elements, plan.dtype, plan.element_bytes,
              snapshot, snapshot_bytes]
    config.extend(op.destination.shape)
    config.extend(op.source.strides_bytes)
    config.extend(op.destination.strides_bytes)
    config.extend(plan.kernel_shape)
    config.extend(plan.strides)
    config.extend(plan.dilations)
    config.append(vector_stage)
    for i, value in enumerate(config):
        image.u32(CONFIG_ADDRESS + 4 * i, value & 0xFFFFFFFF)
    for name, data in buffers.items():
        image.write_bytes(addresses[name], data)
    if snapshot_bytes:
        image.write_zeros(snapshot, snapshot_bytes)
    if vector_stage:
        image.write_zeros(vector_stage, 64)
    # Include the rounded end of every allocation so stack placement is disjoint.
    image.u32(cursor, 0)
    emission = LayerEmission()
    emission.c_source = (Path(__file__).resolve().parents[1] / 'kernels' / f'{kernel_name}.c').read_text()
    emission.dram = image
    emission.output_addr = addresses[op.destination.buffer] + op.destination.offset_bytes
    emission.output_shape = op.destination.shape
    emission.output_elements = plan.elements
    emission.layer_metrics = {
        'map_kind': op.name,
        'execution_kind': ('bf16_vector_or_scalar_pooling'
                           if plan.dtype == 0 else 'scalar_pooling'),
        'kernel_source': kernel_name + '.c',
    }
    return PoolingEmission(emission, addresses, sizes)
