"""Pack only geometry/allocations for dedicated ordinary-C depthwise kernels."""
from dataclasses import dataclass
from pathlib import Path

from codegen.c_emitter import DRAMWriter, LayerEmission
from codegen.layout_emitter import CONFIG_ADDRESS, MAX_DATA_ADDRESS
from graph.depthwise import lower_depthwise


@dataclass(frozen=True)
class DepthwiseEmission:
    emission: LayerEmission
    buffer_addresses: dict
    buffer_sizes: dict


def emit_depthwise(op, buffers):
    plan = lower_depthwise(op, buffers)
    if any(not isinstance(name, str) or not name for name in buffers):
        raise ValueError('buffer allocation names must be nonempty strings')
    if any(not isinstance(data, (bytes, bytearray)) for data in buffers.values()):
        raise TypeError('buffer allocations must contain bytes or bytearray')
    source, weights, output = op.source, op.weights, op.destination
    words = 14 + 2 * len(output.shape) + len(source.shape) + len(weights.shape) + 3 * len(plan.kernel_shape)
    cursor = (max(0x1000, CONFIG_ADDRESS + words * 4) + 3) & -4
    addresses, sizes = {}, {}
    for name in sorted(buffers):
        addresses[name], sizes[name] = cursor, len(buffers[name])
        cursor += max(4, (sizes[name] + 3) & -4)
        if cursor > MAX_DATA_ADDRESS:
            raise OverflowError('depthwise buffers exceed address budget')
    snapshots = {}
    for view in (source, weights):
        if view.buffer == output.buffer and view.buffer not in snapshots:
            size = max(4, (sizes[view.buffer] + 3) & -4)
            snapshots[view.buffer] = cursor, size
            cursor += size
            if cursor > MAX_DATA_ADDRESS:
                raise OverflowError('depthwise snapshots exceed address budget')
    if cursor + 4 > MAX_DATA_ADDRESS:
        raise OverflowError('depthwise image marker exceeds address budget')
    config = [addresses[source.buffer], addresses[weights.buffer], addresses[output.buffer],
              source.offset_bytes, weights.offset_bytes, output.offset_bytes,
              plan.elements, plan.kernel_elements, plan.dtype, plan.element_bytes,
              *snapshots.get(source.buffer, (0, 0)), *snapshots.get(weights.buffer, (0, 0))]
    config.extend(output.shape)
    config.extend(source.strides_bytes)
    config.extend(weights.strides_bytes)
    config.extend(output.strides_bytes)
    config.extend(plan.kernel_shape)
    config.extend(plan.strides)
    config.extend(plan.dilations)
    image = DRAMWriter()
    for i, value in enumerate(config):
        image.u32(CONFIG_ADDRESS + 4 * i, value & 0xffffffff)
    for name, data in buffers.items():
        image.write_bytes(addresses[name], data)
    for address, size in snapshots.values():
        image.write_zeros(address, size)
    image.u32(cursor, 0)
    name = op.name.removeprefix('linalg.')
    emission = LayerEmission()
    emission.c_source = (Path(__file__).resolve().parents[1] / 'kernels' / (name + '.c')).read_text()
    emission.dram = image
    emission.output_addr = addresses[output.buffer] + output.offset_bytes
    emission.output_shape = output.shape
    emission.output_elements = plan.elements
    emission.layer_metrics = {'map_kind': op.name, 'kernel_source': name + '.c',
                              'execution_kind': ('bf16_vector_or_scalar_depthwise'
                                                 if plan.dtype == 0 else 'scalar_depthwise')}
    return DepthwiseEmission(emission, addresses, sizes)
