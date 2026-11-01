"""Compile and execute standalone pooling using actual buffer allocations."""
from dataclasses import dataclass
from pathlib import Path

from codegen.c_emitter import compile_and_assemble, render_in_file
from codegen.pooling import emit_pooling


@dataclass(frozen=True)
class PoolingResult:
    buffers: dict[str, bytes]
    backend: str = 'emulator'


def execute_pooling(op, buffers, work_dir, tag='pooling'):
    from run_graph import _run_emulator

    if not tag or Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('tag must be a single artifact filename component')
    packed = emit_pooling(op, buffers)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    compile_and_assemble(packed.emission, str(work_dir), tag)
    image = work_dir / f'{tag}.in'
    image.write_text(render_in_file(packed.emission))
    memory, _ = _run_emulator(str(image), str(work_dir), tag)
    result = {}
    for name, address in packed.buffer_addresses.items():
        result[name] = bytes((memory.read_data((address + i) & -4)
                             >> (8 * ((address + i) & 3))) & 255
                            for i in range(packed.buffer_sizes[name]))
    return PoolingResult(result)
