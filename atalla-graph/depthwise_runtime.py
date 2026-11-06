"""Compile dedicated depthwise kernels and read actual execution buffers."""
from dataclasses import dataclass
from pathlib import Path

from codegen.c_emitter import compile_and_assemble, render_in_file
from codegen.depthwise import emit_depthwise


@dataclass(frozen=True)
class DepthwiseResult:
    buffers: dict[str, bytes]
    backend: str = 'emulator'


def execute_depthwise(op, buffers, work_dir, tag='depthwise'):
    from run_graph import _run_emulator
    if not tag or Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('tag must be a single artifact filename component')
    packed = emit_depthwise(op, buffers)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    compile_and_assemble(packed.emission, str(work_dir), tag)
    image = work_dir / (tag + '.in')
    image.write_text(render_in_file(packed.emission))
    memory, _ = _run_emulator(str(image), str(work_dir), tag)
    values = {name: bytes((memory.read_data((address+i)&-4) >> (8*((address+i)&3))) & 255
                           for i in range(packed.buffer_sizes[name]))
              for name, address in packed.buffer_addresses.items()}
    return DepthwiseResult(values)
