"""Execute normalized layout operations using current allocation bytes."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from codegen.c_emitter import compile_and_assemble, render_in_file
from codegen.layout_emitter import emit_layout
from graph.layout import LayoutOp


@dataclass(frozen=True)
class LayoutResult:
    buffers: dict[str, bytes]
    backend: str = 'emulator'


def execute_layout(op: LayoutOp, buffers: Mapping[str, bytes], work_dir,
                   tag: str = 'layout') -> LayoutResult:
    """Compile, simulate, and return complete allocations from actual memory.

    Use the returned buffers as inputs to the next operation. No reference or
    host execution path exists here; source snapshotting happens in the C kernel.
    """
    # Reuse the existing register/stack setup without duplicating runtime policy.
    from run_graph import _run_emulator

    if not tag or Path(tag).name != tag or tag in ('.', '..'):
        raise ValueError('tag must be a single artifact filename component')
    packed = emit_layout(op, buffers)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    compile_and_assemble(packed.emission, str(work_dir), tag)
    image = work_dir / f'{tag}.in'
    image.write_text(render_in_file(packed.emission))
    memory, _ = _run_emulator(str(image), str(work_dir), tag)
    result = {}
    for name, address in packed.buffer_addresses.items():
        length = packed.buffer_sizes[name]
        result[name] = bytes((memory.read_data((address + i) & -4)
                              >> (8 * ((address + i) & 3))) & 255 for i in range(length))
    return LayoutResult(result)
