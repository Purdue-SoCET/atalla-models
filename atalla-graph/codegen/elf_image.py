"""Compile one AtallaC kernel and its DRAM image into a linked Atalla ELF.

The compiler owns scheduling: PPCI compiles, packetizes and links the kernel
together with its initialized data section, so the ELF is the single program
artifact. atalla-sim loads the ELF directly. The functional simulator loads an
``.in`` image written from the same ELF's packets and data words, so both
simulators execute identical packets.

Memory layout (byte addresses):

    0x00000000  startup packets the loaders add (jal main; halt.s)
    0x00001004  code  packetized kernel, 20-byte packets (PPCI aligns code to 5)
    0x00080000  data  runtime configuration words (``CONFIG_ADDRESS``, read by
                      kernels through ``ELF_CONFIG`` in kernels/elf_config.h),
    0x00080400        then tensors and scratch buffers

PPCI's ELF writer marks every section except ``data`` executable, so the
configuration lives at the start of the data section rather than at the
``.in`` flow's address 60.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import shutil
import os
from pathlib import Path
import sys

CODE_BASE = 0x1000
CODE_SIZE = 0x40000
DATA_BASE = 0x80000
CONFIG_ADDRESS = DATA_BASE
CONFIG_BYTES = 1024

_GRAPH = Path(__file__).resolve().parents[1]
_MODELS = _GRAPH.parent
KERNEL_DIR = _GRAPH / 'kernels'


def compiler_root() -> Path:
    env = os.environ.get('ATALLA_COMPILER_PATH')
    for cand in ([Path(env)] if env else []) + [_MODELS / 'aihw-ppci-compiler']:
        if (cand / 'ppci').is_dir():
            return cand
    raise FileNotFoundError('aihw-ppci-compiler checkout not found; set ATALLA_COMPILER_PATH')


def _ppci():
    root = str(compiler_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ppci import api
    return api


@dataclass
class DataImage:
    """Little-endian data section: configuration words, then placed buffers."""
    config: list = field(default_factory=list)          # u32 words at CONFIG_ADDRESS
    data: bytearray = field(default_factory=bytearray)  # bytes at DATA_BASE + CONFIG_BYTES

    def place(self, payload: bytes, align: int = 64) -> int:
        """Append ``payload`` after the configuration; returns its byte address."""
        pad = (-len(self.data)) % align
        self.data.extend(b'\0' * pad)
        address = DATA_BASE + CONFIG_BYTES + len(self.data)
        self.data.extend(payload)
        return address

    def reserve(self, size: int, align: int = 64) -> int:
        return self.place(b'\0' * size, align)

    def section_bytes(self) -> bytes:
        cfg = b''.join((int(w) & 0xffffffff).to_bytes(4, 'little') for w in self.config)
        if len(cfg) > CONFIG_BYTES:
            raise OverflowError('runtime configuration exceeds CONFIG_BYTES')
        return cfg + b'\0' * (CONFIG_BYTES - len(cfg)) + bytes(self.data)


def compile_object(source: Path, output: Path, level: str = 'safe') -> None:
    """Compile and packetize one C file into a PPCI object file."""
    api = _ppci()
    from ppci.arch.atalla.registers import get_register
    from ppci.lang.atalla_c.options import COptions
    arch = api.get_arch('atalla')
    # Same callee-save contract as the atalla-inference runner: every register
    # a caller assumes survives a call is saved by the callee.
    arch.callee_save = tuple(get_register(i) for i in (9, 11, *range(18, 28)))
    options = COptions()
    options.add_include_path(str(KERNEL_DIR))
    with Path(source).open() as stream:
        module = api.atalla_c_to_ir(stream, arch, coptions=options)
    if level == 'safe':
        # The O2 sequence without CleanPass, whose phi-edge rewrite miscompiles
        # values merged after a conditional assignment (see README).
        from ppci.irutils import verify_module
        passes = [api.Mem2RegPromotor(), api.RemoveAddZeroPass(), api.ConstantFolder(),
                  api.CommonSubexpressionEliminationPass(), api.TailCallOptimization(),
                  api.LoadAfterStorePass(), api.DeleteUnusedInstructionsPass()]
        verify_module(module)
        for _ in range(3):
            for optimization in passes:
                optimization.run(module)
        verify_module(module)
    else:
        api.optimize(module, level=level)
    obj = api.ir_to_object([module], arch, packetize=True)
    with Path(output).open('w') as stream:
        obj.save(stream)


def _object_key(source: Path, level: str) -> str:
    h = hashlib.sha256(level.encode())
    h.update(Path(source).read_bytes())
    for header in sorted(KERNEL_DIR.glob('*.h')):
        h.update(header.name.encode())
        h.update(header.read_bytes())
    h.update(str(compiler_root().resolve()).encode())
    return h.hexdigest()[:24]


def kernel_object(source: Path, out_dir: Path, tag: str, level: str = 'safe') -> Path:
    """The packetized object for ``source``. With ATALLA_OBJ_CACHE set, objects
    are reused across images while the source, kernel headers and level match."""
    obj = Path(out_dir) / (tag + '.o')
    cache = os.environ.get('ATALLA_OBJ_CACHE')
    if not cache:
        compile_object(source, obj, level)
        return obj
    cached = Path(cache) / f'{Path(source).stem}-{_object_key(source, level)}.o'
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(f'.{os.getpid()}.tmp')
        compile_object(source, tmp, level)
        tmp.replace(cached)
    shutil.copyfile(cached, obj)
    return obj


def link_elf(kernel_source: Path, image: DataImage, out_dir: Path, tag: str,
             level: str = 'safe') -> Path:
    """Compile ``kernel_source`` and link it with ``image``; returns the ELF path."""
    api = _ppci()
    from ppci.binutils.objectfile import ObjectFile, Section
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_obj = kernel_object(kernel_source, out_dir, tag, level)

    memory = ObjectFile(api.get_arch('atalla'))
    payload = image.section_bytes()
    section = Section('data')
    section.add_data(payload)
    memory.add_section(section)
    image_obj = out_dir / (tag + '_image.o')
    with image_obj.open('w') as stream:
        memory.save(stream)

    data_size = (len(payload) + 0x1000 + 0xFFF) & ~0xFFF
    layout = (f'MEMORY code LOCATION=0x{CODE_BASE:X} SIZE=0x{CODE_SIZE:X} {{ SECTION(code) }}\n'
              f'MEMORY data LOCATION=0x{DATA_BASE:X} SIZE=0x{data_size:X} {{ SECTION(data) }}\n')
    linked = api.link([str(image_obj), str(kernel_obj)], layout=io.StringIO(layout), entry='main')
    if linked.get_undefined_symbols():
        raise RuntimeError(f'undefined symbols: {linked.get_undefined_symbols()}')
    code = linked.get_section('code')
    # PPCI aligns code to 5 bytes, so it starts at 0x1004; packets are 20 bytes.
    if code.size % 20:
        raise RuntimeError(f'code section is {code.size} bytes, not whole 20-byte packets')
    if code.address + code.size > CODE_BASE + CODE_SIZE:
        raise OverflowError('kernel code exceeds its region')
    elf = out_dir / (tag + '.elf')
    api.objcopy(linked, None, 'elf', str(elf))
    return elf
