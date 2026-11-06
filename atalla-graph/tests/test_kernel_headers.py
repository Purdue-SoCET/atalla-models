"""Shared C helpers must compile from a work directory outside the source tree."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.c_emitter import LayerEmission, compile_and_assemble, render_in_file
from run_graph import _run_emulator


def test_depthwise_header_works_in_isolated_compile_directory(tmp_path):
    emission = LayerEmission()
    emission.c_source = (Path(__file__).parent/'fixtures/depthwise_helpers.c').read_text()
    for address, value in [(0x1000, 0xaabbccdd), (0x1004, 0), (0x1008, 0), (0x100c, 0)]:
        emission.dram.u32(address, value)
    compile_and_assemble(emission, str(tmp_path), 'helpers')
    image = tmp_path/'helpers.in'
    image.write_text(render_in_file(emission))
    memory, _ = _run_emulator(str(image), str(tmp_path), 'helpers')
    assert memory.read_data(0x1000) == 0xaa4040dd
    assert memory.read_data(0x1004) == 0x4100  # BF16: 2 + 2*3 = 8
    assert memory.read_data(0x1008) == 0xffffffff  # modulo32 overflow
    assert memory.read_data(0x100c) == 0x4040
