from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graph.export_fx import render_graph
from graph.lower_modules import lower_linear_modules
from graph.memoryallocator import allocate_memory, fake_allocate_memory
from model.alexnet import build_alexnet
from scripts.generate_schedule import emit


def main() -> None:
    module = build_alexnet()
    example_input = torch.ones((1, 3, 224, 224), dtype=torch.bfloat16)

    module = module.bfloat16()

    gm = symbolic_trace(module)
    gm = lower_linear_modules(gm)

    ShapeProp(gm).propagate(example_input)

    placeholder_data = {"x": example_input.clone()}  # needed if using allocate_memory

    #gm = fake_allocate_memory(gm)
    gm = allocate_memory(gm, "model/images/dram.bin", placeholder_data)
    render_graph(gm, "model/images/graph.svg", meta_keys=("dram_addr", "bytes"))

    Path("graph_schedule.c").write_text(emit(gm))


if __name__ == "__main__":
    main()
