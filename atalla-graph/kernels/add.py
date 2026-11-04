"""AtallaC add kernel: element-wise C = A + B on tiled data."""

import math
from .common import ADDR_TABLE, TILE, sdma_ctl_expr, sdma_ctl_val


def add_c(total: int, width: int = 32) -> str:
    """Generate AtallaC for element-wise vector add."""
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2

    sdma_sp0 = sdma_ctl_expr("sdma_ctl_sp0", 0, sp_rows, width, width)
    sdma_sp1 = sdma_ctl_expr("sdma_ctl_sp1", 1, sp_rows, width, width)

    return (
        "int main() {\n"
        f"    int* cfg_ptr = (int*){ADDR_TABLE};\n"
        "    int A_GMEM = *(cfg_ptr + 0);\n"
        "    int B_GMEM = *(cfg_ptr + 1);\n"
        "    int C_GMEM = *(cfg_ptr + 2);\n"
        "\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_sp0}"
        f"{sdma_sp1}"
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        "        /* Scratchpad id 0 (matches validated add_4x32.c — variable ``sp`` confused regalloc in multi-tile). */\n"
        "        scpad_load(0, A_GMEM, sdma_ctl_sp0);\n"
        "        scpad_load(0, B_GMEM, sdma_ctl_sp1);\n"
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        f"            vec a = vector_load(0, row, {w_m1}, 0);\n"
        f"            vec b = vector_load(0, row, {w_m1}, 1);\n"
        '            vec c = vec_op_masked("+", a, b, all_mask);\n'
        f"            vector_store(c, 0, row, {w_m1}, 0);\n"
        "            row = row + 1;\n"
        "        }\n"
        "\n"
        "        scpad_store(0, C_GMEM, sdma_ctl_sp0);\n"
        f"        A_GMEM = A_GMEM + {tile_bytes};\n"
        f"        B_GMEM = B_GMEM + {tile_bytes};\n"
        f"        C_GMEM = C_GMEM + {tile_bytes};\n"
        "        tile = tile + 1;\n"
        "    }\n"
        "\n"
        "    atalla_halt();\n"
        "    return 0;\n"
        "}\n"
    )
