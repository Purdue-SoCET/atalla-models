"""AtallaC ReLU kernel: element-wise max(x, 0) via masked zero."""

import math
from kernels.common import ADDR_TABLE, TILE, sdma_ctl_expr


def relu_c(total: int, width: int) -> str:
    """Generate AtallaC for ReLU."""
    rows = math.ceil(total / width)
    w_m1 = width - 1
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    tile_bytes = sp_rows * width * 2
    sdma_s = sdma_ctl_expr("sdma_ctl", 0, sp_rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int OUT_GMEM;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(OUT_GMEM) : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        f"    vec zero_vec = vector_load(sp, 0, {w_m1}, 0);\n"
        '    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);\n'
        "\n"
        "    int tile = 0;\n"
        f"    while (tile < {tile_count}) {{\n"
        "        scpad_load(sp, IN_GMEM, sdma_ctl);\n"
        "\n"
        "        int row = 0;\n"
        f"        while (row < {sp_rows}) {{\n"
        f"            vec v = vector_load(sp, row, {w_m1}, 0);\n"
        "\n"
        '            int m_neg = make_mask("<", v, zero_vec, all_mask);\n'
        '            vec result = vec_op_masked("*", v, 0.0, m_neg);\n'
        "\n"
        f"            vector_store(result, sp, row, {w_m1}, 0);\n"
        "            row = row + 1;\n"
        "        }\n"
        "\n"
        "        scpad_store(sp, OUT_GMEM, sdma_ctl);\n"
        f"        IN_GMEM = IN_GMEM + {tile_bytes};\n"
        f"        OUT_GMEM = OUT_GMEM + {tile_bytes};\n"
        "        tile = tile + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
