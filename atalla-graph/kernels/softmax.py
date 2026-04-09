"""AtallaC softmax: exp(x - max) / sum(exp(x - max)) via 1.0/sum."""

import math
from kernels.common import ADDR_TABLE, sdma_ctl_expr


def softmax_c_batched(num_rows: int, row_len: int) -> str:
    """Softmax independently along each of ``num_rows`` rows of ``row_len`` elements (row-major)."""
    if row_len > 32:
        raise ValueError("softmax_c_batched requires row_len <= 32")
    width = row_len
    w_m1 = width - 1
    mask_val = -1 if width == 32 else (1 << width) - 1
    sdma_s = sdma_ctl_expr("sdma_ctl", 0, 1, width, width)
    stride_b = row_len * 2

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int dummy;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM) : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(dummy)   : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        f"    int mask_val = {mask_val};\n"
        f"{sdma_s}"
        "\n"
        f"    int batch = 0;\n"
        f"    while (batch < {num_rows}) {{\n"
        f"        int row_base = IN_GMEM + batch * {stride_b};\n"
        "        scpad_load(sp, row_base, sdma_ctl);\n"
        "\n"
        f"        vec v = vector_load(sp, 0, {w_m1}, 0);\n"
        '        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);\n'
        '        vec shifted = vec_op_masked("-", v, vmax, mask_val);\n'
        '        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);\n'
        '        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);\n'
        "\n"
        "        float sum_f = sum_v[0];\n"
        "        float inv_sum = 1.0 / sum_f;\n"
        '        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);\n'
        "\n"
        f"        vector_store(result, sp, 0, {w_m1}, 0);\n"
        "        scpad_store(sp, row_base, sdma_ctl);\n"
        "\n"
        "        batch = batch + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )


def softmax_c(length: int) -> str:
    """Single 1D softmax of ``length`` elements; uses vector tiling when length > 32."""
    if length <= 32:
        return softmax_c_batched(1, length)
    width = min(length, 32)
    rows = math.ceil(length / 32)
    w_m1 = width - 1
    mask_val = -1 if width == 32 else (1 << width) - 1
    sdma_s = sdma_ctl_expr("sdma_ctl", 0, rows, width, width)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN_GMEM;\n"
        "    int dummy;\n"
        '    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM) : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)" : "=r"(dummy)   : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        f"    int mask_val = {mask_val};\n"
        "    int ncols = 1;\n"
        f"{sdma_s}"
        "\n"
        "    scpad_load(sp, IN_GMEM, sdma_ctl);\n"
        "\n"
        "    int row = 0;\n"
        f"    while (row < {rows}) {{\n"
        f"        vec v = vector_load(sp, row, {w_m1}, 0);\n"
        "\n"
        '        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);\n'
        '        vec shifted = vec_op_masked("-", v, vmax, mask_val);\n'
        '        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);\n'
        '        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);\n'
        "\n"
        "        float sum_f = sum_v[0];\n"
        "        float inv_sum = 1.0 / sum_f;\n"
        '        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);\n'
        "\n"
        f"        vector_store(result, sp, row, {w_m1}, 0);\n"
        "        row = row + 1;\n"
        "    }\n"
        "\n"
        "    scpad_store(sp, IN_GMEM, sdma_ctl);\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
