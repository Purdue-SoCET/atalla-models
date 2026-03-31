"""AtallaC softmax kernel: exp(x - max) / sum(exp(x - max)) via rcp.bf."""

import math
from kernels.common import ADDR_TABLE, sdma_ctl_expr


def softmax_c(length: int) -> str:
    """Generate AtallaC for softmax using rcp.bf for reciprocal."""
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
        '    asm("scpad_ld %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));\n'
        "\n"
        "    int row = 0;\n"
        f"    while (row < {rows}) {{\n"
        "        vec v;\n"
        f'        asm("vreg_ld %0, %1, %2, {w_m1}, 0" : "=v"(v) : "r"(row), "r"(ncols));\n'
        "\n"
        '        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);\n'
        '        vec shifted = vec_op_masked("-", v, vmax, mask_val);\n'
        '        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);\n'
        '        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);\n'
        "\n"
        "        float sum_f = sum_v[0];\n"
        "        int sum_bits;\n"
        "        int inv_bits;\n"
        '        asm("stbf_s %0, %1, %2" : "=r"(sum_bits) : "r"(sum_f), "r"(0));\n'
        '        asm("rcp_bf %0, %1, %2" : "=r"(inv_bits) : "r"(sum_bits), "r"(0));\n'
        "        float inv_sum;\n"
        '        asm("bfts_s %0, %1, %2" : "=r"(inv_sum) : "r"(inv_bits), "r"(0));\n'
        '        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);\n'
        "\n"
        f'        asm("vreg_st %0, %1, %2, {w_m1}, 0" : : "v"(result), "r"(row), "r"(ncols));\n'
        "        row = row + 1;\n"
        "    }\n"
        "\n"
        '    asm("scpad_st %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));\n'
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
