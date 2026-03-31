"""AtallaC tiled GEMM kernel: C += A × B with 32×32 tiles."""

import math
from kernels.common import ADDR_TABLE, TILE, sdma_ctl_expr


def gemm_c(M: int, N: int, K: int) -> str:
    """Generate AtallaC for tiled GEMM reading config from ADDR_TABLE."""
    tm1 = min(TILE, M) - 1
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)
    sdma_a_s = sdma_ctl_expr("sdma_ctl_a", 0, tile_m, tile_k, K)
    sdma_w_s = sdma_ctl_expr("sdma_ctl_w", 1, tile_k, tile_n, N)
    sdma_c_s = sdma_ctl_expr("sdma_ctl_c", 1, tile_m, tile_n, N)

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int A_GMEM; int W_GMEM; int C_GMEM;\n"
        "    int gM; int gN; int gK;\n"
        "    int M_tiles; int N_tiles; int K_tiles; int tile_sz;\n"
        "\n"
        '    asm("lw_s %0, 0(%1)"  : "=r"(A_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)"  : "=r"(W_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 8(%1)"  : "=r"(C_GMEM)  : "r"(cfg));\n'
        '    asm("lw_s %0, 12(%1)" : "=r"(gM)      : "r"(cfg));\n'
        '    asm("lw_s %0, 16(%1)" : "=r"(gN)      : "r"(cfg));\n'
        '    asm("lw_s %0, 20(%1)" : "=r"(gK)      : "r"(cfg));\n'
        '    asm("lw_s %0, 24(%1)" : "=r"(M_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 28(%1)" : "=r"(N_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 32(%1)" : "=r"(K_tiles) : "r"(cfg));\n'
        '    asm("lw_s %0, 36(%1)" : "=r"(tile_sz) : "r"(cfg));\n'
        "\n"
        "    int all_mask = -1;\n"
        "    int ncols = 1;\n"
        "    int sp_a = 0;\n"
        "    int sp_w = 0;\n"
        "    int sp_c = 0;\n"
        f"{sdma_a_s}"
        f"{sdma_w_s}"
        f"{sdma_c_s}"
        "\n"
        "    int mi = 0;\n"
        "    while (mi < M_tiles) {\n"
        "        int ni = 0;\n"
        "        while (ni < N_tiles) {\n"
        "            int ki = 0;\n"
        "            while (ki < K_tiles) {\n"
        "                int a_off = mi * tile_sz * gK + ki * tile_sz;\n"
        "                int a_byte = a_off * 2;\n"
        "                int a_addr = A_GMEM + a_byte;\n"
        "\n"
        "                int w_off = ki * tile_sz * gN + ni * tile_sz;\n"
        "                int w_byte = w_off * 2;\n"
        "                int w_addr = W_GMEM + w_byte;\n"
        "\n"
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_w), "r"(w_addr), "r"(sdma_ctl_w));\n'
        "\n"
        "                int wi = 0;\n"
        f"                while (wi < {tile_k}) {{\n"
        "                    vec wvec;\n"
        "                    int w_row = sp_w + wi;\n"
        f'                    asm("vreg_ld %0, %1, %2, {tn1}, 1"\n'
        '                        : "=v"(wvec) : "r"(w_row), "r"(ncols));\n'
        '                    asm("lw_vi %0, %1, 0, m0" : "=v"(wvec) : "v"(wvec));\n'
        "                    wi = wi + 1;\n"
        "                }\n"
        "\n"
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_a), "r"(a_addr), "r"(sdma_ctl_a));\n'
        "\n"
        "                int c_off = mi * tile_sz * gN + ni * tile_sz;\n"
        "                int c_byte = c_off * 2;\n"
        "                int c_addr = C_GMEM + c_byte;\n"
        '                asm("scpad_ld %0, %1, %2" : : "r"(sp_c), "r"(c_addr), "r"(sdma_ctl_c));\n'
        "\n"
        "                int ri = 0;\n"
        f"                while (ri < {tile_m}) {{\n"
        "                    vec a_row;\n"
        "                    vec c_row;\n"
        f'                    asm("vreg_ld %0, %1, %2, {tk1}, 0"\n'
        '                        : "=v"(a_row) : "r"(ri), "r"(ncols));\n'
        f'                    asm("vreg_ld %0, %1, %2, {tn1}, 1"\n'
        '                        : "=v"(c_row) : "r"(ri), "r"(ncols));\n'
        "\n"
        "                    vec result = gemm(a_row, c_row, all_mask);\n"
        "\n"
        f'                    asm("vreg_st %0, %1, %2, {tn1}, 1"\n'
        '                        : : "v"(result), "r"(ri), "r"(ncols));\n'
        "                    ri = ri + 1;\n"
        "                }\n"
        "\n"
        '                asm("scpad_st %0, %1, %2" : : "r"(sp_c), "r"(c_addr), "r"(sdma_ctl_c));\n'
        "                ki = ki + 1;\n"
        "            }\n"
        "\n"
        "            ni = ni + 1;\n"
        "        }\n"
        "        mi = mi + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
