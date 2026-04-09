"""AtallaC tiled GEMM kernel: C += A × B with 32×32 tiles."""

import math
from kernels.common import ADDR_TABLE, TILE, sdma_ctl_expr


def gemm_c(M: int, N: int, K: int) -> str:
    """Generate AtallaC for tiled GEMM reading config from ADDR_TABLE.

    RHS DRAM matches ``_write_gemm_rhs_weight``: logical ``(K,N)`` weights are
    stored as ``(N,K)`` row-major with each row padded to ``k_stride`` columns
    (multiple of ``TILE``).  ``A`` uses the same inner stride.  SDMA
    ``full_cols`` must match that stride so row-to-row steps in GMEM are correct
    when ``K < TILE``.
    """
    tm1 = min(TILE, M) - 1
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)
    k_stride = int(math.ceil(max(K, 1) / TILE) * TILE)
    sdma_a_s = sdma_ctl_expr("sdma_ctl_a", 0, tile_m, tile_k, k_stride)
    # W' is (N, k_stride) row-major: each GMEM row is one output channel; tile is
    # tile_n channels × tile_k K coefficients (not tile_k × tile_n).
    sdma_w_s = sdma_ctl_expr("sdma_ctl_w", 1, tile_n, tile_k, k_stride)
    sdma_c_s = sdma_ctl_expr("sdma_ctl_c", 1, tile_m, tile_n, N)
    # SP0: one row of TILE zeros for padding partial-N tails in weight SP1.
    sdma_z_s = sdma_ctl_expr("sdma_ctl_z", 0, 1, TILE, TILE)

    return (
        "void load_weight_tile(int sp_w, int w_addr, int sdma_ctl_w) {\n"
        f"    int cfg_tbl = {ADDR_TABLE};\n"
        "    int z_addr;\n"
        '    asm("lw_s %0, 40(%1)" : "=r"(z_addr) : "r"(cfg_tbl));\n'
        "    int tile_sz;\n"
        '    asm("lw_s %0, 36(%1)" : "=r"(tile_sz) : "r"(cfg_tbl));\n'
        f"{sdma_z_s}"
        "    scpad_load(0, z_addr, sdma_ctl_z);\n"
        f"    vec zrow = vector_load(0, 0, {TILE - 1}, 0);\n"
        "    scpad_load(sp_w, w_addr, sdma_ctl_w);\n"
        f"    int wn = {tile_n};\n"
        "    while (wn < tile_sz) {\n"
        f"        vector_store(zrow, sp_w, wn, {tk1}, 1);\n"
        "        wn = wn + 1;\n"
        "    }\n"
        "    int wi = 0;\n"
        "    while (wi < tile_sz) {\n"
        f"        vec wvec = vector_load(sp_w, wi, {tk1}, 1);\n"
        "        load_weights(wvec);\n"
        "        wi = wi + 1;\n"
        "    }\n"
        "}\n"
        "\n"
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
        "                int w_off = ni * tile_sz * gK + ki * tile_sz;\n"
        "                int w_byte = w_off * 2;\n"
        "                int w_addr = W_GMEM + w_byte;\n"
        "\n"
        "                load_weight_tile(sp_w, w_addr, sdma_ctl_w);\n"
        "\n"
        "                scpad_load(sp_a, a_addr, sdma_ctl_a);\n"
        "\n"
        "                int c_off = mi * tile_sz * gN + ni * tile_sz;\n"
        "                int c_byte = c_off * 2;\n"
        "                int c_addr = C_GMEM + c_byte;\n"
        "                scpad_load(sp_c, c_addr, sdma_ctl_c);\n"
        "\n"
        "                int ri = 0;\n"
        f"                while (ri < {tile_m}) {{\n"
        f"                    vec a_row = vector_load(sp_a, ri, {tk1}, 0);\n"
        f"                    vec c_row = vector_load(sp_c, ri, {tn1}, 1);\n"
        "\n"
        "                    vec result = gemm(a_row, c_row, all_mask);\n"
        "\n"
        f"                    vector_store(result, sp_c, ri, {tn1}, 1);\n"
        "                    ri = ri + 1;\n"
        "                }\n"
        "\n"
        "                scpad_store(sp_c, c_addr, sdma_ctl_c);\n"
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
