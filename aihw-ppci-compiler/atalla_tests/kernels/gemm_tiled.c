/*
 * Tiled GEMM kernel: C += A * W using systolic array.
 *
 * Config at ADDR_TABLE (0x3C):
 *   [0]  A_GMEM    [4]  W_GMEM    [8]  C_GMEM
 *   [12] M         [16] N         [20] K
 *   [24] M_tiles   [28] N_tiles   [32] K_tiles   [36] tile_sz
 *
 * Each tile is TILE x TILE (4x4 for this test).
 * Loads A, W tiles from DRAM into scratchpad, preloads W into systolic array,
 * runs GEMM row by row, stores C tile back.
 */

#define CFG_BASE  0x3C
#define TILE      4
#define TILE_M1   3

int main() {
    int cfg = CFG_BASE;
    int A_GMEM; int W_GMEM; int C_GMEM;
    int gM; int gN; int gK;
    int M_tiles; int N_tiles; int K_tiles; int tile_sz;

    asm("lw_s %0, 0(%1)"  : "=r"(A_GMEM)  : "r"(cfg));
    asm("lw_s %0, 4(%1)"  : "=r"(W_GMEM)  : "r"(cfg));
    asm("lw_s %0, 8(%1)"  : "=r"(C_GMEM)  : "r"(cfg));
    asm("lw_s %0, 12(%1)" : "=r"(gM)      : "r"(cfg));
    asm("lw_s %0, 16(%1)" : "=r"(gN)      : "r"(cfg));
    asm("lw_s %0, 20(%1)" : "=r"(gK)      : "r"(cfg));
    asm("lw_s %0, 24(%1)" : "=r"(M_tiles) : "r"(cfg));
    asm("lw_s %0, 28(%1)" : "=r"(N_tiles) : "r"(cfg));
    asm("lw_s %0, 32(%1)" : "=r"(K_tiles) : "r"(cfg));
    asm("lw_s %0, 36(%1)" : "=r"(tile_sz) : "r"(cfg));

    int all_mask = -1;
    int sp_base = 0;
    int ncols = 1;
    int sdma_ctl_sp0;
    asm("li_s %0, 103809027" : "=r"(sdma_ctl_sp0));
    int sdma_ctl_sp1;
    asm("li_s %0, 1177550851" : "=r"(sdma_ctl_sp1));

    int mi = 0;
    while (mi < M_tiles) {
        int ni = 0;
        while (ni < N_tiles) {
            int c_off = mi * tile_sz * gN + ni * tile_sz;
            int c_byte = c_off * 2;
            int c_addr = C_GMEM + c_byte;

            /* load C tile accumulator into SP1 */
            scpad_load(sp_base, c_addr, sdma_ctl_sp1);

            int ki = 0;
            while (ki < K_tiles) {
                int a_off = mi * tile_sz * gK + ki * tile_sz;
                int a_byte = a_off * 2;
                int a_addr = A_GMEM + a_byte;

                int w_off = ki * tile_sz * gN + ni * tile_sz;
                int w_byte = w_off * 2;
                int w_addr = W_GMEM + w_byte;

                /* load W tile into SP0, preload into systolic array */
                scpad_load(sp_base, w_addr, sdma_ctl_sp0);

                int wi = 0;
                while (wi < TILE) {
                    vec wvec = vector_load(wi, ncols, 3, 0);
                    load_weights(wvec);
                    wi = wi + 1;
                }

                /* load A tile into SP0 (safe: W already in systolic array) */
                scpad_load(sp_base, a_addr, sdma_ctl_sp0);

                /* GEMM: A from SP0, C accumulator from SP1 */
                int ri = 0;
                while (ri < TILE) {
                    vec a_row = vector_load(ri, ncols, 3, 0);
                    vec c_row = vector_load(ri, ncols, 3, 1);

                    vec result = gemm(a_row, c_row, all_mask);

                    vector_store(result, ri, ncols, 3, 1);
                    ri = ri + 1;
                }

                ki = ki + 1;
            }

            /* store C tile from SP1 back to DRAM */
            scpad_store(sp_base, c_addr, sdma_ctl_sp1);
            ni = ni + 1;
        }
        mi = mi + 1;
    }

    asm("halt");
    return 0;
}
