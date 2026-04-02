int main() {
    int cfg = 60;
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
    int ncols = 1;
    int sp_a = 0;
    int sp_w = 0;
    int sp_c = 0;
    int sdma_ctl_a;
    asm("li_s %0, 1067450394" : "=r"(sdma_ctl_a));
    int sdma_ctl_w;
    asm("li_s %0, 1947205633" : "=r"(sdma_ctl_w));
    int sdma_ctl_c;
    asm("li_s %0, 2114977793" : "=r"(sdma_ctl_c));

    int mi = 0;
    while (mi < M_tiles) {
        int ni = 0;
        while (ni < N_tiles) {
            int ki = 0;
            while (ki < K_tiles) {
                int a_off = mi * tile_sz * gK + ki * tile_sz;
                int a_byte = a_off * 2;
                int a_addr = A_GMEM + a_byte;

                int w_off = ki * tile_sz * gN + ni * tile_sz;
                int w_byte = w_off * 2;
                int w_addr = W_GMEM + w_byte;

                scpad_load(sp_w, w_addr, sdma_ctl_w);

                int wi = 0;
                while (wi < 27) {
                    int w_row = sp_w + wi;
                    vec wvec = vector_load(w_row, ncols, 1, 1);
                    load_weights(wvec);
                    wi = wi + 1;
                }

                scpad_load(sp_a, a_addr, sdma_ctl_a);

                int c_off = mi * tile_sz * gN + ni * tile_sz;
                int c_byte = c_off * 2;
                int c_addr = C_GMEM + c_byte;
                scpad_load(sp_c, c_addr, sdma_ctl_c);

                int ri = 0;
                while (ri < 32) {
                    vec a_row = vector_load(ri, ncols, 26, 0);
                    vec c_row = vector_load(ri, ncols, 1, 1);

                    vec result = gemm(a_row, c_row, all_mask);

                    vector_store(result, ri, ncols, 1, 1);
                    ri = ri + 1;
                }

                scpad_store(sp_c, c_addr, sdma_ctl_c);
                ki = ki + 1;
            }

            ni = ni + 1;
        }
        mi = mi + 1;
    }

    asm("halt");
    return 0;
}
