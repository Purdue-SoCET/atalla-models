/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
void load_weight_tile(int sp_w, int w_addr, int sdma_ctl_w) {
    int *cfg_tbl = (int *)60;
    int z_addr;
    z_addr = *(cfg_tbl + 10);
    int tile_sz;
    tile_sz = *(cfg_tbl + 9);
    int sdma_ctl_z = *(int *)(60 + 44);
    int tile_n = *(int *)(60 + 48);
    int k_mask = *(int *)(60 + 52);
    scpad_load(0, z_addr, sdma_ctl_z);
    vec zrow = vector_load(0, 0, 31, 0);
    scpad_load(sp_w, w_addr, sdma_ctl_w);
    int wn = tile_n;
    while (wn < tile_sz) {
        vector_store(zrow, sp_w, wn, 31, 1);
        wn = wn + 1;
    }
    int wi = 0;
    while (wi < tile_sz) {
        vec wvec = vector_load(sp_w, wi, 31, 1);
        vec clean_w = vec_op_masked("+", wvec, zrow, k_mask, zrow);
        load_weights(clean_w);
        wi = wi + 1;
    }
}

int main() {
    int *cfg = (int *)60;
    int A_GMEM; int W_GMEM; int C_GMEM;
    int gM; int gN; int gK;
    int M_tiles; int N_tiles; int K_tiles; int tile_sz;

    A_GMEM = *(cfg + 0);
    W_GMEM = *(cfg + 1);
    C_GMEM = *(cfg + 2);
    gM = *(cfg + 3);
    gN = *(cfg + 4);
    gK = *(cfg + 5);
    M_tiles = *(cfg + 6);
    N_tiles = *(cfg + 7);
    K_tiles = *(cfg + 8);
    tile_sz = *(cfg + 9);

    int all_mask = -1;
    int sp_a = 0;
    int sp_w = 0;
    int sp_c = 0;
    int tile_m = *(int *)(60 + 56);
    int sdma_ctl_a = *(int *)(60 + 60);
    int sdma_ctl_w = *(int *)(60 + 64);
    int sdma_ctl_c = *(int *)(60 + 68);

    int mi = 0;
    while (mi < M_tiles) {
        int ni = 0;
        while (ni < N_tiles) {
            int ki = 0;
            while (ki < K_tiles) {
                int a_off = mi * tile_sz * gK + ki * tile_sz;
                int a_byte = a_off * 2;
                int a_addr = A_GMEM + a_byte;

                int w_off = ni * tile_sz * gK + ki * tile_sz;
                int w_byte = w_off * 2;
                int w_addr = W_GMEM + w_byte;

                load_weight_tile(sp_w, w_addr, sdma_ctl_w);

                scpad_load(sp_a, a_addr, sdma_ctl_a);

                int c_off = mi * tile_sz * gN + ni * tile_sz;
                int c_byte = c_off * 2;
                int c_addr = C_GMEM + c_byte;
                scpad_load(sp_c, c_addr, sdma_ctl_c);

                int ri = 0;
                while (ri < tile_m) {
                    vec a_row = vector_load(sp_a, ri, 31, 0);
                    vec c_row = vector_load(sp_c, ri, 31, 1);

                    vec result = gemm(a_row, c_row, all_mask);

                    vector_store(result, sp_c, ri, 31, 1);
                    ri = ri + 1;
                }

                scpad_store(sp_c, c_addr, sdma_ctl_c);
                ki = ki + 1;
            }

            ni = ni + 1;
        }
        mi = mi + 1;
    }

    atalla_halt();
    return 0;
}
