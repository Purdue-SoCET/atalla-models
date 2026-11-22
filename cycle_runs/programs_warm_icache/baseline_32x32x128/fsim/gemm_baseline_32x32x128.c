void load_weight_tile(int sp_w, int w_addr, int sdma_ctl_w) {
    int* cfg_tbl = (int*)60;
    int z_addr = *(cfg_tbl + 10);
    int tile_sz = *(cfg_tbl + 9);
    int sdma_ctl_z = 32505887;
    scpad_load(0, z_addr, sdma_ctl_z);
    vec zrow = vector_load(0, 0, 31, 0);
    scpad_load(sp_w, w_addr, sdma_ctl_w);
    int wn = 32;
    while (wn < tile_sz) {
        vector_store(zrow, sp_w, wn, 31, 1);
        wn = wn + 1;
    }
    int wi = 0;
    while (wi < tile_sz) {
        vec wvec = vector_load(sp_w, wi, 31, 1);
        load_weights(wvec);
        wi = wi + 1;
    }
}

int main() {
    int* cfg = (int*)60;
    int A_GMEM = *(cfg + 0);
    int W_GMEM = *(cfg + 1);
    int C_GMEM = *(cfg + 2);
    int gM = *(cfg + 3);
    int gN = *(cfg + 4);
    int gK = *(cfg + 5);
    int M_tiles = *(cfg + 6);
    int N_tiles = *(cfg + 7);
    int K_tiles = *(cfg + 8);
    int tile_sz = *(cfg + 9);

    int all_mask = -1;
    int sp_a = 0;
    int sp_w = 0;
    int sp_c = 0;
    int sdma_ctl_a = 1072693375;
    int sdma_ctl_w = 2146435199;
    int sdma_ctl_c = 2146435103;

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
                while (ri < 32) {
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
