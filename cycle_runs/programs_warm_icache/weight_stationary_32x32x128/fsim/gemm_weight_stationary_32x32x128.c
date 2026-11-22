void fetch_weight_tile(int sp_s, int w_addr, int sdma_ctl_w) {
    scpad_load(sp_s, w_addr, sdma_ctl_w);
}

void push_weights(int sp_s, int tile_sz) {
    int wi = 0;
    while (wi < tile_sz) {
        vec w0 = vector_load(sp_s, wi, 31, 1);
        load_weights(w0);
        int wi1 = wi + 1;
        vec w1 = vector_load(sp_s, wi1, 31, 1);
        load_weights(w1);
        wi = wi + 2;
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
    int sp_c = 0;
    int sp_s = 0;
    int k_step = tile_sz * 2;
    int a_mstep = tile_sz * gK * 2;
    int c_mstep = tile_sz * gN * 2;
    int sdma_ctl_a = 2146435199;
    int sdma_ctl_w = 2146435199;
    int sdma_ctl_c = 1072693279;

    int ni = 0;
    while (ni < N_tiles) {
            int w_off = ni * tile_sz * gK;
            int w_byte = w_off * 2;
            int w_addr = W_GMEM + w_byte;
            int c_noff = ni * tile_sz * 2;
            int c_naddr = C_GMEM + c_noff;
            int a_kaddr = A_GMEM;

            int ki = 0;
            while (ki < K_tiles) {
                fetch_weight_tile(sp_s, w_addr, sdma_ctl_w);
                push_weights(sp_s, tile_sz);

                int a_addr = a_kaddr;
                int c_addr = c_naddr;
                int mi = 0;
                while (mi < M_tiles) {
                    scpad_load(sp_c, c_addr, sdma_ctl_c);
                    scpad_load(sp_s, a_addr, sdma_ctl_a);

                    int ri = 0;
                    while (ri < 32) {
                        vec a_row = vector_load(sp_s, ri, 31, 1);
                        vec c_row = vector_load(sp_c, ri, 31, 0);

                        vec result = gemm(a_row, c_row, all_mask);

                        vector_store(result, sp_c, ri, 31, 0);
                        ri = ri + 1;
                    }

                    scpad_store(sp_c, c_addr, sdma_ctl_c);
                    a_addr = a_addr + a_mstep;
                    c_addr = c_addr + c_mstep;
                    mi = mi + 1;
                }

                a_kaddr = a_kaddr + k_step;
                w_addr = w_addr + k_step;
                ki = ki + 1;
            }

        ni = ni + 1;
    }

    atalla_halt();
    return 0;
}
