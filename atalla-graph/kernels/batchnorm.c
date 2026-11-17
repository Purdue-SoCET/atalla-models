#define ADDR_TABLE 60
#define WIDTH_M1   31

int main() {
    int* cfg_ptr       = (int*)ADDR_TABLE;
    int IN_GMEM        = *(cfg_ptr + 0);
    int OUT_GMEM       = *(cfg_ptr + 1);
    int SCALE_GMEM     = *(cfg_ptr + 2);
    int BIAS_GMEM      = *(cfg_ptr + 3);
    int channels       = *(cfg_ptr + 4);
    int sp_rows        = *(cfg_ptr + 5);
    int tile_count     = *(cfg_ptr + 6);
    int sdma_sp0       = *(cfg_ptr + 7);
    int tile_bytes     = *(cfg_ptr + 8);
    int ch_bytes       = *(cfg_ptr + 9);

    int sp = 0;
    int all_mask = -1;

    int ch = 0;
    while (ch < channels) {
        float* A_ptr = (float*)(SCALE_GMEM + ch * 4);
        float A = *A_ptr;
        float* B_ptr = (float*)(BIAS_GMEM + ch * 4);
        float B = *B_ptr;

        int ch_in  = IN_GMEM  + ch * ch_bytes;
        int ch_out = OUT_GMEM + ch * ch_bytes;

        int tile = 0;
        while (tile < tile_count) {
            int in_ptr  = ch_in  + tile * tile_bytes;
            int out_ptr = ch_out + tile * tile_bytes;

            scpad_load(sp, in_ptr, sdma_sp0);

            int row = 0;
            while (row < sp_rows) {
                vec x = vector_load(sp, row, WIDTH_M1, 0);
                vec ax = vec_op_masked("*", x, A, all_mask);
                vec y = vec_op_masked("+", ax, B, all_mask);
                vector_store(y, sp, row, WIDTH_M1, 0);
                row = row + 1;
            }

            scpad_store(sp, out_ptr, sdma_sp0);
            tile = tile + 1;
        }

        ch = ch + 1;
    }

    atalla_halt();
    return 0;
}
