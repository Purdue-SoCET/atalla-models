#define ADDR_TABLE 60
#define WIDTH_M1   31

int main() {
    int* cfg_ptr     = (int*)ADDR_TABLE;
    int X_GMEM       = *(cfg_ptr + 0);
    int Y_GMEM       = *(cfg_ptr + 1);
    int OUT_GMEM     = *(cfg_ptr + 2);
    int tile_count   = *(cfg_ptr + 3);
    int sp_rows      = *(cfg_ptr + 4);
    int sdma_sp0     = *(cfg_ptr + 5);
    int sdma_sp1     = *(cfg_ptr + 6);
    int tile_bytes   = *(cfg_ptr + 7);
    float* fcfg_ptr  = (float*)(ADDR_TABLE + 32);
    float scale      = *fcfg_ptr;
    int sdma_out     = *(cfg_ptr + 9);

    int all_mask = -1;

    vec zero_vec = vector_load(0, 0, WIDTH_M1, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);
    vec sum_vec = zero_vec;

    int tile = 0;
    while (tile < tile_count) {
        scpad_load(0, X_GMEM, sdma_sp0);
        scpad_load(0, Y_GMEM, sdma_sp1);

        int row = 0;
        while (row < sp_rows) {
            vec x = vector_load(0, row, WIDTH_M1, 0);
            vec y = vector_load(0, row, WIDTH_M1, 1);
            vec diff = vec_op_masked("-", x, y, all_mask);
            vec sq = vec_op_masked("*", diff, diff, all_mask);
            sum_vec = vec_op_masked("+", sum_vec, sq, all_mask);
            row = row + 1;
        }

        X_GMEM = X_GMEM + tile_bytes;
        Y_GMEM = Y_GMEM + tile_bytes;
        tile = tile + 1;
    }

    vec rsum = vec_op_masked("RSUM", sum_vec, 0.0, all_mask);
    vec loss = vec_op_masked("*", rsum, scale, all_mask);
    vector_store(loss, 0, 0, WIDTH_M1, 0);
    scpad_store(0, OUT_GMEM, sdma_out);

    atalla_halt();
    return 0;
}
