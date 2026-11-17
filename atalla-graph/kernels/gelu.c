#define ADDR_TABLE 60
#define WIDTH_M1   31

int main() {
    int* cfg_ptr     = (int*)ADDR_TABLE;
    int IN_GMEM      = *(cfg_ptr + 0);
    int OUT_GMEM     = *(cfg_ptr + 1);
    int tile_count   = *(cfg_ptr + 2);
    int sp_rows      = *(cfg_ptr + 3);
    int sdma_sp0     = *(cfg_ptr + 4);
    int tile_bytes   = *(cfg_ptr + 5);

    float* fcfg_ptr  = (float*)ADDR_TABLE;
    float c6         = *(float*)(ADDR_TABLE + 24);
    float c4         = *(float*)(ADDR_TABLE + 28);
    float c2         = *(float*)(ADDR_TABLE + 32);
    float c0         = *(float*)(ADDR_TABLE + 36);
    float c_half     = *(float*)(ADDR_TABLE + 40);
    float bound_pos  = *(float*)(ADDR_TABLE + 44);
    float bound_neg  = *(float*)(ADDR_TABLE + 48);

    int sp = 0;
    int all_mask = -1;

    vec zero_vec = vector_load(sp, 0, WIDTH_M1, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int tile = 0;
    while (tile < tile_count) {
        scpad_load(sp, IN_GMEM, sdma_sp0);

        int row = 0;
        while (row < sp_rows) {
            vec x = vector_load(sp, row, WIDTH_M1, 0);

            vec x2 = vec_op_masked("*", x, x, all_mask);

            /* Horner evaluation: ((c6 * x2 + c4) * x2 + c2) * x2 + c0 */
            vec t1 = vec_op_masked("*", x2, c6, all_mask);
            t1 = vec_op_masked("+", t1, c4, all_mask);

            vec t2 = vec_op_masked("*", t1, x2, all_mask);
            t2 = vec_op_masked("+", t2, c2, all_mask);

            vec t3 = vec_op_masked("*", t2, x2, all_mask);
            t3 = vec_op_masked("+", t3, c0, all_mask);

            vec half_x = vec_op_masked("*", x, c_half, all_mask);
            vec y = vec_op_masked("+", t3, half_x, all_mask);

            /* Boundary clamping: x > 3.0 -> x, x < -3.0 -> 0.0 */
            int mask_gt = make_mask(">", x, bound_pos, all_mask);
            int mask_lt = make_mask("<", x, bound_neg, all_mask);

            y = vec_op_masked("+", zero_vec, x, mask_gt, y);
            y = vec_op_masked("*", zero_vec, 0.0, mask_lt, y);

            vector_store(y, sp, row, WIDTH_M1, 0);
            row = row + 1;
        }

        scpad_store(sp, OUT_GMEM, sdma_sp0);
        IN_GMEM = IN_GMEM + tile_bytes;
        OUT_GMEM = OUT_GMEM + tile_bytes;
        tile = tile + 1;
    }

    atalla_halt();
    return 0;
}
