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
    float k1         = *(float*)(ADDR_TABLE + 24);
    float k3         = *(float*)(ADDR_TABLE + 28);
    float t_c6       = *(float*)(ADDR_TABLE + 32);
    float t_c4       = *(float*)(ADDR_TABLE + 36);
    float t_c2       = *(float*)(ADDR_TABLE + 40);
    float t_c0       = *(float*)(ADDR_TABLE + 44);
    float one        = *(float*)(ADDR_TABLE + 48);
    float neg_one    = *(float*)(ADDR_TABLE + 52);
    float half       = *(float*)(ADDR_TABLE + 56);
    float bound_pos  = *(float*)(ADDR_TABLE + 60);
    float bound_neg  = *(float*)(ADDR_TABLE + 64);

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
            vec x3 = vec_op_masked("*", x2, x, all_mask);

            vec z1 = vec_op_masked("*", x, k1, all_mask);
            vec z2 = vec_op_masked("*", x3, k3, all_mask);
            vec z = vec_op_masked("+", z1, z2, all_mask);

            vec z_sq = vec_op_masked("*", z, z, all_mask);

            /* Horner: ((t_c6 * z_sq + t_c4) * z_sq + t_c2) * z_sq + t_c0 */
            vec p1 = vec_op_masked("*", z_sq, t_c6, all_mask);
            p1 = vec_op_masked("+", p1, t_c4, all_mask);

            vec p2 = vec_op_masked("*", p1, z_sq, all_mask);
            p2 = vec_op_masked("+", p2, t_c2, all_mask);

            vec p3 = vec_op_masked("*", p2, z_sq, all_mask);
            p3 = vec_op_masked("+", p3, t_c0, all_mask);

            vec tanh_val = vec_op_masked("*", z, p3, all_mask);

            /* Clamp tanh to [-1.0, 1.0] */
            int mask_gt = make_mask(">", z, bound_pos, all_mask);
            int mask_lt = make_mask("<", z, bound_neg, all_mask);

            tanh_val = vec_op_masked("+", zero_vec, one, mask_gt, tanh_val);
            tanh_val = vec_op_masked("+", zero_vec, neg_one, mask_lt, tanh_val);

            /* GELU = 0.5 * x * (1.0 + tanh) */
            vec one_plus_tanh = vec_op_masked("+", tanh_val, one, all_mask);
            vec half_x = vec_op_masked("*", x, half, all_mask);
            vec y = vec_op_masked("*", half_x, one_plus_tanh, all_mask);

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
