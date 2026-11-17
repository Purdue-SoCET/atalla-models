#define ADDR_TABLE 60
#define WIDTH_M1   31

int main() {
    int* cfg_ptr = (int*)ADDR_TABLE;
    int GRAD_GMEM  = *(cfg_ptr + 0);
    int X_GMEM     = *(cfg_ptr + 1);
    int OUT_GMEM   = *(cfg_ptr + 2);
    int tile_count = *(cfg_ptr + 3);
    int sp_rows    = *(cfg_ptr + 4);
    int sdma_sp0   = *(cfg_ptr + 5);
    int sdma_sp1   = *(cfg_ptr + 6);
    int tile_bytes = *(cfg_ptr + 7);

    int all_mask = -1;

    vec zero_vec = vector_load(0, 0, WIDTH_M1, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int tile = 0;
    while (tile < tile_count) {
        scpad_load(0, GRAD_GMEM, sdma_sp0);
        scpad_load(0, X_GMEM, sdma_sp1);

        int row = 0;
        while (row < sp_rows) {
            vec grad = vector_load(0, row, WIDTH_M1, 0);
            vec x = vector_load(0, row, WIDTH_M1, 1);
            int m_pos = make_mask(">", x, zero_vec, all_mask);
            vec res = vec_op_masked("+", zero_vec, grad, m_pos);
            vector_store(res, 0, row, WIDTH_M1, 0);
            row = row + 1;
        }

        scpad_store(0, OUT_GMEM, sdma_sp0);
        GRAD_GMEM = GRAD_GMEM + tile_bytes;
        X_GMEM = X_GMEM + tile_bytes;
        OUT_GMEM = OUT_GMEM + tile_bytes;
        tile = tile + 1;
    }

    atalla_halt();
    return 0;
}
