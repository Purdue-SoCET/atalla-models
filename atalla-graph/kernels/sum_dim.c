#define ADDR_TABLE 60
#define WIDTH_M1   31

int main() {
    int* cfg_ptr = (int*)ADDR_TABLE;
    int IN_GMEM        = *(cfg_ptr + 0);
    int OUT_GMEM       = *(cfg_ptr + 1);
    int num_rows       = *(cfg_ptr + 2);
    int col_tiles      = *(cfg_ptr + 3);
    int sdma_ctl_in    = *(cfg_ptr + 4);
    int sdma_ctl_out   = *(cfg_ptr + 5);
    int in_tile_bytes  = *(cfg_ptr + 6);
    int out_tile_bytes = *(cfg_ptr + 7);

    int all_mask = -1;

    int ci = 0;
    while (ci < col_tiles) {
        scpad_load(0, IN_GMEM, sdma_ctl_in);

        vec acc = vector_load(0, 0, WIDTH_M1, 0);
        int r = 1;
        while (r < num_rows) {
            vec v = vector_load(0, r, WIDTH_M1, 0);
            acc = vec_op_masked("+", acc, v, all_mask);
            r = r + 1;
        }

        vector_store(acc, 0, 0, WIDTH_M1, 0);
        scpad_store(0, OUT_GMEM, sdma_ctl_out);

        IN_GMEM = IN_GMEM + in_tile_bytes;
        OUT_GMEM = OUT_GMEM + out_tile_bytes;
        ci = ci + 1;
    }

    atalla_halt();
    return 0;
}
