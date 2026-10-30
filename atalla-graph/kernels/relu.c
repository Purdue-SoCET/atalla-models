/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
int main() {
    int *cfg = (int *)60;
    int IN_GMEM;
    int OUT_GMEM;
    IN_GMEM = *(cfg + 0);
    OUT_GMEM = *(cfg + 1);

    int sp = 0;
    int all_mask = -1;
    int ncols = 1;
    int tile_count = *(int *)(60 + 8);
    int sp_rows = *(int *)(60 + 12);
    int sdma_ctl = *(int *)(60 + 16);
    int tile_bytes = *(int *)(60 + 20);

    vec zero_vec = vector_load(sp, 0, 31, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int tile = 0;
    while (tile < tile_count) {
        scpad_load(sp, IN_GMEM, sdma_ctl);

        int row = 0;
        while (row < sp_rows) {
            vec v = vector_load(sp, row, 31, 0);

            int m_neg = make_mask("<", v, zero_vec, all_mask);
            vec result = vec_op_masked("*", v, 0.0, m_neg);

            vector_store(result, sp, row, 31, 0);
            row = row + 1;
        }

        scpad_store(sp, OUT_GMEM, sdma_ctl);
        IN_GMEM = IN_GMEM + tile_bytes;
        OUT_GMEM = OUT_GMEM + tile_bytes;
        tile = tile + 1;
    }

    atalla_halt();
    return 0;
}
