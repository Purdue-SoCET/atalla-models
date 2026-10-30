/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
int main() {
    int *cfg = (int *)60;
    int A_GMEM;
    int B_GMEM;
    int C_GMEM;
    A_GMEM = *(cfg + 0);
    B_GMEM = *(cfg + 1);
    C_GMEM = *(cfg + 2);

    int all_mask = -1;
    int ncols = 1;
    int tile_count = *(int *)(60 + 12);
    int sp_rows = *(int *)(60 + 16);
    int sdma_ctl_sp0 = *(int *)(60 + 20);
    int sdma_ctl_sp1 = *(int *)(60 + 24);
    int tile_bytes = *(int *)(60 + 28);

    int tile = 0;
    while (tile < tile_count) {
        /* Scratchpad id 0 (matches validated add_4x32.c — variable ``sp`` confused regalloc in multi-tile). */
        scpad_load(0, A_GMEM, sdma_ctl_sp0);
        scpad_load(0, B_GMEM, sdma_ctl_sp1);

        int row = 0;
        while (row < sp_rows) {
            vec a = vector_load(0, row, 31, 0);
            vec b = vector_load(0, row, 31, 1);
            vec c = vec_op_masked("+", a, b, all_mask);
            vector_store(c, 0, row, 31, 0);
            row = row + 1;
        }

        scpad_store(0, C_GMEM, sdma_ctl_sp0);
        A_GMEM = A_GMEM + tile_bytes;
        B_GMEM = B_GMEM + tile_bytes;
        C_GMEM = C_GMEM + tile_bytes;
        tile = tile + 1;
    }

    atalla_halt();
    return 0;
}
