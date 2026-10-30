/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
int main() {
    int *cfg = (int *)60;
    int IN_BASE;
    int OUT_BASE;
    IN_BASE = *(cfg + 0);
    OUT_BASE = *(cfg + 1);

    int sp = 0;
    int all_mask = -1;
    int ncols = 1;
    int C = *(int *)(60 + 8);
    int H_out = *(int *)(60 + 12);
    int pool = *(int *)(60 + 16);
    int stride = *(int *)(60 + 20);
    int in_ch_bytes = *(int *)(60 + 24);
    int out_ch_bytes = *(int *)(60 + 28);
    int sdma_in = *(int *)(60 + 32);
    int sdma_out = *(int *)(60 + 36);

    vec zero_vec = vector_load(sp, 0, 31, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int ch = 0;
    while (ch < C) {
        int in_ptr = IN_BASE + ch * in_ch_bytes;
        int out_ptr = OUT_BASE + ch * out_ch_bytes;
        scpad_load(sp, in_ptr, sdma_in);

        int oh = 0;
        while (oh < H_out) {
            int in_row = oh * stride;
            vec best = vector_load(sp, in_row, 31, 0);
            int p = 1;
            while (p < pool) {
                int r = in_row + p;
                vec v = vector_load(sp, r, 31, 0);
                int gt = make_mask(">", v, best, all_mask);
                best = vec_op_masked("+", zero_vec, v, gt, best);
                p = p + 1;
            }
            vector_store(best, sp, oh, 31, 0);
            oh = oh + 1;
        }

        scpad_store(sp, out_ptr, sdma_out);
        ch = ch + 1;
    }

    atalla_halt();
    return 0;
}
