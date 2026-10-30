/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
int main() {
    int *cfg = (int *)60;
    int IN_GMEM;
    int dummy;
    IN_GMEM = *(cfg + 0);
    dummy = *(cfg + 1);

    int sp = 0;
    int mask_val = *(int *)(60 + 20);
    int num_rows = *(int *)(60 + 8);
    int stride_b = *(int *)(60 + 12);
    int sdma_ctl = *(int *)(60 + 16);

    int batch = 0;
    while (batch < num_rows) {
        int row_base = IN_GMEM + batch * stride_b;
        scpad_load(sp, row_base, sdma_ctl);

        vec v = vector_load(sp, 0, 31, 0);
        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);
        vec shifted = vec_op_masked("-", v, vmax, mask_val);
        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);
        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);

        float sum_f = sum_v[0];
        float inv_sum = 1.0 / sum_f;
        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);

        vector_store(result, sp, 0, 31, 0);
        scpad_store(sp, row_base, sdma_ctl);

        batch = batch + 1;
    }

    atalla_halt();
    return 0;
}
