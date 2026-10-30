/* Runtime configuration ABI: see kernels/common.py and codegen/c_emitter.py.
 * Vector widths and scratchpad IDs are fixed ISA immediates. */
int main() {
    int *cfg = (int *)60;
    int IN0;
    int OUT0;
    int GM;
    int BM;
    int M;
    IN0 = *(cfg + 0);
    OUT0 = *(cfg + 1);
    GM = *(cfg + 2);
    BM = *(cfg + 3);
    M = *(cfg + 4);

    int sp = 0;
    float inv_d = *(float *)(60 + 24);
    float eps = *(float *)(60 + 28);
    int D = *(int *)(60 + 20);
    int nr = *(int *)(60 + 32);
    int sdma_row = *(int *)(60 + 36);

    /* Two affine vectors share SP1 when they fit. Keep them resident across
     * groups; SP0 then holds the input and output for all three passes. */
    int resident = nr <= 16;
    int resident_ctl = sdma_row | ((nr - 1) << 25);
    if (resident) {
        scpad_load(0, GM, resident_ctl | 1073741824);
        scpad_load(nr * 64, BM, resident_ctl | 1073741824);
    }
    int grp = 0;
    while (grp < M) {
        int base = grp * D * 2;
        if (resident) { scpad_load(0, IN0 + base, resident_ctl); }
        float sum = 0.0;
        int rr = 0;
        while (rr < nr) {
            int rows = nr - rr;
            if (rows > 32) { rows = 32; }
            int control = sdma_row | ((rows - 1) << 25);
            if (!resident) { scpad_load(0, IN0 + base + rr * 64, control); }
            int row = 0;
            while (row < rows) {
                vec v = vector_load(sp, row, 31, 0);
                vec sv = vec_op_masked("RSUM", v, 0.0, -1);
                sum = sum + sv[0];
                row = row + 1;
            }
            rr = rr + rows;
        }
        float mean = sum * inv_d;

        float var_acc = 0.0;
        rr = 0;
        while (rr < nr) {
            int rows = nr - rr;
            if (rows > 32) { rows = 32; }
            int control = sdma_row | ((rows - 1) << 25);
            if (!resident) { scpad_load(0, IN0 + base + rr * 64, control); }
            int row = 0;
            while (row < rows) {
                vec v = vector_load(sp, row, 31, 0);
                vec c = vec_op_masked("-", v, mean, -1);
                vec sq = vec_op_masked("*", c, c, -1);
                vec sv2 = vec_op_masked("RSUM", sq, 0.0, -1);
                var_acc = var_acc + sv2[0];
                row = row + 1;
            }
            rr = rr + rows;
        }
        float var = var_acc * inv_d + eps;
        float inv_std = 1.0 / sqrt(var);

        rr = 0;
        while (rr < nr) {
            int rows = nr - rr;
            if (rows > 32) { rows = 32; }
            int control = sdma_row | ((rows - 1) << 25);
            if (!resident) {
                scpad_load(0, IN0 + base + rr * 64, control);
                scpad_load(0, GM + rr * 64, control | 1073741824);
            }
            int row = 0;
            while (row < rows) {
                vec v = vector_load(0, row, 31, 0);
                vec c = vec_op_masked("-", v, mean, -1);
                vec n = vec_op_masked("*", c, inv_std, -1);
                vec g = vector_load(0, row, 31, 1);
                vec ng = vec_op_masked("*", n, g, -1);
                vector_store(ng, 0, row, 31, 0);
                row = row + 1;
            }
            if (!resident) { scpad_load(0, BM + rr * 64, control | 1073741824); }
            int bias_base = 0;
            if (resident) { bias_base = nr * 64; }
            row = 0;
            while (row < rows) {
                vec ng = vector_load(0, row, 31, 0);
                vec bvec = vector_load(bias_base, row, 31, 1);
                vec o = vec_op_masked("+", ng, bvec, -1);
                vector_store(o, 0, row, 31, 0);
                row = row + 1;
            }
            scpad_store(0, OUT0 + base + rr * 64, control);
            rr = rr + rows;
        }
        grp = grp + 1;
    }

    atalla_halt();
    return 0;
}
