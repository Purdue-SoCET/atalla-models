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

    int grp = 0;
    while (grp < M) {
        int base = grp * D * 2;
        float sum = 0.0;
        int rr = 0;
        while (rr < nr) {
            scpad_load(sp, IN0 + base + rr * 64, sdma_row);
            vec v = vector_load(sp, 0, 31, 0);
            vec sv = vec_op_masked("RSUM", v, 0.0, -1);
            sum = sum + sv[0];
            rr = rr + 1;
        }
        float mean = sum * inv_d;

        float var_acc = 0.0;
        rr = 0;
        while (rr < nr) {
            scpad_load(sp, IN0 + base + rr * 64, sdma_row);
            vec v = vector_load(sp, 0, 31, 0);
            vec c = vec_op_masked("-", v, mean, -1);
            vec sq = vec_op_masked("*", c, c, -1);
            vec sv2 = vec_op_masked("RSUM", sq, 0.0, -1);
            var_acc = var_acc + sv2[0];
            rr = rr + 1;
        }
        float var = var_acc * inv_d + eps;
        float inv_std = 1.0 / sqrt(var);

        rr = 0;
        while (rr < nr) {
            scpad_load(sp, IN0 + base + rr * 64, sdma_row);
            vec v = vector_load(sp, 0, 31, 0);
            vec c = vec_op_masked("-", v, mean, -1);
            vec n = vec_op_masked("*", c, inv_std, -1);
            scpad_load(sp, GM + rr * 64, sdma_row);
            vec g = vector_load(sp, 0, 31, 0);
            vec ng = vec_op_masked("*", n, g, -1);
            scpad_load(sp, BM + rr * 64, sdma_row);
            vec bvec = vector_load(sp, 0, 31, 0);
            vec o = vec_op_masked("+", ng, bvec, -1);
            vector_store(o, sp, 0, 31, 0);
            scpad_store(sp, OUT0 + base + rr * 64, sdma_row);
            rr = rr + 1;
        }
        grp = grp + 1;
    }

    atalla_halt();
    return 0;
}
