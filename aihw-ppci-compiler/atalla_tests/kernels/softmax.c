/*
 * Softmax kernel: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * Config at ADDR_TABLE (0x3C):
 *   [0] IN_GMEM    [4] unused
 *
 * Fixed: 1 row x 32 elements. Result written back in-place.
 * Uses rmax, exp, rsum reductions, then 1.0/sum (compiles to rcp_bf+mul_bf).
 */

#define CFG_BASE  0x3C
#define WIDTH_M1  31
#define ROWS      1
#define ROWS_M1   0
#define MASK_VAL  0xFFFFFFFF

int main() {
    int cfg = CFG_BASE;
    int IN_GMEM;
    int dummy;
    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM) : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(dummy)   : "r"(cfg));

    int sp = 0;
    int mask_val = MASK_VAL;
    int ncols = 1;
    int sdma_ctl;
    asm("li_s %0, 32505887" : "=r"(sdma_ctl));

    /* load input from DRAM */
    scpad_load(sp, IN_GMEM, sdma_ctl);

    int row = 0;
    while (row < ROWS) {
        vec v = vector_load(row, ncols, 31, 0);

        /* subtract max for numerical stability */
        vec vmax = vec_op_masked("RMAX", v, 0.0, mask_val);
        vec shifted = vec_op_masked("-", v, vmax, mask_val);

        /* exponentiate */
        vec exp_v = vec_op_masked("EXP", shifted, 0.0, mask_val);

        /* sum reduction */
        vec sum_v = vec_op_masked("RSUM", exp_v, 0.0, mask_val);

        /* scalar reciprocal: 1.0/x compiles to rcp_bf + mul_bf */
        float sum_f = sum_v[0];
        float inv_sum = 1.0 / sum_f;

        vec result = vec_op_masked("*", exp_v, inv_sum, mask_val);

        vector_store(result, row, ncols, 31, 0);
        row = row + 1;
    }

    /* store result back in-place */
    scpad_store(sp, IN_GMEM, sdma_ctl);

    asm("halt");
    return 0;
}
