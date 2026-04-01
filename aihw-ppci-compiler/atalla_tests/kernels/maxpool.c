/*
 * MaxPool2D kernel: per-channel 2x2 pool, stride 2, on 8x8 spatial tile.
 *
 * Config at ADDR_TABLE (0x3C):
 *   [0] IN_BASE    [4] OUT_BASE
 *
 * Fixed: 1 channel, 8x8 input -> 4x8 output (vertical max only).
 * Horizontal stride-select + pool-width max done in post-processing
 * because ppci lacks shift.vi / vmov.vts / stbf.s inline asm support.
 *
 * Strategy: load pool_size adjacent input rows as vectors, pairwise
 * vertical max via make_mask(">") + vec_op_masked blend, store result row.
 */

#define CFG_BASE  0x3C
#define WIDTH_M1  7
#define H_IN      8
#define H_OUT     4
#define CHANNELS  1
#define STRIDE    2
#define POOL      2
#define IN_CH_BYTES   (8 * 8 * 2)
#define OUT_CH_BYTES  (4 * 8 * 2)

int main() {
    int cfg = CFG_BASE;
    int IN_BASE;
    int OUT_BASE;
    asm("lw_s %0, 0(%1)" : "=r"(IN_BASE)  : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(OUT_BASE) : "r"(cfg));

    int sp = 0;
    int all_mask = -1;
    int ncols = 1;

    /* sdma_ctl for 8x8 input tile on SP0: (sid=0, rows=8, cols=8, full=8) */
    int sdma_in;
    asm("li_s %0, 242221063" : "=r"(sdma_in));

    /* sdma_ctl for 4x8 output tile on SP0: (sid=0, rows=4, cols=8, full=8) */
    int sdma_out;
    asm("li_s %0, 108003335" : "=r"(sdma_out));

    /* zero vector for blend */
    vec zero_vec = vector_load(0, ncols, 7, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int ch = 0;
    while (ch < CHANNELS) {
        int in_ptr = IN_BASE + ch * IN_CH_BYTES;
        int out_ptr = OUT_BASE + ch * OUT_CH_BYTES;

        /* load input spatial tile */
        scpad_load(sp, in_ptr, sdma_in);

        int oh = 0;
        while (oh < H_OUT) {
            int in_row = oh * STRIDE;

            /* load first row of pool window */
            vec best = vector_load(in_row, ncols, 7, 0);

            /* load second row, vertical max */
            int r1 = in_row + 1;
            vec v1 = vector_load(r1, ncols, 7, 0);
            int gt1 = make_mask(">", v1, best, all_mask);
            best = vec_op_masked("+", v1, zero_vec, gt1);

            /* store vertically-maxed row */
            vector_store(best, oh, ncols, 7, 0);
            oh = oh + 1;
        }

        /* store output tile back to DRAM */
        scpad_store(sp, out_ptr, sdma_out);
        ch = ch + 1;
    }

    asm("halt");
    return 0;
}
