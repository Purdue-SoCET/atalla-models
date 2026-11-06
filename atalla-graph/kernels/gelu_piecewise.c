/* BF16 GELU: four centered cubic intervals, with ratio evaluation near zero.
 * Python supplies coefficients/configuration, never computes GELU outputs.
 * Header at byte60: input, output, tile count, rows, DMA, tile bytes, zero page,
 * approximation mode (0=erf reference, 1=tanh reference). Floats at +32: zero.value,
 * one.value, minus one.value, positive infinity; then four rows [lo,hi.value,center.value,c3.value,c2.value,c1.value,c0].
 */
/* Configuration floats occupy FP32 words. Use 4-byte union storage rather
 * than BF16-sized float pointer indexing or adjacent float local slots. */
union FloatBits { unsigned int bits; float value; };

int main() {
    int *cfg = (int *)60;
    int input = cfg[0];
    int output = cfg[1];
    union FloatBits zero;
    zero.bits = cfg[8];
    union FloatBits one;
    one.bits = cfg[9];
    union FloatBits minus_one;
    minus_one.bits = cfg[10];
    union FloatBits infinity;
    infinity.bits = cfg[11];
    union FloatBits half;
    half.bits = cfg[40];
    scpad_load(0, cfg[6], 32505887);
    vec zeros = vector_load(0, 0, 31, 0);
    int tile = 0;
    while (tile < cfg[2]) {
        scpad_load(0, input, cfg[4]);
        int row = 0;
        while (row < cfg[3]) {
            vec x = vector_load(0, row, 31, 0);
            int negative = make_mask("<", x, zeros, -1);
            vec magnitude = vec_op_masked("*", x, minus_one.value, negative, x);
            /* Finite tails: identity for positive x, signed zero for negative x.
             * Multiplication also propagates NaN and the negative-infinity NaN. */
            vec y = vec_op_masked("*", x, zero.value, -1);
            int positive = make_mask(">", x, zeros, -1);
            y = vec_op_masked("*", x, one.value, positive, y);
            if (cfg[7] == 0) {
                /* Match the erf reference's positive-tail overflow ordering. */
                y = vec_op_masked("+", x, x, positive, y);
                y = vec_op_masked("*", y, half.value, positive, y);
                int infinite = make_mask("==", magnitude, infinity.value, -1);
                y = vec_op_masked("*", y, zero.value, infinite, y);
            }
            int segment = 0;
            while (segment < 4) {
                unsigned int *coeff = (unsigned int *)(60 + 48 + segment * 28);
                union FloatBits lo;
                lo.bits = coeff[0];
                union FloatBits hi;
                hi.bits = coeff[1];
                union FloatBits center;
                center.bits = coeff[2];
                union FloatBits c3;
                c3.bits = coeff[3];
                union FloatBits c2;
                c2.bits = coeff[4];
                union FloatBits c1;
                c1.bits = coeff[5];
                union FloatBits c0;
                c0.bits = coeff[6];
                int below_hi = make_mask("<", magnitude, hi.value, -1);
                int below_lo = make_mask("<", magnitude, lo.value, -1);
                int active = below_hi & ~below_lo;
                vec t = vec_op_masked("-", magnitude, center.value, active, zeros);
                vec p = vec_op_masked("*", t, c3.value, active, zeros);
                p = vec_op_masked("+", p, c2.value, active, zeros);
                p = vec_op_masked("*", p, t, active, zeros);
                p = vec_op_masked("+", p, c1.value, active, zeros);
                p = vec_op_masked("*", p, t, active, zeros);
                p = vec_op_masked("+", p, c0.value, active, zeros);
                vec value;
                if (segment == 0) {
                    /* p=Phi(abs(x)); x*Phi(x) avoids cancellation near zero. */
                    vec factor = vec_op_masked("-", p, one.value, active & negative, p);
                    value = vec_op_masked("*", magnitude, factor, active, zeros);
                } else {
                    /* p=abs(x)*(1-Phi(abs(x))); avoid negative-tail cancellation. */
                    value = vec_op_masked("-", x, p, active, zeros);
                    value = vec_op_masked("*", p, minus_one.value, active & negative, value);
                }
                y = vec_op_masked("*", value, one.value, active, y);
                segment = segment + 1;
            }
            vector_store(y, 0, row, 31, 0);
            row = row + 1;
        }
        scpad_store(0, output, cfg[4]);
        input = input + cfg[5];
        output = output + cfg[5];
        tile = tile + 1;
    }
    atalla_halt();
    return 0;
}
