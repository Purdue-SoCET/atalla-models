/*
 * LayerNorm kernel: (x - mean) / sqrt(var + eps) over an NxN tile.
 *
 * Config at ADDR_TABLE (0x3C):
 *   [0] IN_GMEM   [4] SCPAD_BASE
 *
 * Additional constants in low memory:
 *   addr 20: epsilon (fp32)
 *   addr 24: 1.0 / (N*N) (fp32)
 *
 * Fixed: 4 rows x 32 cols (N=4). Result written back in-place.
 *
 * Algorithm:
 *   1. Row-wise RSUM -> tree-reduce -> mean = total_sum * inv_n2
 *   2. centered = x - mean
 *   3. Row-wise RSUM(centered^2) -> tree-reduce -> var = sum_sq * inv_n2
 *   4. denom = sqrt(var + eps)
 *   5. inv_denom = rcp(denom)
 *   6. result = centered * inv_denom
 *
 * NOTE: ppci currently lacks sqrti_vi and rcp_bf inline asm support,
 *       so this kernel may not compile end-to-end. It serves as the
 *       reference AtallaC implementation matching build_layernorm.py asm.
 */

#define CFG_BASE   0x3C
#define EPS_ADDR   20
#define INV_N2_ADDR 24
#define WIDTH_M1   31
#define ROWS       4
#define COLS       4
#define SID        0
#define RSUM_IMM   64
#define MASK_ALL   0xF

int main() {
    int cfg = CFG_BASE;
    int IN_GMEM;
    int SCPAD_BASE;
    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)    : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(SCPAD_BASE)  : "r"(cfg));

    /* load epsilon and 1/N^2 from data memory */
    int eps_addr = EPS_ADDR;
    float epsilon;
    asm("lw_s %0, 0(%1)" : "=r"(epsilon) : "r"(eps_addr));

    int inv_addr = INV_N2_ADDR;
    float inv_n2;
    asm("lw_s %0, 0(%1)" : "=r"(inv_n2) : "r"(inv_addr));

    int sp = 0;
    int mask_val = MASK_ALL;
    int ncols = 1;
    int sdma_ctl;
    asm("li_s %0, 133169183" : "=r"(sdma_ctl));

    /* load tile from DRAM to scratchpad */
    asm("scpad_ld %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));

    /* load rows into vector registers */
    vec r0, r1, r2, r3;
    int row0 = 0; int row1 = 1; int row2 = 2; int row3 = 3;
    asm("vreg_ld %0, %1, %2, 31, 0" : "=v"(r0) : "r"(row0), "r"(ncols));
    asm("vreg_ld %0, %1, %2, 31, 0" : "=v"(r1) : "r"(row1), "r"(ncols));
    asm("vreg_ld %0, %1, %2, 31, 0" : "=v"(r2) : "r"(row2), "r"(ncols));
    asm("vreg_ld %0, %1, %2, 31, 0" : "=v"(r3) : "r"(row3), "r"(ncols));

    /* --- compute mean --- */
    vec s0 = vec_op_masked("RSUM", r0, 0.0, mask_val);
    vec s1 = vec_op_masked("RSUM", r1, 0.0, mask_val);
    vec s2 = vec_op_masked("RSUM", r2, 0.0, mask_val);
    vec s3 = vec_op_masked("RSUM", r3, 0.0, mask_val);

    vec p01 = vec_op_masked("+", s0, s1, mask_val);
    vec p23 = vec_op_masked("+", s2, s3, mask_val);
    vec total = vec_op_masked("+", p01, p23, mask_val);

    vec mean = vec_op_masked("*", total, inv_n2, mask_val);

    /* --- compute centered = x - mean --- */
    vec c0 = vec_op_masked("-", r0, mean, mask_val);
    vec c1 = vec_op_masked("-", r1, mean, mask_val);
    vec c2 = vec_op_masked("-", r2, mean, mask_val);
    vec c3 = vec_op_masked("-", r3, mean, mask_val);

    /* --- compute variance = sum(centered^2) / N^2 --- */
    vec sq0 = vec_op_masked("*", c0, c0, mask_val);
    vec sq1 = vec_op_masked("*", c1, c1, mask_val);
    vec sq2 = vec_op_masked("*", c2, c2, mask_val);
    vec sq3 = vec_op_masked("*", c3, c3, mask_val);

    vec vs0 = vec_op_masked("RSUM", sq0, 0.0, mask_val);
    vec vs1 = vec_op_masked("RSUM", sq1, 0.0, mask_val);
    vec vs2 = vec_op_masked("RSUM", sq2, 0.0, mask_val);
    vec vs3 = vec_op_masked("RSUM", sq3, 0.0, mask_val);

    vec vp01 = vec_op_masked("+", vs0, vs1, mask_val);
    vec vp23 = vec_op_masked("+", vs2, vs3, mask_val);
    vec var_sum = vec_op_masked("+", vp01, vp23, mask_val);

    vec variance = vec_op_masked("*", var_sum, inv_n2, mask_val);

    /* --- denominator = sqrt(variance + epsilon) --- */
    vec denom_seed = vec_op_masked("+", variance, epsilon, mask_val);
    vec denom = vec_op_masked("SQRT", denom_seed, 0.0, mask_val);

    /* extract scalar denominator, compute reciprocal */
    float denom_f = denom[0];
    int denom_bits;
    int inv_bits;
    asm("stbf_s %0, %1, %2" : "=r"(denom_bits) : "r"(denom_f), "r"(0));
    asm("rcp_bf %0, %1, %2" : "=r"(inv_bits) : "r"(denom_bits), "r"(0));
    float inv_denom;
    asm("bfts_s %0, %1, %2" : "=r"(inv_denom) : "r"(inv_bits), "r"(0));

    /* --- result = centered * inv_denom --- */
    vec out0 = vec_op_masked("*", c0, inv_denom, mask_val);
    vec out1 = vec_op_masked("*", c1, inv_denom, mask_val);
    vec out2 = vec_op_masked("*", c2, inv_denom, mask_val);
    vec out3 = vec_op_masked("*", c3, inv_denom, mask_val);

    /* store normalized rows back to scratchpad */
    asm("vreg_st %0, %1, %2, 31, 0" : : "v"(out0), "r"(row0), "r"(ncols));
    asm("vreg_st %0, %1, %2, 31, 0" : : "v"(out1), "r"(row1), "r"(ncols));
    asm("vreg_st %0, %1, %2, 31, 0" : : "v"(out2), "r"(row2), "r"(ncols));
    asm("vreg_st %0, %1, %2, 31, 0" : : "v"(out3), "r"(row3), "r"(ncols));

    /* store result back to DRAM */
    asm("scpad_st %0, %1, %2" : : "r"(sp), "r"(IN_GMEM), "r"(sdma_ctl));

    asm("halt");
    return 0;
}
