/* Shared engine for the named linalg.conv_* kernels (kernels/conv_*.c).
 *
 * graph/conv.py maps each named op onto one canonical loop nest
 *     O[n,g,o0,o1,o2,f] += I[n,g,o0*s0+k0*d0,o1*s1+k1*d1,o2*s2+k2*d2,c] * K[g,f,k0,k1,k2,c]
 * through byte strides, and picks the path; codegen/conv.py packs the
 * configuration words documented there at ELF_CONFIG. The entry file selects
 * which paths are compiled in with CONV_ARRAY / CONV_LANES / CONV_SCALAR.
 *
 * Scratchpads (32 rows x 32 bf16 each in the functional sim):
 *   SP0  weight tile, then activation tile (time-shared within a step)
 *   SP1  output tile, resident across every tap and channel tile
 *
 * Ordering: an SDMA signals completion only by holding its rs1 register, and
 * the compiler may keep the pad base in other registers too, so every SDMA
 * goes through dma_load or the inline tile-store asm, which wait for completion. The data cache
 * is not coherent with SDMA and has no flush, so no scalar store ever
 * produces data a DMA reads.
 */
#ifndef ATALLA_CONV_ENGINE_H
#define ATALLA_CONV_ENGINE_H

#include "elf_config.h"
#include "depthwise_common.h"

#define CFG_PATH 0
#define CFG_DT 1
#define CFG_ZP 4
#define CFG_BASE 6
#define CFG_EXT 9
#define CFG_STEP 19
#define CFG_DIL 22
#define CFG_IS 25
#define CFG_KS 31
#define CFG_OS 37
#define CFG_RED 43
#define CFG_ZERO 48
#define CFG_IDENT 49
#define CFG_NPRE 50
#define CFG_NPOST 51
#define CFG_RESTAGE 52
#define CFG_SNAP 124
#define CFG_SP0 131
#define CFG_SP1 132
#define CFG_FOLD 133
#define SID1 1073741824

int sdma_ctl(int rows, int cols, int pitch) {
    return ((rows - 1) << 25) | ((cols - 1) << 20) | (pitch - 1);
}

/* SDMA, then wait: the addi reads the register the SDMA holds until the
 * transfer completes, so the in-order pipeline stalls here and every later
 * scratchpad access is ordered after it, whatever register the compiler
 * keeps the pad base in (the atalla-inference spatial_tiles.c idiom). */
int dma_load(int sp, int dram, int ctl) {
    int done;
    asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(done) : "r"(dram), "r"(ctl), "r"(sp));
    return done;
}

/* Storing an SP1 tile (written inline below; PPCI's preprocessor mangles asm
 * strings inside macros): the RTL orders an SDMA after earlier vreg.st only
 * when both name the same rs1 register, and the compiler may keep the base in
 * several registers, so the tile's last row (already stored) is stored again
 * on the SDMA's own register in the same asm block; the scratchpad's write
 * queue drains earlier rows first. */

int min32(int a) {
    if (a > 32) { return 32; }
    return a;
}

/* Lanes at or above `count` (0 when the tile is full). */
int tail_mask(int count) {
    if (count >= 32) { return 0; }
    return ~((1 << count) - 1);
}

/* ---------------------------------------------------------------- copies */

void run_snapshots(int *cfg) {
    int k = 0;
    while (k < cfg[CFG_SNAP]) {
        int *s = cfg + CFG_SNAP + 1 + 3 * k;
        int byte = 0;
        while (byte < s[2]) {
            *(int *)(s[1] + byte) = *(int *)(s[0] + byte);
            byte = byte + 4;
        }
        k = k + 1;
    }
}

#ifdef CONV_ARRAY
/* Transpose one tile: rt rows (stride srs bytes) x ct contiguous bf16 columns
 * at src -> ct rows (stride drs) x rt columns at dst. The rows enter the array
 * as weight columns; identity rows then read them back as columns. A gemm on a
 * zero activation flags rows holding NaN/Inf (0*Inf is NaN), which the
 * identity product would smear across their row; each flagged element is put
 * back into the SP1 tile before its DMA store (-0 + x == x for every x), so the
 * transpose is exact and no scalar store ever feeds a DMA. */
void transpose_tile(int sp0, int sp1, int src, int srs, int dst, int drs, int rt, int ct,
                    int zero, int ident) {
    int spz = dma_load(sp1 + 1984, zero, sdma_ctl(1, 32, 32) | SID1);
    sp0 = dma_load(sp0, src, sdma_ctl(rt, ct, srs / 2));
    vec z = vector_load(spz, 0, 31, 1);
    int tail = tail_mask(ct);
    /* Count down without a negative compare: the functional sim's branches
     * compare registers unsigned, the RTL's signed. */
    int r = rt;
    while (r > 0) {
        r = r - 1;
        vec w = vector_load(sp0, r, 31, 0);
        if (tail != 0) { w = vec_op_masked("+", z, z, tail, w); }
        load_weights(w);
    }
    vec chk = gemm(z, z, -1);
    int bad = make_mask("!=", chk, chk, -1);
    if (rt < 32) { bad = bad & ((1 << rt) - 1); }
    sp0 = dma_load(sp0, ident, sdma_ctl(ct, 32, 32));
    int q = 0;
    vec t = z;
    if (bad == 0) {
        while (q < ct) {
            vec e = vector_load(sp0, q, 31, 0);
            t = gemm(e, z, -1);
            vector_store(t, sp1, q, 31, 1);
            q = q + 1;
        }
    } else {
        vec nz = vec_op_masked("*", z, -1082130432, -1);      /* -0.0 */
        while (q < ct) {
            vec e = vector_load(sp0, q, 31, 0);
            t = gemm(e, z, -1);
            int j = 0;
            while (j < rt) {
                if (((bad >> j) & 1) != 0) {
                    int bits = read_half(src + j * srs + q * 2) << 16;
                    t = vec_op_masked("+", nz, bits, 1 << j, t);
                }
                j = j + 1;
            }
            vector_store(t, sp1, q, 31, 1);
            q = q + 1;
        }
    }
    int last_row = ct - 1;
    int dctl = sdma_ctl(ct, rt, drs / 2) | SID1;
    asm("vreg_st %4, %3, %5, 31, 1\nscpad_st %3, %1, %2\naddi_s %0, %3, 0"
        : "=r"(sp1) : "r"(dst), "r"(dctl), "r"(sp1), "v"(t), "r"(last_row));
}

void run_restage(int *d, int sp0, int sp1, int zero, int ident) {
    int b0 = 0;
    while (b0 < d[2]) {
        int b1 = 0;
        while (b1 < d[3]) {
            int b2 = 0;
            while (b2 < d[4]) {
                int b3 = 0;
                while (b3 < d[5]) {
                    int s = d[0] + b0 * d[6] + b1 * d[7] + b2 * d[8] + b3 * d[9];
                    int t = d[1] + b0 * d[10] + b1 * d[11] + b2 * d[12] + b3 * d[13];
                    int r0 = 0;
                    while (r0 < d[14]) {
                        int c0 = 0;
                        while (c0 < d[15]) {
                            transpose_tile(sp0, sp1, s + r0 * d[16] + c0 * 2, d[16],
                                           t + c0 * d[17] + r0 * 2, d[17],
                                           min32(d[14] - r0), min32(d[15] - c0), zero, ident);
                            c0 = c0 + 32;
                        }
                        r0 = r0 + 32;
                    }
                    b3 = b3 + 1;
                }
                b2 = b2 + 1;
            }
            b1 = b1 + 1;
        }
        b0 = b0 + 1;
    }
}

/* Implicit GEMM. A tile is up to 32 output pixels taken in row-major order
 * over (o1, o2), so short rows still fill it, x up to 32 filters, resident in
 * SP1. Each step (tap, reduction tile) loads ft weight columns with lw.vi (last
 * filter first, so column j holds filter j) and runs one gemm.vv per pixel.
 * Pixels that cross an o1 row are moved by one SDMA per row segment.
 *
 * Hot-loop shape: no calls, every DMA waits inline, one bottom-tested loop per
 * level and no masked ops. Reduction lanes past a tail are zero because SP0 is
 * zero-filled by DMA before any tail tile (once per kernel when the whole
 * reduction fits in one tile), so a short DMA leaves zeros, never stale data.
 */
void array_conv(int *cfg, int sp0, int sp1) {
    int *ext = cfg + CFG_EXT;
    int *is = cfg + CFG_IS;
    int *ks = cfg + CFG_KS;
    int *os = cfg + CFG_OS;
    int *st = cfg + CFG_STEP;
    int *dl = cfg + CFG_DIL;
    int C = ext[3];
    int F = ext[2];
    int O2 = ext[6];
    int P = ext[5] * O2;
    int zero = cfg[CFG_ZERO];
    int zctl = (31 << 25) | (31 << 20) | 31;
    int one_tile = C <= 32;
    if (one_tile && C < 32) {
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp0) : "r"(zero), "r"(zctl), "r"(sp0));
    }
    /* With k2 folded into the reduction (kf > 1), neighbouring pixels' windows
     * overlap and one SDMA cannot move them (row pitch must cover the row), so
     * each row segment moves in kf residue classes at kf times the pitch. Output
     * rows use the same order, so gemm row r always pairs with psum row r. */
    int kf = cfg[CFG_FOLD];
    int in_pitch = kf * st[2] * is[4] / 2 - 1;
    int k_pitch = ks[1] / 2 - 1;
    int o_pitch = kf * os[4] / 2 - 1;
    int n = 0;
    do {
        int g = 0;
        do {
            int o0 = 0;
            do {
                int obase = cfg[CFG_BASE + 2] + n * os[0] + g * os[1] + o0 * os[2];
                int ibase0 = cfg[CFG_BASE] + n * is[0] + g * is[1] + o0 * st[0] * is[2];
                int p0 = 0;
                do {
                    int pt = P - p0;
                    if (pt > 32) { pt = 32; }
                    int f0 = 0;
                    do {
                        int ft = F - f0;
                        if (ft > 32) { ft = 32; }
                        int fcols = ((ft - 1) << 20) | SID1;
                        /* Output tile in, one DMA per o1 row segment. */
                        int r = 0;
                        int q = p0;
                        do {
                            int o1 = q / O2;
                            int o2 = q - o1 * O2;
                            int len = O2 - o2;
                            if (len > pt - q + p0) { len = pt - q + p0; }
                            int rho = 0;
                            do {
                                int cnt = (len - rho + kf - 1) / kf;
                                int dst = sp1 + r * 64;
                                int src = obase + o1 * os[3] + (o2 + rho) * os[4] + f0 * 2;
                                int ctl = ((cnt - 1) << 25) | fcols | o_pitch;
                                asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(dst) : "r"(src), "r"(ctl), "r"(dst));
                                r = r + cnt;
                                rho = rho + 1;
                            } while (rho < kf && rho < len);
                            q = q + len;
                        } while (q < p0 + pt);
                        vec last = vector_load(sp1, 0, 31, 1);
                        int k0 = 0;
                        do {
                            int k1 = 0;
                            do {
                                int k2 = 0;
                                do {
                                    int wbase = cfg[CFG_BASE + 1] + g * ks[0] + f0 * ks[1]
                                                + k0 * ks[2] + k1 * ks[3] + k2 * ks[4];
                                    int ibase = ibase0 + k0 * dl[0] * is[2] + k1 * dl[1] * is[3]
                                                + k2 * dl[2] * is[4];
                                    int c0 = 0;
                                    do {
                                        int ct = C - c0;
                                        if (ct > 32) { ct = 32; }
                                        if (ct < 32 && one_tile == 0) {
                                            asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp0) : "r"(zero), "r"(zctl), "r"(sp0));
                                        }
                                        int ccols = (ct - 1) << 20;
                                        int wsrc = wbase + c0 * 2;
                                        int wctl = ((ft - 1) << 25) | ccols | k_pitch;
                                        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp0) : "r"(wsrc), "r"(wctl), "r"(sp0));
                                        int j = ft;
                                        do {
                                            j = j - 1;
                                            load_weights(vector_load(sp0, j, 31, 0));
                                        } while (j > 0);
                                        /* Activations, one DMA per o1 row segment. */
                                        r = 0;
                                        q = p0;
                                        do {
                                            int o1 = q / O2;
                                            int o2 = q - o1 * O2;
                                            int len = O2 - o2;
                                            if (len > pt - q + p0) { len = pt - q + p0; }
                                            int rho = 0;
                                            do {
                                                int cnt = (len - rho + kf - 1) / kf;
                                                int dst = sp0 + r * 64;
                                                int src = ibase + o1 * st[1] * is[3] + (o2 + rho) * st[2] * is[4] + c0 * 2;
                                                int ctl = ((cnt - 1) << 25) | ccols | in_pitch;
                                                asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(dst) : "r"(src), "r"(ctl), "r"(dst));
                                                r = r + cnt;
                                                rho = rho + 1;
                                            } while (rho < kf && rho < len);
                                            q = q + len;
                                        } while (q < p0 + pt);
                                        r = 0;
                                        do {
                                            vec a = vector_load(sp0, r, 31, 0);
                                            vec acc = vector_load(sp1, r, 31, 1);
                                            last = gemm(a, acc, -1);
                                            vector_store(last, sp1, r, 31, 1);
                                            r = r + 1;
                                        } while (r < pt);
                                        c0 = c0 + 32;
                                    } while (c0 < C);
                                    k2 = k2 + 1;
                                } while (k2 < ext[9]);
                                k1 = k1 + 1;
                            } while (k1 < ext[8]);
                            k0 = k0 + 1;
                        } while (k0 < ext[7]);
                        /* Output tile out. The first segment's SDMA re-stores the
                         * tile's last row on its own rs1 so it orders after every
                         * vreg.st; later segments follow the waited SDMA. */
                        r = 0;
                        q = p0;
                        int lrow = pt - 1;
                        do {
                            int o1 = q / O2;
                            int o2 = q - o1 * O2;
                            int len = O2 - o2;
                            if (len > pt - q + p0) { len = pt - q + p0; }
                            int rho = 0;
                            do {
                                int cnt = (len - rho + kf - 1) / kf;
                                int seg = sp1 + r * 64;
                                int dst = obase + o1 * os[3] + (o2 + rho) * os[4] + f0 * 2;
                                int ctl = ((cnt - 1) << 25) | fcols | o_pitch;
                                if (r == 0) {
                                    asm("vreg_st %4, %3, %5, 31, 1\nscpad_st %3, %1, %2\naddi_s %0, %3, 0"
                                        : "=r"(seg) : "r"(dst), "r"(ctl), "r"(seg), "v"(last), "r"(lrow));
                                } else {
                                    asm("scpad_st %3, %1, %2\naddi_s %0, %3, 0" : "=r"(seg) : "r"(dst), "r"(ctl), "r"(seg));
                                }
                                r = r + cnt;
                                rho = rho + 1;
                            } while (rho < kf && rho < len);
                            q = q + len;
                        } while (q < p0 + pt);
                        f0 = f0 + 32;
                    } while (f0 < F);
                    p0 = p0 + pt;
                } while (p0 < P);
                o0 = o0 + 1;
            } while (o0 < ext[4]);
            g = g + 1;
        } while (g < ext[1]);
        n = n + 1;
    } while (n < ext[0]);
}
#endif

#ifdef CONV_LANES
/* Channel-free bf16 conv: 32 outputs along o2 per vector. Each tap DMAs its
 * shifted input window into SP0 and costs one mul.vs (scalar = filter bf16 as
 * fp32 bits) and one add.vv, in Linalg order, so every product and sum rounds
 * to bf16 as specified. */
void lanes_conv(int *cfg, int sp0, int sp1) {
    int *ext = cfg + CFG_EXT;
    int *is = cfg + CFG_IS;
    int *ks = cfg + CFG_KS;
    int *os = cfg + CFG_OS;
    int K2 = ext[9];
    int o0 = 0;
    while (o0 < ext[4]) {
        int o1 = 0;
        while (o1 < ext[5]) {
            int p0 = 0;
            while (p0 < ext[6]) {
                int pt = min32(ext[6] - p0);
                int out = cfg[CFG_BASE + 2] + o0 * os[2] + o1 * os[3] + p0 * 2;
                int out_ctl = sdma_ctl(1, pt, pt) | SID1;
                int in_ctl = sdma_ctl(1, pt, pt);
                sp1 = dma_load(sp1, out, out_ctl);
                vec acc = vector_load(sp1, 0, 31, 1);
                int k0 = 0;
                while (k0 < ext[7]) {
                    int k1 = 0;
                    while (k1 < ext[8]) {
                        int row = cfg[CFG_BASE] + (o0 + k0) * is[2] + (o1 + k1) * is[3] + p0 * 2;
                        int kb = cfg[CFG_BASE + 1] + k0 * ks[2] + k1 * ks[3];
                        int q = 0;
                        while (q < K2) {
                            int coeff = read_half(kb + q * ks[4]) << 16;
                            sp0 = dma_load(sp0, row + q * 2, in_ctl);
                            vec v = vector_load(sp0, 0, 31, 0);
                            vec prod = vec_op_masked("*", v, coeff, -1);
                            acc = vec_op_masked("+", acc, prod, -1);
                            q = q + 1;
                        }
                        k1 = k1 + 1;
                    }
                    k0 = k0 + 1;
                }
                int row0 = 0;
                vector_store(acc, sp1, 0, 31, 1);
                asm("vreg_st %4, %3, %5, 31, 1\nscpad_st %3, %1, %2\naddi_s %0, %3, 0"
                    : "=r"(sp1) : "r"(out), "r"(out_ctl), "r"(sp1), "v"(acc), "r"(row0));
                p0 = p0 + pt;
            }
            o1 = o1 + 1;
        }
        o0 = o0 + 1;
    }
}
#endif

#ifdef CONV_SCALAR
int element_width(int code) {
    if (code == 1) { return 1; }
    if (code == 3) { return 4; }
    return 2;
}

/* Integer element, sign-extended; bf16 is returned as raw bits. */
int load_value(int address, int code) {
    unsigned int bits = read_element(address, element_width(code));
    if (code == 1) { return (int)(bits << 24) >> 24; }
    if (code == 2) { return (int)(bits << 16) >> 16; }
    return bits;
}

/* Every operand, layout and type: scalar loops, reduction in Linalg order. */
void scalar_conv(int *cfg) {
    int *ext = cfg + CFG_EXT;
    int *is = cfg + CFG_IS;
    int *ks = cfg + CFG_KS;
    int *os = cfg + CFG_OS;
    int *st = cfg + CFG_STEP;
    int *dl = cfg + CFG_DIL;
    int dti = cfg[CFG_DT];
    int dtk = cfg[CFG_DT + 1];
    int float_math = dti == 0;
    int width = element_width(cfg[CFG_DT + 2]);
    int izp = cfg[CFG_ZP];
    int kzp = cfg[CFG_ZP + 1];
    /* Reduction axes in Linalg order; sizes and per-step byte strides. */
    int rs0 = 1; int rs1 = 1; int rs2 = 1; int rs3 = 1;
    int ri0 = 0; int ri1 = 0; int ri2 = 0; int ri3 = 0;
    int rk0 = 0; int rk1 = 0; int rk2 = 0; int rk3 = 0;
    int a = 0;
    while (a < cfg[CFG_RED]) {
        int code = cfg[CFG_RED + 1 + a];
        int size = ext[3];
        int istr = is[5];
        int kstr = ks[5];
        if (code != 0) {
            size = ext[6 + code];
            istr = dl[code - 1] * is[1 + code];
            kstr = ks[1 + code];
        }
        if (a == 0) { rs0 = size; ri0 = istr; rk0 = kstr; }
        if (a == 1) { rs1 = size; ri1 = istr; rk1 = kstr; }
        if (a == 2) { rs2 = size; ri2 = istr; rk2 = kstr; }
        if (a == 3) { rs3 = size; ri3 = istr; rk3 = kstr; }
        a = a + 1;
    }
    int n = 0;
    while (n < ext[0]) {
        int g = 0;
        while (g < ext[1]) {
            int f = 0;
            while (f < ext[2]) {
                int o0 = 0;
                while (o0 < ext[4]) {
                    int o1 = 0;
                    while (o1 < ext[5]) {
                        int o2 = 0;
                        while (o2 < ext[6]) {
                            int out = cfg[CFG_BASE + 2] + n * os[0] + g * os[1] + o0 * os[2]
                                      + o1 * os[3] + o2 * os[4] + f * os[5];
                            int ib = cfg[CFG_BASE] + n * is[0] + g * is[1] + o0 * st[0] * is[2]
                                     + o1 * st[1] * is[3] + o2 * st[2] * is[4];
                            int kb = cfg[CFG_BASE + 1] + g * ks[0] + f * ks[1];
                            unsigned int acc = read_element(out, width);
                            int x0 = 0;
                            while (x0 < rs0) {
                                int x1 = 0;
                                while (x1 < rs1) {
                                    int x2 = 0;
                                    while (x2 < rs2) {
                                        int x3 = 0;
                                        while (x3 < rs3) {
                                            int ia = ib + x0 * ri0 + x1 * ri1 + x2 * ri2 + x3 * ri3;
                                            int ka = kb + x0 * rk0 + x1 * rk1 + x2 * rk2 + x3 * rk3;
                                            int u = load_value(ia, dti);
                                            int v = load_value(ka, dtk);
                                            if (float_math) {
                                                acc = multiply_add(acc, u, v, 0);
                                            } else {
                                                acc = acc + (u - izp) * (v - kzp);
                                            }
                                            x3 = x3 + 1;
                                        }
                                        x2 = x2 + 1;
                                    }
                                    x1 = x1 + 1;
                                }
                                x0 = x0 + 1;
                            }
                            write_element(out, width, acc);
                            o2 = o2 + 1;
                        }
                        o1 = o1 + 1;
                    }
                    o0 = o0 + 1;
                }
                f = f + 1;
            }
            g = g + 1;
        }
        n = n + 1;
    }
}
#endif

void conv_main() {
    int *cfg = (int *)ELF_CONFIG;
    int path = cfg[CFG_PATH];
    int sp0 = cfg[CFG_SP0];
    int sp1 = cfg[CFG_SP1];
    run_snapshots(cfg);
#ifdef CONV_SCALAR
    if (path == 1) { scalar_conv(cfg); }
#endif
#ifdef CONV_ARRAY
    if (path == 2) {
        int k = 0;
        while (k < cfg[CFG_NPRE]) {
            run_restage(cfg + CFG_RESTAGE + 18 * k, sp0, sp1, cfg[CFG_ZERO], cfg[CFG_IDENT]);
            k = k + 1;
        }
        array_conv(cfg, sp0, sp1);
        int m = 0;
        while (m < cfg[CFG_NPOST]) {
            run_restage(cfg + CFG_RESTAGE + 18 * (cfg[CFG_NPRE] + m), sp0, sp1, cfg[CFG_ZERO],
                        cfg[CFG_IDENT]);
            m = m + 1;
        }
    }
#endif
#ifdef CONV_LANES
    if (path == 3) { lanes_conv(cfg, sp0, sp1); }
#endif
}

#endif
