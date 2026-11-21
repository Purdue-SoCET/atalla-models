/* linalg.conv_2d_nchw_fchw_q
 *   O[n, f, oh, ow] += (I[n, c, oh*SH + kh*DH, ow*SW + kw*DW] - IZp) * (K[f, c, kh, kw] - KZp)
 * Reduction order: c, kh, kw.
 * Integer inputs (i8/i16/i32), i32 accumulation modulo 2^32, zero points
 * subtracted after the signed cast: scalar loops in Linalg order.
 * Geometry and path come from graph/conv.py; see kernels/conv_engine.h.
 */
#define CONV_SCALAR 1
#include "conv_engine.h"

int main() {
    conv_main();
    atalla_halt();
    return 0;
}
