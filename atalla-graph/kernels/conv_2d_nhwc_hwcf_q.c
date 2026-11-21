/* linalg.conv_2d_nhwc_hwcf_q
 *   O[n, oh, ow, f] += (I[n, oh*SH + kh*DH, ow*SW + kw*DW, c] - IZp) * (K[kh, kw, c, f] - KZp)
 * Reduction order: kh, kw, c.
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
