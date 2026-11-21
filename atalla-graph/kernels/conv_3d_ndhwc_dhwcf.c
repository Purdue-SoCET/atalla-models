/* linalg.conv_3d_ndhwc_dhwcf
 *   O[n, od, oh, ow, f] += I[n, od*SD + kd*DD, oh*SH + kh*DH, ow*SW + kw*DW, c] * K[kd, kh, kw, c, f]
 * Reduction order: kd, kh, kw, c.
 * bf16: implicit GEMM on the systolic array, reduction over c per tap.
 * Restaged on device (array transposes): filter.
 * Integer types, odd alignment or unsupported strides: scalar loops.
 * Geometry and path come from graph/conv.py; see kernels/conv_engine.h.
 */
#define CONV_ARRAY 1
#define CONV_SCALAR 1
#include "conv_engine.h"

int main() {
    conv_main();
    atalla_halt();
    return 0;
}
