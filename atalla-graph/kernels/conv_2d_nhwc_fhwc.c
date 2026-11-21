/* linalg.conv_2d_nhwc_fhwc
 *   O[n, oh, ow, f] += I[n, oh*SH + kh*DH, ow*SW + kw*DW, c] * K[f, kh, kw, c]
 * Reduction order: kh, kw, c.
 * bf16: implicit GEMM on the systolic array, reduction over c per tap.
 * No restaging: operands are already channel-contiguous.
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
