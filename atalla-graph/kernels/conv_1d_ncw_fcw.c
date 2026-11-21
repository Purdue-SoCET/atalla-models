/* linalg.conv_1d_ncw_fcw
 *   O[n, f, ow] += I[n, c, ow*SW + kw*DW] * K[f, c, kw]
 * Reduction order: c, kw.
 * bf16: implicit GEMM on the systolic array, reduction over c per tap.
 * Restaged on device (array transposes): input, filter, output.
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
