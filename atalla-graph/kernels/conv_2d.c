/* linalg.conv_2d
 *   O[oh, ow] += I[oh + kh, ow + kw] * K[kh, kw]
 * Reduction order: kh, kw.
 * bf16: vector lanes, 32 outputs per vector, one mul.vs + add.vv per tap in
 * Linalg order (bit-exact bf16 rounding). Other types: scalar loops.
 * Geometry and path come from graph/conv.py; see kernels/conv_engine.h.
 */
#define CONV_LANES 1
#define CONV_SCALAR 1
#include "conv_engine.h"

int main() {
    conv_main();
    atalla_halt();
    return 0;
}
