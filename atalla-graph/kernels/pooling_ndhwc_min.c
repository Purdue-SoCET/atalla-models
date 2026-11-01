/* pooling_ndhwc_min: fixed NDHWC layout, 3D min reduction.
 * Runtime config at byte 60: input/output allocation, input/output byte offset,
 * output/kernel element count, dtype, element bytes, alias snapshot address/size;
 * then physical output shape, input/output byte strides, window, steps, dilations.
 * Only tensor geometry and supported dtype vary. No rank/reduction selector.
 */
unsigned int read_element(int address, int width) {
    unsigned int value = 0;
    int byte = 0;
    while (byte < width) {
        int a = address + byte;
        unsigned int word = *(volatile unsigned int *)(a & -4);
        unsigned int part = (word >> ((a & 3) * 8)) & 255;
        value = value | (part << (byte * 8));
        byte = byte + 1;
    }
    return value;
}

void write_element(int address, int width, unsigned int value) {
    int byte = 0;
    while (byte < width) {
        int a = address + byte;
        int shift = (a & 3) * 8;
        unsigned int mask = 255;
        mask = mask << shift;
        unsigned int part = (value >> (byte * 8)) & 255;
        unsigned int old = *(volatile unsigned int *)(a & -4);
        *(volatile unsigned int *)(a & -4) = (old & ~mask) | (part << shift);
        byte = byte + 1;
    }
}

unsigned int reduce_value(unsigned int a, unsigned int b, int dtype) {
    int less;
    if (dtype == 0 || dtype == 1 || dtype == 4) {
        int sign_shift = 15;
        int magnitude_mask = 32767;
        int infinity = 32640;
        int quiet_nan = 32704;
        if (dtype == 1) {
            sign_shift = 31;
            magnitude_mask = 2147483647;
            infinity = 2139095040;
            quiet_nan = 2143289344;
        } else if (dtype == 4) {
            infinity = 31744;
            quiet_nan = 32256;
        }
        int ma = a & magnitude_mask;
        int mb = b & magnitude_mask;
        if (ma > infinity || mb > infinity) {
            return quiet_nan;
        }
        int sa = (a >> sign_shift) & 1;
        int sb = (b >> sign_shift) & 1;
        if (sa != sb) {
            less = sa; /* -0 < +0 */
        } else if (sa) {
            less = ma > mb;
        } else {
            less = ma < mb;
        }
    } else {
        int sa = (a >> 31) & 1;
        int sb = (b >> 31) & 1;
        int ma = a & 2147483647;
        int mb = b & 2147483647;
        if (sa != sb) {
            less = sa; /* signed i32 */
        } else {
            less = ma < mb;
        }
    }
    if (less) {
        return a;
    }
    return b;
}

int main() {
    int *cfg = (int *)60;
    int input = cfg[0];
    int output = cfg[1];
    int dtype = cfg[6];
    int width = cfg[7];
    int *shape = cfg + 10;
    int *input_strides = cfg + 15;
    int *output_strides = cfg + 20;
    int *kernel = cfg + 25;
    int *steps = cfg + 28;
    int *dilations = cfg + 31;
    if (cfg[8] != 0) {
        int byte = 0;
        while (byte < cfg[9]) {
            *(int *)(cfg[8] + byte) = *(int *)(input + byte);
            byte = byte + 4;
        }
        input = cfg[8];
    }
    /* BF16 vector fast path. Lanes span the physical innermost axis.
     * Geometry remains scalar; each window reduction operates on 32 lanes.
     * DMA transfers only active tail lanes, preserving adjacent allocation bytes.
     */
    int vectorized = dtype == 0 && input_strides[4] == 2
        && output_strides[4] == 2 && cfg[5] > 0;
    int axis = 0;
    while (axis < 5) {
        if ((input_strides[axis] & 1) != 0 || (output_strides[axis] & 1) != 0) {
            vectorized = 0;
        }
        axis = axis + 1;
    }
    if (((cfg[2] | cfg[3]) & 1) != 0) {
        vectorized = 0;
    }
    union VectorConstant { unsigned int bits; float value; };
    union VectorConstant one;
    one.bits = 1065353216;
    if (vectorized && cfg[4] > 0) {
        int outer = 0;
        int outer_count = cfg[4] / shape[4];
        while (outer < outer_count) {
            int index = outer;
            int i3 = index % shape[3];
            index = index / shape[3];
            int i2 = index % shape[2];
            index = index / shape[2];
            int i1 = index % shape[1];
            index = index / shape[1];
            int i0 = index % shape[0];
            index = index / shape[0];
            int lane_start = 0;
            while (lane_start < shape[4]) {
                int lanes = shape[4] - lane_start;
                if (lanes > 32) { lanes = 32; }
                int control = ((lanes - 1) << 20) | (lanes - 1);
                int destination = output + cfg[3] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2] + i3 * output_strides[3] + lane_start * output_strides[4];
                int origin = input + cfg[2] + i0 * input_strides[0] + i1 * steps[0] * input_strides[1] + i2 * steps[1] * input_strides[2] + i3 * steps[2] * input_strides[3] + lane_start * input_strides[4];
                scpad_load(0, destination, control);
                vec acc = vector_load(0, 0, 31, 0);
                int k0 = 0;
                while (k0 < kernel[0]) {
                    int k1 = 0;
                    while (k1 < kernel[1]) {
                        int k2 = 0;
                        while (k2 < kernel[2]) {
                            int address = origin + k0 * dilations[0] * input_strides[1] + k1 * dilations[1] * input_strides[2] + k2 * dilations[2] * input_strides[3];
                            scpad_load(0, address, control);
                            vec value = vector_load(0, 0, 31, 0);
                            int take = make_mask("<", value, acc, -1);
                            acc = vec_op_masked("*", value, one.value, take, acc);
                            int nan_mask = make_mask("!=", value, value, -1);
                            union VectorNan { unsigned int bits; float value; };
                            union VectorNan qnan;
                            qnan.bits = 2143289344;
                            acc = vec_op_masked("*", acc, qnan.value, nan_mask, acc);
                            k2 = k2 + 1;
                        }
                        k1 = k1 + 1;
                    }
                    k0 = k0 + 1;
                }
                vector_store(acc, 0, 0, 31, 0);
                scpad_store(0, destination, control);
                lane_start = lane_start + 32;
            }
            outer = outer + 1;
        }
    }
    int element = 0;
    while (element < cfg[4]) {
        int remaining = element;
        int c = remaining % shape[4];
        remaining = remaining / shape[4];
        int w = remaining % shape[3];
        remaining = remaining / shape[3];
        int h = remaining % shape[2];
        remaining = remaining / shape[2];
        int d = remaining % shape[1];
        remaining = remaining / shape[1];
        int n = remaining;
        int out_address = output + cfg[3] + n * output_strides[0] + c * output_strides[4] + d * output_strides[1] + h * output_strides[2] + w * output_strides[3];
        unsigned int best;
        if (vectorized) {
            /* BF16 vector-path addresses are halfword aligned. */
            unsigned int word = *(unsigned int *)(out_address & -4);
            best = (word >> ((out_address & 2) * 8)) & 65535;
        } else {
            best = read_element(out_address, width);
        }
        if (vectorized) {
            int magnitude = best & 32767;
            if (magnitude > 32640) {
                write_element(out_address, width, 32704);
                element = element + 1;
                continue;
            }
            if (magnitude != 0 || (int)best == 32768) {
                element = element + 1;
                continue;
            }
            /* The vector result is zero: only zero sign bits can change it.
             * A preferred-sign zero is decisive; all other magnitudes can be
             * ignored because the vector reduction already compared them. */
        }
        if (cfg[5] > 0) {
            int origin = input + cfg[2] + n * input_strides[0] + c * input_strides[4] + d * steps[0] * input_strides[1] + h * steps[1] * input_strides[2] + w * steps[2] * input_strides[3];
            int kd = 0;
            while (kd < kernel[0]) {
                int kh = 0;
                while (kh < kernel[1]) {
                    int kw = 0;
                    while (kw < kernel[2]) {
                        int address = origin + kd * dilations[0] * input_strides[1] + kh * dilations[1] * input_strides[2] + kw * dilations[2] * input_strides[3];
                        unsigned int value;
                        if (vectorized) {
                            unsigned int word = *(unsigned int *)(address & -4);
                            value = (word >> ((address & 2) * 8)) & 65535;
                            if ((int)value == 32768) { best = value; }
                        } else {
                            value = read_element(address, width);
                            best = reduce_value(best, value, dtype);
                        }
                        kw = kw + 1;
                    }
                    kh = kh + 1;
                }
                kd = kd + 1;
            }
        }
        if (vectorized) {
            /* Leave unchanged zeros in place; repair only a changed sign. */
            if ((int)best == 32768) {
                unsigned int word = *(unsigned int *)(out_address & -4);
                int shift = (out_address & 2) * 8;
                unsigned int mask = 65535;
                mask = mask << shift;
                *(unsigned int *)(out_address & -4) = (word & ~mask) | (best << shift);
            }
        } else {
            write_element(out_address, width, best);
        }
        element = element + 1;
    }
    atalla_halt();
    return 0;
}
