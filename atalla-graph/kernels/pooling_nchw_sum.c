/* pooling_nchw_sum: fixed NCHW layout, 2D sum reduction.
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

/* Explicit BF16 storage rounding after each addition, ties to even. */
union ScalarBits {
    unsigned int bits;
    float value;
};

unsigned int reduce_value(unsigned int a, unsigned int b, int dtype) {
    if (dtype == 0) {
        union ScalarBits left;
        union ScalarBits right;
        union ScalarBits result;
        left.bits = a << 16;
        right.bits = b << 16;
        result.value = left.value + right.value;
        unsigned int bits = result.bits;
        int magnitude = bits & 2147483647;
        if (magnitude > 2139095040) {
            return 32704;
        }
        bits = bits + 32767 + ((bits >> 16) & 1);
        return (bits >> 16) & 65535;
    }
    return a + b; /* modulo-32 integer addition */
}

int main() {
    int *cfg = (int *)60;
    int input = cfg[0];
    int output = cfg[1];
    int dtype = cfg[6];
    int width = cfg[7];
    int *shape = cfg + 10;
    int *input_strides = cfg + 14;
    int *output_strides = cfg + 18;
    int *kernel = cfg + 22;
    int *steps = cfg + 24;
    int *dilations = cfg + 26;
    if (cfg[8] != 0) {
        int byte = 0;
        while (byte < cfg[9]) {
            *(int *)(cfg[8] + byte) = *(int *)(input + byte);
            byte = byte + 4;
        }
        input = cfg[8];
    }
    /* Appended config word 28: 64-byte staging buffer for stride two. */
    /* BF16 vector fast path. Lanes span the physical innermost axis.
     * Geometry remains scalar; each window reduction operates on 32 lanes.
     * DMA transfers only active tail lanes, preserving adjacent allocation bytes.
     */
    int vectorized = dtype == 0 && input_strides[3] == 2
        && output_strides[3] == 2 && cfg[5] > 0 && (steps[1] == 1 || (steps[1] == 2 && cfg[28] != 0));
    int axis = 0;
    while (axis < 4) {
        if ((input_strides[axis] & 1) != 0 || (output_strides[axis] & 1) != 0) {
            vectorized = 0;
        }
        axis = axis + 1;
    }
    if (((cfg[2] | cfg[3]) & 1) != 0) {
        vectorized = 0;
    }
    if (vectorized && cfg[4] > 0) {
        int outer = 0;
        int outer_count = cfg[4] / shape[3];
        while (outer < outer_count) {
            int index = outer;
            int i2 = index % shape[2];
            index = index / shape[2];
            int i1 = index % shape[1];
            index = index / shape[1];
            int i0 = index % shape[0];
            index = index / shape[0];
            int lane_start = 0;
            while (lane_start < shape[3]) {
                /* Batch adjacent full vectors. Each window position occupies
                 * a disjoint region of SP1; SP0 accumulates the output tile. */
                if (steps[1] == 1 && shape[3] - lane_start >= 64 && cfg[5] <= 16) {
                    int blocks = (shape[3] - lane_start) / 32;
                    int capacity = 32 / cfg[5];
                    if (blocks > capacity) { blocks = capacity; }
                    int control = ((blocks - 1) << 25) | (31 << 20) | 31;
                    int origin = input + cfg[2] + i0 * input_strides[0] + i1 * input_strides[1] + i2 * steps[0] * input_strides[2] + lane_start * steps[1] * input_strides[3];
                    int destination = output + cfg[3] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2] + lane_start * output_strides[3];
                    scpad_load(0, destination, control);
                    int staged = 0;
                    {
                        int k0 = 0;
                        while (k0 < kernel[0]) {
                            int k1 = 0;
                            while (k1 < kernel[1]) {
                                int source_address = origin + k0 * dilations[0] * input_strides[2] + k1 * dilations[1] * input_strides[3];
                                scpad_load(staged * blocks * 64, source_address, control | 1073741824);
                                staged = staged + 1;
                                k1 = k1 + 1;
                            }
                            k0 = k0 + 1;
                        }
                    }
                    staged = 0;
                    int k0 = 0;
                    while (k0 < kernel[0]) {
                        int k1 = 0;
                        while (k1 < kernel[1]) {
                            int block = 0;
                            while (block < blocks) {
                                vec acc = vector_load(0, block, 31, 0);
                                vec value = vector_load(0, staged * blocks + block, 31, 1);
                                acc = vec_op_masked("+", acc, value, -1);
                                vector_store(acc, 0, block, 31, 0);
                                block = block + 1;
                            }
                            staged = staged + 1;
                            k1 = k1 + 1;
                        }
                        k0 = k0 + 1;
                    }
                    scpad_store(0, destination, control);
                    lane_start = lane_start + blocks * 32;
                    continue;
                }
                int lanes = shape[3] - lane_start;
                if (lanes > 32) { lanes = 32; }
                if (steps[1] == 2 && lanes > 16) { lanes = 16; }
                int dense_lanes = lanes;
                if (steps[1] == 2) { dense_lanes = lanes * 2 - 1; }
                int input_control = ((dense_lanes - 1) << 20) | (dense_lanes - 1);
                int stage = cfg[28];
                int control = ((lanes - 1) << 20) | (lanes - 1);
                int destination = output + cfg[3] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2] + lane_start * output_strides[3];
                int origin = input + cfg[2] + i0 * input_strides[0] + i1 * input_strides[1] + i2 * steps[0] * input_strides[2] + lane_start * steps[1] * input_strides[3];
                if (steps[1] == 2) {
                    /* Expand initialized outputs into even lanes. Odd lanes do
                     * not contribute to any stored output. No host arithmetic. */
                    int j = 0;
                    while (j < lanes) {
                        int address = destination + j * 2;
                        unsigned int word = *(unsigned int *)(address & -4);
                        unsigned int bits = (word >> ((address & 2) * 8)) & 65535;
                        *(unsigned int *)(stage + j * 4) = bits;
                        j = j + 1;
                    }
                    scpad_load(0, stage, input_control);
                } else {
                    scpad_load(0, destination, control);
                }
                vec acc = vector_load(0, 0, 31, 0);
                int k0 = 0;
                while (k0 < kernel[0]) {
                    int k1 = 0;
                    while (k1 < kernel[1]) {
                        int address = origin + k0 * dilations[0] * input_strides[2] + k1 * dilations[1] * input_strides[3];
                        scpad_load(0, address, input_control);
                        vec value = vector_load(0, 0, 31, 0);
                        acc = vec_op_masked("+", acc, value, -1);
                        k1 = k1 + 1;
                    }
                    k0 = k0 + 1;
                }
                vector_store(acc, 0, 0, 31, 0);
                if (steps[1] == 2) {
                    scpad_store(0, stage, input_control);
                    /* Gather even result lanes, preserving adjacent halfwords. */
                    int j = 0;
                    while (j < lanes) {
                        unsigned int bits = *(unsigned int *)(stage + j * 4);
                        bits = bits & 65535;
                        int address = destination + j * 2;
                        int shift = (address & 2) * 8;
                        unsigned int word = *(unsigned int *)(address & -4);
                        unsigned int mask = 65535;
                        mask = mask << shift;
                        *(unsigned int *)(address & -4) = (word & ~mask) | (bits << shift);
                        j = j + 1;
                    }
                } else {
                    scpad_store(0, destination, control);
                }
                lane_start = lane_start + lanes;
            }
            outer = outer + 1;
        }
    }
    int element = 0;
    while (element < cfg[4]) {
        int remaining = element;
        int c = remaining % shape[1];
        remaining = remaining / shape[1];
        int w = remaining % shape[3];
        remaining = remaining / shape[3];
        int h = remaining % shape[2];
        remaining = remaining / shape[2];
        int n = remaining;
        int out_address = output + cfg[3] + n * output_strides[0] + c * output_strides[1] + h * output_strides[2] + w * output_strides[3];
        unsigned int best;
        if (vectorized) {
            /* BF16 vector-path addresses are halfword aligned. */
            unsigned int word = *(unsigned int *)(out_address & -4);
            best = (word >> ((out_address & 2) * 8)) & 65535;
        } else {
            best = read_element(out_address, width);
        }
        if (vectorized) {
            /* Canonicalize NaNs after vector BF16 rounding, matching the scalar ABI. */
            int magnitude = best & 32767;
            if (magnitude > 32640) { write_element(out_address, width, 32704); }
            element = element + 1;
            continue;
        }
        if (cfg[5] > 0) {
            int origin = input + cfg[2] + n * input_strides[0] + c * input_strides[1] + h * steps[0] * input_strides[2] + w * steps[1] * input_strides[3];
            int kh = 0;
            while (kh < kernel[0]) {
                int kw = 0;
                while (kw < kernel[1]) {
                    int address = origin + kh * dilations[0] * input_strides[2] + kw * dilations[1] * input_strides[3];
                    unsigned int value = read_element(address, width);
                    best = reduce_value(best, value, dtype);
                    kw = kw + 1;
                }
                kh = kh + 1;
            }
        }
        write_element(out_address, width, best);
        element = element + 1;
    }
    atalla_halt();
    return 0;
}
