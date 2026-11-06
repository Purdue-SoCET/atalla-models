/* depthwise_conv_1d_ncw_cw: fixed NCW, 1 spatial reductions, multiplier=0.
 * Header at byte60: input/filter/output bases, three view byte offsets,
 * output/kernel counts, dtype/width, then input and filter snapshot addr/bytes.
 * Metadata: physical output shape, input/filter/output byte strides, kernel
 * spatial shape, spatial steps/dilations. No layout/rank/operation dispatcher.
 */
#include "depthwise_common.h"

int main() {
    int *cfg = (int *)60;
    int input = cfg[0];
    int weights = cfg[1];
    int output = cfg[2];
    int dtype = cfg[8];
    int width = cfg[9];
    int *shape = cfg + 14;
    int *input_strides = cfg + 17;
    int *weight_strides = cfg + 20;
    int *output_strides = cfg + 22;
    int *kernel = cfg + 25;
    int *steps = cfg + 26;
    int *dilations = cfg + 27;
    if (cfg[6] == 0 || cfg[7] == 0) { atalla_halt(); return 0; }
    if (cfg[10] != 0) {
        int byte = 0;
        while (byte < cfg[11]) {
            *(int *)(cfg[10] + byte) = *(int *)(input + byte);
            byte = byte + 4;
        }
        input = cfg[10];
    }
    if (cfg[12] != 0) {
        if (cfg[12] != cfg[10]) {
            int byte = 0;
            while (byte < cfg[13]) {
                *(int *)(cfg[12] + byte) = *(int *)(weights + byte);
                byte = byte + 4;
            }
        }
        weights = cfg[12];
    }
    int vectorized = dtype == 0 && output_strides[2] == 2 && input_strides[2] == 2 && steps[0] == 1;
    if (((cfg[3] | cfg[4] | cfg[5]) & 1) != 0) { vectorized = 0; }
    int axis_input_strides = 0;
    while (axis_input_strides < 3) {
        if ((input_strides[axis_input_strides] & 1) != 0) { vectorized = 0; }
        axis_input_strides = axis_input_strides + 1;
    }
    int axis_weight_strides = 0;
    while (axis_weight_strides < 2) {
        if ((weight_strides[axis_weight_strides] & 1) != 0) { vectorized = 0; }
        axis_weight_strides = axis_weight_strides + 1;
    }
    int axis_output_strides = 0;
    while (axis_output_strides < 3) {
        if ((output_strides[axis_output_strides] & 1) != 0) { vectorized = 0; }
        axis_output_strides = axis_output_strides + 1;
    }
    if (vectorized) {
        int outer = 0;
        int outer_count = cfg[6] / shape[2];
        while (outer < outer_count) {
            int index = outer;
            int i1 = index % shape[1];
            index = index / shape[1];
            int i0 = index % shape[0];
            index = index / shape[0];
            int lane_start = 0;
            while (lane_start < shape[2]) {
                /* Batch adjacent full vectors. Each window position occupies
                 * a disjoint region of SP1; SP0 accumulates the output tile. */
                if (steps[0] == 1 && shape[2] - lane_start >= 64 && cfg[7] <= 16) {
                    int blocks = (shape[2] - lane_start) / 32;
                    int capacity = 32 / cfg[7];
                    if (blocks > capacity) { blocks = capacity; }
                    int control = ((blocks - 1) << 25) | (31 << 20) | 31;
                    int origin = input + cfg[3] + i0 * input_strides[0] + i1 * input_strides[1] + lane_start * steps[0] * input_strides[2];
                    int filter_origin = weights + cfg[4] + i1 * weight_strides[0];
                    int destination = output + cfg[5] + i0 * output_strides[0] + i1 * output_strides[1] + lane_start * output_strides[2];
                    scpad_load(0, destination, control);
                    int staged = 0;
                    {
                        int k0 = 0;
                        while (k0 < kernel[0]) {
                            int source_address = origin + k0 * dilations[0] * input_strides[2];
                            scpad_load(staged * blocks * 64, source_address, control | 1073741824);
                            staged = staged + 1;
                            k0 = k0 + 1;
                        }
                    }
                    staged = 0;
                    int k0 = 0;
                    while (k0 < kernel[0]) {
                        int weight_address = filter_origin + k0 * weight_strides[1];
                        union ScalarBits coefficient;
                        coefficient.bits = read_half(weight_address) << 16;
                        int block = 0;
                        while (block < blocks) {
                            vec acc = vector_load(0, block, 31, 0);
                            vec value = vector_load(0, staged * blocks + block, 31, 1);
                            vec product = vec_op_masked("*", value, coefficient.value, -1);
                            acc = vec_op_masked("+", acc, product, -1);
                            vector_store(acc, 0, block, 31, 0);
                            block = block + 1;
                        }
                        staged = staged + 1;
                        k0 = k0 + 1;
                    }
                    scpad_store(0, destination, control);
                    int j = 0;
                    while (j < blocks * 32) {
                        int address = destination + j * 2;
                        unsigned int bits = read_half(address);
                        int magnitude = bits & 32767;
                        if (magnitude > 32640) { write_element(address, 2, 32704); }
                        j = j + 1;
                    }
                    lane_start = lane_start + blocks * 32;
                    continue;
                }
                int lanes = shape[2] - lane_start;
                if (lanes > 32) { lanes = 32; }
                int control = ((lanes - 1) << 20) | (lanes - 1);
                int origin = input + cfg[3] + i0 * input_strides[0] + i1 * input_strides[1] + lane_start * steps[0] * input_strides[2];
                int filter_origin = weights + cfg[4] + i1 * weight_strides[0];
                int destination = output + cfg[5] + i0 * output_strides[0] + i1 * output_strides[1] + lane_start * output_strides[2];
                scpad_load(0, destination, control);
                vec acc = vector_load(0, 0, 31, 0);
                int k0 = 0;
                while (k0 < kernel[0]) {
                    int source_address = origin + k0 * dilations[0] * input_strides[2];
                    int weight_address = filter_origin + k0 * weight_strides[1];
                    union ScalarBits coefficient;
                    coefficient.bits = read_half(weight_address) << 16;
                    scpad_load(0, source_address, control);
                    vec value = vector_load(0, 0, 31, 0);
                    vec product = vec_op_masked("*", value, coefficient.value, -1);
                    acc = vec_op_masked("+", acc, product, -1);
                    k0 = k0 + 1;
                }
                vector_store(acc, 0, 0, 31, 0);
                scpad_store(0, destination, control);
                int j = 0;
                while (j < lanes) {
                    int address = destination + j * 2;
                    unsigned int bits = read_half(address);
                    int magnitude = bits & 32767;
                    if (magnitude > 32640) { write_element(address, 2, 32704); }
                    j = j + 1;
                }
                lane_start = lane_start + lanes;
            }
            outer = outer + 1;
        }
        atalla_halt();
        return 0;
    }
    int element = 0;
    while (element < cfg[6]) {
        int index = element;
        int i2 = index % shape[2];
        index = index / shape[2];
        int i1 = index % shape[1];
        index = index / shape[1];
        int i0 = index % shape[0];
        index = index / shape[0];
        int origin = input + cfg[3] + i0 * input_strides[0] + i1 * input_strides[1] + i2 * steps[0] * input_strides[2];
        int filter_origin = weights + cfg[4] + i1 * weight_strides[0];
        int destination = output + cfg[5] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2];
        unsigned int acc = read_element(destination, width);
        int k0 = 0;
        while (k0 < kernel[0]) {
            int source_address = origin + k0 * dilations[0] * input_strides[2];
            int weight_address = filter_origin + k0 * weight_strides[1];
            unsigned int value = read_element(source_address, width);
            unsigned int coefficient = read_element(weight_address, width);
            acc = multiply_add(acc, value, coefficient, dtype);
            k0 = k0 + 1;
        }
        write_element(destination, width, acc);
        element = element + 1;
    }
    atalla_halt();
    return 0;
}
