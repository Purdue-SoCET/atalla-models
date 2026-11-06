/* depthwise_conv_3d_ndhwc_dhwc: fixed NDHWC, 3 spatial reductions, multiplier=0.
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
    int *input_strides = cfg + 19;
    int *weight_strides = cfg + 24;
    int *output_strides = cfg + 28;
    int *kernel = cfg + 33;
    int *steps = cfg + 36;
    int *dilations = cfg + 39;
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
    int vectorized = dtype == 0 && output_strides[4] == 2 && input_strides[4] == 2 && weight_strides[3] == 2;
    if (((cfg[3] | cfg[4] | cfg[5]) & 1) != 0) { vectorized = 0; }
    int axis_input_strides = 0;
    while (axis_input_strides < 5) {
        if ((input_strides[axis_input_strides] & 1) != 0) { vectorized = 0; }
        axis_input_strides = axis_input_strides + 1;
    }
    int axis_weight_strides = 0;
    while (axis_weight_strides < 4) {
        if ((weight_strides[axis_weight_strides] & 1) != 0) { vectorized = 0; }
        axis_weight_strides = axis_weight_strides + 1;
    }
    int axis_output_strides = 0;
    while (axis_output_strides < 5) {
        if ((output_strides[axis_output_strides] & 1) != 0) { vectorized = 0; }
        axis_output_strides = axis_output_strides + 1;
    }
    if (vectorized) {
        /* SP0 holds initialized/output rows. SP1 holds a reusable width
         * strip; depthwise reserves its final row for filter coefficients.
         * Reduction order within each output is unchanged. */
        int outer = 0;
        int outer_count = cfg[6] / (shape[3] * shape[4]);
        while (outer < outer_count) {
            int index = outer;
            int i2 = index % shape[2];
            index = index / shape[2];
            int i1 = index % shape[1];
            index = index / shape[1];
            int i0 = index % shape[0];
            index = index / shape[0];
            int i3 = 0;
            while (i3 < shape[3]) {
                int width_rows = shape[3] - i3;
                if (width_rows > 32) { width_rows = 32; }
                int lane_start = 0;
                int max_lanes = shape[4];
                if (max_lanes > 32) { max_lanes = 32; }
                int output_pitch = output_strides[3] / 2;
                int batchable = output_pitch >= max_lanes && output_pitch <= 1048576;
                int input_pitch = input_strides[3] / 2;
                int halo = (kernel[2] - 1) * dilations[2];
                int strip = batchable && input_pitch >= max_lanes && input_pitch <= 1048576 && halo < 31;
                if (strip) {
                    int capacity = (30 - halo) / steps[2] + 1;
                    if (width_rows > capacity) { width_rows = capacity; }
                } else { width_rows = 1; }
                /* Avoid fetching holes when dilation/stride makes a dense
                 * strip more expensive than loading the used window rows. */
                if (strip && (width_rows - 1) * steps[2] + halo + 1 > width_rows * kernel[2]) {
                    strip = 0;
                    width_rows = 1;
                }
                while (lane_start < shape[4]) {
                    int lanes = shape[4] - lane_start;
                    if (lanes > 32) { lanes = 32; }
                    int control = ((lanes - 1) << 20) | (lanes - 1);
                    int output_control = control;
                    if (width_rows > 1) {
                        output_control = ((width_rows - 1) << 25) | ((lanes - 1) << 20) | (output_pitch - 1);
                    }
                    int origin = input + cfg[3] + i0 * input_strides[0] + lane_start * input_strides[4] + i1 * steps[0] * input_strides[1] + i2 * steps[1] * input_strides[2] + i3 * steps[2] * input_strides[3];
                    int filter_origin = weights + cfg[4] + lane_start * weight_strides[3];
                    int destination = output + cfg[5] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2] + i3 * output_strides[3] + lane_start * output_strides[4];
                    if (width_rows == 1) {
                        scpad_load(0, destination, control);
                        vec acc = vector_load(0, 0, 31, 0);
                        int k0 = 0;
                        while (k0 < kernel[0]) {
                            int k1 = 0;
                            while (k1 < kernel[1]) {
                                int k2 = 0;
                                while (k2 < kernel[2]) {
                                    int source_address = origin + k0 * dilations[0] * input_strides[1] + k1 * dilations[1] * input_strides[2] + k2 * dilations[2] * input_strides[3];
                                    int weight_address = filter_origin + k0 * weight_strides[0] + k1 * weight_strides[1] + k2 * weight_strides[2];
                                    scpad_load(0, source_address, control);
                                    vec value = vector_load(0, 0, 31, 0);
                                    scpad_load(0, weight_address, control);
                                    vec coefficient = vector_load(0, 0, 31, 0);
                                    vec product = vec_op_masked("*", value, coefficient, -1);
                                    acc = vec_op_masked("+", acc, product, -1);
                                    k2 = k2 + 1;
                                }
                                k1 = k1 + 1;
                            }
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
                        continue;
                    }
                    scpad_load(0, destination, output_control);
                    int k0 = 0;
                    while (k0 < kernel[0]) {
                        int k1 = 0;
                        while (k1 < kernel[1]) {
                            int k2 = 0;
                            if (strip) {
                                int source_address = origin + k0 * dilations[0] * input_strides[1] + k1 * dilations[1] * input_strides[2] + k2 * dilations[2] * input_strides[3];
                                int source_rows = (width_rows - 1) * steps[2] + halo + 1;
                                int source_control = 1073741824 | ((source_rows - 1) << 25)
                                    | ((lanes - 1) << 20) | (input_pitch - 1);
                                scpad_load(0, source_address, source_control);
                            }
                            while (k2 < kernel[2]) {
                                int weight_address = filter_origin + k0 * weight_strides[0] + k1 * weight_strides[1] + k2 * weight_strides[2];
                                scpad_load(1984, weight_address, control | 1073741824);
                                vec coefficient = vector_load(1984, 0, 31, 1);
                                int row = 0;
                                while (row < width_rows) {
                                    vec acc = vector_load(0, row, 31, 0);
                                    int source_row = row * steps[2] + k2 * dilations[2];
                                    if (!strip) {
                                        int source_address = origin + k0 * dilations[0] * input_strides[1] + k1 * dilations[1] * input_strides[2] + k2 * dilations[2] * input_strides[3];
                                        scpad_load(0, source_address, control | 1073741824);
                                        source_row = 0;
                                    }
                                    vec value = vector_load(0, source_row, 31, 1);
                                    vec product = vec_op_masked("*", value, coefficient, -1);
                                    acc = vec_op_masked("+", acc, product, -1);
                                    vector_store(acc, 0, row, 31, 0);
                                    row = row + 1;
                                }
                                k2 = k2 + 1;
                            }
                            k1 = k1 + 1;
                        }
                        k0 = k0 + 1;
                    }
                    scpad_store(0, destination, output_control);
                    int row = 0;
                    while (row < width_rows) {
                        int j = 0;
                        while (j < lanes) {
                            int address = destination + row * output_strides[3] + j * 2;
                            unsigned int bits = read_half(address);
                            int magnitude = bits & 32767;
                            if (magnitude > 32640) { write_element(address, 2, 32704); }
                            j = j + 1;
                        }
                        row = row + 1;
                    }
                    lane_start = lane_start + lanes;
                }
                i3 = i3 + width_rows;
            }
            outer = outer + 1;
        }
        atalla_halt();
        return 0;
    }
    int element = 0;
    while (element < cfg[6]) {
        int index = element;
        int i4 = index % shape[4];
        index = index / shape[4];
        int i3 = index % shape[3];
        index = index / shape[3];
        int i2 = index % shape[2];
        index = index / shape[2];
        int i1 = index % shape[1];
        index = index / shape[1];
        int i0 = index % shape[0];
        index = index / shape[0];
        int origin = input + cfg[3] + i0 * input_strides[0] + i4 * input_strides[4] + i1 * steps[0] * input_strides[1] + i2 * steps[1] * input_strides[2] + i3 * steps[2] * input_strides[3];
        int filter_origin = weights + cfg[4] + i4 * weight_strides[3];
        int destination = output + cfg[5] + i0 * output_strides[0] + i1 * output_strides[1] + i2 * output_strides[2] + i3 * output_strides[3] + i4 * output_strides[4];
        unsigned int acc = read_element(destination, width);
        int k0 = 0;
        while (k0 < kernel[0]) {
            int k1 = 0;
            while (k1 < kernel[1]) {
                int k2 = 0;
                while (k2 < kernel[2]) {
                    int source_address = origin + k0 * dilations[0] * input_strides[1] + k1 * dilations[1] * input_strides[2] + k2 * dilations[2] * input_strides[3];
                    int weight_address = filter_origin + k0 * weight_strides[0] + k1 * weight_strides[1] + k2 * weight_strides[2];
                    unsigned int value = read_element(source_address, width);
                    unsigned int coefficient = read_element(weight_address, width);
                    acc = multiply_add(acc, value, coefficient, dtype);
                    k2 = k2 + 1;
                }
                k1 = k1 + 1;
            }
            k0 = k0 + 1;
        }
        write_element(destination, width, acc);
        element = element + 1;
    }
    atalla_halt();
    return 0;
}
