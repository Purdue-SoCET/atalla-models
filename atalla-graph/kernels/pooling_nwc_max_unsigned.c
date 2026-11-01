/* pooling_nwc_max_unsigned: fixed NWC layout, 1D max reduction.
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

unsigned int reduce_value(unsigned int a, unsigned int b) {
    int sa = (a >> 31) & 1;
    int sb = (b >> 31) & 1;
    int ma = a & 2147483647;
    int mb = b & 2147483647;
    int less;
    if (sa != sb) {
        less = sb;
    } else {
        less = ma < mb;
    }
    if (less) {
        return b;
    }
    return a;
}

int main() {
    int *cfg = (int *)60;
    int input = cfg[0];
    int output = cfg[1];
    int width = cfg[7];
    int *shape = cfg + 10;
    int *input_strides = cfg + 13;
    int *output_strides = cfg + 16;
    int *kernel = cfg + 19;
    int *steps = cfg + 20;
    int *dilations = cfg + 21;
    if (cfg[8] != 0) {
        int byte = 0;
        while (byte < cfg[9]) {
            *(int *)(cfg[8] + byte) = *(int *)(input + byte);
            byte = byte + 4;
        }
        input = cfg[8];
    }
    int element = 0;
    while (element < cfg[4]) {
        int remaining = element;
        int c = remaining % shape[2];
        remaining = remaining / shape[2];
        int w = remaining % shape[1];
        remaining = remaining / shape[1];
        int n = remaining;
        int out_address = output + cfg[3] + n * output_strides[0] + c * output_strides[2] + w * output_strides[1];
        unsigned int best = read_element(out_address, width);
        if (cfg[5] > 0) {
            int origin = input + cfg[2] + n * input_strides[0] + c * input_strides[2] + w * steps[0] * input_strides[1];
            int kw = 0;
            while (kw < kernel[0]) {
                int address = origin + kw * dilations[0] * input_strides[1];
                unsigned int value = read_element(address, width);
                best = reduce_value(best, value);
                kw = kw + 1;
            }
        }
        write_element(out_address, width, best);
        element = element + 1;
    }
    atalla_halt();
    return 0;
}
