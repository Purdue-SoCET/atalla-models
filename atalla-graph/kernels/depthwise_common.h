/* Shared storage and scalar arithmetic for dedicated depthwise kernels.
 * Layout/rank/multiplier loops stay in each operation's .c source.
 */
#ifndef ATALLA_DEPTHWISE_COMMON_H
#define ATALLA_DEPTHWISE_COMMON_H

unsigned int read_element(int address, int width) {
    unsigned int value = 0;
    int byte = 0;
    while (byte < width) {
        int a = address + byte;
        unsigned int word = *(unsigned int *)(a & -4);
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
        unsigned int old = *(unsigned int *)(a & -4);
        unsigned int part = (value >> (byte * 8)) & 255;
        *(unsigned int *)(a & -4) = (old & ~mask) | (part << shift);
        byte = byte + 1;
    }
}

unsigned int read_half(int address) {
    unsigned int word = *(unsigned int *)(address & -4);
    return (word >> ((address & 2) * 8)) & 65535;
}

union ScalarBits { unsigned int bits; float value; };

unsigned int round_bf16(unsigned int bits) {
    int magnitude = bits & 2147483647;
    if (magnitude > 2139095040) { return 32704; }
    bits = bits + 32767 + ((bits >> 16) & 1);
    return (bits >> 16) & 65535;
}

unsigned int multiply_add(unsigned int acc, unsigned int a, unsigned int b, int dtype) {
    if (dtype == 0) {
        union ScalarBits left;
        union ScalarBits right;
        union ScalarBits result;
        left.bits = a << 16;
        right.bits = b << 16;
        result.value = left.value * right.value;
        unsigned int product = round_bf16(result.bits);
        left.bits = product << 16;
        right.bits = acc << 16;
        result.value = left.value + right.value;
        return round_bf16(result.bits);
    }
    return acc + a * b; /* modulo32, no signed C overflow */
}

#endif
