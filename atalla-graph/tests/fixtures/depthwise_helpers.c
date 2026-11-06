#include "depthwise_common.h"

int main() {
    write_element(4097, 2, 16448);
    unsigned int *output = (unsigned int *)4100;
    output[0] = multiply_add(16384, 16384, 16448, 0);
    output[1] = multiply_add(1, 2147483647, 2, 1);
    output[2] = read_element(4097, 2);
    atalla_halt();
    return 0;
}
