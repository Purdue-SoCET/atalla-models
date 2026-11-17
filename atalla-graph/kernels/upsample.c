#define ADDR_TABLE 60

int main() {
    int* cfg_ptr       = (int*)ADDR_TABLE;
    int IN_GMEM        = *(cfg_ptr + 0);
    int OUT_GMEM       = *(cfg_ptr + 1);
    int total_rows     = *(cfg_ptr + 2);
    int row_words      = *(cfg_ptr + 3);
    int in_row_bytes   = *(cfg_ptr + 4);
    int out_row_bytes  = *(cfg_ptr + 5);

    int r = 0;
    while (r < total_rows) {
        int in_row_addr  = IN_GMEM  + r * in_row_bytes;
        int out_r0_addr  = OUT_GMEM + (2 * r) * out_row_bytes;
        int out_r1_addr  = out_r0_addr + out_row_bytes;

        int w = 0;
        while (w < row_words) {
            int in_addr = in_row_addr + w * 4;
            int val = *(int*)in_addr;
            int x0 = val & 65535;
            int x1 = (val >> 16) & 65535;
            int dup0 = (x0 << 16) | x0;
            int dup1 = (x1 << 16) | x1;

            int out_offset = w * 8;
            int out0 = out_r0_addr + out_offset;
            int out1 = out_r1_addr + out_offset;

            *(int*)(out0 + 0) = dup0;
            *(int*)(out0 + 4) = dup1;
            *(int*)(out1 + 0) = dup0;
            *(int*)(out1 + 4) = dup1;

            w = w + 1;
        }
        r = r + 1;
    }

    atalla_halt();
    return 0;
}
