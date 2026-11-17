#define ADDR_TABLE    60
#define WIDTH_M1      7
#define OUT_WIDTH_M1  3
#define LANE_MASK     255

#define M01  0x3
#define M23  0xC
#define M45  0x30
#define M67  0xC0

int main() {
    int* cfg_ptr     = (int*)ADDR_TABLE;
    int IN_BASE      = *(cfg_ptr + 0);
    int OUT_BASE     = *(cfg_ptr + 1);
    int channels     = *(cfg_ptr + 2);
    int H_IN         = *(cfg_ptr + 3);
    int H_OUT        = *(cfg_ptr + 4);
    int sdma_in      = *(cfg_ptr + 5);
    int sdma_out     = *(cfg_ptr + 6);
    int in_ch_bytes  = *(cfg_ptr + 7);
    int out_ch_bytes = *(cfg_ptr + 8);
    float* fcfg_ptr  = (float*)(ADDR_TABLE + 36);
    float scale      = *fcfg_ptr;

    int sp = 0;
    int lane_mask = LANE_MASK;

    vec zero_vec = vector_load(sp, 0, WIDTH_M1, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, lane_mask);

    int ch = 0;
    while (ch < channels) {
        int in_ptr = IN_BASE + ch * in_ch_bytes;
        int out_ptr = OUT_BASE + ch * out_ch_bytes;

        scpad_load(sp, in_ptr, sdma_in);

        int oh = 0;
        while (oh < H_OUT) {
            int in_row = oh * 2;
            vec v0 = vector_load(sp, in_row, WIDTH_M1, 0);
            int r1 = in_row + 1;
            vec v1 = vector_load(sp, r1, WIDTH_M1, 0);
            vec vsum = vec_op_masked("+", v0, v1, lane_mask);
            vector_store(vsum, sp, oh, WIDTH_M1, 0);
            oh = oh + 1;
        }

        int row = 0;
        while (row < H_OUT) {
            vec r = vector_load(sp, row, WIDTH_M1, 0);

            vec p01 = vec_op_masked("RSUM", r, 0.0, M01);
            vec acc = vec_op_masked("+", zero_vec, p01, 0x1);

            vec p23 = vec_op_masked("RSUM", r, 0.0, M23);
            acc = vec_op_masked("+", zero_vec, p23, 0x2, acc);

            vec p45 = vec_op_masked("RSUM", r, 0.0, M45);
            acc = vec_op_masked("+", zero_vec, p45, 0x4, acc);

            vec p67 = vec_op_masked("RSUM", r, 0.0, M67);
            acc = vec_op_masked("+", zero_vec, p67, 0x8, acc);

            acc = vec_op_masked("*", acc, scale, 0xF);

            vector_store(acc, sp, row, OUT_WIDTH_M1, 0);
            row = row + 1;
        }

        scpad_store(sp, out_ptr, sdma_out);
        ch = ch + 1;
    }

    atalla_halt();
    return 0;
}
