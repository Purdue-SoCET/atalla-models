int main() {
    int cfg = 60;
    int IN_GMEM;
    int OUT_GMEM;
    asm("lw_s %0, 0(%1)" : "=r"(IN_GMEM)  : "r"(cfg));
    asm("lw_s %0, 4(%1)" : "=r"(OUT_GMEM) : "r"(cfg));

    int sp = 0;
    int all_mask = -1;
    int ncols = 1;
    int sdma_ctl;
    asm("li_s %0, 66060319" : "=r"(sdma_ctl));

    vec zero_vec = vector_load(0, ncols, 31, 0);
    zero_vec = vec_op_masked("*", zero_vec, 0.0, all_mask);

    int tile = 0;
    while (tile < 1) {
        scpad_load(sp, IN_GMEM, sdma_ctl);

        int row = 0;
        while (row < 2) {
            vec v = vector_load(row, ncols, 31, 0);

            int m_neg = make_mask("<", v, zero_vec, all_mask);
            vec result = vec_op_masked("*", v, 0.0, m_neg);

            vector_store(result, row, ncols, 31, 0);
            row = row + 1;
        }

        scpad_store(sp, OUT_GMEM, sdma_ctl);
        IN_GMEM = IN_GMEM + 128;
        OUT_GMEM = OUT_GMEM + 128;
        tile = tile + 1;
    }

    asm("halt");
    return 0;
}
