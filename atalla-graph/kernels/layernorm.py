"""AtallaC LayerNorm: per-group normalize over D features (D multiple of 32).

One scratchpad tile; streams each 32-wide row to GMEM for mean/variance and again
for normalize + gamma + beta. Config table at ADDR_TABLE (see emit_layernorm).
"""

from __future__ import annotations

from .common import ADDR_TABLE, sdma_ctl_expr


def layernorm_c(M: int, D: int, eps: float) -> str:
    """Layer norm over M groups × D features; D must be divisible by 32."""
    if D % 32 != 0:
        raise ValueError(f"layernorm_c requires D % 32 == 0, got D={D}")
    nr = D // 32
    ctl = sdma_ctl_expr("sdma_row", 0, 1, 32, 32)
    inv_d = 1.0 / float(D)
    # AtallaC lexer rejects scientific notation in float literals.
    eps_lit = f"{float(eps):.8f}"
    inv_d_lit = f"{inv_d:.10f}"

    return (
        "int main() {\n"
        f"    int cfg = {ADDR_TABLE};\n"
        "    int IN0;\n"
        "    int OUT0;\n"
        "    int GM;\n"
        "    int BM;\n"
        "    int M;\n"
        '    asm("lw_s %0, 0(%1)"  : "=r"(IN0)  : "r"(cfg));\n'
        '    asm("lw_s %0, 4(%1)"  : "=r"(OUT0) : "r"(cfg));\n'
        '    asm("lw_s %0, 8(%1)"  : "=r"(GM)    : "r"(cfg));\n'
        '    asm("lw_s %0, 12(%1)" : "=r"(BM)    : "r"(cfg));\n'
        '    asm("lw_s %0, 16(%1)" : "=r"(M)     : "r"(cfg));\n'
        "\n"
        "    int sp = 0;\n"
        f"    float inv_d = {inv_d_lit};\n"
        f"    float eps = {eps_lit};\n"
        f"{ctl}"
        "\n"
        "    int grp = 0;\n"
        f"    while (grp < M) {{\n"
        f"        int base = grp * {D} * 2;\n"
        "        float sum = 0.0;\n"
        "        int rr = 0;\n"
        f"        while (rr < {nr}) {{\n"
        "            scpad_load(sp, IN0 + base + rr * 64, sdma_row);\n"
        "            vec v = vector_load(sp, 0, 31, 0);\n"
        '            vec sv = vec_op_masked("RSUM", v, 0.0, -1);\n'
        "            sum = sum + sv[0];\n"
        "            rr = rr + 1;\n"
        "        }\n"
        "        float mean = sum * inv_d;\n"
        "\n"
        "        float var_acc = 0.0;\n"
        "        rr = 0;\n"
        f"        while (rr < {nr}) {{\n"
        "            scpad_load(sp, IN0 + base + rr * 64, sdma_row);\n"
        "            vec v = vector_load(sp, 0, 31, 0);\n"
        '            vec c = vec_op_masked("-", v, mean, -1);\n'
        '            vec sq = vec_op_masked("*", c, c, -1);\n'
        '            vec sv2 = vec_op_masked("RSUM", sq, 0.0, -1);\n'
        "            var_acc = var_acc + sv2[0];\n"
        "            rr = rr + 1;\n"
        "        }\n"
        "        float var = var_acc * inv_d + eps;\n"
        "        float inv_std = 1.0 / sqrt(var);\n"
        "\n"
        "        rr = 0;\n"
        f"        while (rr < {nr}) {{\n"
        "            scpad_load(sp, IN0 + base + rr * 64, sdma_row);\n"
        "            vec v = vector_load(sp, 0, 31, 0);\n"
        '            vec c = vec_op_masked("-", v, mean, -1);\n'
        '            vec n = vec_op_masked("*", c, inv_std, -1);\n'
        "            scpad_load(sp, GM + rr * 64, sdma_row);\n"
        "            vec g = vector_load(sp, 0, 31, 0);\n"
        '            vec ng = vec_op_masked("*", n, g, -1);\n'
        "            scpad_load(sp, BM + rr * 64, sdma_row);\n"
        "            vec bvec = vector_load(sp, 0, 31, 0);\n"
        '            vec o = vec_op_masked("+", ng, bvec, -1);\n'
        "            vector_store(o, sp, 0, 31, 0);\n"
        "            scpad_store(sp, OUT0 + base + rr * 64, sdma_row);\n"
        "            rr = rr + 1;\n"
        "        }\n"
        "        grp = grp + 1;\n"
        "    }\n"
        "\n"
        '    asm("halt");\n'
        "    return 0;\n"
        "}\n"
    )
