"""AtallaC tiled GEMM kernel: C += A × B with 32×32 tiles.

Two schedules are generated from the same DRAM layout (``_write_gemm_*`` in
``codegen/c_emitter.py``), selected by ``ATALLA_GEMM_SCHEDULE``:

``baseline`` (default, unchanged output)
    Loop order ``mi > ni > ki``.  Every K step DMAs the zero row (SP0), the
    weight tile (SP1), the A tile (SP0) and the C tile (SP1), runs ``tile_m``
    ``gemm`` rows and stores the C tile back.  Scratchpad roles: SP0 = zero row
    + A, SP1 = W then C.

``double_buffer``
    Same loop order and the same arithmetic, but the C tile is resident in SP0
    for the whole K sweep (one load before ``ki = 0``, one store after the last
    K step) while W and A stream through SP1.  The next K step's weight tile is
    DMA'd into SP1 as soon as the current tile's ``gemm`` rows have consumed
    the A tile, before the loop bookkeeping, so the only DMA issued between
    two row loops is the A tile.  The inner ``load_weights`` loop is unrolled
    by 2.  See ``docs/experiments/gemm_double_buffer_2026-11.md``.

Why the buffers are assigned this way: the core has two single-tile
scratchpads (32 rows × 32 lanes each) and the systolic array holds one weight
tile, so a 32×32×32 K step needs three tile buffers (A, C, next W) for a true
DMA/compute overlap and only two exist.  The functional simulator also resets
its ``lw.vi`` weight-column counter only on an SP1 DMA load, so weight tiles
must stream through SP1.  C therefore owns SP0 and W/A share SP1.

Weight-load order: ``push_weights`` walks rows ``0 .. tile_sz-1`` first-to-
last.  When the upstream ``investigate-weight-load-order`` change flips
``lw.vi`` to last-to-first, reverse the loop in ``_push_weights_c`` (and the
one in ``_gemm_c_baseline``); nothing else depends on the order.
"""

import math
import os

from .common import ADDR_TABLE, TILE, sdma_ctl_expr

SCHEDULES = ("baseline", "double_buffer")
SCHEDULE_ENV = "ATALLA_GEMM_SCHEDULE"


def gemm_schedule() -> str:
    """Schedule selected by ``ATALLA_GEMM_SCHEDULE`` (unset/empty = ``baseline``)."""
    v = os.environ.get(SCHEDULE_ENV, "").strip().lower()
    if not v:
        return "baseline"
    if v not in SCHEDULES:
        raise ValueError(f"{SCHEDULE_ENV}={v!r}: expected one of {', '.join(SCHEDULES)}")
    return v


def gemm_c(M: int, N: int, K: int, schedule: str | None = None) -> str:
    """Generate AtallaC for tiled GEMM reading config from ADDR_TABLE.

    RHS DRAM matches ``_write_gemm_rhs_weight``: logical ``(K,N)`` weights are
    stored as ``(N,K)`` row-major with each row padded to ``k_stride`` columns
    (multiple of ``TILE``).  ``A`` uses the same inner stride.  SDMA
    ``full_cols`` must match that stride so row-to-row steps in GMEM are correct
    when ``K < TILE``.

    ``schedule`` overrides ``ATALLA_GEMM_SCHEDULE`` (see module docstring).
    """
    sched = gemm_schedule() if schedule is None else schedule
    if sched == "baseline":
        return _gemm_c_baseline(M, N, K)
    if sched == "double_buffer":
        return _gemm_c_double_buffer(M, N, K)
    raise ValueError(f"unknown GEMM schedule {sched!r}; expected one of {', '.join(SCHEDULES)}")


def _gemm_c_baseline(M: int, N: int, K: int) -> str:
    tm1 = min(TILE, M) - 1
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)
    k_stride = int(math.ceil(max(K, 1) / TILE) * TILE)
    sdma_a_s = sdma_ctl_expr("sdma_ctl_a", 0, tile_m, tile_k, k_stride)
    # W' is (N, k_stride) row-major: each GMEM row is one output channel; tile is
    # tile_n channels × tile_k K coefficients (not tile_k × tile_n).
    sdma_w_s = sdma_ctl_expr("sdma_ctl_w", 1, tile_n, tile_k, k_stride)
    sdma_c_s = sdma_ctl_expr("sdma_ctl_c", 1, tile_m, tile_n, N)
    # SP0: one row of TILE zeros for padding partial-N tails in weight SP1.
    sdma_z_s = sdma_ctl_expr("sdma_ctl_z", 0, 1, TILE, TILE)

    return (
        "void load_weight_tile(int sp_w, int w_addr, int sdma_ctl_w) {\n"
        f"    int* cfg_tbl = (int*){ADDR_TABLE};\n"
        "    int z_addr = *(cfg_tbl + 10);\n"
        "    int tile_sz = *(cfg_tbl + 9);\n"
        f"{sdma_z_s}"
        "    scpad_load(0, z_addr, sdma_ctl_z);\n"
        f"    vec zrow = vector_load(0, 0, {TILE - 1}, 0);\n"
        "    scpad_load(sp_w, w_addr, sdma_ctl_w);\n"
        f"    int wn = {tile_n};\n"
        "    while (wn < tile_sz) {\n"
        f"        vector_store(zrow, sp_w, wn, {tk1}, 1);\n"
        "        wn = wn + 1;\n"
        "    }\n"
        "    int wi = 0;\n"
        "    while (wi < tile_sz) {\n"
        f"        vec wvec = vector_load(sp_w, wi, {tk1}, 1);\n"
        "        load_weights(wvec);\n"
        "        wi = wi + 1;\n"
        "    }\n"
        "}\n"
        "\n"
        "int main() {\n"
        f"    int* cfg = (int*){ADDR_TABLE};\n"
        "    int A_GMEM = *(cfg + 0);\n"
        "    int W_GMEM = *(cfg + 1);\n"
        "    int C_GMEM = *(cfg + 2);\n"
        "    int gM = *(cfg + 3);\n"
        "    int gN = *(cfg + 4);\n"
        "    int gK = *(cfg + 5);\n"
        "    int M_tiles = *(cfg + 6);\n"
        "    int N_tiles = *(cfg + 7);\n"
        "    int K_tiles = *(cfg + 8);\n"
        "    int tile_sz = *(cfg + 9);\n"
        "\n"
        "    int all_mask = -1;\n"
        "    int sp_a = 0;\n"
        "    int sp_w = 0;\n"
        "    int sp_c = 0;\n"
        f"{sdma_a_s}"
        f"{sdma_w_s}"
        f"{sdma_c_s}"
        "\n"
        "    int mi = 0;\n"
        "    while (mi < M_tiles) {\n"
        "        int ni = 0;\n"
        "        while (ni < N_tiles) {\n"
        "            int ki = 0;\n"
        "            while (ki < K_tiles) {\n"
        "                int a_off = mi * tile_sz * gK + ki * tile_sz;\n"
        "                int a_byte = a_off * 2;\n"
        "                int a_addr = A_GMEM + a_byte;\n"
        "\n"
        "                int w_off = ni * tile_sz * gK + ki * tile_sz;\n"
        "                int w_byte = w_off * 2;\n"
        "                int w_addr = W_GMEM + w_byte;\n"
        "\n"
        "                load_weight_tile(sp_w, w_addr, sdma_ctl_w);\n"
        "\n"
        "                scpad_load(sp_a, a_addr, sdma_ctl_a);\n"
        "\n"
        "                int c_off = mi * tile_sz * gN + ni * tile_sz;\n"
        "                int c_byte = c_off * 2;\n"
        "                int c_addr = C_GMEM + c_byte;\n"
        "                scpad_load(sp_c, c_addr, sdma_ctl_c);\n"
        "\n"
        "                int ri = 0;\n"
        f"                while (ri < {tile_m}) {{\n"
        f"                    vec a_row = vector_load(sp_a, ri, {tk1}, 0);\n"
        f"                    vec c_row = vector_load(sp_c, ri, {tn1}, 1);\n"
        "\n"
        "                    vec result = gemm(a_row, c_row, all_mask);\n"
        "\n"
        f"                    vector_store(result, sp_c, ri, {tn1}, 1);\n"
        "                    ri = ri + 1;\n"
        "                }\n"
        "\n"
        "                scpad_store(sp_c, c_addr, sdma_ctl_c);\n"
        "                ki = ki + 1;\n"
        "            }\n"
        "\n"
        "            ni = ni + 1;\n"
        "        }\n"
        "        mi = mi + 1;\n"
        "    }\n"
        "\n"
        "    atalla_halt();\n"
        "    return 0;\n"
        "}\n"
    )


def _fetch_weight_tile_c(tile_n: int, tk1: int, sdma_z_s: str) -> str:
    """``fetch_weight_tile``: DMA one W' tile into SP1 (and zero-pad partial-N rows).

    The zero row is only DMA'd when ``tile_n < TILE``; a full tile needs no
    padding, so the baseline's unconditional 64-byte zero-row DMA per K step is
    dropped.  The zero row goes through SP1 row 0 *before* the weight DMA (which
    then overwrites it) so SP0, which holds the resident C tile, is never touched.
    """
    if tile_n >= TILE:
        return (
            "void fetch_weight_tile(int sp_s, int w_addr, int sdma_ctl_w) {\n"
            "    scpad_load(sp_s, w_addr, sdma_ctl_w);\n"
            "}\n"
        )
    return (
        "void fetch_weight_tile(int sp_s, int w_addr, int sdma_ctl_w) {\n"
        f"    int* cfg_tbl = (int*){ADDR_TABLE};\n"
        "    int z_addr = *(cfg_tbl + 10);\n"
        "    int tile_sz = *(cfg_tbl + 9);\n"
        f"{sdma_z_s}"
        "    scpad_load(sp_s, z_addr, sdma_ctl_z);\n"
        f"    vec zrow = vector_load(sp_s, 0, {TILE - 1}, 1);\n"
        "    scpad_load(sp_s, w_addr, sdma_ctl_w);\n"
        f"    int wn = {tile_n};\n"
        "    while (wn < tile_sz) {\n"
        f"        vector_store(zrow, sp_s, wn, {tk1}, 1);\n"
        "        wn = wn + 1;\n"
        "    }\n"
        "}\n"
    )


def _push_weights_c(tk1: int) -> str:
    """``push_weights``: 32 × ``load_weights`` from SP1, unrolled by 2.

    ``tile_sz`` is always ``TILE`` (32, even).  Order is first-to-last; to
    reverse it for the pending ``lw.vi`` change start at ``tile_sz - 2`` and
    step by ``-2`` (loading ``wi + 1`` before ``wi``).
    """
    return (
        "void push_weights(int sp_s, int tile_sz) {\n"
        "    int wi = 0;\n"
        "    while (wi < tile_sz) {\n"
        f"        vec w0 = vector_load(sp_s, wi, {tk1}, 1);\n"
        "        load_weights(w0);\n"
        "        int wi1 = wi + 1;\n"
        f"        vec w1 = vector_load(sp_s, wi1, {tk1}, 1);\n"
        "        load_weights(w1);\n"
        "        wi = wi + 2;\n"
        "    }\n"
        "}\n"
    )


def _gemm_c_double_buffer(M: int, N: int, K: int) -> str:
    tn1 = min(TILE, N) - 1
    tk1 = min(TILE, K) - 1
    tile_m = min(TILE, M)
    tile_n = min(TILE, N)
    tile_k = min(TILE, K)
    k_stride = int(math.ceil(max(K, 1) / TILE) * TILE)
    # SP1 streams W (consumed into the systolic array) and then A for each K step.
    sdma_a_s = sdma_ctl_expr("sdma_ctl_a", 1, tile_m, tile_k, k_stride)
    sdma_w_s = sdma_ctl_expr("sdma_ctl_w", 1, tile_n, tile_k, k_stride)
    # SP0 holds the C tile for the whole K sweep.
    sdma_c_s = sdma_ctl_expr("sdma_ctl_c", 0, tile_m, tile_n, N)
    sdma_z_s = sdma_ctl_expr("sdma_ctl_z", 1, 1, TILE, TILE)

    return (
        f"{_fetch_weight_tile_c(tile_n, tk1, sdma_z_s)}"
        "\n"
        f"{_push_weights_c(tk1)}"
        "\n"
        "int main() {\n"
        f"    int* cfg = (int*){ADDR_TABLE};\n"
        "    int A_GMEM = *(cfg + 0);\n"
        "    int W_GMEM = *(cfg + 1);\n"
        "    int C_GMEM = *(cfg + 2);\n"
        "    int gM = *(cfg + 3);\n"
        "    int gN = *(cfg + 4);\n"
        "    int gK = *(cfg + 5);\n"
        "    int M_tiles = *(cfg + 6);\n"
        "    int N_tiles = *(cfg + 7);\n"
        "    int K_tiles = *(cfg + 8);\n"
        "    int tile_sz = *(cfg + 9);\n"
        "\n"
        "    int all_mask = -1;\n"
        "    int sp_c = 0;\n"
        "    int sp_s = 0;\n"
        # K-step byte stride along a row of A / W' (tile_sz BF16).  Kept in a
        # register: the compiler encodes `reg + const` below 4096 as addi_s with a
        # signed 12-bit immediate (docs/add_kernel_multitile_fix.md).
        "    int k_step = tile_sz * 2;\n"
        f"{sdma_a_s}"
        f"{sdma_w_s}"
        f"{sdma_c_s}"
        "\n"
        "    int mi = 0;\n"
        "    while (mi < M_tiles) {\n"
        "        int ni = 0;\n"
        "        while (ni < N_tiles) {\n"
        "            int c_off = mi * tile_sz * gN + ni * tile_sz;\n"
        "            int c_byte = c_off * 2;\n"
        "            int c_addr = C_GMEM + c_byte;\n"
        "            scpad_load(sp_c, c_addr, sdma_ctl_c);\n"
        "\n"
        "            int a_off = mi * tile_sz * gK;\n"
        "            int a_byte = a_off * 2;\n"
        "            int a_addr = A_GMEM + a_byte;\n"
        "            int w_off = ni * tile_sz * gK;\n"
        "            int w_byte = w_off * 2;\n"
        "            int w_addr = W_GMEM + w_byte;\n"
        "\n"
        "            fetch_weight_tile(sp_s, w_addr, sdma_ctl_w);\n"
        "\n"
        "            int ki = 0;\n"
        "            while (ki < K_tiles) {\n"
        "                push_weights(sp_s, tile_sz);\n"
        "\n"
        "                scpad_load(sp_s, a_addr, sdma_ctl_a);\n"
        "\n"
        "                int ri = 0;\n"
        f"                while (ri < {tile_m}) {{\n"
        f"                    vec a_row = vector_load(sp_s, ri, {tk1}, 1);\n"
        f"                    vec c_row = vector_load(sp_c, ri, {tn1}, 0);\n"
        "\n"
        "                    vec result = gemm(a_row, c_row, all_mask);\n"
        "\n"
        f"                    vector_store(result, sp_c, ri, {tn1}, 0);\n"
        "                    ri = ri + 1;\n"
        "                }\n"
        "\n"
        "                ki = ki + 1;\n"
        "                a_addr = a_addr + k_step;\n"
        "                w_addr = w_addr + k_step;\n"
        "                if (ki < K_tiles) {\n"
        "                    fetch_weight_tile(sp_s, w_addr, sdma_ctl_w);\n"
        "                }\n"
        "            }\n"
        "\n"
        "            scpad_store(sp_c, c_addr, sdma_ctl_c);\n"
        "            ni = ni + 1;\n"
        "        }\n"
        "        mi = mi + 1;\n"
        "    }\n"
        "\n"
        "    atalla_halt();\n"
        "    return 0;\n"
        "}\n"
    )
