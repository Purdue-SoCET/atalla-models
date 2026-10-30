"""Runtime configuration ABI shared by Python emitters and standalone C kernels.

Dictionary keys are byte offsets from ADDR_TABLE. Integer values are u32
words, floats are IEEE FP32 words. Configuration never contains C source.
"""
import math

ADDR_TABLE = 60
TILE = 32


def sdma_ctl_val(sid: int, num_rows: int, num_cols: int, full_cols: int) -> int:
    return (sid << 30) | ((num_rows - 1) << 25) | ((num_cols - 1) << 20) | (full_cols - 1)


def elementwise_params(total: int, width: int, binary: bool):
    rows = math.ceil(total / width)
    sp_rows = min(rows, TILE)
    tile_count = math.ceil(rows / sp_rows)
    ctl0 = sdma_ctl_val(0, sp_rows, width, width)
    values = [tile_count, sp_rows, ctl0]
    if binary:
        values.append(sdma_ctl_val(1, sp_rows, width, width))
    values.append(sp_rows * width * 2)
    start = 12 if binary else 8
    return {start + 4*i: value for i,value in enumerate(values)}


def padded_elementwise_rows(total: int, width: int):
    rows = math.ceil(total / width)
    sp_rows = min(rows, TILE)
    return math.ceil(rows / sp_rows) * sp_rows


def gemm_params(M: int, N: int, K: int):
    tm, tn, tk = min(TILE,M), min(TILE,N), min(TILE,K)
    ks = math.ceil(max(K,1)/TILE)*TILE
    return {44: sdma_ctl_val(0,1,TILE,TILE), 48: tn, 52: (1 << tk)-1,
            56: tm, 60: sdma_ctl_val(0,tm,tk,ks),
            64: sdma_ctl_val(1,tn,tk,ks), 68: sdma_ctl_val(1,tm,tn,N)}


def softmax_params(rows: int, width: int):
    return {8: rows, 12: width*2, 16: sdma_ctl_val(0,1,width,width), 20: (1 << width)-1}


def maxpool_params(H: int, W: int, C: int, pool: int, stride: int):
    ho = (H-pool)//stride+1
    return {8:C,12:ho,16:pool,20:stride,24:H*W*2,28:ho*W*2,
            32:sdma_ctl_val(0,H,W,W),36:sdma_ctl_val(0,ho,W,W)}


def layernorm_params(D: int, eps: float):
    # Preserve the decimal precision of the former generated float literals.
    return {20:D,24:float(f'{1.0/D:.10f}'),28:float(f'{eps:.8f}'),
            32:D//TILE,36:sdma_ctl_val(0,1,TILE,TILE)}
