| shape | schedule | rc | cycles | packets | gemm | lw.vi | sdma ld/st | array MAC util | DRAM B (scpad / icache / dcache) | DRAM B/cyc | fsim bytes_moved | dep findings | cos(atalla-sim, ref) | cos(fsim, ref) | atalla-sim == fsim bits | == baseline bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128x32x32 | baseline | 0 | 39635 | 7496 | 128 | 128 | 16/4 | 0.323% | 33024 / 25664 / 640 | 1.497 | 33024 | 20 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | baseline | 0 | 39625 | 7418 | 128 | 128 | 16/4 | 0.323% | 33024 / 25664 / 640 | 1.497 | 33024 | 20 | -0.002035 | 0.999995 | False | True |
| 128x32x32 | double_buffer | 0 | 39280 | 7205 | 128 | 128 | 12/4 | 0.326% | 32768 / 26880 / 512 | 1.532 | 32768 | 12 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | double_buffer | 0 | 36696 | 6980 | 128 | 128 | 9/1 | 0.349% | 20480 / 27008 / 512 | 1.308 | 20480 | 12 | -0.002035 | 0.999995 | False | True |
| 128x32x32 | weight_stationary | 0 | 28763 | 5421 | 128 | 32 | 9/4 | 0.445% | 26624 / 11328 / 512 | 1.337 | 26624 | 12 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | weight_stationary | 0 | 38546 | 7068 | 128 | 128 | 12/4 | 0.332% | 32768 / 27072 / 512 | 1.566 | 32768 | 12 | -0.002035 | 0.999995 | False | True |
