| shape | schedule | rc | cycles | packets | gemm | lw.vi | sdma ld/st | array MAC util | DRAM B (scpad / icache / dcache) | DRAM B/cyc | fsim bytes_moved | dep findings | cos(atalla-sim, ref) | cos(fsim, ref) | atalla-sim == fsim bits | == baseline bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128x32x32 | baseline | 0 | 39053 | 7496 | 128 | 128 | 16/4 | 0.328% | 33024 / 20096 / 640 | 1.377 | 33024 | 20 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | baseline | 0 | 39053 | 7418 | 128 | 128 | 16/4 | 0.328% | 33024 / 20096 / 640 | 1.377 | 33024 | 20 | -0.002035 | 0.999995 | False | True |
| 128x32x32 | double_buffer | 0 | 38722 | 7205 | 128 | 128 | 12/4 | 0.331% | 32768 / 21888 / 512 | 1.425 | 32768 | 12 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | double_buffer | 0 | 36136 | 6980 | 128 | 128 | 9/1 | 0.354% | 20480 / 22016 / 512 | 1.190 | 20480 | 12 | -0.002035 | 0.999995 | False | True |
| 128x32x32 | weight_stationary | 0 | 28220 | 5421 | 128 | 32 | 9/4 | 0.454% | 26624 / 6400 / 512 | 1.188 | 26624 | 12 | -0.040543 | 0.999999 | False | True |
| 32x32x128 | weight_stationary | 0 | 38004 | 7068 | 128 | 128 | 12/4 | 0.337% | 32768 / 22144 / 512 | 1.458 | 32768 | 12 | -0.002035 | 0.999995 | False | True |
