       .section data
       .section data
       .section code
       global load_weight_tile
       type load_weight_tile func
       .align 5
 load_weight_tile:
       addi_s x2, x2, -48
       sw_s x1, 4(x2)
       sw_s x8, 0(x2)
       sw_s x33, 8(x2)
       addi_s x8, x2, 8
       addi_s x2, x2, -16
       addi_s x33, x32, 0
       addi_s x32, x32, -128
 load_weight_tile_block0:
 load_weight_tile_block1:
       sw_s x12, 36(x8)
       sw_s x13, 32(x8)
       sw_s x14, 28(x8)
       addi_s x9, x0, 60
       sw_s x9, 24(x8)
       lw_s x9, 24(x8)
       addi_s x10, x0, 10
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 20(x8)
       lw_s x9, 24(x8)
       addi_s x10, x0, 9
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 16(x8)
       lui_s x9, 253952
       addi_s x9, x9, 31
       sw_s x9, 12(x8)
       lw_s x10, 20(x8)
       lw_s x9, 12(x8)
       lui_s x11, 0
       scpad_ld x11, x10, x9
       lui_s x10, 0
       lui_s x9, 0
       vreg_ld v1, x10, x9, 31, 0
       addi_s x10, x33, -64
       addi_s x9, x0, 1
       vreg_st v1, x10, x9, 31, 3
       lw_s x11, 36(x8)
       lw_s x10, 32(x8)
       lw_s x9, 28(x8)
       scpad_ld x11, x10, x9
       addi_s x9, x0, 32
       sw_s x9, 8(x8)
 load_weight_tile_block2:
       lw_s x10, 8(x8)
       lw_s x9, 16(x8)
       bge_s x10, x9, load_weight_tile_block4
 load_weight_tile_block3:
       addi_s x10, x33, -64
       addi_s x9, x0, 1
       vreg_ld v1, x10, x9, 31, 3
       lw_s x10, 36(x8)
       lw_s x9, 8(x8)
       vreg_st v1, x10, x9, 31, 1
       lw_s x9, 8(x8)
       addi_s x9, x9, 1
       sw_s x9, 8(x8)
       jal x0, load_weight_tile_block2
 load_weight_tile_block4:
       lui_s x9, 0
       sw_s x9, 4(x8)
 load_weight_tile_block5:
       lw_s x10, 4(x8)
       lw_s x9, 16(x8)
       bge_s x10, x9, load_weight_tile_block7
 load_weight_tile_block6:
       lw_s x10, 36(x8)
       lw_s x9, 4(x8)
       vreg_ld v1, x10, x9, 31, 1
       addi_s x10, x33, -128
       addi_s x9, x0, 1
       vreg_st v1, x10, x9, 31, 3
       addi_s x10, x33, -128
       addi_s x9, x0, 1
       vreg_ld v1, x10, x9, 31, 3
       lw_vi v0, v1, 0, m0
       lw_s x9, 4(x8)
       addi_s x9, x9, 1
       sw_s x9, 4(x8)
       jal x0, load_weight_tile_block5
 load_weight_tile_block7:
 load_weight_tile_epilog:
       addi_s x32, x33, 0
       addi_s x2, x2, 16
       lw_s x33, 8(x2)
       lw_s x1, 4(x2)
       lw_s x8, 0(x2)
       addi_s x2, x2, 48
       jalr x0,x1, 0
       .section data
       .section code
       global main
       type main func
       .align 5
 main:
       lui_s x32, 15625
       addi_s x32, x32, 0
       addi_s x33, x32, 0
       addi_s x2, x2, -144
       sw_s x1, 4(x2)
       sw_s x8, 0(x2)
       sw_s x33, 8(x2)
       addi_s x8, x2, 8
       addi_s x2, x2, -16
       addi_s x33, x32, 0
       addi_s x32, x32, -192
 main_block0:
 main_block1:
       addi_s x9, x0, 60
       sw_s x9, 132(x8)
       lw_s x9, 132(x8)
       lui_s x10, 0
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 128(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 1
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 124(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 2
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 120(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 3
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 116(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 4
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 112(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 5
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 108(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 6
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 104(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 7
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 100(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 8
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 96(x8)
       lw_s x9, 132(x8)
       addi_s x10, x0, 9
       muli_s x10, x10, 4
       add_s x9, x9, x10
       lw_s x9, 0(x9)
       sw_s x9, 92(x8)
       addi_s x9, x0, 1
       sub_s x9, x0, x9
       sw_s x9, 88(x8)
       lui_s x9, 0
       sw_s x9, 84(x8)
       lui_s x9, 0
       sw_s x9, 80(x8)
       lui_s x9, 0
       sw_s x9, 76(x8)
       lui_s x9, 8380416
       addi_s x9, x9, 127
       sw_s x9, 72(x8)
       lui_s x9, 16769024
       addi_s x9, x9, 127
       sw_s x9, 68(x8)
       lui_s x9, 16769024
       addi_s x9, x9, 31
       sw_s x9, 64(x8)
       lui_s x9, 0
       sw_s x9, 60(x8)
 main_block2:
       lw_s x10, 60(x8)
       lw_s x9, 104(x8)
       bge_s x10, x9, main_block4
 main_block3:
       lui_s x9, 0
       sw_s x9, 56(x8)
       jal x0, main_block5
 main_block4:
       halt
       lui_s x10, 0
       jal x0, main_epilog
 main_block5:
       lw_s x10, 56(x8)
       lw_s x9, 100(x8)
       bge_s x10, x9, main_block7
 main_block6:
       lui_s x9, 0
       sw_s x9, 52(x8)
       jal x0, main_block8
 main_block7:
       lw_s x9, 60(x8)
       addi_s x9, x9, 1
       sw_s x9, 60(x8)
       jal x0, main_block2
 main_block8:
       lw_s x10, 52(x8)
       lw_s x9, 96(x8)
       bge_s x10, x9, main_block10
 main_block9:
       lw_s x13, 60(x8)
       lw_s x12, 92(x8)
       lw_s x11, 108(x8)
       lw_s x10, 52(x8)
       lw_s x9, 92(x8)
       mul_s x12, x13, x12
       mul_s x11, x12, x11
       mul_s x9, x10, x9
       add_s x9, x11, x9
       sw_s x9, 48(x8)
       lw_s x9, 48(x8)
       muli_s x9, x9, 2
       sw_s x9, 44(x8)
       lw_s x10, 128(x8)
       lw_s x9, 44(x8)
       add_s x9, x10, x9
       sw_s x9, 40(x8)
       lw_s x13, 56(x8)
       lw_s x12, 92(x8)
       lw_s x11, 108(x8)
       lw_s x10, 52(x8)
       lw_s x9, 92(x8)
       mul_s x12, x13, x12
       mul_s x11, x12, x11
       mul_s x9, x10, x9
       add_s x9, x11, x9
       sw_s x9, 36(x8)
       lw_s x9, 36(x8)
       muli_s x9, x9, 2
       sw_s x9, 32(x8)
       lw_s x10, 124(x8)
       lw_s x9, 32(x8)
       add_s x9, x10, x9
       sw_s x9, 28(x8)
       lw_s x12, 80(x8)
       lw_s x13, 28(x8)
       lw_s x14, 68(x8)
       jal x1, load_weight_tile
       lw_s x11, 84(x8)
       lw_s x10, 40(x8)
       lw_s x9, 72(x8)
       scpad_ld x11, x10, x9
       lw_s x13, 60(x8)
       lw_s x12, 92(x8)
       lw_s x11, 112(x8)
       lw_s x10, 56(x8)
       lw_s x9, 92(x8)
       mul_s x12, x13, x12
       mul_s x11, x12, x11
       mul_s x9, x10, x9
       add_s x9, x11, x9
       sw_s x9, 24(x8)
       lw_s x9, 24(x8)
       muli_s x9, x9, 2
       sw_s x9, 20(x8)
       lw_s x10, 120(x8)
       lw_s x9, 20(x8)
       add_s x9, x10, x9
       sw_s x9, 16(x8)
       lw_s x11, 76(x8)
       lw_s x10, 16(x8)
       lw_s x9, 64(x8)
       scpad_ld x11, x10, x9
       lui_s x9, 0
       sw_s x9, 12(x8)
       jal x0, main_block11
 main_block10:
       lw_s x9, 56(x8)
       addi_s x9, x9, 1
       sw_s x9, 56(x8)
       jal x0, main_block5
 main_block11:
       lw_s x9, 12(x8)
       addi_s x10, x0, 32
       bge_s x9, x10, main_block13
 main_block12:
       lw_s x10, 84(x8)
       lw_s x9, 12(x8)
       vreg_ld v1, x10, x9, 31, 0
       addi_s x10, x33, -64
       addi_s x9, x0, 1
       vreg_st v1, x10, x9, 31, 3
       lw_s x10, 76(x8)
       lw_s x9, 12(x8)
       vreg_ld v1, x10, x9, 31, 1
       addi_s x10, x33, -128
       addi_s x9, x0, 1
       vreg_st v1, x10, x9, 31, 3
       addi_s x10, x33, -64
       addi_s x9, x0, 1
       vreg_ld v2, x10, x9, 31, 3
       addi_s x10, x33, -128
       addi_s x9, x0, 1
       vreg_ld v1, x10, x9, 31, 3
       lw_s x9, 88(x8)
       addi_s x10, x33, -192
       mv_stm m1, x9
       gemm_vv v2, v2, v0, m1
       add_vv v1, v2, v1, m1
       addi_s x9, x0, 1
       vreg_st v1, x10, x9, 31, 3
       addi_s x10, x33, -192
       addi_s x9, x0, 1
       vreg_ld v1, x10, x9, 31, 3
       lw_s x10, 76(x8)
       lw_s x9, 12(x8)
       vreg_st v1, x10, x9, 31, 1
       lw_s x9, 12(x8)
       addi_s x9, x9, 1
       sw_s x9, 12(x8)
       jal x0, main_block11
 main_block13:
       lw_s x11, 76(x8)
       lw_s x10, 16(x8)
       lw_s x9, 64(x8)
       scpad_st x11, x10, x9
       lw_s x9, 52(x8)
       addi_s x9, x9, 1
       sw_s x9, 52(x8)
       jal x0, main_block8
 main_epilog:
       addi_s x32, x33, 0
       addi_s x2, x2, 16
       lw_s x33, 8(x2)
       lw_s x1, 4(x2)
       lw_s x8, 0(x2)
       addi_s x2, x2, 144
       jalr x0,x1, 0
       .section data
       .section code
