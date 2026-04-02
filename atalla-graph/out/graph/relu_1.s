       .section data
       .section data
       .section code
       global main
       type main func
 main:
       addi_s x2, x2, -64
       sw_s x1, 4(x2)
       sw_s x8, 0(x2)
       addi_s x8, x2, 8
       addi_s x2, x2, -16
 main_block0:
       jal x0, main_block1
 main_block1:
       li_s x9, 60
       sw_s x9, 52(x8)
       lw_s x9, 52(x8)
       lw_s x9, 0(x9)
       sw_s x9, 48(x8)
       lw_s x9, 52(x8)
       lw_s x9, 4(x9)
       sw_s x9, 44(x8)
       li_s x9, 0
       sw_s x9, 40(x8)
       li_s x9, 1
       sub_s x9, x0, x9
       sw_s x9, 36(x8)
       li_s x9, 1
       sw_s x9, 32(x8)
       li_s x9, 267386911
       sw_s x9, 28(x8)
       lw_s x9, 32(x8)
       addi_s x11, x9, 0
       addi_s x10, x33, -64
       li_s x9, 0
       vreg_ld v1, x9, x11, 31, 0
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -64
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 36(x8)
       addi_s x11, x9, 0
       addi_s x10, x33, -64
       li_s x9, 0
       mv_stm m1, x11
       mul_vs v1, v1, x9, m1
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       li_s x9, 0
       sw_s x9, 24(x8)
       jal x0, main_block2
 main_block2:
       lw_s x9, 24(x8)
       addi_s x10, x9, 0
       li_s x9, 1
       blt_s x10, x9, main_block3
       jal x0, main_block4
 main_block3:
       lw_s x9, 40(x8)
       addi_s x11, x9, 0
       lw_s x9, 48(x8)
       addi_s x10, x9, 0
       lw_s x9, 28(x8)
       scpad_ld x11, x10, x9
       li_s x9, 0
       sw_s x9, 20(x8)
       jal x0, main_block5
 main_block4:
       halt
       li_s x9, 0
       jal x0, main_epilog
 main_block5:
       lw_s x9, 20(x8)
       addi_s x10, x9, 0
       li_s x9, 8
       blt_s x10, x9, main_block6
       jal x0, main_block7
 main_block6:
       lw_s x9, 20(x8)
       addi_s x11, x9, 0
       lw_s x9, 32(x8)
       addi_s x10, x33, -128
       vreg_ld v1, x11, x9, 31, 0
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -128
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       add_vv v2, v1, v0, m0
       addi_s x10, x33, -64
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 36(x8)
       mv_stm m1, x9
       mlt_mvv m1, v2, v1, m1
       mv_mts x9, m1
       sw_s x9, 16(x8)
       addi_s x10, x33, -128
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 16(x8)
       addi_s x11, x9, 0
       addi_s x10, x33, -192
       li_s x9, 0
       mv_stm m1, x11
       mul_vs v1, v1, x9, m1
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -192
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 20(x8)
       addi_s x10, x9, 0
       lw_s x9, 32(x8)
       vreg_st v1, x10, x9, 31, 0
       lw_s x9, 20(x8)
       addi_s x9, x9, 1
       sw_s x9, 20(x8)
       jal x0, main_block5
 main_block7:
       lw_s x9, 40(x8)
       addi_s x11, x9, 0
       lw_s x9, 44(x8)
       addi_s x10, x9, 0
       lw_s x9, 28(x8)
       scpad_st x11, x10, x9
       lw_s x9, 48(x8)
       addi_s x9, x9, 512
       sw_s x9, 48(x8)
       lw_s x9, 44(x8)
       addi_s x9, x9, 512
       sw_s x9, 44(x8)
       lw_s x9, 24(x8)
       addi_s x9, x9, 1
       sw_s x9, 24(x8)
       jal x0, main_block2
 main_epilog:
       addi_s x10, x9, 0
       addi_s x2, x2, 16
       lw_s x1, 4(x2)
       lw_s x8, 0(x2)
       addi_s x2, x2, 64
       jalr x0,x1, 0
       .section data
       .section code
       .align 4
