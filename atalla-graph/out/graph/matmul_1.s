       .section data
       .section data
       .section code
       global main
       type main func
 main:
       addi_s x2, x2, -160
       sw_s x1, 4(x2)
       sw_s x8, 0(x2)
       addi_s x8, x2, 8
       addi_s x2, x2, -16
 main_block0:
       jal x0, main_block1
 main_block1:
       li_s x9, 60
       sw_s x9, 148(x8)
       lw_s x9, 148(x8)
       lw_s x9, 0(x9)
       sw_s x9, 144(x8)
       lw_s x9, 148(x8)
       lw_s x9, 4(x9)
       sw_s x9, 140(x8)
       lw_s x9, 148(x8)
       lw_s x9, 8(x9)
       sw_s x9, 136(x8)
       lw_s x9, 148(x8)
       lw_s x9, 12(x9)
       sw_s x9, 132(x8)
       lw_s x9, 148(x8)
       lw_s x9, 16(x9)
       sw_s x9, 128(x8)
       lw_s x9, 148(x8)
       lw_s x9, 20(x9)
       sw_s x9, 124(x8)
       lw_s x9, 148(x8)
       lw_s x9, 24(x9)
       sw_s x9, 120(x8)
       lw_s x9, 148(x8)
       lw_s x9, 28(x9)
       sw_s x9, 116(x8)
       lw_s x9, 148(x8)
       lw_s x9, 32(x9)
       sw_s x9, 112(x8)
       lw_s x9, 148(x8)
       lw_s x9, 36(x9)
       sw_s x9, 108(x8)
       li_s x9, 1
       sub_s x9, x0, x9
       sw_s x9, 104(x8)
       li_s x9, 1
       sw_s x9, 100(x8)
       li_s x9, 0
       sw_s x9, 96(x8)
       li_s x9, 0
       sw_s x9, 92(x8)
       li_s x9, 0
       sw_s x9, 88(x8)
       li_s x9, 32505895
       sw_s x9, 84(x8)
       li_s x9, 2146435111
       sw_s x9, 80(x8)
       li_s x9, 1106247719
       sw_s x9, 76(x8)
       li_s x9, 0
       sw_s x9, 72(x8)
       jal x0, main_block2
 main_block2:
       lw_s x9, 72(x8)
       addi_s x10, x9, 0
       lw_s x9, 120(x8)
       blt_s x10, x9, main_block3
       jal x0, main_block4
 main_block3:
       li_s x9, 0
       sw_s x9, 68(x8)
       jal x0, main_block5
 main_block4:
       halt
       li_s x9, 0
       jal x0, main_epilog
 main_block5:
       lw_s x9, 68(x8)
       addi_s x10, x9, 0
       lw_s x9, 116(x8)
       blt_s x10, x9, main_block6
       jal x0, main_block7
 main_block6:
       li_s x9, 0
       sw_s x9, 64(x8)
       jal x0, main_block8
 main_block7:
       lw_s x9, 72(x8)
       addi_s x9, x9, 1
       sw_s x9, 72(x8)
       jal x0, main_block2
 main_block8:
       lw_s x9, 64(x8)
       addi_s x10, x9, 0
       lw_s x9, 112(x8)
       blt_s x10, x9, main_block9
       jal x0, main_block10
 main_block9:
       lw_s x9, 72(x8)
       addi_s x14, x9, 0
       lw_s x9, 108(x8)
       addi_s x13, x9, 0
       lw_s x9, 124(x8)
       addi_s x10, x9, 0
       lw_s x9, 64(x8)
       addi_s x12, x9, 0
       lw_s x9, 108(x8)
       addi_s x11, x9, 0
       mul_s x9, x14, x13
       mul_s x10, x9, x10
       mul_s x9, x12, x11
       add_s x9, x10, x9
       sw_s x9, 60(x8)
       lw_s x9, 60(x8)
       muli_s x9, x9, 2
       sw_s x9, 56(x8)
       lw_s x9, 144(x8)
       addi_s x10, x9, 0
       lw_s x9, 56(x8)
       add_s x9, x10, x9
       sw_s x9, 52(x8)
       lw_s x9, 64(x8)
       addi_s x14, x9, 0
       lw_s x9, 108(x8)
       addi_s x13, x9, 0
       lw_s x9, 128(x8)
       addi_s x10, x9, 0
       lw_s x9, 68(x8)
       addi_s x12, x9, 0
       lw_s x9, 108(x8)
       addi_s x11, x9, 0
       mul_s x9, x14, x13
       mul_s x10, x9, x10
       mul_s x9, x12, x11
       add_s x9, x10, x9
       sw_s x9, 48(x8)
       lw_s x9, 48(x8)
       muli_s x9, x9, 2
       sw_s x9, 44(x8)
       lw_s x9, 140(x8)
       addi_s x10, x9, 0
       lw_s x9, 44(x8)
       add_s x9, x10, x9
       sw_s x9, 40(x8)
       lw_s x9, 92(x8)
       addi_s x11, x9, 0
       lw_s x9, 40(x8)
       addi_s x10, x9, 0
       lw_s x9, 80(x8)
       scpad_ld x11, x10, x9
       li_s x9, 0
       sw_s x9, 36(x8)
       jal x0, main_block11
 main_block10:
       lw_s x9, 68(x8)
       addi_s x9, x9, 1
       sw_s x9, 68(x8)
       jal x0, main_block5
 main_block11:
       lw_s x9, 36(x8)
       addi_s x10, x9, 0
       li_s x9, 32
       blt_s x10, x9, main_block12
       jal x0, main_block13
 main_block12:
       lw_s x9, 92(x8)
       addi_s x10, x9, 0
       lw_s x9, 36(x8)
       add_s x9, x10, x9
       sw_s x9, 32(x8)
       lw_s x9, 32(x8)
       addi_s x11, x9, 0
       lw_s x9, 100(x8)
       addi_s x10, x33, -64
       vreg_ld v1, x11, x9, 31, 1
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -64
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_vi v0, v1, 0, m0
       lw_s x9, 36(x8)
       addi_s x9, x9, 1
       sw_s x9, 36(x8)
       jal x0, main_block11
 main_block13:
       lw_s x9, 96(x8)
       addi_s x11, x9, 0
       lw_s x9, 52(x8)
       addi_s x10, x9, 0
       lw_s x9, 84(x8)
       scpad_ld x11, x10, x9
       lw_s x9, 72(x8)
       addi_s x14, x9, 0
       lw_s x9, 108(x8)
       addi_s x13, x9, 0
       lw_s x9, 128(x8)
       addi_s x10, x9, 0
       lw_s x9, 68(x8)
       addi_s x12, x9, 0
       lw_s x9, 108(x8)
       addi_s x11, x9, 0
       mul_s x9, x14, x13
       mul_s x10, x9, x10
       mul_s x9, x12, x11
       add_s x9, x10, x9
       sw_s x9, 28(x8)
       lw_s x9, 28(x8)
       muli_s x9, x9, 2
       sw_s x9, 24(x8)
       lw_s x9, 136(x8)
       addi_s x10, x9, 0
       lw_s x9, 24(x8)
       add_s x9, x10, x9
       sw_s x9, 20(x8)
       lw_s x9, 88(x8)
       addi_s x11, x9, 0
       lw_s x9, 20(x8)
       addi_s x10, x9, 0
       lw_s x9, 76(x8)
       scpad_ld x11, x10, x9
       li_s x9, 0
       sw_s x9, 16(x8)
       jal x0, main_block14
 main_block14:
       lw_s x9, 16(x8)
       addi_s x10, x9, 0
       li_s x9, 1
       blt_s x10, x9, main_block15
       jal x0, main_block16
 main_block15:
       lw_s x9, 16(x8)
       addi_s x11, x9, 0
       lw_s x9, 100(x8)
       addi_s x10, x33, -128
       vreg_ld v1, x11, x9, 31, 0
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       lw_s x9, 16(x8)
       addi_s x11, x9, 0
       lw_s x9, 100(x8)
       addi_s x10, x33, -192
       vreg_ld v1, x11, x9, 31, 1
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -128
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       add_vv v2, v1, v0, m0
       addi_s x10, x33, -192
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 104(x8)
       addi_s x10, x33, -256
       mv_stm m1, x9
       gemm_vv v1, v2, v1, m1
       li_s x9, 1
       vreg_st v1, x10, x9, 31, 0
       addi_s x10, x33, -256
       li_s x9, 1
       vreg_ld v1, x10, x9, 31, 0
       lw_s x9, 16(x8)
       addi_s x10, x9, 0
       lw_s x9, 100(x8)
       vreg_st v1, x10, x9, 31, 1
       lw_s x9, 16(x8)
       addi_s x9, x9, 1
       sw_s x9, 16(x8)
       jal x0, main_block14
 main_block16:
       lw_s x9, 88(x8)
       addi_s x11, x9, 0
       lw_s x9, 20(x8)
       addi_s x10, x9, 0
       lw_s x9, 76(x8)
       scpad_st x11, x10, x9
       lw_s x9, 64(x8)
       addi_s x9, x9, 1
       sw_s x9, 64(x8)
       jal x0, main_block8
 main_epilog:
       addi_s x10, x9, 0
       addi_s x2, x2, 16
       lw_s x1, 4(x2)
       lw_s x8, 0(x2)
       addi_s x2, x2, 160
       jalr x0,x1, 0
       .section data
       .section code
       .align 4
