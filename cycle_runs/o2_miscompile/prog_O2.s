       .section data
       .section data
       .section code
       global load_weight_tile
       type load_weight_tile func
       ; packetization: {"instructions_before_packetization": 35, "nop_packets": 13, "packets": 45}
       .align 5
 load_weight_tile:
       addi_s x2, x2, -16
       nop
       nop
       nop
       sw_s x1, 4(x2)
       nop
       nop
       nop
       sw_s x8, 0(x2)
       nop
       nop
       nop
       addi_s x8, x2, 8
       nop
       nop
       nop
       addi_s x2, x2, -16
       nop
       nop
       nop
 load_weight_tile_block0:
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 40
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 60
       nop
       nop
       addi_s x10, x10, 36
       nop
       nop
       nop
       lui_s x11, 253952
       lw_s x10, 0(x10)
       nop
       nop
       lui_s x15, 0
       nop
       nop
       nop
       addi_s x11, x11, 31
       nop
       nop
       nop
       scpad_ld x15, x9, x11
       nop
       nop
       nop
       lui_s x11, 0
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       vreg_ld v1, x11, x9, 31, 0
       nop
       nop
       nop
       scpad_ld x12, x13, x14
       addi_s x9, x0, 32
       nop
       nop
 load_weight_tile_block2:
       bge_s x9, x10, load_weight_tile_block4
       nop
       nop
       nop
 load_weight_tile_block3:
       vreg_st v1, x12, x9, 31, 1
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       jal x0, load_weight_tile_block2
       nop
       nop
       nop
 load_weight_tile_block4:
       lui_s x9, 0
       nop
       nop
       nop
 load_weight_tile_block5:
       bge_s x9, x10, load_weight_tile_block7
       nop
       nop
       nop
 load_weight_tile_block6:
       vreg_ld v1, x12, x9, 31, 1
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       lw_vi v0, v1, 0, m0
       nop
       nop
       nop
       jal x0, load_weight_tile_block5
       nop
       nop
       nop
 load_weight_tile_block7:
 load_weight_tile_epilog:
       addi_s x2, x2, 16
       nop
       nop
       nop
       lw_s x1, 4(x2)
       nop
       nop
       nop
       lw_s x8, 0(x2)
       nop
       nop
       nop
       addi_s x2, x2, 16
       nop
       nop
       nop
       jalr x0,x1, 0
       nop
       nop
       nop
       .section data
       .section code
       global main
       type main func
       ; packetization: {"instructions_before_packetization": 110, "nop_packets": 79, "packets": 180}
       .align 5
 main:
       addi_s x2, x2, -32
       nop
       nop
       nop
       sw_s x1, 4(x2)
       lui_s x32, 15625
       nop
       nop
       sw_s x8, 0(x2)
       addi_s x32, x32, 0
       nop
       nop
       addi_s x8, x2, 8
       nop
       nop
       nop
       addi_s x33, x32, 0
       nop
       nop
       nop
       addi_s x2, x2, -16
       nop
       nop
       nop
 main_block0:
       addi_s x9, x0, 60
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x11, x0, 1
       nop
       nop
       lui_s x20, 0
       nop
       nop
       nop
       sub_s x11, x0, x11
       nop
       nop
       nop
       sw_s x9, 12(x8)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 4
       nop
       nop
       nop
       lw_s x9, 0(x9)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 16(x8)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 8
       nop
       nop
       nop
       lw_s x9, 0(x9)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 20(x8)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 16
       nop
       nop
       nop
       lw_s x27, 0(x9)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 20
       nop
       nop
       nop
       lw_s x23, 0(x9)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 24
       nop
       nop
       nop
       lw_s x26, 0(x9)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 28
       nop
       nop
       nop
       lw_s x22, 0(x9)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 32
       nop
       nop
       nop
       lw_s x21, 0(x9)
       nop
       nop
       nop
       addi_s x9, x0, 60
       nop
       nop
       nop
       addi_s x9, x9, 36
       nop
       nop
       nop
       lw_s x19, 0(x9)
       nop
       nop
       nop
 main_block2:
       bge_s x20, x26, main_block4
       nop
       nop
       nop
 main_block3:
       lui_s x18, 0
       nop
       nop
       nop
       jal x0, main_block5
       nop
       nop
       nop
 main_block4:
       halt
       nop
       nop
       nop
       lui_s x10, 0
       nop
       nop
       nop
       jal x0, main_epilog
       nop
       nop
       nop
 main_block5:
       bge_s x18, x22, main_block7
       nop
       nop
       nop
 main_block6:
       lui_s x9, 0
       nop
       nop
       nop
       jal x0, main_block8
       nop
       nop
       nop
 main_block7:
       addi_s x20, x20, 1
       nop
       nop
       nop
       jal x0, main_block2
       nop
       nop
       nop
 main_block8:
       bge_s x9, x21, main_block10
       nop
       nop
       nop
 main_block9:
       mul_s x25, x18, x19
       lui_s x14, 16769024
       nop
       nop
       mul_s x24, x9, x19
       lui_s x12, 0
       nop
       nop
       addi_s x14, x14, 127
       nop
       nop
       nop
       mul_s x10, x25, x23
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x10, x10, x24
       nop
       nop
       nop
       muli_s x13, x10, 2
       nop
       nop
       nop
       lw_s x10, 16(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x13, x10, x13
       nop
       nop
       nop
       jal x1, load_weight_tile
       nop
       nop
       nop
       mul_s x12, x20, x19
       lui_s x14, 0
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x10, x12, x23
       nop
       nop
       nop
       mul_s x12, x12, x27
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x10, x10, x24
       nop
       nop
       nop
       muli_s x13, x10, 2
       add_s x12, x12, x25
       nop
       nop
       lw_s x10, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x10, x10, x13
       nop
       nop
       nop
       lui_s x13, 8380416
       nop
       nop
       nop
       addi_s x13, x13, 127
       nop
       nop
       nop
       scpad_ld x14, x10, x13
       nop
       nop
       nop
       muli_s x13, x12, 2
       nop
       nop
       nop
       lw_s x12, 20(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x13, x12, x13
       nop
       nop
       nop
       lui_s x12, 16769024
       nop
       nop
       nop
       addi_s x12, x12, 31
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       lui_s x14, 0
       nop
       nop
       nop
       scpad_ld x14, x13, x12
       nop
       nop
       nop
       lui_s x12, 0
       nop
       nop
       nop
       jal x0, main_block11
       nop
       nop
       nop
 main_block10:
       addi_s x18, x18, 1
       nop
       nop
       nop
       jal x0, main_block5
       nop
       nop
       nop
 main_block11:
       addi_s x14, x0, 32
       nop
       nop
       nop
       bge_s x12, x14, main_block13
       nop
       nop
       nop
 main_block12:
       lui_s x14, 0
       mv_stm m1, x11
       nop
       nop
       vreg_ld v1, x14, x12, 31, 0
       nop
       nop
       nop
       lui_s x14, 0
       nop
       nop
       nop
       vreg_ld v2, x14, x12, 31, 1
       nop
       nop
       nop
       lui_s x14, 0
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       gemm_vv v3, v1, v0, m1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_vv v2, v3, v2, m1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v2, x14, x12, 31, 1
       nop
       nop
       nop
       addi_s x12, x12, 1
       nop
       nop
       nop
       jal x0, main_block11
       nop
       nop
       nop
 main_block13:
       lui_s x14, 16769024
       nop
       nop
       nop
       lui_s x15, 0
       nop
       nop
       nop
       addi_s x14, x14, 31
       nop
       nop
       nop
       scpad_st x15, x13, x14
       addi_s x9, x9, 1
       nop
       nop
       jal x0, main_block8
       nop
       nop
       nop
 main_epilog:
       addi_s x2, x2, 16
       nop
       nop
       nop
       lw_s x1, 4(x2)
       nop
       nop
       nop
       lw_s x8, 0(x2)
       nop
       nop
       nop
       addi_s x2, x2, 32
       nop
       nop
       nop
       jalr x0,x1, 0
       nop
       nop
       nop
       .section data
       .section code
