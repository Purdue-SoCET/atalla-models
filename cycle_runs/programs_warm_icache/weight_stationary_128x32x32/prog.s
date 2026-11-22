       .section data
       .section data
       .section code
       global fetch_weight_tile
       type fetch_weight_tile func
       ; packetization: {"instructions_before_packetization": 17, "nop_packets": 2, "packets": 19}
       .align 5
 fetch_weight_tile:
       addi_s x2, x2, -32
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
 fetch_weight_tile_block0:
 fetch_weight_tile_block1:
       sw_s x12, 20(x8)
       nop
       nop
       nop
       sw_s x13, 16(x8)
       nop
       nop
       nop
       sw_s x14, 12(x8)
       nop
       nop
       nop
       lw_s x11, 20(x8)
       nop
       nop
       nop
       lw_s x10, 16(x8)
       nop
       nop
       nop
       lw_s x9, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       scpad_ld x11, x10, x9
       nop
       nop
       nop
 fetch_weight_tile_epilog:
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
       global push_weights
       type push_weights func
       ; packetization: {"instructions_before_packetization": 49, "nop_packets": 62, "packets": 109}
       .align 5
 push_weights:
       addi_s x2, x2, -32
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
       sw_s x33, 8(x2)
       addi_s x8, x2, 8
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
       addi_s x32, x32, -128
       nop
       nop
       nop
 push_weights_block0:
 push_weights_block1:
       sw_s x12, 20(x8)
       lui_s x9, 0
       nop
       nop
       sw_s x13, 16(x8)
       nop
       nop
       nop
       sw_s x9, 12(x8)
       nop
       nop
       nop
 push_weights_block2:
       lw_s x10, 12(x8)
       nop
       nop
       nop
       lw_s x9, 16(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       bge_s x10, x9, push_weights_block4
       nop
       nop
       nop
 push_weights_block3:
       lw_s x10, 20(x8)
       nop
       nop
       nop
       lw_s x9, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 1
       nop
       nop
       nop
       addi_s x10, x33, -64
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 3
       nop
       nop
       nop
       addi_s x10, x33, -64
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 3
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
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
       lw_s x9, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       sw_s x9, 8(x8)
       nop
       nop
       nop
       lw_s x10, 20(x8)
       nop
       nop
       nop
       lw_s x9, 8(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 1
       nop
       nop
       nop
       addi_s x10, x33, -128
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 3
       nop
       nop
       nop
       addi_s x10, x33, -128
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 3
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
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
       lw_s x9, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 2
       nop
       nop
       nop
       sw_s x9, 12(x8)
       nop
       nop
       nop
       jal x0, push_weights_block2
       nop
       nop
       nop
 push_weights_block4:
 push_weights_epilog:
       addi_s x32, x33, 0
       nop
       nop
       nop
       addi_s x2, x2, 16
       nop
       nop
       nop
       lw_s x33, 8(x2)
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
       global main
       type main func
       ; packetization: {"branch_lowering_instructions": 1, "instructions_before_packetization": 245, "nop_packets": 210, "packets": 440}
       .align 5
 main:
       lui_s x32, 15625
       nop
       nop
       nop
       addi_s x2, x2, -144
       nop
       nop
       nop
       addi_s x32, x32, 0
       sw_s x1, 4(x2)
       nop
       nop
       addi_s x33, x32, 0
       sw_s x8, 0(x2)
       nop
       nop
       sw_s x33, 8(x2)
       addi_s x8, x2, 8
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
       addi_s x32, x32, -192
       nop
       nop
       nop
 main_block0:
 main_block1:
       addi_s x9, x0, 60
       nop
       nop
       nop
       sw_s x9, 132(x8)
       lui_s x10, 0
       nop
       nop
       lw_s x9, 132(x8)
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 1
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 128(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 2
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 124(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 3
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 120(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 4
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 116(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 5
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 112(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 6
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 108(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 7
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 104(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 8
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 100(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
       nop
       nop
       nop
       lw_s x9, 0(x9)
       addi_s x10, x0, 9
       nop
       nop
       muli_s x10, x10, 4
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 96(x8)
       nop
       nop
       nop
       lw_s x9, 132(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x9, x10
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
       sw_s x9, 92(x8)
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       sub_s x9, x0, x9
       nop
       nop
       nop
       sw_s x9, 88(x8)
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 84(x8)
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 80(x8)
       nop
       nop
       nop
       lw_s x9, 92(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       muli_s x9, x9, 2
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 76(x8)
       nop
       nop
       nop
       lw_s x10, 92(x8)
       nop
       nop
       nop
       lw_s x9, 108(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x9, x10, x9
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       muli_s x9, x9, 2
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 72(x8)
       nop
       nop
       nop
       lw_s x10, 92(x8)
       nop
       nop
       nop
       lw_s x9, 112(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x9, x10, x9
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       muli_s x9, x9, 2
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 68(x8)
       nop
       nop
       nop
       lui_s x9, 16769024
       nop
       nop
       nop
       addi_s x9, x9, 31
       nop
       nop
       nop
       sw_s x9, 64(x8)
       nop
       nop
       nop
       lui_s x9, 16769024
       nop
       nop
       nop
       addi_s x9, x9, 31
       nop
       nop
       nop
       sw_s x9, 60(x8)
       nop
       nop
       nop
       lui_s x9, 8380416
       nop
       nop
       nop
       addi_s x9, x9, 31
       nop
       nop
       nop
       sw_s x9, 56(x8)
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 52(x8)
       nop
       nop
       nop
 main_block2:
       lw_s x10, 52(x8)
       nop
       nop
       nop
       lw_s x9, 100(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       bge_s x10, x9, main_block4
       nop
       nop
       nop
 main_block3:
       lw_s x11, 52(x8)
       nop
       nop
       nop
       lw_s x10, 92(x8)
       nop
       nop
       nop
       lw_s x9, 108(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x10, x11, x10
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x9, x10, x9
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 48(x8)
       nop
       nop
       nop
       lw_s x9, 48(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       muli_s x9, x9, 2
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 44(x8)
       nop
       nop
       nop
       lw_s x10, 124(x8)
       nop
       nop
       nop
       lw_s x9, 44(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 40(x8)
       nop
       nop
       nop
       lw_s x10, 52(x8)
       nop
       nop
       nop
       lw_s x9, 92(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mul_s x9, x10, x9
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       muli_s x9, x9, 2
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 36(x8)
       nop
       nop
       nop
       lw_s x10, 120(x8)
       nop
       nop
       nop
       lw_s x9, 36(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 32(x8)
       nop
       nop
       nop
       lw_s x9, 128(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       sw_s x9, 28(x8)
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 24(x8)
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
       lw_s x10, 24(x8)
       nop
       nop
       nop
       lw_s x9, 96(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       bge_s x10, x9, main_block7
       nop
       nop
       nop
 main_block6:
       lw_s x12, 80(x8)
       nop
       nop
       nop
       lw_s x13, 40(x8)
       nop
       nop
       nop
       lw_s x14, 60(x8)
       nop
       nop
       nop
       jal x1, fetch_weight_tile
       nop
       nop
       nop
       lw_s x12, 80(x8)
       nop
       nop
       nop
       lw_s x13, 92(x8)
       nop
       nop
       nop
       jal x1, push_weights
       nop
       nop
       nop
       lw_s x9, 28(x8)
       nop
       nop
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
       lw_s x9, 32(x8)
       nop
       nop
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
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 12(x8)
       nop
       nop
       nop
       jal x0, main_block8
       nop
       nop
       nop
 main_block7:
       lw_s x9, 52(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       sw_s x9, 52(x8)
       nop
       nop
       nop
       jal x0, main_block2
       nop
       nop
       nop
 main_block8:
       lw_s x10, 12(x8)
       nop
       nop
       nop
       lw_s x9, 104(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       bge_s x10, x9, main_block10
       nop
       nop
       nop
 main_block9:
       lw_s x11, 84(x8)
       nop
       nop
       nop
       lw_s x10, 16(x8)
       nop
       nop
       nop
       lw_s x9, 56(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       scpad_ld x11, x10, x9
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       lw_s x11, 80(x8)
       nop
       nop
       nop
       lw_s x10, 20(x8)
       nop
       nop
       nop
       lw_s x9, 64(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       scpad_ld x11, x10, x9
       nop
       nop
       nop
       lui_s x9, 0
       nop
       nop
       nop
       sw_s x9, 8(x8)
       nop
       nop
       nop
       jal x0, main_block11
       nop
       nop
       nop
 main_block10:
       lw_s x10, 28(x8)
       nop
       nop
       nop
       lw_s x9, 76(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 28(x8)
       nop
       nop
       nop
       lw_s x10, 40(x8)
       nop
       nop
       nop
       lw_s x9, 76(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 40(x8)
       nop
       nop
       nop
       lw_s x9, 24(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       sw_s x9, 24(x8)
       nop
       nop
       nop
       jal x0, main_block5
       nop
       nop
       nop
 main_block11:
       lw_s x9, 8(x8)
       addi_s x10, x0, 32
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       blt_s x9, x10, main_block13__branch_skip_0
       nop
       nop
       nop
       jal x0, main_block13
       nop
       nop
       nop
 main_block13__branch_skip_0:
 main_block12:
       lw_s x10, 80(x8)
       nop
       nop
       nop
       lw_s x9, 8(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 1
       nop
       nop
       nop
       addi_s x10, x33, -64
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 3
       nop
       nop
       nop
       lw_s x10, 84(x8)
       nop
       nop
       nop
       lw_s x9, 8(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 0
       nop
       nop
       nop
       addi_s x10, x33, -128
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 3
       nop
       nop
       nop
       addi_s x10, x33, -64
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       vreg_ld v2, x10, x9, 31, 3
       nop
       nop
       nop
       addi_s x10, x33, -128
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 3
       nop
       nop
       nop
       lw_s x9, 88(x8)
       addi_s x10, x33, -192
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       mv_stm m1, x9
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       gemm_vv v2, v2, v0, m1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_vv v1, v2, v1, m1
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 3
       nop
       nop
       nop
       addi_s x10, x33, -192
       nop
       nop
       nop
       addi_s x9, x0, 1
       nop
       nop
       nop
       vreg_ld v1, x10, x9, 31, 3
       nop
       nop
       nop
       lw_s x10, 84(x8)
       nop
       nop
       nop
       lw_s x9, 8(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       vreg_st v1, x10, x9, 31, 0
       nop
       nop
       nop
       lw_s x9, 8(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       sw_s x9, 8(x8)
       nop
       nop
       nop
       jal x0, main_block11
       nop
       nop
       nop
 main_block13:
       lw_s x11, 84(x8)
       nop
       nop
       nop
       lw_s x10, 16(x8)
       nop
       nop
       nop
       lw_s x9, 56(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       scpad_st x11, x10, x9
       nop
       nop
       nop
       lw_s x10, 20(x8)
       nop
       nop
       nop
       lw_s x9, 72(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 20(x8)
       nop
       nop
       nop
       lw_s x10, 16(x8)
       nop
       nop
       nop
       lw_s x9, 68(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       add_s x9, x10, x9
       nop
       nop
       nop
       sw_s x9, 16(x8)
       nop
       nop
       nop
       lw_s x9, 12(x8)
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       nop
       addi_s x9, x9, 1
       nop
       nop
       nop
       sw_s x9, 12(x8)
       nop
       nop
       nop
       jal x0, main_block8
       nop
       nop
       nop
 main_epilog:
       addi_s x32, x33, 0
       nop
       nop
       nop
       addi_s x2, x2, 16
       nop
       nop
       nop
       lw_s x33, 8(x2)
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
       addi_s x2, x2, 144
       nop
       nop
       nop
       jalr x0,x1, 0
       nop
       nop
       nop
       .section data
       .section code
