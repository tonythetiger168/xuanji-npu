// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU 向量核的指令格式 (npu_vec / npu_vec_eng 與驗證共用)。語意見 docs/npu_vec.md。
`ifndef NPU_VEC_PKG_SV
`define NPU_VEC_PKG_SV
package npu_vec_pkg;
  localparam int VL = 16;                          // lane 數: 一個 64 B L2 line = 16 × FP32
  typedef enum logic [3:0] {
    VOP_NOP = 4'd0, VOP_LD = 4'd1, VOP_ST = 4'd2, VOP_ADD = 4'd3, VOP_SUB = 4'd4, VOP_MUL = 4'd5,
    VOP_MAX = 4'd6, VOP_SFU = 4'd7, VOP_SPLAT = 4'd8, VOP_RSUM = 4'd9, VOP_RMAX = 4'd10,
    VOP_TOPK = 4'd11
  } vop_e;
  // LD 的 fn: 0 FP32, 1 BF16, 2 FP16, 3 INT32, 4 E4M3, 5 E5M2, 6 E2M1 (MX 比例 = imm[7:0], E8M0)
  // ST 的 fn: 0 FP32 (va), 1 BF16 (va → 元素 0..15, vb → 16..31, RNE)
  // SFU 的 fn: 0 recip, 1 rsqrt, 2 exp2, 3 exp, 4 sinπ, 5 cosπ (rtl/fp/sfu.sv)
  // TOPK 的 fn: 0 值 (vd ← 前 16 大的值), 1 索引 (vd ← 前 16 大的索引; 目前的索引表 = vd 原內容); 候選 = va (索引 imm + lane),
  //   目前的前 16 = vb。排序: FP32 位元的 totalOrder 降序, 同值時索引小者在前 (docs/npu_vec.md)
  typedef struct packed {
    vop_e        op;
    logic [2:0]  fn;
    logic [2:0]  vd, va, vb;
    logic [2:0]  part;                             // LD: 取 line 中第 part 組 16 個元素 (16 / 8 / 4 bit 格式)
    logic [31:0] imm;                              // SPLAT 的值; LD 的 MX 比例
    logic [25:0] line;                             // LD / ST 的 L2 line 位址
  } npu_vec_ins_t;                                 // 77 bit
endpackage : npu_vec_pkg
`endif
