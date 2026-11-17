// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// Xuanji NPU package
// File: rtl/common/npu_pkg.sv
// Description: NPU parameters, data types, command format and L2 address
//              mapping. Split out of xuanji-core's orca_pkg when the NPU moved
//              to its own repository; xuanji-core imports this package for the
//              legacy AI tile and the cloud SoCs.
//=============================================================================

`ifndef NPU_PKG_SV
`define NPU_PKG_SV

package npu_pkg;

  // ---------------------------------------------------------------------------
  // AI Tile Parameters (ORCA-NPU v3)
  // ---------------------------------------------------------------------------
  parameter int AI_TILES          = 4;     // 4 AI tiles
  parameter int CLUSTERS_PER_TILE = 16;    // 16 clusters per tile
  parameter int CUS_PER_CLUSTER   = 4;     // 4 compute units per cluster
  parameter int PE_ARRAY_DIM      = 64;    // 64x64 systolic array
  parameter int AI_L2_SIZE_MB     = 64;    // 64 MB L2 SRAM per AI tile
  parameter int HBM3_STACKS       = 3;     // 3 HBM3 stacks per AI tile
  parameter int HBM3_CHANNELS     = 16;    // 16 channels per stack
  // Data type encodings (4-bit, npu_pe/npu_acc/systolic/attn)
  // Weights/activations per byte lane (packed low-bit formats):
  //   INT8/FP8/BF8 : 1/byte | INT4/NF4/FP4 : 2/byte | BITNET(1.58b) : 4/byte
  // Integer types (sub-byte packed where noted)
  parameter int AI_DTYPE_INT8     = 4'd0;  // 1/byte
  parameter int AI_DTYPE_INT4     = 4'd1;  // 2/byte
  parameter int AI_DTYPE_INT2     = 4'd2;  // 4/byte (sign+magnitude quad-pack)
  parameter int AI_DTYPE_INT3     = 4'd3;  // 2/byte (2x3=6b packed)
  parameter int AI_DTYPE_INT5     = 4'd4;  // 1/byte
  parameter int AI_DTYPE_INT6     = 4'd5;  // 1/byte
  // Floating-point types
  parameter int AI_DTYPE_BF16     = 4'd6;  // bfloat16, 1/2 byte
  parameter int AI_DTYPE_FP16     = 4'd7;  // IEEE half, 1/2 byte
  parameter int AI_DTYPE_FP32     = 4'd8;  // IEEE single, 1/4 byte
  parameter int AI_DTYPE_FP8_E4M3 = 4'd9;  // FP8 E4M3 (3 mantissa), 1/byte
  parameter int AI_DTYPE_FP8_E5M2 = 4'd10; // FP8 E5M2 (2 mantissa), 1/byte
  parameter int AI_DTYPE_BF8      = 4'd11; // Brain-float 8 ~= FP8 E5M2, 1/byte
  // Sub-byte float / ternary
  parameter int AI_DTYPE_NF4      = 4'd12; // NormalFloat-4 (QLoRA), 2/byte
  parameter int AI_DTYPE_FP4      = 4'd13; // 4-bit float (E2M1), 2/byte
  parameter int AI_DTYPE_BITNET   = 4'd14; // BitNet 1.58b ternary {-1,0,+1}, 4/byte
  parameter int AI_DTYPE_MXFP4    = 4'd15; // Microscaling FP4 (training, block-scaled E2M1), 2/byte + scale

  // Legacy alias: AI_DTYPE_FP8 == FP8 E4M3 (training default)
  parameter int AI_DTYPE_FP8      = AI_DTYPE_FP8_E4M3;

  // ---------------------------------------------------------------------------
  // Data-type helpers
  // ---------------------------------------------------------------------------
  // Number of packed weights per byte for a given dtype
  function automatic int dtype_weights_per_byte(input int dt);
    case (dt)
      AI_DTYPE_INT2:                             return 4;
      AI_DTYPE_INT3, AI_DTYPE_INT4,
      AI_DTYPE_NF4,  AI_DTYPE_FP4:               return 2;
      AI_DTYPE_BITNET:                           return 4;
      AI_DTYPE_FP32:                             return 0;  // 4 bytes per weight (unpacked)
      default:                                   return 1;  // INT8/INT5/INT6/FP8/BF8/FP16/BF16
    endcase
  endfunction

  // True for sub-8-bit packed formats (need unpack before MAC)
  function automatic logic dtype_is_packed(input int dt);
    return (dt == AI_DTYPE_INT2) || (dt == AI_DTYPE_INT3) ||
           (dt == AI_DTYPE_INT4) || (dt == AI_DTYPE_NF4) ||
           (dt == AI_DTYPE_FP4) || (dt == AI_DTYPE_BITNET);
  endfunction

  // Sub-byte integer sign-extension width (0 = full 8-bit)
  function automatic int dtype_int_bits(input int dt);
    case (dt)
      AI_DTYPE_INT2: return 2;
      AI_DTYPE_INT3: return 3;
      AI_DTYPE_INT4: return 4;
      AI_DTYPE_INT5: return 5;
      AI_DTYPE_INT6: return 6;
      default:       return 8;
    endcase
  endfunction

  // NF4 (NormalFloat-4) dequantization LUT, QLoRA values x64 (±6-bit signed)
  // idx: 0..15 -> {-1.0,-0.696,-0.525,-0.395,-0.284,-0.185,-0.091,0,.080,.161,.246,.338,.441,.563,.723,1.0}
  function automatic logic signed [7:0] nf4_dequant_lut(input logic [3:0] idx);
    case (idx)
      4'd0:  return -8'sd64;
      4'd1:  return -8'sd45;
      4'd2:  return -8'sd34;
      4'd3:  return -8'sd25;
      4'd4:  return -8'sd18;
      4'd5:  return -8'sd12;
      4'd6:  return -8'sd6;
      4'd7:  return  8'sd0;
      4'd8:  return  8'sd5;
      4'd9:  return  8'sd10;
      4'd10: return  8'sd16;
      4'd11: return  8'sd22;
      4'd12: return  8'sd28;
      4'd13: return  8'sd36;
      4'd14: return  8'sd46;
      4'd15: return  8'sd64;
      default: return 8'sd0;
    endcase
  endfunction

  // BitNet 1.58b ternary decode from 2-bit code
  // 01 -> +1, 11 -> -1, 00/10 -> 0
  function automatic logic signed [2:0] bitnet_decode(input logic [1:0] code);
    case (code)
      2'b01:   return 3'sd1;
      2'b11:   return -3'sd1;
      default: return 3'sd0;
    endcase
  endfunction

  // NPU Subsystem 命令 (v6.3.9, orca_npu_ss; 語意見 docs/npu.md 「NPU Subsystem」)
  typedef enum logic [2:0] {
    NPU_OP_GEMM   = 3'd0,   // C[c] = A[a] × B[b], K = k (npu_core)
    NPU_OP_DMA_LD = 3'd1,   // 外部 ext.. → L2 a.., k 個 line
    NPU_OP_DMA_ST = 3'd2,   // L2 a.. → 外部 ext.., k 個 line
    NPU_OP_NOP    = 3'd3,   // 只回報完成 (wt = 1 時即 fence)
    NPU_OP_VEC    = 3'd4,   // 向量程式: 從 L2 line a 起的 k 個指令 (npu_vec_eng)
    NPU_OP_KV     = 3'd5,   // v6.3.13: paged KV cache 頁管理 / 間接 DMA 索引表 (npu_llm_eng, docs/npu_llm.md)
    NPU_OP_MOE    = 3'd6    // v6.3.13: MoE dispatch 表 (npu_llm_eng); 7 保留 (當 NOP)
  } npu_op_e;
  typedef struct packed {
    npu_op_e     op;
    logic        wt;        // 等待: 先前所有命令完成後才分派
    logic [7:0]  id;
    logic [31:0] ext;       // 外部 line 位址 (DMA)
    logic [25:0] a, b, c;   // L2 line 位址 (GEMM: A / B / C; DMA: a)
    logic [15:0] k;         // GEMM: K; DMA: line 數
    logic        sa, sb;    // GEMM: A / B 有號
  } npu_tile_cmd_t;

  // ---------------------------------------------------------------------------
  // Cloud LLM 參數 (npu_llm_eng 與 xuanji-core 的 orca_v63_garuda / cloud_card 共用)
  // ---------------------------------------------------------------------------
  parameter int PE_ARRAY_DIM_128  = 128;   // 雲端 systolic 陣列維度
  parameter int KV_PAGE_BITS      = 9;     // KV-cache 頁表 (512 頁)
  parameter int MOE_MAX_EXPERTS   = 256;   // MoE 最大專家數

  // ---------------- NPU L2 位址對應 (npu_l2_xbar 與 TB 後門共用) ----------------
  // bank = line 低 bw 位元 XOR (列位址各 bw 位元段) — 對齊的區域 (例如每個命令一個對齊 slot) 不會全部從同一 bank
  // 開始而在多個 CU 間形成同步衝突 (convoy); 連續 NB 個 line 仍落在 NB 個不同 bank。只摺疊列位址 [bw, bw+dw)
  // 內的位元, 所以 line 與 line + 容量仍對應同一位置 (超出容量的位址模數環繞)。列 = line[bw +: dw] (不變)。
  function automatic int unsigned npu_l2_bank(logic [25:0] l, int bw, int dw);
    logic [25:0] b, m;
    m = (26'(1) << bw) - 1'b1;
    b = l & m;
    for (int s = bw; s < bw + dw; s += bw) b ^= (l >> s) & m & ((26'(1) << (bw + dw - s)) - 1'b1);
    return int'(b);
  endfunction
  function automatic int unsigned npu_l2_row(logic [25:0] l, int bw, int dw);
    return int'((l >> bw) & ((26'(1) << dw) - 1'b1));
  endfunction
endpackage : npu_pkg

`endif // NPU_PKG_SV
