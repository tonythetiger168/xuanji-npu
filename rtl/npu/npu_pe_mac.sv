// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU PE 的乘加 (v6.3.16): INT8 (INT32 累加) / FP8 E4M3·E5M2 / BF16 (FP32 累加)。組合邏輯, npu_mma 的每個 PE 一份
// (獨立成模組: coverage 以模組合併, 不必在 N×N 個 generate 範圍各自收斂)。
//
// 格式 (token 旗標, 整個命令相同):
//   fp = 0: INT8 × INT8 (sa / sb = 有號), sum = acc + a × b (模 2^32)
//   fp = 1, bf = 0: FP8, sa / sb 選 A / B 的子格式 (0 = E4M3, 1 = E5M2; OCP MX v1.0: E4M3 只有 S.1111.111 為 NaN、無 inf,
//                   E5M2 為 IEEE 式); 次正規數精確 (同 rtl/fp/fp_small_dec.sv)
//   fp = 1, bf = 1: BF16, 每個元素佔兩步 (byte-planar: 先低位元組、再高位元組): h = 1 的步驟以 {a, a_lo} 為運算元
//                   (a_lo = 前一步鎖存的低位元組); h = 0 的步驟只在 last 時運算 (奇數步: 運算元 = {0x00, a}, 即 ±0)
// 乘積: 解碼後的有效位數 ≤ 8 bit, 乘積 ≤ 16 bit, 以 FP32 表示時精確; 小於最小正規數 → ±0 (FTZ), 溢位 → ±inf;
//   BF16 輸入 DAZ (指數 0 當 ±0); 任一為 NaN 或 inf × 0 → qNaN 0x7FC00000; 0 的符號 = 兩者符號 xor。
// 累加: sum = first ? p : fp_add(acc, p) (DAZ / FTZ / RNE, 依 k 的順序) — 即逐步以 FP32 捨入的內積。
module npu_pe_mac (
  input  logic [7:0]  a, b,                 // 本步的 A / B 位元組
  input  logic [7:0]  a_lo, b_lo,           // BF16: 前一步 (低位元組) 鎖存的值
  input  logic        sa, sb,               // INT8: 有號; FP8: 1 = E5M2
  input  logic        fp, bf, h,            // FP 格式 / BF16 / BF16 的高位元組步驟
  input  logic        first,                // 本命令的第一個乘積 (累加器重新開始)
  input  logic [31:0] acc,
  output logic [31:0] sum
);
  // ---------------- INT8 ----------------
  wire signed [8:0]  ax = {sa & a[7], a};
  wire signed [8:0]  bx = {sb & b[7], b};
  wire signed [17:0] iprod = ax * bx;
  wire [31:0] isum = (first ? 32'd0 : acc) + 32'(iprod);

  // ---------------- FP 運算元解碼: 值 = (-1)^s × m × 2^x ----------------
  typedef struct packed {
    logic              s, z, inf, nan;
    logic [7:0]        m;                   // 有效位數 (含隱藏位元; 次正規 / 0 時隱藏位元為 0)
    logic signed [9:0] x;                   // m 最低位的指數
  } dec_t;
  function automatic dec_t dec(logic [7:0] v, logic [7:0] lo, logic e5, logic bf16, logic hi);
    dec_t d;
    logic [15:0] w = hi ? {v, lo} : {8'h00, v};
    d = '0;
    if (bf16) begin                          // BF16: 1.8.7, bias 127; DAZ
      d.s = w[15];
      d.z = (w[14:7] == 8'd0);
      d.inf = (w[14:7] == 8'hFF) && (w[6:0] == '0);
      d.nan = (w[14:7] == 8'hFF) && (w[6:0] != '0);
      d.m = {1'b1, w[6:0]};
      d.x = 10'(signed'({2'b0, w[14:7]})) - 10'sd134;
    end else if (e5) begin                   // E5M2: 1.5.2, bias 15
      d.s = v[7];
      d.z = (v[6:0] == '0);
      d.inf = (v[6:2] == 5'h1F) && (v[1:0] == '0);
      d.nan = (v[6:2] == 5'h1F) && (v[1:0] != '0);
      d.m = {5'd0, v[6:2] != '0, v[1:0]};
      d.x = ((v[6:2] == '0) ? 10'sd1 : 10'(signed'({5'b0, v[6:2]}))) - 10'sd17;
    end else begin                           // E4M3: 1.4.3, bias 7; 只有 S.1111.111 為 NaN
      d.s = v[7];
      d.z = (v[6:0] == '0);
      d.nan = (v[6:0] == 7'h7F);
      d.m = {4'd0, v[6:3] != '0, v[2:0]};
      d.x = ((v[6:3] == '0) ? 10'sd1 : 10'(signed'({6'b0, v[6:3]}))) - 10'sd10;
    end
    return d;
  endfunction
  dec_t da, db;
  assign da = dec(a, a_lo, sa, bf, h);
  assign db = dec(b, b_lo, sb, bf, h);

  // ---------------- 乘積 → FP32 (精確; FTZ / 溢位) ----------------
  wire        ps = da.s ^ db.s;
  wire [15:0] pm = da.m * db.m;
  logic [3:0] lz;                           // pm 最高的 1 的位置
  always_comb begin
    lz = '0;
    for (int i = 0; i < 16; i++) if (pm[i]) lz = 4'(i);
  end
  wire signed [10:0] pe = 11'(da.x) + 11'(db.x) + 11'(lz) + 11'sd127;   // FP32 偏置指數
  // 乘積最多 16 個有效位元: FP32 尾數的低 8 bit 恆為 0, 只保留高位 (p = 乘積的 bit 31..8)
  wire [14:0] pn = 15'(pm << (4'd15 - lz));                             // 隱藏位元移到 bit 15 後的尾數 (隱藏位元捨去)
  logic [23:0] p;
  always_comb begin
    if (da.nan || db.nan || (da.inf && db.z) || (db.inf && da.z)) p = 24'h7FC0_00;
    else if (da.inf || db.inf)                                    p = {ps, 8'hFF, 15'd0};
    else if (da.z || db.z || pm == '0)                            p = {ps, 23'd0};
    else if (pe > 11'sd254)                                       p = {ps, 8'hFF, 15'd0};
    else if (pe < 11'sd1)                                         p = {ps, 23'd0};
    else                                                          p = {ps, pe[7:0], pn[14:0]};
  end
  logic [31:0] fsum;
  fp_add #(.E(8), .M(23)) u_add (.a(acc), .b({p, 8'h00}), .y(fsum));

  assign sum = !fp ? isum : first ? {p, 8'h00} : fsum;
endmodule : npu_pe_mac
