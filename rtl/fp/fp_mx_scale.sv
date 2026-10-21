// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫: FP32 值 × 2^(scale-127) (OCP MX 的 E8M0 共用比例)。
//   輸入 / 輸出為 FP32 的高 9+M 個位元 {符號, 指數, M bit 尾數} (fp_small_dec 的輸出; 低位恆 0)。
//   乘上 2 的冪是精確的, 只判斷範圍: 指數 > 254 → ±inf, < 1 → ±0 (FTZ, 保留符號)。
//   scale = 0xFF (E8M0 的 NaN) → 標準 qNaN; inf / NaN / ±0 輸入不變。
module fp_mx_scale #(
  parameter int M = 3
)(
  input  logic [8+M:0] x,
  input  logic [7:0]   scale,
  output logic [8+M:0] y
);
  localparam logic [8+M:0] QN = (9 + M)'(10'b0_1111_1111_1) << (M - 1);   // 標準 qNaN 的高位
  wire        s  = x[8+M];
  wire  [7:0] e  = x[M+7:M];
  // 新指數 = e + scale - 127 ∈ [-126, 381]
  wire signed [9:0] be = signed'({2'b0, e}) + signed'({2'b0, scale}) - 10'sd127;
  always_comb begin
    if (scale == 8'hFF)            y = QN;
    else if (e == 8'hFF || e == '0) y = x;
    else if (be > 10'sd254)        y = {s, 8'hFF, M'(0)};
    else if (be < 10'sd1)          y = {s, 8'd0, M'(0)};
    else                           y = {s, be[7:0], x[M-1:0]};
  end
endmodule : fp_mx_scale
