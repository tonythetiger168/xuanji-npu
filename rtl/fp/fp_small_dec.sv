// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫: 小格式元素 → FP32 (精確)。OCP MX v1.0 的 FP8 E4M3 / E5M2、FP4 E2M1, 以及 IEEE FP16。
//   - 次正規元素精確解碼 (MX 規格要求; E2M1 的 0.5 就是次正規數), 正規化後成為 FP32 正規數
//     (最小的 FP16 次正規數 2^-24 仍在 FP32 正規範圍內, 不會 FTZ)。
//   - 元素 NaN → 標準 qNaN; inf (只有 IEEE 式格式) → ±inf; ±0 保留符號。
//   - 輸出 y = FP32 的高 9+M 個位元 {符號, 8 bit 指數, M bit 尾數}; 其餘 23-M 個低位恆為 0, 由使用端補上
//     (避免恆為 0 的訊號)。
//   SP: 0 = IEEE 式 (指數全 1 為 inf / NaN: FP16、E5M2), 1 = E4M3 (只有 S.1111.111 為 NaN, 無 inf),
//       2 = 無特殊值 (E2M1)。
module fp_small_dec #(
  parameter int E    = 4,
  parameter int M    = 3,
  parameter int BIAS = 7,
  parameter int SP   = 1
)(
  input  logic [E+M:0] x,
  output logic [8+M:0] y
);
  localparam int SW = $clog2(M + 1);               // 次正規數的正規化移位 1 .. M
  localparam logic [8+M:0] QN = (9 + M)'(10'b0_1111_1111_1) << (M - 1);   // 標準 qNaN 的高位
  logic          s;
  logic [E-1:0]  ex;
  logic [M-1:0]  fr;
  assign {s, ex, fr} = x;

  logic [SW-1:0] sh;                               // 次正規: 最高的 1 移到隱藏位元所需的位移
  always_comb begin
    sh = '0;
    for (int i = 0; i < M; i++) if (fr[i]) sh = SW'(M - i);
  end
  // 數值結果 (非特殊值): FP32 指數 = 元素無偏指數 + 127; 正規: ex - BIAS, 次正規: 1 - BIAS - sh
  logic [8+M:0] yv;
  always_comb begin
    if (ex == '0 && fr == '0) yv = {s, 8'd0, M'(0)};
    else if (ex == '0)        yv = {s, 8'(128 - BIAS) - 8'(sh), M'(fr << sh)};
    else                      yv = {s, 8'(ex) + 8'(127 - BIAS), fr};
  end
  if (SP == 0) begin : g_ieee
    assign y = (ex == '1) ? ((fr != '0) ? QN : {s, 8'hFF, M'(0)}) : yv;
  end else if (SP == 1) begin : g_e4m3
    assign y = (ex == '1 && fr == '1) ? QN : yv;
  end else begin : g_none
    assign y = yv;
  end
endmodule : fp_small_dec
