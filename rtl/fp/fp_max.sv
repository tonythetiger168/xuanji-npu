// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫: FP32 max (softmax 的最大值、ReLU)。語意同函式庫 (docs/fp.md):
//   任一輸入 NaN → 標準 qNaN; DAZ (次正規數當作同號的 0, 結果以 ±0 表示); max(+0, -0) = +0。
module fp_max (
  input  logic [31:0] a, b,
  output logic [31:0] y
);
  wire nan_a = (a[30:23] == 8'hFF) && (a[22:0] != '0);
  wire nan_b = (b[30:23] == 8'hFF) && (b[22:0] != '0);
  wire [31:0] ad = (a[30:23] == '0) ? {a[31], 31'd0} : a;      // DAZ
  wire [31:0] bd = (b[30:23] == '0) ? {b[31], 31'd0} : b;
  // 符號不同: 正者較大 (含 +0 > -0); 同號: 正數取大的大小, 負數取小的大小
  wire a_big = (ad[31] != bd[31]) ? !ad[31] : (ad[31] ? (ad[30:0] < bd[30:0]) : (ad[30:0] > bd[30:0]));
  assign y = (nan_a || nan_b) ? 32'h7FC0_0000 : a_big ? ad : bd;
endmodule : fp_max
