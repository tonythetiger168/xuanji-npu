// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - 同指數寬度的縮減轉換 (組合式): MI 位元尾數 → MO 位元尾數 (例如 FP32 → BF16)。
//   RNE 捨入 (捨入進位可能溢位成 inf); DAZ (指數 0 → ±0)、NaN → 標準 qNaN、inf 保留符號
//   (語意與 fp_add / fp_mul 相同, 見 docs/fp.md)。
module fp_narrow #(
  parameter int E  = 8,
  parameter int MI = 23,
  parameter int MO = 7
)(
  input  logic [E+MI:0] a,
  output logic [E+MO:0] y
);
  localparam logic [E-1:0] EMAX = '1;
  localparam int D = MI - MO;                    // 捨去的位元數
  if (D < 2) begin : g_bad
    $error("fp_narrow: 須捨去 >= 2 個尾數位元 (MI - MO >= 2)");
  end
  logic          s;
  logic [E-1:0]  e;
  logic [MI-1:0] f;
  assign {s, e, f} = a;
  wire [MO-1:0] keep = f[MI-1 -: MO];
  wire          rb   = f[D-1];
  wire          st   = |f[D-2:0];
  // {指數, 尾數} 一起加 1: 尾數進位自然進到指數, 全 1 指數即 inf
  wire [E+MO-1:0] r = {e, keep} + (E+MO)'(rb && (st || keep[0]));

  always_comb begin
    if (e == EMAX && f != '0) y = {1'b0, EMAX, 1'b1, (MO - 1)'(0)};
    else if (e == EMAX)       y = {s, EMAX, MO'(0)};
    else if (e == '0)         y = {s, E'(0), MO'(0)};
    else                      y = {s, r};
  end
endmodule : fp_narrow
