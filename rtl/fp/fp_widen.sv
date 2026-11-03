// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - 同指數寬度的擴展轉換 (組合式): MI 位元尾數 → MO 位元尾數 (例如 BF16 → FP32), 精確 (補 0)。
//   DAZ (指數 0 → ±0)、NaN → 標準 qNaN、inf 保留符號 (語意與 fp_add / fp_mul 相同, 見 docs/fp.md)。
module fp_widen #(
  parameter int E  = 8,
  parameter int MI = 7,
  parameter int MO = 23
)(
  input  logic [E+MI:0] a,
  output logic [E+MO:0] y
);
  localparam logic [E-1:0] EMAX = '1;
  if (MO <= MI) begin : g_bad
    $error("fp_widen: MO 須 > MI");
  end
  logic          s;
  logic [E-1:0]  e;
  logic [MI-1:0] f;
  assign {s, e, f} = a;
  always_comb begin
    if (e == EMAX && f != '0) y = {1'b0, EMAX, 1'b1, (MO - 1)'(0)};
    else if (e == EMAX)       y = {s, EMAX, MO'(0)};
    else if (e == '0)         y = {s, E'(0), MO'(0)};
    else                      y = {s, e, f, (MO - MI)'(0)};
  end
endmodule : fp_widen
