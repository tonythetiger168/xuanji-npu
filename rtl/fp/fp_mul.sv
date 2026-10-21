// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - 乘法器 (組合式, 參數化格式: FP32 = E 8 / M 23, BF16 = E 8 / M 7)
//
// 語意與 fp_add 相同 (DAZ / FTZ / RNE / 溢位 → inf / 標準 qNaN; 0 × inf → qNaN; 符號 = XOR)。
// BF16 × BF16 的乘積 (8 bit × 8 bit 有效位元) 放得進 FP32 而不需捨入: NPU 的 BF16 MAC 以
// fp_mul #(8, 23) 對擴展後的 BF16 運算, 再以 fp_add #(8, 23) 累加。
module fp_mul #(
  parameter int E = 8,
  parameter int M = 23
)(
  input  logic [E+M:0] a, b,
  output logic [E+M:0] y
);
  localparam int EW = E + 3;
  localparam logic [E-1:0] EMAX = '1;
  localparam int BIAS = (1 << (E - 1)) - 1;
  if (M < 2) begin : g_bad
    $error("fp_mul: M 須 >= 2");
  end

  logic          sa, sb;
  logic [E-1:0]  ea, eb;
  logic [M-1:0]  fa, fb;
  assign {sa, ea, fa} = a;
  assign {sb, eb, fb} = b;
  wire a_nan = (ea == EMAX) && (fa != '0), b_nan = (eb == EMAX) && (fb != '0);
  wire a_inf = (ea == EMAX) && (fa == '0), b_inf = (eb == EMAX) && (fb == '0);
  wire a_zero = (ea == '0), b_zero = (eb == '0);                  // DAZ
  wire s = sa ^ sb;

  logic [2*M+1:0] p;                             // 有效位元乘積 (值在 [1, 4))
  logic [M-1:0]   q, fr;                         // 正規化後的尾數 (不含恆為 1 的隱藏位元) / 捨入後
  logic           rb, st;
  logic [M:0]     qs;                            // {進位, 尾數} + 捨入
  logic signed [EW-1:0] k, kr;
  logic [E+M:0]   y_fin;
  always_comb begin
    p = {1'b1, fa} * {1'b1, fb};
    k = EW'(signed'({3'b0, ea})) + EW'(signed'({3'b0, eb})) - EW'(2 * BIAS);
    if (p[2*M+1]) begin                          // 乘積在 [2, 4): 指數 + 1
      q = p[2*M -: M];   rb = p[M];   st = |p[M-1:0]; k = k + EW'(1);
    end else begin
      q = p[2*M-1 -: M]; rb = p[M-1]; st = |p[M-2:0];
    end
    // RNE 捨入; 尾數全 1 進位時尾數歸零、指數 + 1 (隱藏位元恆為 1, 不必計算)
    qs = {1'b0, q} + (M+1)'(rb && (st || q[0]));
    fr = qs[M-1:0];
    kr = qs[M] ? k + EW'(1) : k;
    if (kr > EW'(BIAS))            y_fin = {s, EMAX, M'(0)};
    else if (kr < EW'(1 - BIAS))   y_fin = {s, E'(0), M'(0)};
    else                           y_fin = {s, E'(kr + EW'(BIAS)), fr};
  end

  always_comb begin
    if (a_nan || b_nan || (a_inf && b_zero) || (a_zero && b_inf)) y = {1'b0, EMAX, 1'b1, (M - 1)'(0)};
    else if (a_inf || b_inf)       y = {s, EMAX, M'(0)};
    else if (a_zero || b_zero)     y = {s, E'(0), M'(0)};
    else                           y = y_fin;
  end
endmodule : fp_mul
