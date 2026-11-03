// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - 加法器 (組合式, 參數化格式: FP32 = E 8 / M 23, BF16 = E 8 / M 7)
//
// 語意 (與 dv/fp/fp_ref.py 位元一致; docs/fp.md):
//   DAZ (指數 0 的輸入當作 ±0)、FTZ (結果小於最小正規數 → ±0, 保留符號)、round-to-nearest-even、
//   溢位 → ±inf、NaN / inf - inf → 標準 qNaN; 精確和為 0 → +0, 但 (-0) + (-0) = -0。
// 作法: 大數在前 (比較 |a|, |b|), 小數右移對齊並把移出的位元收成 sticky, 放在寬度 2M+5 的資料路徑上相加 / 減,
//   正規化 (前導 1 對齊最高位) 後取 M+1 位元以 RNE 捨入。資料路徑在大數之下保留 M+2 個保護位元: 對齊移位
//   不超過 M+2 時和 / 差完全精確; 更遠時結果必為大數本身。所以結果與精確和捨入相同 (含大量相消的減法)。
module fp_add #(
  parameter int E = 8,
  parameter int M = 23
)(
  input  logic [E+M:0] a, b,
  output logic [E+M:0] y
);
  localparam int L  = 2 * M + 4;                 // {進位, 大數 M+1 位元, M+2 個保護位元}
  localparam int EW = E + 2;                     // 有號指數運算寬度
  localparam logic [E-1:0] EMAX = '1;            // 全 1 = inf / NaN
  localparam int BIAS = (1 << (E - 1)) - 1;
  if (M < 2) begin : g_bad
    $error("fp_add: M 須 >= 2");
  end

  logic          sa, sb;
  logic [E-1:0]  ea, eb;
  logic [M-1:0]  fa, fb;
  assign {sa, ea, fa} = a;
  assign {sb, eb, fb} = b;
  wire a_nan = (ea == EMAX) && (fa != '0), b_nan = (eb == EMAX) && (fb != '0);
  wire a_inf = (ea == EMAX) && (fa == '0), b_inf = (eb == EMAX) && (fb == '0);
  wire a_zero = (ea == '0), b_zero = (eb == '0);                  // DAZ

  // 大數 / 小數 (以指數與尾數比較大小; 零的指數欄位為 0, 自然是較小者)
  wire swap = {eb, fb} > {ea, fa};
  wire          s_big = swap ? sb : sa;
  wire [E-1:0]  e_big = swap ? eb : ea, e_sml = swap ? ea : eb;
  // DAZ: 指數 0 的輸入整個有效位元為 0 (不只隱藏位元; 次正規數的尾數不可帶入)
  wire [M:0]    ma = a_zero ? '0 : {1'b1, fa}, mb = b_zero ? '0 : {1'b1, fb};
  wire [M:0]    m_big = swap ? mb : ma;
  wire [M:0]    m_sml = swap ? ma : mb;
  wire          sub   = sa ^ sb;

  logic [L-2:0] ys;                              // 對齊後的小數 (不含最高的進位位元)
  logic [L-1:0] z;
  logic [L-1:1] zn;                              // 正規化後 (最低位恆為 0, 見下方說明)
  logic [E-1:0] d;                               // 指數差 (大數在前, 不會為負)
  localparam int LZW = $clog2(M + 3);
  logic [LZW-1:0] lz;                            // z 的前導零數 (0 .. M+2, 見下方說明)
  logic [M:0]   q;
  logic         rb, st, z_zero;
  logic signed [EW-1:0] k;                       // 結果的無偏指數 (捨入前)
  always_comb begin
    d = e_big - e_sml;
    // 小數對齊: 移位 <= M + 2 時不會移出任何位元 (下方有 M + 2 個保護位元), 和 / 差精確。
    // 移位 >= M + 3 時小數 < 大數的 1/4 ulp; 即使減法讓結果落到下一個 binade, 仍 < 該處的 1/2 ulp,
    // RNE 結果必定就是大數本身 (不可能平手) → 小數直接當 0 (不需 sticky; mutation 驗證過 sticky 在此不影響結果)
    if (d < E'(M + 3)) ys = {m_sml, (M + 2)'(0)} >> d;
    else               ys = '0;
    z = sub ? ({1'b0, m_big, (M + 2)'(0)} - {1'b0, ys}) : ({1'b0, m_big, (M + 2)'(0)} + {1'b0, ys});
    z_zero = (z == '0);
    // 正規化。前導 1 的位置 >= M + 1: 大量相消只發生在指數差 <= 1, 此時兩數的最低位元都在第 M + 1 位以上;
    // 指數差更大時差值 >= 大數的一半。所以前導零數 <= M + 2, 而且 z 的最低位只有在 lz = 0 (加法進位) 時才會
    // 留在最低位 — 那時小數 < 2^(M+2) 不可能使大數進位而讓 z[0] 為 1 — 故正規化後最低位恆為 0, 不需保留。
    lz = '0;
    for (int i = M + 1; i < L; i++) if (z[i]) lz = LZW'(L - 1 - i);
    zn = (L - 1)'((z << lz) >> 1);
    q  = zn[L-1 -: M+1];
    rb = zn[L-2-M];
    st = |zn[L-3-M:1];
    // 前導 1 在 z 的第 L-1-lz 位; 大數的隱藏位元在第 L-2 位
    k  = EW'(signed'({2'b0, e_big})) - EW'(BIAS) + EW'(1) - EW'(signed'({1'b0, lz}));
  end

  // RNE 捨入: {進位, 隱藏位元, 尾數} + 1; 進位 (尾數全 1 捨入成 2.0) 時尾數歸零、指數 + 1
  logic [M+1:0] qs;
  logic [M-1:0] fr;
  logic signed [EW-1:0] kr;
  logic [E+M:0] y_fin;
  always_comb begin
    qs = {1'b0, q} + (M+2)'(rb && (st || q[0]));
    fr = qs[M+1] ? '0 : qs[M-1:0];
    kr = qs[M+1] ? k + EW'(1) : k;
    if (kr > EW'(BIAS))            y_fin = {s_big, EMAX, M'(0)};              // 溢位 → inf
    else if (kr < EW'(1 - BIAS))   y_fin = {s_big, E'(0), M'(0)};             // FTZ
    else                           y_fin = {s_big, E'(kr + EW'(BIAS)), fr};
  end

  always_comb begin
    if (a_nan || b_nan || (a_inf && b_inf && sub)) y = {1'b0, EMAX, 1'b1, (M - 1)'(0)};   // 標準 qNaN
    else if (a_inf)                                y = {sa, EMAX, M'(0)};
    else if (b_inf)                                y = {sb, EMAX, M'(0)};
    else if (a_zero && b_zero)                     y = {sa & sb, E'(0), M'(0)};
    else if (z_zero)                               y = '0;                                 // 精確相消 → +0
    else                                           y = y_fin;
  end
endmodule : fp_add
