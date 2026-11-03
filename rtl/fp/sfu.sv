// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - SFU (special function unit, 組合式, FP32 輸入 / 輸出)
//   op 0: recip(x) = 1/x      op 1: rsqrt(x) = 1/sqrt(x)
//   op 2: exp2(x)  = 2^x      op 3: exp(x)   = e^x (x·log2 e 以 48 bit 定點計算後走 exp2)
//   op 4: sinπ(x)  = sin(πx)  op 5: cosπ(x)  = cos(πx)
//     (範圍縮減在 48 位元小數定點上精確進行, 折到 r ∈ [0, 1/2]; 結果 = 4 · g(r) · r, g = sin(πr)/(4r) 查表,
//      r 保持全精度, 所以 r 很小 (x 接近零點) 時仍有相對精度)
// 作法: 範圍縮減到 [1, 2) / [2, 4) 的有效位元 (recip / rsqrt) 或 [0, 1) 的小數 (exp2 / exp), 以前 7 個位元查表
//   取得該段的二次多項式係數 (rtl/fp/sfu_tables.svh, 由 dv/fp/sfu_ref.py 產生), 對段內偏移 t 以定點計算
//   y = c0 + (c1·t >> TW) + (c2·(t·t >> TW) >> TW) (y ∈ [2^29, 2^31), 只需兩種正規化), RNE 捨入。整數運算與 dv/fp/sfu_ref.py 位元一致;
//   誤差界 (對雙精度參考) 見 docs/fp.md。語意同 fp_add (DAZ / FTZ / 標準 qNaN)。
// 特殊值: recip(±0) = ±inf, recip(±inf) = ±0; rsqrt(±0) = ±inf, rsqrt(+inf) = +0, rsqrt(負數 / -inf) = NaN;
//         exp2 / exp: +inf → +inf, -inf → +0, |x| >= 128 → +inf (x > 0) / +0 (x < 0); NaN → 標準 qNaN;
//         sinπ / cosπ: ±inf / NaN → qNaN; sinπ(整數) = ±0 (x 的符號), cosπ(n + 1/2) = +0, cosπ(±0) = 1。
module sfu (
  input  logic [2:0]  op,
  input  logic [31:0] x,
  output logic [31:0] y
);
  localparam int S = 7, F = 30;
  localparam logic [31:0] QNAN = 32'h7FC0_0000, PINF = 32'h7F80_0000;
  `include "sfu_tables.svh"                     // 係數表與 SFU_LOG2E (由 dv/fp/sfu_ref.py 產生)

  logic        s;
  logic [7:0]  ex;
  logic [22:0] fr;
  assign {s, ex, fr} = x;
  wire x_nan = (ex == 8'hFF) && (fr != '0), x_inf = (ex == 8'hFF) && (fr == '0), x_zero = (ex == '0);

  // ---------------- exp2 / exp 的定點指數 (XF = 32 位元小數) ----------------
  logic signed [41:0] v;                         // x · 2^32 (|x| < 128)
  logic signed [41:0] w;                         // 2^w 的指數 (32 位元小數)
  always_comb begin
    logic [23:0] m;
    m = {1'b1, fr};
    // x · 2^32 = m · 2^(ex - 118); |x| < 128 時 ex <= 133, 左移 <= 15 (ex >= 134 的結果由特殊值路徑取代)。
    // ex = 0 (±0 / 次正規, DAZ) 時 m >> 118 = 0, 不需另外處理 (mutation 驗證)
    if (ex >= 8'd118) v = 42'(m) << (ex - 8'd118);
    else              v = 42'(m >> (8'd118 - ex));
    if (s) v = -v;
    // exp: w = floor(x · log2(e) · 2^32) (log2(e) 常數去掉尾端 0, 少移同樣位數, 結果與 sfu_ref.py 相同)
    if (op == 3'd3) w = 42'((91'(v) * signed'({1'b0, SFU_LOG2E})) >>> SFU_LOG2E_SH);
    else            w = v;
  end

  // ---------------- sinπ / cosπ 的範圍縮減 ----------------
  wire          trig = op[2];
  wire          tcos = op[0];
  wire          t_fix = (ex >= 8'd102);          // 指數 >= -25: 48 位元小數定點可精確表示 |x|
  logic [48:0]  v0, v1;
  logic [47:0]  v2, v3;                          // 折到 [0, 1) / [0, 1/2]
  logic         tneg, tflip;
  logic [47:0]  rr;                              // r 的整數表示 (r = rr · 2^-er)
  logic [7:0]   er;                              // 48 (定點) 或 150 - ex (|x| < 2^-25 的 sinπ, 不需縮減)
  logic [46:15] rs;                              // r · 2^48 的查表用位元 (r < 1/2; r = 1/2 另外處理, 低 15 位元不用)
  always_comb begin
    // |x| · 2^48 mod 2^49 (= |x| mod 2)。|x| >= 2^23 (整數) 時同樣精確: 2^23 <= |x| < 2^24 只剩最低位 (奇偶),
    // 更大時為 0 — 一般路徑即給出 sinπ = ±0、cosπ = (-1)^x, 不需另外處理 (mutation 驗證)
    v0 = 49'(73'({1'b1, fr}) << (ex - 8'd102));
    v1 = tcos ? v0 + 49'(1 << 47) : v0;          // cosπ(x) = sinπ(x + 1/2) (偶函數, 不看符號)
    tflip = v1[48];                              // u >= 1: 變號, u -= 1
    v2 = v1[47:0];
    v3 = (v2 > 48'(1 << 47)) ? 48'(1 << 48) - v2 : v2;   // u > 1/2: u = 1 - u
    tneg = (tcos ? 1'b0 : s) ^ tflip;
    if (t_fix) begin rr = v3;              er = 8'd48;        rs = v3[46:15]; end
    else       begin rr = 48'({1'b1, fr}); er = 8'd150 - ex;  rs = 32'((32'({1'b1, fr}) >> (8'd102 - ex)) >> 15); end
  end

  // ---------------- 查表 + 二次多項式 ----------------
  // 係數: c0 無號 31 bit (恆正), c1 / c2 有號 25 / 17 bit — 寬度由 dv/fp/sfu_ref.py 檢查
  logic [S-1:0]  idx;
  logic [24:0]   t;                              // 段內偏移 (recip / rsqrt: 16 bit, exp2 / exp: 25 bit)
  logic [72:0]   c;
  logic signed [7:0] eu;                         // 無偏指數 ex - 127
  always_comb begin
    eu  = 8'(signed'({1'b0, ex}) - 9'sd127);
    if (trig)       begin idx = rs[46:40]; t = rs[39:15]; end   // r · 256 的整數 / 小數 (r < 1/2: 索引 < 128)
    else if (op[1]) begin idx = w[31:25]; t = w[24:0]; end
    else            begin idx = fr[22:16]; t = {9'b0, fr[15:0]}; end
    unique case (op)
      3'd0:    c = SFU_TAB_RECIP[idx];
      3'd1:    c = eu[0] ? SFU_TAB_RSQRT_O[idx] : SFU_TAB_RSQRT_E[idx];
      3'd2, 3'd3: c = SFU_TAB_EXP2[idx];
      default: c = SFU_TAB_SINPI[idx];
    endcase
  end
  logic [30:0]        c0;
  logic signed [24:0] c1;
  logic signed [16:0] c2, p2;                    // |c2 · t² / 2^2TW| < 2^16
  logic signed [25:0] p1;                        // |c1 · t / 2^TW| < 2^24
  logic [24:0]        t2;                        // t² / 2^TW < 2^TW
  logic [30:0]        yu;                        // y · 2^-F ∈ [2^29, 2^31) (sfu_ref.py 檢查)
  always_comb begin
    {c0, c1, c2} = c;
    // TW = 25 (exp2 / exp / sinπ / cosπ) 或 16 (recip / rsqrt)
    if (op[1] || op[2]) begin
      p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 25);
      t2 = 25'((50'(t) * 50'(t)) >> 25);
      p2 = 17'((43'(c2) * signed'({18'b0, t2})) >>> 25);
    end else begin
      p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 16);
      t2 = 25'((50'(t) * 50'(t)) >> 16);
      p2 = 17'((43'(c2) * signed'({18'b0, t2})) >>> 16);
    end
    yu = 31'(33'(c0) + 33'(p1) + 33'(p2));
  end

  // ---------------- 結果指數 k (值 = y · 2^(k - F)) ----------------
  logic signed [9:0] k;
  always_comb begin
    unique case (op)
      3'd0:    k = 10'sd127 - 10'(signed'({1'b0, ex}));            // 1/(m·2^e) = (1/m)·2^-e
      3'd1:    k = -10'(eu >>> 1);                                  // rsqrt: 2^(-floor(e/2))
      default: k = 10'(w >>> 32);                                   // floor(w) (sinπ / cosπ 不用 k)
    endcase
  end

  // ---------------- 正規化 + RNE 捨入 ----------------
  // y ∈ [2^29, 2^31): 最高位在第 30 位 (hi) 時捨去 7 位, 否則捨去 6 位; 保留 24 個有效位元
  wire          hi   = yu[30];
  wire [22:0]   q    = hi ? yu[29:7] : yu[28:6];  // 尾數 (不含恆為 1 的最高位)
  wire          rb   = hi ? yu[6] : yu[5];        // 捨入位元
  wire          st   = hi ? (yu[5:0] != '0) : (yu[4:0] != '0);
  wire          inc  = rb && (st || q[0]);
  wire [23:0]   qs   = {1'b0, q} + 24'(inc);      // 尾數全 1 進位時 qs[23] = 1 (尾數歸零)
  logic signed [10:0] e;
  logic [30:0]  y_fin;                           // {指數, 尾數} (符號另外處理)
  always_comb begin
    e = 11'(k) - 11'(F) + (hi ? 11'sd7 : 11'sd6) + 11'sd23 + (qs[23] ? 11'sd1 : 11'sd0);
    if (e > 11'sd127)        y_fin = PINF[30:0];
    else if (e < -11'sd126)  y_fin = '0;
    else                     y_fin = {8'(e + 11'sd127), qs[22:0]};
  end

  // ---------------- sinπ / cosπ: 4 · g · r 的正規化 + RNE (g · rr 最多 30 + 48 bit) ----------------
  logic [77:0]  tp;                              // g · rr (yu 的低 30 bit 即 g · 2^30 ∈ [2^29, 2^30))
  logic [6:0]   tnb, tsh;
  logic [23:0]  tq;
  logic         tinc;
  logic [24:0]  tqs;
  logic signed [10:0] te;
  logic [30:0]  t_fin;
  always_comb begin
    tp  = 78'(yu[29:0]) * 78'(rr);
    tnb = '0;
    for (int i = 0; i < 78; i++) if (tp[i]) tnb = 7'(i + 1);
    tsh = tnb - 7'd24;                           // g >= 2^29 且 rr >= 2^23 → 位元長 >= 53, 恆 > 0
    tq  = 24'(tp >> tsh);
    tinc = tp[tsh - 7'd1] && ((tp & ((78'(1) << (tsh - 7'd1)) - 78'(1))) != '0 || tq[0]);
    tqs = {1'b0, tq} + 25'(tinc);
    // 值 = tp · 2^(2 - er - 30); 結果指數 = 2 - er - 30 + tsh + 23 (+ 進位)
    te  = 11'sd2 - 11'(er) - 11'sd30 + 11'(tsh) + 11'sd23 + (tqs[24] ? 11'sd1 : 11'sd0);
    if (te < -11'sd126) t_fin = '0;              // |sinπ x| >= π · 2^-126 > 最小正規數: 不會發生, 保留 FTZ 語意
    else                t_fin = {8'(te + 11'sd127), tqs[24] ? 23'(tqs >> 1) : tqs[22:0]};
  end

  always_comb begin
    unique case (op)
      3'd0: begin                                // recip
        if (x_nan)       y = QNAN;
        else if (x_inf)  y = {s, 31'b0};
        else if (x_zero) y = {s, PINF[30:0]};
        else             y = {s, y_fin};
      end
      3'd1: begin                                // rsqrt
        if (x_nan || (s && !x_zero)) y = QNAN;
        else if (x_inf)  y = '0;
        else if (x_zero) y = {s, PINF[30:0]};
        else             y = {1'b0, y_fin};
      end
      3'd2, 3'd3: begin                          // exp2 / exp
        if (x_nan)                   y = QNAN;
        else if (x_inf || ex >= 8'd134) y = s ? '0 : PINF;
        else                         y = {1'b0, y_fin};
      end
      default: begin                             // sinπ / cosπ
        if (ex == 8'hFF)             y = QNAN;
        else if (x_zero)             y = tcos ? 32'h3F80_0000 : {s, 31'b0};
        else if (tcos && !t_fix)     y = 32'h3F80_0000;                 // |x| < 2^-25: cosπ 捨入為 1
        else if (t_fix && v3 == '0)  y = tcos ? '0 : {s, 31'b0};       // sinπ(整數) / cosπ(n + 1/2)
        else if (t_fix && v3 == 48'(1 << 47)) y = {tneg, 31'h3F80_0000};  // r = 1/2: ±1
        else                         y = {tneg, t_fin};
      end
    endcase
  end
endmodule : sfu
