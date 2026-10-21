// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫 - SFU (special function unit, 組合式, FP32 輸入 / 輸出)
//   op 0: recip(x) = 1/x      op 1: rsqrt(x) = 1/sqrt(x)
//   op 2: exp2(x)  = 2^x      op 3: exp(x)   = e^x (x·log2 e 以 48 bit 定點計算後走 exp2)
// 作法: 範圍縮減到 [1, 2) / [2, 4) 的有效位元 (recip / rsqrt) 或 [0, 1) 的小數 (exp2 / exp), 以前 7 個位元查表
//   取得該段的二次多項式係數 (rtl/fp/sfu_tables.svh, 由 dv/fp/sfu_ref.py 產生), 對段內偏移 t 以定點計算
//   y = c0 + (c1·t >> TW) + (c2·(t·t >> TW) >> TW) (y ∈ [2^29, 2^31), 只需兩種正規化), RNE 捨入。整數運算與 dv/fp/sfu_ref.py 位元一致;
//   誤差界 (對雙精度參考) 見 docs/fp.md。語意同 fp_add (DAZ / FTZ / 標準 qNaN)。
// 特殊值: recip(±0) = ±inf, recip(±inf) = ±0; rsqrt(±0) = ±inf, rsqrt(+inf) = +0, rsqrt(負數 / -inf) = NaN;
//         exp2 / exp: +inf → +inf, -inf → +0, |x| >= 128 → +inf (x > 0) / +0 (x < 0); NaN → 標準 qNaN。
module sfu (
  input  logic [1:0]  op,
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
    if (op == 2'd3) w = 42'((91'(v) * signed'({1'b0, SFU_LOG2E})) >>> SFU_LOG2E_SH);
    else            w = v;
  end

  // ---------------- 查表 + 二次多項式 ----------------
  // 係數: c0 無號 31 bit (恆正), c1 有號 25 bit, c2 無號 16 bit (恆正) — 寬度由 dv/fp/sfu_ref.py 檢查
  logic [S-1:0]  idx;
  logic [24:0]   t;                              // 段內偏移 (recip / rsqrt: 16 bit, exp2 / exp: 25 bit)
  logic [71:0]   c;
  logic signed [7:0] eu;                         // 無偏指數 ex - 127
  always_comb begin
    eu  = 8'(signed'({1'b0, ex}) - 9'sd127);
    if (op[1]) begin idx = w[31:25]; t = w[24:0]; end
    else       begin idx = fr[22:16]; t = {9'b0, fr[15:0]}; end
    unique case (op)
      2'd0:    c = SFU_TAB_RECIP[idx];
      2'd1:    c = eu[0] ? SFU_TAB_RSQRT_O[idx] : SFU_TAB_RSQRT_E[idx];
      default: c = SFU_TAB_EXP2[idx];
    endcase
  end
  logic [30:0]        c0;
  logic signed [24:0] c1;
  logic [15:0]        c2, p2;
  logic signed [25:0] p1;                        // |c1 · t / 2^TW| < 2^24
  logic [24:0]        t2;                        // t² / 2^TW < 2^TW
  logic [30:0]        yu;                        // y · 2^-F ∈ [2^29, 2^31) (sfu_ref.py 檢查)
  always_comb begin
    {c0, c1, c2} = c;
    // TW = 25 (exp2 / exp) 或 16 (recip / rsqrt)
    if (op[1]) begin
      p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 25);
      t2 = 25'((50'(t) * 50'(t)) >> 25);
      p2 = 16'((41'(c2) * 41'(t2)) >> 25);
    end else begin
      p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 16);
      t2 = 25'((50'(t) * 50'(t)) >> 16);
      p2 = 16'((41'(c2) * 41'(t2)) >> 16);
    end
    yu = 31'(33'(c0) + 33'(p1) + 33'(p2));
  end

  // ---------------- 結果指數 k (值 = y · 2^(k - F)) ----------------
  logic signed [9:0] k;
  always_comb begin
    unique case (op)
      2'd0:    k = 10'sd127 - 10'(signed'({1'b0, ex}));            // 1/(m·2^e) = (1/m)·2^-e
      2'd1:    k = -10'(eu >>> 1);                                  // rsqrt: 2^(-floor(e/2))
      default: k = 10'(w >>> 32);                                   // floor(w)
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

  always_comb begin
    unique case (op)
      2'd0: begin                                // recip
        if (x_nan)       y = QNAN;
        else if (x_inf)  y = {s, 31'b0};
        else if (x_zero) y = {s, PINF[30:0]};
        else             y = {s, y_fin};
      end
      2'd1: begin                                // rsqrt
        if (x_nan || (s && !x_zero)) y = QNAN;
        else if (x_inf)  y = '0;
        else if (x_zero) y = {s, PINF[30:0]};
        else             y = {1'b0, y_fin};
      end
      default: begin                             // exp2 / exp
        if (x_nan)                   y = QNAN;
        else if (x_inf || ex >= 8'd134) y = s ? '0 : PINF;
        else                         y = {1'b0, y_fin};
      end
    endcase
  end
endmodule : sfu
