// SPDX-License-Identifier: Apache-2.0
// NPU GEMM 浮點格式 (v6.3.16: FP8 E4M3 / E5M2、BF16, FP32 累加) 的參考函式 (npu_mma / npu_gemm_cu / npu_core / npu_ss 的
// TB 共用; 與 RTL 分開撰寫: 運算元以 real 求值, 乘積以 double 精確相乘後轉成 FP32 位元; 累加用 npu_mx_ref.svh 的
// mx_add, 該函式與 dv/fp/fp_ref.py 比對過)。
`ifndef NPU_FP_REF_SVH
`define NPU_FP_REF_SVH
`include "npu_mx_ref.svh"
typedef struct { bit nan, inf, neg; real v; } fpr_t;          // v = |值|
// 一個運算元: bf = 1 時為 BF16 {hi, lo} (DAZ), 否則 FP8 的位元組 hi (e5: 1 = E5M2, 0 = E4M3)
function automatic fpr_t fpr_dec(logic [7:0] hi, logic [7:0] lo, bit e5, bit bf);
  fpr_t r;
  int e, m;
  r.nan = 0; r.inf = 0; r.v = 0.0;
  r.neg = hi[7];
  if (bf) begin
    e = int'(hi[6:0]) * 2 + int'(lo[7]); m = int'(lo[6:0]);
    if (e == 255) begin r.nan = (m != 0); r.inf = (m == 0); end
    else if (e != 0) r.v = (1.0 + real'(m) / 128.0) * (2.0 ** (e - 127));
  end else if (e5) begin
    e = int'(hi[6:2]); m = int'(hi[1:0]);
    if (e == 31) begin r.nan = (m != 0); r.inf = (m == 0); end
    else if (e == 0) r.v = real'(m) / 4.0 * (2.0 ** -14);
    else r.v = (1.0 + real'(m) / 4.0) * (2.0 ** (e - 15));
  end else begin
    e = int'(hi[6:3]); m = int'(hi[2:0]);
    if (e == 15 && m == 7) r.nan = 1;
    else if (e == 0) r.v = real'(m) / 8.0 * (2.0 ** -6);
    else r.v = (1.0 + real'(m) / 8.0) * (2.0 ** (e - 7));
  end
  return r;
endfunction
// 精確乘積 → FP32 (FTZ: |p| < 2^-126 → ±0; |p| ≥ 2^128 → ±inf; NaN / inf × 0 → qNaN)
function automatic logic [31:0] fpr_mul(fpr_t a, fpr_t b);
  bit s = a.neg ^ b.neg;
  real p;
  logic [63:0] d;
  int e;
  if (a.nan || b.nan || (a.inf && !b.inf && b.v == 0.0) || (b.inf && !a.inf && a.v == 0.0)) return 32'h7FC0_0000;
  if (a.inf || b.inf) return {s, 8'hFF, 23'd0};
  p = a.v * b.v;                                                       // ≤ 16 個有效位元: double 精確
  if (p == 0.0 || p < 2.0 ** -126) return {s, 31'd0};
  if (p >= 2.0 ** 128) return {s, 8'hFF, 23'd0};
  d = $realtobits(p);
  e = int'(d[62:52]) - 1023 + 127;
  if (d[28:0] != 0) $display("ERROR: fpr_mul inexact");
  return {s, 8'(e), d[51:29]};
endfunction
// 一個 C 元素: 步驟 0..k-1 的位元組 a[t] / b[t] (第 t 步); fp = 0: INT8 (sa / sb 有號), fp = 1: FP8 (sa / sb = E5M2)
// 或 BF16 (bf = 1: 兩步一個元素, 先低後高; 奇數步的最後一步 = {0x00, 該位元組}), FP32 依序累加
function automatic logic [31:0] fpr_dot(int k, const ref logic [7:0] a[], const ref logic [7:0] b[], input logic sa, sb, fp, bf);
  logic [31:0] acc = 32'd0, p;
  longint s = 0;
  int t = 0;
  bit first = 1;
  if (!fp) begin
    for (int i = 0; i < k; i++) s += (sa ? longint'($signed(a[i])) : longint'(a[i])) * (sb ? longint'($signed(b[i])) : longint'(b[i]));
    return 32'(s);
  end
  while (t < k) begin
    if (bf) begin
      if (t + 1 < k) begin p = fpr_mul(fpr_dec(a[t+1], a[t], 0, 1), fpr_dec(b[t+1], b[t], 0, 1)); t += 2; end
      else           begin p = fpr_mul(fpr_dec(8'h00, a[t], 0, 1), fpr_dec(8'h00, b[t], 0, 1)); t += 1; end
    end else begin p = fpr_mul(fpr_dec(a[t], 8'h00, sa, 0), fpr_dec(b[t], 8'h00, sb, 0)); t += 1; end
    acc = first ? p : mx_add(acc, p);
    first = 0;
  end
  return acc;
endfunction
// 寫回器: 一個 C 元素的 GEMM 結果 p (fp = 0: INT32, 1: FP32) → (MX 比例) → (累加舊值)
function automatic logic [31:0] fpr_elem(bit fp, logic [31:0] p, bit mx, logic [7:0] ea, logic [7:0] eb, bit acc, logic [31:0] old);
  logic [31:0] g;
  if (!fp) return mx ? mx_elem(p, ea, eb, acc, old) : p + (acc ? old : 32'd0);
  g = mx ? mx_scale(p, ea, eb) : p;
  return acc ? mx_add(old, g) : g;
endfunction
// 刺激: FP8 位元組 (e5 = E5M2) / BF16 (16 bit)。mode 0: 有限值、指數集中 (累加不常溢位); 1: 含極值 / 次正規 / 0 / 特殊值;
// 2: 完全隨機位元
function automatic logic [7:0] fpr_pick8(bit e5, int mode);
  logic [7:0] v;
  if (mode == 2) return 8'($urandom());
  if (mode == 1 && $urandom_range(3, 0) == 0)
    case ($urandom_range(5, 0))
      0: return {1'($urandom()), 7'h00};                               // ±0
      1: return {1'($urandom()), 7'h7F};                               // E4M3 NaN / E5M2 NaN
      2: return {1'($urandom()), e5 ? 7'h7C : 7'h7E};                  // E5M2 inf / E4M3 最大值 448
      3: return {1'($urandom()), 7'(e5 ? $urandom_range(3, 1) : $urandom_range(7, 1))};   // 次正規
      default: return {1'($urandom()), e5 ? 7'h7B : 7'h08};            // E5M2 最大有限值 / E4M3 最小正規數
    endcase
  do v = 8'($urandom()); while (e5 ? (v[6:2] == 5'h1F) : (v[6:0] == 7'h7F));   // 有限值
  if (mode == 0) v[6] = e5 ? (v[6] && v[5:2] < 4'hC) : (v[6] && v[5:3] < 3'h6);  // 指數偏小
  return v;
endfunction
function automatic logic [15:0] fpr_pick16(int mode);
  logic [15:0] v = 16'($urandom());
  if (mode == 2) return v;
  if (mode == 1 && $urandom_range(3, 0) == 0)
    case ($urandom_range(5, 0))
      0: return {1'($urandom()), 15'h0000};                            // ±0
      1: return {1'($urandom()), 8'hFF, 7'($urandom_range(127, 1))};   // NaN
      2: return {1'($urandom()), 8'hFF, 7'h00};                        // inf
      3: return {1'($urandom()), 8'h00, 7'($urandom())};               // 次正規 (DAZ)
      4: return {1'($urandom()), 8'(($urandom_range(1, 0) == 0) ? $urandom_range(3, 1) : $urandom_range(254, 250)), 7'($urandom())};  // 極小 / 極大
      default: return {1'($urandom()), 8'($urandom_range(254, 1)), 7'($urandom())};
    endcase
  v[14:7] = 8'($urandom_range(147, 107));                              // 2^-20 .. 2^20
  return v;
endfunction
`endif
