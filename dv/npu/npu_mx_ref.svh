// SPDX-License-Identifier: Apache-2.0
// NPU GEMM 的 MX 區塊比例參考函式 (npu_gemm_cu / npu_core / npu_ss 的 TB 共用; 與 RTL 分開撰寫)。
// 語意同 dv/fp/fp_ref.py: FP32 DAZ (指數 0 當 ±0)、FTZ (結果小於最小正規數 → ±0, 保留符號)、RNE、溢位 → ±inf、
// NaN / inf - inf → qNaN 0x7FC00000; 精確和為 0 → +0, 但 (-0) + (-0) = -0。
`ifndef NPU_MX_REF_SVH
`define NPU_MX_REF_SVH
// 正數 m (< 2^62) × 2^(ex - 150) 依 RNE 捨入成 FP32 (ex: 對應 m 的 bit 23 為隱藏位元時的偏置指數)
function automatic logic [31:0] mx_round(bit s, longint unsigned m, int ex);
  int p = 63;
  longint unsigned q, rem, half;
  int e;
  while (p > 0 && !m[p]) p--;
  if (p > 23) begin
    q = m >> (p - 23); rem = m & ((64'd1 << (p - 23)) - 1); half = 64'd1 << (p - 24);
    if (rem > half || (rem == half && q[0])) q++;
  end else q = m << (23 - p);
  e = ex + p - 23;
  if (q == (64'd1 << 24)) begin q >>= 1; e++; end
  if (e > 254) return {s, 8'hFF, 23'd0};
  if (e < 1)   return {s, 31'd0};
  return {s, 8'(e), 23'(q)};
endfunction
function automatic logic [31:0] mx_i2f(logic [31:0] a);
  longint v = longint'($signed(a));
  if (v == 0) return 32'd0;
  return mx_round(v < 0, (v < 0) ? longint'(-v) : longint'(v), 150);   // m × 2^0: 隱藏位元在 bit 23 時指數 = 150
endfunction
// x × 2^(ea + eb - 254) (E8M0 比例; 任一為 0xFF → qNaN)
function automatic logic [31:0] mx_scale(logic [31:0] x, logic [7:0] ea, logic [7:0] eb);
  int e;
  if (ea == 8'hFF || eb == 8'hFF) return 32'h7FC0_0000;
  if (x[30:23] == 8'd0 || x[30:23] == 8'hFF) return x;                   // 0 / inf / NaN 不變 (整數命令只會有 +0)
  e = int'(x[30:23]) + int'(ea) + int'(eb) - 254;
  if (e > 254) return {x[31], 8'hFF, 23'd0};
  if (e < 1)   return {x[31], 31'd0};
  return {x[31], 8'(e), x[22:0]};
endfunction
function automatic logic [31:0] mx_add(logic [31:0] a, logic [31:0] b);
  bit an = a[30:23] == 8'hFF && a[22:0] != 0, bn = b[30:23] == 8'hFF && b[22:0] != 0;
  bit ai = a[30:23] == 8'hFF && a[22:0] == 0, bi = b[30:23] == 8'hFF && b[22:0] == 0;
  bit az = a[30:23] == 8'd0, bz = b[30:23] == 8'd0;
  int ea, eb, d;
  longint ma, mb, v;
  if (an || bn) return 32'h7FC0_0000;
  if (ai && bi) return (a[31] == b[31]) ? a : 32'h7FC0_0000;
  if (ai) return a;
  if (bi) return b;
  if (az && bz) return {a[31] && b[31], 31'd0};
  if (az) return b;
  if (bz) return a;
  ea = int'(a[30:23]); eb = int'(b[30:23]);
  ma = longint'({1'b1, a[22:0]}); mb = longint'({1'b1, b[22:0]});
  if (a[31]) ma = -ma;
  if (b[31]) mb = -mb;
  // 對齊到較小的指數 (差 > 30 時小的一方小於大數 1/4 ulp, 結果為大數)
  if (ea >= eb) begin
    d = ea - eb; if (d > 30) return a;
    v = (ma <<< d) + mb;
    if (v == 0) return 32'd0;
    return mx_round(v < 0, (v < 0) ? longint'(-v) : longint'(v), eb);
  end else begin
    d = eb - ea; if (d > 30) return b;
    v = (mb <<< d) + ma;
    if (v == 0) return 32'd0;
    return mx_round(v < 0, (v < 0) ? longint'(-v) : longint'(v), ea);
  end
endfunction
// 一個元素: INT32 的 A×B 結果 → FP32 × 比例 (+ 舊的 C)
// 邊界比例: 讓部分和 p 縮放後的指數落在 t (例: 0 / 1 下溢邊界, 254 / 255 溢位邊界); p = 0 或比例超出 0..254 時回傳 0
function automatic bit mx_edge_sc_f(logic [31:0] f, int t, output logic [7:0] ea, output logic [7:0] eb);   // f: FP32
  int fe = int'(f >> 23) & 255, a;
  eb = (t < 128) ? 8'd0 : 8'd254;
  a = t + 254 - fe - int'(eb);
  ea = 8'(a);
  return fe != 0 && fe != 255 && a >= 0 && a <= 254;
endfunction
function automatic bit mx_edge_sc(logic [31:0] p, int t, output logic [7:0] ea, output logic [7:0] eb);       // p: INT32
  return mx_edge_sc_f(mx_i2f(p), t, ea, eb);
endfunction
function automatic logic [31:0] mx_elem(logic [31:0] p, logic [7:0] ea, logic [7:0] eb, bit acc, logic [31:0] old);
  logic [31:0] g = mx_scale(mx_i2f(p), ea, eb);
  return acc ? mx_add(old, g) : g;
endfunction
`endif
