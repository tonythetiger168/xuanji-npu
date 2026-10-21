// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// SFU TB:
//   1. 位元一致: 讀入 dv/fp/gen_sfu_vectors.py 的向量 (期望值來自 dv/fp/sfu_ref.py), 逐筆比對。
//   2. 精度: 對雙精度參考 ($sqrt / $exp / $pow) 量測 ULP 誤差 — recip / rsqrt 窮舉所有尾數
//      (rsqrt 含指數奇偶), exp2 / exp 掃描 +exp_n 個點; 最大誤差須 <= +max_ulp (預設 1.0)。
// plusargs: +vec=<檔案>  +exp_n=N (預設 2000000)  +max_ulp=R  +no_sweep
//=============================================================================
module sfu_tb;
  logic [1:0]  op;
  logic [31:0] x, y;
  sfu dut (.op(op), .x(x), .y(y));
  // 每個係數表的每一項都被用到 (表的內容由精度掃描驗證)
  covergroup cg_tab with function sample(int o, bit odd, logic [6:0] i);
    cp_recip:  coverpoint i iff (o == 0)          { bins e[] = {[0:127]}; }
    cp_rsq_e:  coverpoint i iff (o == 1 && !odd)  { bins e[] = {[0:127]}; }
    cp_rsq_o:  coverpoint i iff (o == 1 && odd)   { bins e[] = {[0:127]}; }
    cp_exp2:   coverpoint i iff (o >= 2)          { bins e[] = {[0:127]}; }
  endgroup
  cg_tab u_cg = new;
  always @(op, x) #0 if (op != 2'd1 || !x[31]) u_cg.sample(int'(op), dut.eu[0], dut.idx);

  int n_err = 0, n_vec = 0;
  string vec;
  real max_ulp = 1.0;
  real worst [4];

  // FP32 位元 ↔ real (Verilator 5.052 的 $bitstoshortreal / $shortrealtobits 回傳 0, 不可用)
  function automatic real b2r(logic [31:0] b);
    real m;
    if (b[30:23] == 8'h00) return 0.0;
    if (b[30:23] == 8'hFF) return b[31] ? -1.0e308 * 10.0 : 1.0e308 * 10.0;   // ±inf
    m = (1.0 + real'(b[22:0]) / 8388608.0) * $pow(2.0, real'(int'(b[30:23]) - 127));
    return b[31] ? -m : m;
  endfunction
  function automatic logic [31:0] r2b(real v);        // 截斷取 FP32 (只用來產生輸入樣本)
    real a; int e; logic s;
    s = (v < 0); a = s ? -v : v;
    if (a < 1.1754943508222875e-38) return {s, 31'b0};
    e = int'($floor($ln(a) / $ln(2.0)));
    if ($pow(2.0, e) > a) e--;
    if ($pow(2.0, e + 1) <= a) e++;
    return {s, 8'(e + 127), 23'(longint'($floor((a / $pow(2.0, e) - 1.0) * 8388608.0)))};
  endfunction

  function automatic real ulp_err(int o, logic [31:0] xi, logic [31:0] yi);
    real xr, r, g, a;
    int e;
    xr = b2r(xi);
    g  = b2r(yi);
    case (o)
      0: r = 1.0 / xr;
      1: r = 1.0 / $sqrt(xr);
      2: r = $pow(2.0, xr);
      default: r = $exp(xr);
    endcase
    a = (r < 0) ? -r : r;
    if (a < 1.1754943508222875e-38) return (g == 0.0) ? 0.0 : 1e9;      // 參考值在 FTZ 範圍
    if (a >= 3.4028235677973366e38) return (yi[30:23] == 8'hFF) ? 0.0 : 1e9;
    e = int'($floor($ln(a) / $ln(2.0)));
    if ($pow(2.0, e) > a) e--;
    if ($pow(2.0, e + 1) <= a) e++;
    return ((g > r) ? (g - r) : (r - g)) / $pow(2.0, e - 23);
  endfunction

  task automatic sweep_check(int o, logic [31:0] xi);
    real u;
    op = 2'(o); x = xi; #1;
    u = ulp_err(o, xi, y);
    if (u > worst[o]) worst[o] = u;
  endtask

  initial begin
    int fd, r, n_exp = 2000000;
    logic [31:0] xv, ev, ov;
    if (!$value$plusargs("vec=%s", vec)) vec = "build/sfu_vec.hex";
    void'($value$plusargs("exp_n=%d", n_exp));
    void'($value$plusargs("max_ulp=%f", max_ulp));
    fd = $fopen(vec, "r");
    if (fd == 0) begin $display("ERROR: cannot open %s", vec); n_err++; end
    else begin
      while (!$feof(fd)) begin
        r = $fscanf(fd, "%h %h %h\n", ov, xv, ev);
        if (r != 3) break;
        op = ov[1:0]; x = xv; #1; n_vec++;
        if (y !== ev) begin
          if (n_err < 20) $display("MISMATCH op=%0d x=%h dut=%h ref=%h", ov, xv, y, ev);
          n_err++;
        end
      end
      $fclose(fd);
    end
    if (!$test$plusargs("no_sweep")) begin
      for (int o = 0; o < 4; o++) worst[o] = 0.0;
      for (int f = 0; f < (1 << 23); f++) begin
        sweep_check(0, {1'b0, 8'd127, 23'(f)});
        sweep_check(1, {1'b0, 8'd127, 23'(f)});
        sweep_check(1, {1'b0, 8'd128, 23'(f)});
      end
      for (int i = 0; i < n_exp; i++) begin
        real v2 = real'($urandom()) / 4294967295.0 * 254.0 - 127.0;
        real ve = real'($urandom()) / 4294967295.0 * 174.0 - 87.0;
        sweep_check(2, r2b(v2));
        sweep_check(3, r2b(ve));
        if (i % 4 == 0) begin                  // 靠近 0 (小數部分精度)
          sweep_check(2, r2b(v2 / 128.0));
          sweep_check(3, r2b(ve / 87.0));
        end
      end
      $display("SFU accuracy (max ulp): recip=%0.3f rsqrt=%0.3f exp2=%0.3f exp=%0.3f (limit %0.2f)",
               worst[0], worst[1], worst[2], worst[3], max_ulp);
      for (int o = 0; o < 4; o++) if (worst[o] > max_ulp) begin $display("ERROR: op %0d error %0.3f ulp > %0.2f", o, worst[o], max_ulp); n_err++; end
    end
    $display("SFU: vectors=%0d errors=%0d", n_vec, n_err);
    if (n_err == 0 && n_vec > 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
