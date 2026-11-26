// SPDX-License-Identifier: Apache-2.0
// npu_pe_mac 單元測試: FP8 (四種子格式組合) 與 INT8 的所有位元組對窮舉, BF16 / 累加隨機, 對照 npu_fp_ref.svh
// (獨立撰寫: real 求值 + double 精確乘積; 累加用與 fp_ref.py 比對過的 mx_add)。
// plusargs: +seed=N  +n=隨機向量數 (預設 400000)
`include "npu_fp_ref.svh"
module npu_pe_mac_tb;
  logic [7:0]  a, b, a_lo, b_lo;
  logic        sa, sb, fp, bf, h, first;
  logic [31:0] acc, sum;
  npu_pe_mac dut (.*);
  int unsigned seed, n, n_err = 0, n_chk = 0;

  function automatic logic [31:0] ref_sum();
    logic [31:0] p;
    longint s;
    if (!fp) begin
      s = (sa ? longint'($signed(a)) : longint'(a)) * (sb ? longint'($signed(b)) : longint'(b));
      return (first ? 32'd0 : acc) + 32'(s);
    end
    if (bf) p = fpr_mul(fpr_dec(h ? a : 8'h00, h ? a_lo : a, 0, 1), fpr_dec(h ? b : 8'h00, h ? b_lo : b, 0, 1));
    else    p = fpr_mul(fpr_dec(a, 8'h00, sa, 0), fpr_dec(b, 8'h00, sb, 0));
    return first ? p : mx_add(acc, p);
  endfunction
  task automatic chk();
    logic [31:0] r;
    #1 r = ref_sum(); n_chk++;
    if (sum !== r) begin
      if (n_err < 20) $display("MISMATCH fp=%0d bf=%0d h=%0d sa=%0d sb=%0d first=%0d a=%h b=%h a_lo=%h b_lo=%h acc=%h dut=%h ref=%h",
                               fp, bf, h, sa, sb, first, a, b, a_lo, b_lo, acc, sum, r);
      n_err++;
    end
  endtask
  function automatic logic [31:0] pick_acc();
    case ($urandom_range(4, 0))
      0: return 32'($urandom());
      1: return {1'($urandom()), 8'hFF, 23'(($urandom_range(1, 0) == 0) ? 0 : $urandom())};   // inf / NaN
      2: return {1'($urandom()), 8'h00, 23'($urandom())};                                         // 0 / 次正規 (DAZ)
      default: return {1'($urandom()), 8'($urandom_range(160, 90)), 23'($urandom())};
    endcase
  endfunction

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n)) n = 400000;
    void'($urandom(seed));
    a_lo = 8'h00; b_lo = 8'h00; h = 0; acc = 0;
    // 窮舉: INT8 (sa, sb) 與 FP8 (E4M3 / E5M2 組合) 的所有位元組對, first = 1 (sum = 乘積)
    for (int f = 0; f < 2; f++) for (int s = 0; s < 4; s++) for (int i = 0; i < 256; i++) for (int j = 0; j < 256; j++) begin
      fp = f[0]; bf = 0; sa = s[1]; sb = s[0]; first = 1; a = 8'(i); b = 8'(j);
      chk();
    end
    // 隨機: BF16 (高 / 低位元組步驟)、累加 (first = 0)、旗標任意
    repeat (n) begin
      logic [15:0] x, y;                                                // 帶初值的區域變數只初始化一次: 分開賦值
      x = fpr_pick16($urandom_range(2, 0)); y = fpr_pick16($urandom_range(2, 0));
      fp = ($urandom_range(5, 0) != 0); bf = 1'($urandom()); h = 1'($urandom()); sa = 1'($urandom()); sb = 1'($urandom());
      first = ($urandom_range(3, 0) == 0); acc = pick_acc();
      if (bf && h) begin a = x[15:8]; a_lo = x[7:0]; b = y[15:8]; b_lo = y[7:0]; end
      else if (bf) begin a = x[7:0]; b = y[7:0]; a_lo = 8'($urandom()); b_lo = 8'($urandom()); end
      else begin a = fpr_pick8(sa, $urandom_range(2, 0)); b = fpr_pick8(sb, $urandom_range(2, 0)); a_lo = 8'($urandom()); b_lo = 8'($urandom()); end
      chk();
    end
    // 邊界: BF16 乘積的指數落在下溢 (0 / 1) 與溢位 (254 / 255) 附近
    fp = 1; bf = 1; h = 1; first = 1;
    repeat (n / 8) begin
      int ex, t, ey;
      logic [15:0] x, y;
      ex = $urandom_range(254, 1); t = ($urandom_range(1, 0) == 0) ? int'($urandom_range(3, 0)) - 1 : $urandom_range(256, 253);
      ey = t + 127 - ex - $urandom_range(1, 0);                         // 尾數相乘可能進位 1
      if (ey < 1 || ey > 254) continue;
      x = {1'($urandom()), 8'(ex), 7'($urandom())}; y = {1'($urandom()), 8'(ey), 7'($urandom())};
      a = x[15:8]; a_lo = x[7:0]; b = y[15:8]; b_lo = y[7:0];
      chk();
    end
    $display("PEMAC: seed=%0d checked=%0d errors=%0d", seed, n_chk, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
