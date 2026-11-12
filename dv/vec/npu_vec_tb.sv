// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_vec TB: 讀入 gen_vec_vectors.py / vec_kernels.py 產生的 L2 內容與指令串 (期望值來自 vec_ref.py),
// 逐指令比對: 寫入暫存器的值、寫入 L2 的資料、err。L2 模型的讀 / 寫 grant 以 +gnt_p 機率隨機給 (預設 70%),
// 指令之間隨機插入空拍 (+idle_p), test_en 隨機切換 (功能必須相同)。reset 記錄: 中途 reset (暫存器檔保持);
// err_clr 記錄: 一拍 err_clr。
// plusargs: +vec=<base> (讀 <base>.mem / <base>.ins)  +gnt_p=N  +idle_p=N  +seed=N  +perf (grant 全給、無空拍、
//           test_en = 0, 回報拍數)
//=============================================================================
module npu_vec_tb;
  import npu_vec_pkg::*;
  logic clk = 0, rst_n = 0, test_en = 0;
  always #5 clk = ~clk;
  logic ins_valid = 0, ins_ready, err, err_clr = 0;
  npu_vec_ins_t ins = '0;
  logic rd_req, rd_gnt, wr_req, wr_gnt;
  logic [25:0] rd_line, wr_line;
  logic [511:0] rd_data, wr_data;
  npu_vec dut (.*);

  logic [511:0] mem [logic [25:0]];
  logic g_rd = 0, g_wr = 0;
  assign rd_gnt  = rd_req && g_rd;
  assign wr_gnt  = wr_req && g_wr;
  // 沒有 grant 時讀資料為垃圾 (未等 grant 就取資料的錯誤會造成不符)
  assign rd_data = !rd_gnt ? {16{32'hDEAD_BEEF}} : mem.exists(rd_line) ? mem[rd_line] : '0;
  always @(posedge clk) if (wr_gnt) mem[wr_line] = wr_data;

  int gnt_p = 70, idle_p = 10, seed = 1, n_err = 0, n_ins = 0, n_cyc = 0;
  bit perf = 0;
  string base;
  int fm, fi;

  task automatic check(int kind, int e_err, logic [511:0] exp, npu_vec_ins_t ci);
    // 在 fire 的拍 (posedge 之前) 檢查組合式輸出; err 在下一拍檢查
    if (kind == 1 && !(dut.we && dut.wd == exp)) begin
      n_err++; if (n_err < 10) $display("MISMATCH reg ins=%h op=%0d fn=%0d\n  got %h\n  exp %h", ci, ci.op, ci.fn, dut.wd, exp);
    end
    if (kind == 2 && wr_data !== exp) begin
      n_err++; if (n_err < 10) $display("MISMATCH st ins=%h\n  got %h\n  exp %h", ci, wr_data, exp);
    end
    if ((kind == 0 || kind == 3) && (dut.we || rd_req || wr_req)) begin      // NOP / 非法: 不寫暫存器、不存取 L2
      n_err++; if (n_err < 10) $display("MISMATCH unexpected write ins=%h", ci);
    end
  endtask

  initial begin
    logic [25:0] ln;
    logic [511:0] d, exp;
    logic [76:0] iv;
    int kind, e_err, c0;
    if (!$value$plusargs("vec=%s", base)) $fatal(1, "need +vec=");
    void'($value$plusargs("gnt_p=%d", gnt_p));
    void'($value$plusargs("idle_p=%d", idle_p));
    void'($value$plusargs("seed=%d", seed));
    perf = $test$plusargs("perf");
    if (perf) begin gnt_p = 100; idle_p = 0; end
    void'($urandom(seed));
    fm = $fopen({base, ".mem"}, "r");
    while ($fscanf(fm, "%h %h\n", ln, d) == 2) mem[ln] = d;
    $fclose(fm);
    fi = $fopen({base, ".ins"}, "r");
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(negedge clk);
    c0 = n_cyc;
    while ($fscanf(fi, "%h %d %d %h\n", iv, kind, e_err, exp) == 4) begin
      if (kind == 4) begin                         // reset (暫存器檔保持)
        ins_valid = 0; rst_n = 0; @(negedge clk); rst_n = 1; @(negedge clk);
        continue;
      end
      if (kind == 5) begin                         // err_clr 一拍 (沒有指令)
        ins_valid = 0; err_clr = 1; @(negedge clk); err_clr = 0;
        if (err !== 1'b0) begin n_err++; $display("MISMATCH err after err_clr"); end
        continue;
      end
      while (!perf && $urandom_range(99) < idle_p) begin
        ins_valid = 0; g_rd = 0; g_wr = 0; @(negedge clk);
      end
      ins_valid = 1; ins = npu_vec_ins_t'(iv);
      if (!perf) test_en = ($urandom_range(99) < 10);
      forever begin
        g_rd = ($urandom_range(99) < gnt_p); g_wr = ($urandom_range(99) < gnt_p);
        #1;
        if (ins_ready) break;
        @(negedge clk);
      end
      check(kind, e_err, exp, ins);
      n_ins++;
      @(negedge clk);
      if (err !== e_err[0]) begin
        n_err++; if (n_err < 10) $display("MISMATCH err ins=%h got %0d exp %0d", ins, err, e_err);
      end
    end
    ins_valid = 0;
    $fclose(fi);
    $display("VEC: ins=%0d cycles=%0d errors=%0d", n_ins, n_cyc - c0, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
  always @(posedge clk) if (rst_n) n_cyc++;
endmodule
