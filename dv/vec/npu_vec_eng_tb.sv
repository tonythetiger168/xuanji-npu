// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_vec_eng TB: 讀入 gen_eng_vectors.py 產生的 L2 內容 (含程式) 與命令串, 逐一下命令 (命令間隨機空拍),
// 檢查每個命令的 done / derr (命令之間偶爾 reset; 30% 的命令在上一個的 done 拍就提出), 最後比對資料區的 L2 內容 (期望值來自 vec_ref.py)。L2 讀 / 寫 grant 隨機 (+gnt_p),
// 沒有 grant 時讀資料為垃圾; test_en 隨機切換。plusargs: +vec=<base>  +gnt_p=N  +seed=N  +perf
//=============================================================================
module npu_vec_eng_tb;
  logic clk = 0, rst_n = 0, test_en = 0;
  always #5 clk = ~clk;
  logic cmd_valid = 0, cmd_ready, done, derr, busy;
  logic [25:0] cmd_line = '0;
  logic [15:0] cmd_n = '0;
  logic rd_req, rd_gnt, wr_req, wr_gnt;
  logic [25:0] rd_line, wr_line;
  logic [511:0] rd_data, wr_data;
  npu_vec_eng dut (.*);

  logic [511:0] mem [logic [25:0]];
  logic g_rd = 0, g_wr = 0;
  assign rd_gnt  = rd_req && g_rd;
  assign wr_gnt  = wr_req && g_wr;
  assign rd_data = !rd_gnt ? {16{32'hDEAD_BEEF}} : mem.exists(rd_line) ? mem[rd_line] : '0;
  int gnt_p = 70, seed = 1, n_err = 0, n_cmd = 0, n_ins = 0, cyc = 0, busy_cyc = 0, n_rst = 0;
  bit perf = 0, early = 0;
  always @(posedge clk) begin
    if (wr_gnt) mem[wr_line] = wr_data;
    g_rd <= ($urandom_range(99) < gnt_p); g_wr <= ($urandom_range(99) < gnt_p);
    if (!perf && $urandom_range(99) < 5) test_en <= !test_en;
    cyc++;
    if (busy) busy_cyc++;
  end
  // done 只在命令執行中出現, 每個命令一次
  int pend = 0;
  always @(posedge clk) if (rst_n) begin
    if (cmd_valid && cmd_ready) pend++;
    if (done) begin
      if (pend != 1) begin n_err++; $display("ERROR: done without command"); end
      pend--;
    end
  end

  string base;
  initial begin
    logic [25:0] ln; logic [511:0] d; int n, e, fh;
    if (!$value$plusargs("vec=%s", base)) $fatal(1, "need +vec=");
    void'($value$plusargs("gnt_p=%d", gnt_p));
    void'($value$plusargs("seed=%d", seed));
    perf = $test$plusargs("perf");
    if (perf) gnt_p = 100;
    void'($urandom(seed));
    fh = $fopen({base, ".mem"}, "r");
    while ($fscanf(fh, "%h %h\n", ln, d) == 2) mem[ln] = d;
    $fclose(fh);
    repeat (3) @(posedge clk);
    rst_n = 1;
    fh = $fopen({base, ".cmd"}, "r");
    while ($fscanf(fh, "%h %d %d\n", ln, n, e) == 3) begin
      int t;
      t = 0;
      if (!early) begin
        repeat (perf ? 0 : $urandom_range(3)) @(negedge clk);
        if (!perf && $urandom_range(99) < 5) begin        // 命令之間 reset (引擎狀態清除; 向量暫存器檔保持)
          @(negedge clk);                                  // 先讓 done 在 posedge 被監看到
          rst_n = 0; @(negedge clk); rst_n = 1; n_rst++;
        end
        @(negedge clk);
      end
      cmd_valid = 1; cmd_line = ln; cmd_n = 16'(n);       // early: 在上一個命令的 done 拍就提出 (此拍不可接受)
      do @(negedge clk); while (!(cmd_ready === 1'b0 && busy));       // 已接受 (act)
      cmd_valid = 0;
      t = 0;
      while (!done && t < 400000) begin @(negedge clk); t++; end
      if (!done) begin n_err++; $display("ERROR: command %0d timeout", n_cmd); break; end
      if (derr !== e[0]) begin n_err++; $display("MISMATCH derr cmd %0d got %0d exp %0d", n_cmd, derr, e); end
      n_cmd++; n_ins += n;
      early = !perf && ($urandom_range(99) < 30);
    end
    $fclose(fh);
    repeat (2) @(negedge clk);
    fh = $fopen({base, ".exp"}, "r");
    while ($fscanf(fh, "%h %h\n", ln, d) == 2) begin
      logic [511:0] g;
      g = mem.exists(ln) ? mem[ln] : '0;                // (宣告時給初值在 Verilator 是 static, 只算一次)
      if (g !== d) begin n_err++; if (n_err < 10) $display("MISMATCH mem %h\n  got %h\n  exp %h", ln, g, d); end
    end
    $fclose(fh);
    $display("VECENG: commands=%0d ins=%0d busy_cycles=%0d resets=%0d errors=%0d", n_cmd, n_ins, busy_cyc, n_rst, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
