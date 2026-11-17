// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_llm_e2e_tb — orca_npu_ss 端到端: 依情境檔執行一段命令程式, 把指定的外部記憶體區域寫到輸出檔
//   情境與期望值由 dv/npu_llm/llm_e2e.py (numpy) 產生與比對 (paged attention 的 KV 寫入 / gather + QK^T、
//   MoE 的 router top-k → dispatch → token gather → combine)。
//   情境檔 (每行一筆, 16 進位):
//     E <ext line 位址> <512 bit 資料>              外部記憶體初值 (未給的 line 為 0)
//     C <op> <wt> <id> <ext> <a> <b> <c> <k> <sa> <sb> 命令 (依序送出)
//     O <名稱> <ext line 位址> <line 數>              結束後輸出的區域
//   輸出檔: 每個輸出 line 一行 "<名稱> <索引> <512 bit 資料>"。
//   外部記憶體模型: 讀 grant 隨機、回應依序 (延遲 1..8 拍), 寫 grant 隨機; 完成 ready 隨機。
// plusargs: +scn=<情境檔> +out=<輸出檔> +seed=N
//=============================================================================
module npu_llm_e2e_tb #(parameter int N = 8, parameter int NCU = 4, parameter int NB = 4, parameter int DEPTH = 1024,
                         parameter int OUT = 64);
  import npu_pkg::*;
  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready;
  npu_tile_cmd_t cmd = '0;
  logic cpl_valid, cpl_ready = 1'b0, irq, busy, vec_err;
  logic [7:0] cpl_id;
  logic m_rd_req, m_rd_gnt = 1'b0, m_rd_rvalid = 1'b0, m_wr_req, m_wr_gnt = 1'b0;
  logic [31:0] m_rd_addr, m_wr_addr;
  logic [511:0] m_rd_rdata = '0, m_wr_data;
  logic perf_wr = 1'b0;
  logic [47:0] perf_wdata_busy = '0, perf_wdata_steps = '0, perf_busy, perf_steps;
  always #5 clk = ~clk;

  orca_npu_ss #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .OUT(OUT)) dut (.*);

  logic [511:0] emem [int unsigned];
  function automatic logic [511:0] erd(int unsigned a); return emem.exists(a) ? emem[a] : '0; endfunction
  typedef struct { logic [511:0] d; longint t; } rsp_t;
  rsp_t rq [$];
  longint cyc = 0, last_t = 0;
  always @(posedge clk) begin
    cyc++;
    if (rst_n) begin
      if (m_rd_req && m_rd_gnt) begin
        automatic longint t = cyc + longint'($urandom_range(8, 1));
        if (t <= last_t) t = last_t + 1;
        last_t = t;
        rq.push_back('{erd(m_rd_addr), t});
      end
      if (rq.size() != 0 && rq[0].t <= cyc) begin m_rd_rvalid <= 1'b1; m_rd_rdata <= rq.pop_front().d; end
      else begin m_rd_rvalid <= 1'b0; m_rd_rdata <= {16{$urandom()}}; end
      m_rd_gnt <= ($urandom_range(99, 0) < 70);
      if (m_wr_req && m_wr_gnt) emem[m_wr_addr] = m_wr_data;
      m_wr_gnt <= ($urandom_range(99, 0) < 70);
      cpl_ready <= ($urandom_range(99, 0) < 80);
    end
  end

  npu_tile_cmd_t prog [$];
  string onames [$];
  int unsigned oaddr [$], on [$];
  int ncpl = 0, nacc = 0;
  bit seen [256];
  always @(posedge clk) if (rst_n && cpl_valid && cpl_ready) begin ncpl++; seen[cpl_id] = 1'b1; end
  always @(posedge clk) if (rst_n && cmd_valid && cmd_ready) nacc++;
  bit fail = 0;

  initial begin
    string scn, outf, ln, tag, nm;
    int fd, seed, r;
    int unsigned a32, n, op, wt, id, ext, a, b, c, k, sa, sb;
    logic [511:0] d;
    longint t0;
    if (!$value$plusargs("scn=%s", scn)) begin $display("TB FAIL: +scn=<file> required"); $finish; end
    if (!$value$plusargs("out=%s", outf)) outf = "e2e_out.txt";
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    void'($urandom(seed));
    fd = $fopen(scn, "r");
    if (fd == 0) begin $display("TB FAIL: cannot open %s", scn); $finish; end
    while ($fgets(ln, fd)) begin
      tag = "";
      r = $sscanf(ln, "%s", tag);
      if (tag == "E") begin r = $sscanf(ln, "%s %h %h", tag, a32, d); emem[a32] = d; end
      else if (tag == "C") begin
        npu_tile_cmd_t x;
        r = $sscanf(ln, "%s %h %h %h %h %h %h %h %h %h %h", tag, op, wt, id, ext, a, b, c, k, sa, sb);
        x = '0; x.op = npu_op_e'(op); x.wt = wt[0]; x.id = 8'(id); x.ext = ext; x.a = 26'(a); x.b = 26'(b); x.c = 26'(c); x.k = 16'(k);
        x.sa = sa[0]; x.sb = sb[0];
        prog.push_back(x);
      end
      else if (tag == "O") begin r = $sscanf(ln, "%s %s %h %h", tag, nm, a32, n); onames.push_back(nm); oaddr.push_back(a32); on.push_back(n); end
    end
    $fclose(fd);
    $display("E2E: %0d commands, %0d ext lines, %0d output regions", prog.size(), emem.size(), onames.size());
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
    t0 = cyc;
    foreach (prog[i]) begin                               // 在負緣驅動, 以正緣的握手計數確認被接受
      @(negedge clk);
      cmd = prog[i]; cmd_valid = 1'b1;
      while (nacc != i + 1) @(negedge clk);
      cmd_valid = 1'b0;
    end
    for (int w = 0; w < 2_000_000 && (ncpl != prog.size() || busy); w++) @(posedge clk);
    if (ncpl != prog.size() || busy) begin $display("TB FAIL: %0d / %0d completions, busy %0b", ncpl, prog.size(), busy); fail = 1; end
    foreach (prog[i]) if (!seen[prog[i].id]) begin $display("TB FAIL: no completion for id %0d", prog[i].id); fail = 1; end
    $display("E2E: done in %0d cycles", cyc - t0);
    fd = $fopen(outf, "w");
    foreach (onames[j]) for (int unsigned i = 0; i < on[j]; i++) $fdisplay(fd, "%s %0d %0128h", onames[j], i, erd(oaddr[j] + i));
    $fclose(fd);
    if (!fail) $display("TB PASS");
    $finish;
  end
endmodule
