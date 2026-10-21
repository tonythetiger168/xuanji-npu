// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_gemm_cu — constrained-random TB
//   L2 行為模型 (64 B line, 組合式讀取) + 隨機 rd_gnt / wr_gnt 延遲 (模擬與其他 L2 使用者的仲裁);
//   每個命令隨機 K (含 >= 32768 的長命令, 累加器模 2^32 環繞)、有號性、A / B / C line 位址 (互不重疊;
//   高位元隨機 — 模型只解碼低 15 bit), 以整數 GEMM 參考模型比對寫回的 C, 並檢查 C 區以外沒有被寫。
//   命令之間隨機間隔, 偶爾在運算中途 reset。
// plusargs: +seed=N  +cmds=N (預設 200)
//=============================================================================
module npu_gemm_cu_tb #(parameter int N = 8);
  localparam int RP = (N == 32) ? 2 : 1;         // CU 讀取埠數 (與 npu_gemm_cu 相同公式)
  localparam int KMAX = 65535;
  localparam int AMAX = (KMAX * N + 63) / 64;    // A / B 各最多佔的 line 數 (N=16: 16384, N=32: 32768)
  localparam int LW = (N == 32) ? 17 : 16, LINES = 1 << LW;   // 模型只解碼低 LW bit
  localparam int CL = (4 * N * N + 63) / 64;     // C 佔的 line 數
  localparam int RS = (N == 32) ? 43600 : 21800; // 每塊區域的 line 數 (>= AMAX + 2, 三塊 <= LINES)
  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready, done, feeding;
  logic [25:0] cmd_a_line = '0, cmd_b_line = '0, cmd_c_line = '0;
  logic [15:0] cmd_k = 16'd1;
  logic cmd_a_signed = 1'b0, cmd_b_signed = 1'b0;
  logic [RP-1:0] rd_req, rd_gnt;
  logic wr_req, wr_gnt;
  logic [RP-1:0][25:0] rd_line;
  logic [25:0] wr_line;
  logic [RP-1:0][511:0] rd_data;
  logic [511:0] wr_data;
  always #5 clk = ~clk;

  npu_gemm_cu #(.N(N)) dut (.*);

  // ---------------- L2 模型 ----------------
  logic [511:0] mem [LINES];
  int gnt_pct = 100;                           // 每拍給 grant 的機率 (%)
  logic [RP-1:0] rg;                           // 每個讀取埠各自隨機 grant
  logic wg;
  always @(negedge clk) begin
    for (int p = 0; p < RP; p++) rg[p] = ($urandom_range(99, 0) < gnt_pct);
    wg = ($urandom_range(99, 0) < gnt_pct);
  end
  assign rd_gnt  = rd_req & rg;
  assign wr_gnt  = wr_req && wg;
  for (genvar p = 0; p < RP; p++) begin : g_rd
    assign rd_data[p] = mem[rd_line[p][LW-1:0]];
  end
  int n_wr_bad = 0;
  logic [25:0] c_lo, c_hi;                     // 本命令的 C 區 [c_lo, c_hi) (line)
  always @(posedge clk) if (rst_n && wr_gnt) begin
    if (wr_line < c_lo || wr_line >= c_hi) begin
      $display("ERROR: write outside C region line=%h [%h,%h)", wr_line, c_lo, c_hi); n_wr_bad++;
    end
    mem[wr_line[LW-1:0]] <= wr_data;
  end

  // ---------------- 參考模型與檢查 ----------------
  int unsigned seed, ncmds, n_ok = 0, n_rst = 0, n_err = 0, n_done = 0;
  logic [7:0] A[N][KMAX], B[KMAX][N];
  always @(posedge clk) if (rst_n && done) n_done++;

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic logic [7:0] pick(int mode);
    case (mode)
      1: case (ur(0, 4)) 0: return 8'h00; 1: return 8'h7F; 2: return 8'h80; 3: return 8'hFF; default: return 8'($urandom()); endcase
      2: return (ur(0, 5) == 0) ? 8'($urandom()) : 8'h00;
      default: return 8'($urandom());
    endcase
  endfunction
  function automatic void put8(int unsigned line, int unsigned off, logic [7:0] v);   // off: line 起算的 byte 偏移
    mem[(line + off / 64) % LINES][(off % 64) * 8 +: 8] = v;
  endfunction
  function automatic logic [31:0] get32(int unsigned line, int unsigned off);
    return mem[(line + off / 64) % LINES][(off % 64) * 8 +: 32];
  endfunction

  task automatic run_cmd(int rst_at, int k);
    int mode = ur(0, 2);
    bit sa = 1'($urandom()), sb = 1'($urandom());
    int al = (k * N + 63) / 64;                // A / B 各佔的 line 數 (最多 AMAX)
    // line 位址: 低 LW bit 分成三塊互不重疊的區域 (各 RS line),
    // A / B / C 隨機分配到其中一塊; 高位元隨機
    int perm = ur(0, 5);
    int ra = (perm < 2) ? 0 : (perm < 4) ? 1 : 2;
    int rb = (ra + 1 + perm % 2) % 3, rc = 3 - ra - rb;
    logic [25:0] a0 = {(26-LW)'($urandom()), LW'(ur(RS * ra, RS * ra + RS - 2 - al))};
    logic [25:0] b0 = {(26-LW)'($urandom()), LW'(ur(RS * rb, RS * rb + RS - 2 - al))};
    logic [25:0] c0 = {(26-LW)'($urandom()), LW'(ur(RS * rc, RS * rc + RS - 2 - CL))};
    int cyc = 0, timeout = 40 * (k + 3 * N) + 400;
    longint s;
    for (int l = 0; l < al + 1; l++) begin mem[(a0[LW-1:0] + l) % LINES] = {16{$urandom()}}; mem[(b0[LW-1:0] + l) % LINES] = {16{$urandom()}}; end
    for (int l = 0; l < CL; l++) mem[c0[LW-1:0] + l] = {16{$urandom()}};   // 背景垃圾 (含 A / B 之後的 line 尾端)
    for (int kk = 0; kk < k; kk++) for (int i = 0; i < N; i++) begin
      A[i][kk] = pick(mode); B[kk][i] = pick(mode);
      put8(a0[LW-1:0], kk * N + i, A[i][kk]); put8(b0[LW-1:0], kk * N + i, B[kk][i]);
    end
    c_lo = c0; c_hi = c0 + 26'(CL);
    test_en = (ur(0, 3) == 0);
    gnt_pct = (ur(0, 2) == 0) ? 100 : ur(20, 90);
    // 送命令
    cmd_valid = 1'b1; cmd_a_line = a0; cmd_b_line = b0; cmd_c_line = c0; cmd_k = 16'(k);
    cmd_a_signed = sa; cmd_b_signed = sb;
    do @(posedge clk); while (!cmd_ready);
    #1 cmd_valid = 1'b0; cmd_a_line = 26'($urandom()); cmd_b_line = 26'($urandom()); cmd_c_line = 26'($urandom());  // 命令只在受理拍取樣
    cmd_a_signed = 1'($urandom()); cmd_b_signed = 1'($urandom()); cmd_k = 16'($urandom());
    while (!done) begin
      @(posedge clk); #1; cyc++;
      if (rst_at > 0 && cyc == rst_at) begin
        rst_n = 1'b0; repeat (ur(1, 3)) @(posedge clk); #1 rst_n = 1'b1; n_rst++;
        if (!cmd_ready) begin $display("ERROR: not idle after reset"); n_err++; end
        return;
      end
      if (cyc > timeout) begin $display("ERROR: timeout k=%0d", k); n_err++; return; end
    end
    // 比對 C
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      s = 0;
      for (int kk = 0; kk < k; kk++)
        s += (sa ? longint'($signed(A[i][kk])) : longint'(A[i][kk])) * (sb ? longint'($signed(B[kk][j])) : longint'(B[kk][j]));
      if (get32(c0[LW-1:0], 4 * (i * N + j)) !== 32'(s)) begin
        if (n_err < 20) $display("MISMATCH k=%0d sa=%0d sb=%0d C[%0d][%0d] dut=%h ref=%h", k, sa, sb, i, j, get32(c0[LW-1:0], 4 * (i * N + j)), 32'(s));
        n_err++;
      end
    end
    n_ok++;
    repeat (ur(0, 3)) @(posedge clk);
    #1;
  endtask

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cmds=%d", ncmds)) ncmds = 200;
    void'($urandom(seed));
    c_lo = 0; c_hi = 0;
    repeat (3) @(posedge clk); #1 rst_n = 1'b1;
    run_cmd(0, ur(32768, KMAX));               // 長命令: K 計數器高位元、累加器溢位環繞
    run_cmd(0, KMAX);                          // K 最大值 (line 計數器最高位元)
    for (int t = 0; t < ncmds; t++) begin
      run_cmd((ur(0, 15) == 0) ? ur(1, 150) : 0, (ur(0, 5) == 0) ? 1 : (ur(0, 3) == 0) ? ur(60, 300) : ur(1, 70));
      if (n_err + n_wr_bad > 30) break;
    end
    repeat (2) @(posedge clk);                 // 讓最後一個 done 脈衝被計數
    n_err += n_wr_bad;
    $display("GEMMCU: seed=%0d cmds_ok=%0d resets=%0d done_pulses=%0d errors=%0d", seed, n_ok, n_rst, n_done, n_err);
    if (n_done < n_ok) begin $display("ERROR: done pulses %0d < completed %0d", n_done, n_ok); n_err++; end
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
