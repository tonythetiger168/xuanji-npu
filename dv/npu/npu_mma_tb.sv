// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_mma — constrained-random TB
//   每個 tile: 隨機 K (1..KMAX)、有號 / 無號組合、資料樣式 (隨機 / 邊界值 / 稀疏 / 全滿 / 單位矩陣),
//   driver 隨機插入空拍與「忙碌中的 start / 未就緒時的 in_valid」(應被忽略, 搭配垃圾資料),
//   monitor 隨機 out_ready 回壓; scoreboard 以整數 GEMM 參考模型逐列比對。
//   每 tile 隨機 test_en (ICG 強制開啟 / 實際閘控) — 兩者結果必須相同; 偶爾在運算中途 reset。
// plusargs: +seed=N  +tiles=N (預設 400)
//=============================================================================
module npu_mma_tb;
  localparam int N = 8, KMAX = 80;
  logic clk = 1'b0, rst_n = 1'b0;
  logic test_en = 1'b0, start = 1'b0, a_signed = 1'b0, b_signed = 1'b0;
  logic in_valid = 1'b0, in_last = 1'b0, out_ready = 1'b0;
  logic [N-1:0][7:0] a_vec = '0, b_vec = '0;
  logic busy, in_ready, out_valid;
  logic [$clog2(N)-1:0] out_row;
  logic [N-1:0][31:0] out_vec;
  always #5 clk = ~clk;

  npu_mma #(.N(N)) dut (.*);

  logic [7:0] A[N][KMAX], B[KMAX][N];
  logic [31:0] C[N][N];
  int unsigned seed, ntiles, n_ok = 0, n_rst = 0, n_err = 0, n_rows = 0;

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic logic [7:0] pick(int mode);
    case (mode)
      1: case (ur(0, 5)) 0: return 8'h00; 1: return 8'h7F; 2: return 8'h80; 3: return 8'hFF; 4: return 8'h01; default: return 8'($urandom()); endcase
      2: return (ur(0, 7) == 0) ? 8'($urandom()) : 8'h00;
      3: return 8'hFF;
      default: return 8'($urandom());
    endcase
  endfunction

  task automatic gen(int k, int mode, bit sa, bit sb);
    for (int i = 0; i < N; i++) for (int kk = 0; kk < k; kk++) A[i][kk] = (mode == 4) ? ((i == kk % N) ? 8'h01 : 8'h00) : pick(mode);
    for (int kk = 0; kk < k; kk++) for (int j = 0; j < N; j++) B[kk][j] = (mode == 4) ? 8'($urandom()) : pick(mode);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      longint s = 0;
      for (int kk = 0; kk < k; kk++)
        s += (sa ? longint'($signed(A[i][kk])) : longint'(A[i][kk])) * (sb ? longint'($signed(B[kk][j])) : longint'(B[kk][j]));
      C[i][j] = 32'(s);
    end
  endtask

  task automatic do_reset(int cyc);
    rst_n = 1'b0; start = 1'b0; in_valid = 1'b0; out_ready = 1'b0;
    repeat (cyc) @(posedge clk);
    #1 rst_n = 1'b1;
  endtask

  // 一個 tile; rst_at > 0 時在第 rst_at 拍中途 reset (放棄本 tile)
  task automatic run_tile(int k, int mode, bit sa, bit sb, int bub, int bp, int rst_at);
    int ks = 0, rows = 0, cyc = 0;
    bit started = 0;
    gen(k, mode, sa, sb);
    if (busy) begin $display("ERROR: busy at tile start"); n_err++; end
    while (rows < N) begin
      // ---- 驅動 (posedge 後 #1) ----
      start    = !started || (ur(0, 7) == 0);           // 忙碌中的 start 應被忽略
      a_signed = started ? 1'($urandom()) : sa;          // 忽略期間給垃圾值
      b_signed = started ? 1'($urandom()) : sb;
      in_valid = (started && ks < k && ur(0, 99) >= bub) || (ur(0, 15) == 0);
      in_last  = (ks == k - 1);
      for (int i = 0; i < N; i++) begin
        a_vec[i] = (started && ks < k) ? A[i][ks] : 8'($urandom());
        b_vec[i] = (started && ks < k) ? B[ks][i] : 8'($urandom());
      end
      out_ready = (ur(0, 99) >= bp);
      // ---- 邊沿前取樣握手 ----
      #4;
      if (in_valid && in_ready) begin
        if (!(started && ks < k)) begin $display("ERROR: in_ready accepted garbage (ks=%0d)", ks); n_err++; end
        ks++;
      end
      if (out_valid && out_ready) begin
        if (out_row != rows[$clog2(N)-1:0]) begin $display("ERROR: out_row %0d expected %0d", out_row, rows); n_err++; end
        for (int j = 0; j < N; j++)
          if (out_vec[j] !== C[rows][j]) begin
            if (n_err < 20) $display("MISMATCH tile k=%0d mode=%0d sa=%0d sb=%0d te=%0d C[%0d][%0d] dut=%h ref=%h",
                                     k, mode, sa, sb, test_en, rows, j, out_vec[j], C[rows][j]);
            n_err++;
          end
        rows++; n_rows++;
      end
      if (start && !busy) started = 1;
      @(posedge clk); #1;
      cyc++;
      if (rst_at > 0 && cyc == rst_at) begin
        do_reset(ur(1, 3)); n_rst++;
        if (busy) begin $display("ERROR: busy after reset"); n_err++; end
        return;
      end
      if (cyc > 20 * (k + 4 * N) + 2000) begin $display("ERROR: timeout k=%0d", k); n_err++; return; end
    end
    start = 1'b0; in_valid = 1'b0;
    n_ok++;
  endtask

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("tiles=%d", ntiles)) ntiles = 400;
    void'($urandom(seed));
    do_reset(3);
    // directed: K=1 / K=N / 最大量級 (有號 -128*-128, 無號 255*255) / 兩種 test_en
    run_tile(1, 0, 1, 1, 0, 0, 0);
    run_tile(N, 4, 0, 0, 0, 0, 0);
    run_tile(KMAX, 3, 0, 0, 0, 0, 0);
    run_tile(KMAX, 1, 1, 1, 30, 30, 0);
    for (int t = 0; t < ntiles; t++) begin
      int k, mode, bub, bp, ra;
      test_en = (ur(0, 3) == 0);
      case (ur(0, 5)) 0: k = 1; 1: k = ur(2, N); 2: k = ur(N + 1, 2 * N); 3: k = ur(64, KMAX); default: k = ur(1, KMAX); endcase
      mode = ur(0, 4);
      bub  = (ur(0, 2) == 0) ? 0 : ur(0, 70);
      bp   = (ur(0, 2) == 0) ? 0 : ur(0, 70);
      ra   = (ur(0, 19) == 0) ? ur(1, k + 3 * N) : 0;
      run_tile(k, mode, 1'($urandom()), 1'($urandom()), bub, bp, ra);
      if (n_err > 50) break;
    end
    $display("MMA: seed=%0d tiles_ok=%0d resets=%0d rows=%0d errors=%0d", seed, n_ok, n_rst, n_rows, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
