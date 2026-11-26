// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_mma — constrained-random TB
//   每個 tile: 隨機 K (1..KMAX)、有號 / 無號組合、資料樣式 (隨機 / 邊界值 / 稀疏 / 全滿 / 單位矩陣)。
//   v6.3.12 tile 管線化: driver 與 monitor 同時運作 — driver 在輸入端閒置 (in_idle) 時就 start 下一個 tile
//   (多半緊接著, 與前一個 tile 的排空 / 輸出重疊), 隨機插入空拍與「輸入端忙碌時的 start / 未就緒時的
//   in_valid」(應被忽略, 搭配垃圾資料); monitor 隨機 out_ready 回壓, 依序以整數 GEMM 參考模型逐列比對。
//   隨機 test_en (ICG 強制開啟 / 實際閘控) — 兩者結果必須相同; 偶爾在運算中途 reset (丟棄在途的 tile)。
//   v6.3.16: 約一半的 tile 為浮點 (FP8 E4M3 / E5M2 混合、BF16 byte-planar, 含奇數步), 參考模型 npu_fp_ref.svh。
// plusargs: +seed=N  +tiles=N (預設 400)
//=============================================================================
`include "npu_fp_ref.svh"
module npu_mma_tb #(parameter int N = 8);
  localparam int KMAX = 80;
  logic clk = 1'b0, rst_n = 1'b0;
  logic test_en = 1'b0, start = 1'b0, a_signed = 1'b0, b_signed = 1'b0, fp = 1'b0, bf = 1'b0;
  logic in_valid = 1'b0, in_last = 1'b0, out_ready = 1'b0;
  logic [N-1:0][7:0] a_vec = '0, b_vec = '0;
  logic busy, in_idle, in_ready, out_valid;
  logic [$clog2(N)-1:0] out_row;
  logic [N-1:0][31:0] out_vec;
  always #5 clk = ~clk;

  npu_mma #(.N(N)) dut (.*);

  typedef struct { int k, mode; bit sa, sb, fp, bf; logic [7:0] A[N][KMAX]; logic [7:0] B[KMAX][N]; logic [31:0] C[N][N]; } tile_t;
  tile_t q [$];                          // 已產生、尚未輸出完的 tile (q[0] = 正在輸出的)
  int unsigned seed, ntiles, n_ok = 0, n_rst = 0, n_err = 0, n_rows = 0, n_ovl = 0, n_lstall = 0;

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic logic [7:0] pick(int mode);
    case (mode)
      1: case (ur(0, 5)) 0: return 8'h00; 1: return 8'h7F; 2: return 8'h80; 3: return 8'hFF; 4: return 8'h01; default: return 8'($urandom()); endcase
      2: return (ur(0, 7) == 0) ? 8'($urandom()) : 8'h00;
      3: return 8'hFF;
      default: return 8'($urandom());
    endcase
  endfunction

  // fp = 1: mode 0..2 = fpr_pick8 / fpr_pick16 的樣式; BF16 時 A[i][2t] / A[i][2t+1] = 元素 t 的低 / 高位元組
  function automatic tile_t gen(int k, int mode, bit sa, bit sb, bit fp = 0, bit bf = 0);
    tile_t t;
    logic [7:0] av[], bv[];
    t.k = k; t.mode = mode; t.sa = sa; t.sb = sb; t.fp = fp; t.bf = bf;            // bf 在整數 tile 也隨機 (應被忽略)
    for (int i = 0; i < N; i++) for (int kk = 0; kk < k; kk++)
      if (!fp) t.A[i][kk] = (mode == 4) ? ((i == kk % N) ? 8'h01 : 8'h00) : pick(mode);
      else if (!bf) t.A[i][kk] = fpr_pick8(sa, mode);
      else if (kk % 2 == 0) {t.A[i][kk + 1 < KMAX ? kk + 1 : kk], t.A[i][kk]} = fpr_pick16(mode);
    for (int kk = 0; kk < k; kk++) for (int j = 0; j < N; j++)
      if (!fp) t.B[kk][j] = (mode == 4) ? 8'($urandom()) : pick(mode);
      else if (!bf) t.B[kk][j] = fpr_pick8(sb, mode);
      else if (kk % 2 == 0) {t.B[kk + 1 < KMAX ? kk + 1 : kk][j], t.B[kk][j]} = fpr_pick16(mode);
    av = new[k]; bv = new[k];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      for (int kk = 0; kk < k; kk++) begin av[kk] = t.A[i][kk]; bv[kk] = t.B[kk][j]; end
      t.C[i][j] = fpr_dot(k, av, bv, sa, sb, fp, bf);
    end
    return t;
  endfunction
  function automatic tile_t rand_tile();
    int k;
    case (ur(0, 5)) 0: k = 1; 1: k = ur(2, N); 2: k = ur(N + 1, 2 * N); 3: k = ur(64, KMAX); default: k = ur(1, KMAX); endcase
    if (ur(0, 1) == 0) return gen(k, ur(0, 4), 1'($urandom()), 1'($urandom()), 0, 1'($urandom()));
    return gen(k, ur(0, 2), 1'($urandom()), 1'($urandom()), 1, 1'($urandom()));
  endfunction

  task automatic do_reset(int cyc);
    rst_n = 1'b0; start = 1'b0; in_valid = 1'b0; out_ready = 1'b0;
    repeat (cyc) @(posedge clk);
    #1 rst_n = 1'b1;
  endtask

  // 一段連續的 tile 串流 (nt 個); bub / bp / gap: 輸入空拍 / 輸出回壓 / tile 之間的間隔機率 (%);
  // rst_at > 0: 第 rst_at 拍中途 reset, 丟棄在途的 tile
  tile_t dir [$];                         // directed tile (優先使用)
  task automatic run_stream(int nt, int bub, int bp, int gap, int rst_at);
    int d_n = 0, d_ks = 0, cyc = 0, rows = 0, d_t = 0;   // d_t: driver 目前 tile 在 q 中的位置
    bit feeding = 0;
    q.delete();
    if (busy) begin $display("ERROR: busy at stream start"); n_err++; end
    while ((d_n < nt && n_ok < ntiles + 4) || q.size() != 0) begin     // 達到數量後只停止開新 tile, 在途的要收完
      bit want_start;
      // ---- 驅動 (posedge 後 #1) ----
      want_start = !feeding && d_n < nt && n_ok < ntiles + 4 && ur(0, 99) >= gap;
      if (want_start && d_t == q.size()) q.push_back((dir.size() != 0) ? dir.pop_front() : rand_tile());
      start    = want_start || (ur(0, 7) == 0);                         // 輸入端忙碌時的 start 應被忽略
      a_signed = want_start ? q[d_t].sa : 1'($urandom());
      b_signed = want_start ? q[d_t].sb : 1'($urandom());
      fp       = want_start ? q[d_t].fp : 1'($urandom());
      bf       = want_start ? q[d_t].bf : 1'($urandom());
      in_valid = (feeding && ur(0, 99) >= bub) || (ur(0, 15) == 0);
      in_last  = feeding ? (d_ks == q[d_t].k - 1) : 1'($urandom());
      for (int i = 0; i < N; i++) begin
        a_vec[i] = feeding ? q[d_t].A[i][d_ks] : 8'($urandom());
        b_vec[i] = feeding ? q[d_t].B[d_ks][i] : 8'($urandom());
      end
      out_ready = (ur(0, 99) >= bp);
      // ---- 邊沿前取樣握手 ----
      #4;
      if (start && !feeding && !want_start && in_idle) begin            // 輸入端閒置時的垃圾 start 也會開始: 不要給
        start = 1'b0; #0;
      end
      if (feeding && in_valid && in_last && !in_ready && busy && dut.ost != 0) n_lstall++;
      if (in_valid && in_ready) begin
        if (!feeding) begin $display("ERROR: in_ready accepted garbage"); n_err++; end
        else begin
          d_ks++;
          if (d_ks == q[d_t].k) begin feeding = 0; d_t++; d_n++; end
        end
      end
      if (start && in_idle) begin
        if (!want_start) begin $display("ERROR: start accepted without request"); n_err++; end
        if (out_valid || dut.ost != 0) n_ovl++;                         // 與前一個 tile 的排空 / 輸出重疊
        feeding = 1; d_ks = 0;
      end
      if (out_valid && out_ready) begin
        if (q.size() == 0 || d_t == 0) begin $display("ERROR: output without a finished tile"); n_err++; end
        else begin
          if (out_row != rows[$clog2(N)-1:0]) begin $display("ERROR: out_row %0d expected %0d", out_row, rows); n_err++; end
          for (int j = 0; j < N; j++)
            if (out_vec[j] !== q[0].C[rows][j]) begin
              if (n_err < 20) $display("MISMATCH tile k=%0d mode=%0d sa=%0d sb=%0d fp=%0d bf=%0d te=%0d C[%0d][%0d] dut=%h ref=%h",
                                       q[0].k, q[0].mode, q[0].sa, q[0].sb, q[0].fp, q[0].bf, test_en, rows, j, out_vec[j], q[0].C[rows][j]);
              n_err++;
            end
          rows++; n_rows++;
          if (rows == N) begin rows = 0; void'(q.pop_front()); d_t--; n_ok++; end
        end
      end
      @(posedge clk); #1;
      cyc++;
      if (ur(0, 199) == 0) test_en = !test_en;
      if (rst_at > 0 && cyc == rst_at) begin
        do_reset(ur(1, 3)); n_rst++; q.delete();
        if (busy) begin $display("ERROR: busy after reset"); n_err++; end
        return;
      end
      if (cyc > 40 * nt * (KMAX + 4 * N) + 2000) begin $display("ERROR: timeout"); n_err++; return; end
    end
    start = 1'b0; in_valid = 1'b0;
    repeat (2) @(posedge clk); #1;
    if (busy || !in_idle) begin $display("ERROR: busy after stream"); n_err++; end
  endtask

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("tiles=%d", ntiles)) ntiles = 400;
    void'($urandom(seed));
    do_reset(3);
    // directed: K=1 / K=N / 最大量級 (有號 -128*-128, 無號 255*255), 背靠背 (K=1 緊接 K=1: 第二個的 in_last 等前一個輸出完)
    dir.push_back(gen(1, 0, 1, 1)); dir.push_back(gen(N, 4, 0, 0)); dir.push_back(gen(KMAX, 3, 0, 0));
    dir.push_back(gen(KMAX, 1, 1, 1)); dir.push_back(gen(1, 1, 0, 1)); dir.push_back(gen(1, 1, 1, 0));
    // 浮點: BF16 K = 1 (奇數步) / 2 / 3, FP8 E4M3 × E5M2, 背靠背
    dir.push_back(gen(1, 0, 0, 0, 1, 1)); dir.push_back(gen(2, 0, 0, 0, 1, 1)); dir.push_back(gen(3, 1, 0, 0, 1, 1));
    dir.push_back(gen(1, 0, 0, 1, 1, 0)); dir.push_back(gen(KMAX, 0, 1, 0, 1, 0));
    run_stream(11, 0, 0, 0, 0);
    while (n_ok < ntiles) begin
      int nt, bub, bp, gap, ra;
      test_en = (ur(0, 3) == 0);
      nt  = ur(1, 8);
      bub = (ur(0, 2) == 0) ? 0 : ur(0, 70);
      bp  = (ur(0, 2) == 0) ? 0 : ur(0, 70);
      gap = (ur(0, 2) == 0) ? 0 : ur(0, 60);
      ra  = (ur(0, 9) == 0) ? ur(1, nt * (KMAX + 3 * N)) : 0;
      run_stream(nt, bub, bp, gap, ra);
      if (n_err > 50) break;
    end
    $display("MMA: seed=%0d tiles_ok=%0d resets=%0d rows=%0d overlapped_starts=%0d last_stalls=%0d errors=%0d",
             seed, n_ok, n_rst, n_rows, n_ovl, n_lstall, n_err);
    if (n_err == 0 && n_ovl > 0 && n_lstall > 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
