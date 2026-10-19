// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_core — 子系統 CRV TB (每個產品組態各跑一次, 參數由 -G 指定)
//   producer: 經 host 寫埠把 A / B 寫進一個空閒 slot (隨機起點, 跨 bank), 再送出命令 (隨機 id);
//   consumer: 隨機 cpl_ready 回壓; 收到完成後經 host 讀埠讀回 C, 與整數 GEMM 參考模型比對, 釋放 slot。
//   兩者並行 → host 讀寫與各 CU 的 L2 存取互相競爭 bank。
//   檢查: 每個 id 恰好完成一次、C 正確、CU 只寫入執行中命令的 C 區、perf_steps = 所有命令 K 的總和。
//   隨機 test_en; 偶爾在 batch 中途 reset (丟棄在途命令後重新開始)。
//   burst: 先準備數個命令的資料再連續送出 (填滿命令佇列; host 寫入與 CU 讀取重疊)。
//   +long: 單一長命令 (K >= 32768, A / B 位址超出 L2 容量而折返), 參考值由 L2 行為模型後門讀取計算。
//   效能計數器寫入: 每個 batch 前隨機寫入 (含接近 2^48 的值, 驗證進位與環繞), 檢查之後的增量。
//   +perf: 效能量測 — 預先載入 2×NCU 個長命令 (K 取 slot 能容納的最大值) 後一次送出,
//          量測 MAC 利用率 = perf_steps / (perf_busy × NCU), 須 >= +min_util (百分比, 預設 0)。
// plusargs: +seed=N  +cmds=N (預設 200)  +perf [+perf_k=K]  +min_util=P  +long
//=============================================================================
module npu_core_tb #(parameter int N = 8, parameter int NCU = 4, parameter int NB = 4, parameter int DEPTH = 1024);
  localparam int QD = 4;                                  // 與 npu_core 預設一致
  localparam int SPL = 64 / N, CL = (4 * N * N + 63) / 64;
  localparam int LINES = NB * DEPTH, NSLOT = 16, SLOT = LINES / NSLOT;   // slot 數 > NCU + QD: 全部 CU 同時忙碌且佇列滿
  localparam int KCAP = ((SLOT - CL - 2) / 2) * SPL;     // slot 能容納的最大 K
  localparam int KMAX = (KCAP > 2000) ? 2000 : KCAP;
  localparam int LB = $clog2(LINES);

  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready, cmd_a_signed = 1'b0, cmd_b_signed = 1'b0;
  logic [7:0] cmd_id = '0;
  logic [25:0] cmd_a_line = '0, cmd_b_line = '0, cmd_c_line = '0;
  logic [15:0] cmd_k = 16'd1;
  logic cpl_valid, cpl_ready = 1'b0;
  logic [7:0] cpl_id;
  logic h_rd_req = 1'b0, h_rd_gnt, h_wr_req = 1'b0, h_wr_gnt;
  logic [25:0] h_rd_line = '0, h_wr_line = '0;
  logic [511:0] h_rd_data, h_wr_data = '0;
  logic busy, perf_wr = 1'b0;
  logic [47:0] perf_busy, perf_steps, perf_wdata_busy = '0, perf_wdata_steps = '0;
  longint unsigned pb0 = 0, ps0 = 0;    // 計數器寫入的基準值
  always #5 clk = ~clk;

  npu_core #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH)) dut (.*);

  // ---------------- 命令記錄 ----------------
  typedef struct { int k; bit sa, sb; int slot, ns; logic [25:0] a, b, c; logic [31:0] C []; } rec_t;
  rec_t inflight [int];                 // id -> 記錄
  bit   pend_ids [int];                 // 已準備、尚未送出的 id
  bit   slot_busy [NSLOT];
  int unsigned seed, ncmds, n_ok = 0, n_err = 0, n_rst = 0, n_issued = 0;
  longint k_sum = 0;                    // 本次 reset 以來所有已完成命令的 K 總和 (= perf_steps)
  bit rst_fired;
  int min_util = 0;
  int cpl_bp = 0;                       // cpl_ready 回壓機率 (%)

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic logic [7:0] pick(int mode);
    case (mode)
      1: case (ur(0, 4)) 0: return 8'h00; 1: return 8'h7F; 2: return 8'h80; 3: return 8'hFF; default: return 8'($urandom()); endcase
      2: return (ur(0, 5) == 0) ? 8'($urandom()) : 8'h00;
      default: return 8'($urandom());
    endcase
  endfunction

  // 統計 (報告用): 最大佇列深度、host 與 CU 同時存取 L2 的拍數
  int max_q = 0, n_ovl = 0;
  always @(posedge clk) if (rst_n) begin
    if (int'(dut.q_cnt) > max_q) max_q = int'(dut.q_cnt);
    if ((h_rd_req || h_wr_req) && dut.rd_req[NCU-1:0] != '0) n_ovl++;
  end

  // watchdog: 長時間沒有任何命令送出或完成 → 視為卡死 (例如 id 錯誤導致 slot 永不釋放, 系統閒置但 TB 永遠等待)
  longint last_prog = 0;
  int n_ok_q = 0, n_iss_q = 0;
  always @(posedge clk) begin
    if (n_ok != n_ok_q || n_issued != n_iss_q) begin n_ok_q = n_ok; n_iss_q = n_issued; last_prog = $time; end
    else if ($time - last_prog > 64'd20_000_000) begin
      $display("ERROR: watchdog - no completion for 2M cycles"); $display("TB FAIL"); $finish;
    end
  end

  // CU 只能寫入執行中命令的 C 區
  always @(posedge clk) if (rst_n)
    for (int m = 0; m < NCU; m++) if (dut.wr_gnt[m]) begin
      automatic bit ok = 0;
      foreach (inflight[id]) if (dut.wr_line[m] >= inflight[id].c && dut.wr_line[m] < inflight[id].c + 26'(CL)) ok = 1;
      if (!ok) begin $display("ERROR: CU%0d write outside any in-flight C region line=%h", m, dut.wr_line[m]); n_err++; end
    end

  task automatic host_write(logic [25:0] line, logic [511:0] d);
    h_wr_req = 1'b1; h_wr_line = line; h_wr_data = d;
    do begin #4; if (h_wr_gnt) break; @(posedge clk); #1; end while (1);
    @(posedge clk); #1 h_wr_req = 1'b0; h_wr_line = 26'($urandom()); h_wr_data = {16{$urandom()}};
  endtask
  task automatic host_read(logic [25:0] line, output logic [511:0] d);
    h_rd_req = 1'b1; h_rd_line = line;
    do begin #4; if (h_rd_gnt) break; @(posedge clk); #1; end while (1);
    d = h_rd_data;
    @(posedge clk); #1 h_rd_req = 1'b0; h_rd_line = 26'($urandom());
  endtask

  // 產生一個命令: 資料寫入 slot, 回傳記錄 (尚未送出)
  task automatic prepare(int k, output rec_t r, output logic [7:0] id, input int ns = 1);   // ns: 佔用連續 ns 個 slot
    int s, al = (k + SPL - 1) / SPL, mode = ur(0, 2), off;
    logic [7:0] A [][], B [][];
    logic [511:0] ln;
    do begin s = ns * ur(0, NSLOT / ns - 1); if (slot_busy[s]) @(posedge clk); end while (slot_busy[s]);
    for (int i = 0; i < ns; i++) slot_busy[s + i] = 1;
    r.ns = ns;
    off = ur(0, ns * SLOT - 2 * al - CL);                // slot 內隨機起點: A / B / C 落在不同 bank
    r.k = k; r.sa = 1'($urandom()); r.sb = 1'($urandom()); r.slot = s;
    // 低位: slot 內位置; 高位 (xbar 只解碼低 LB bit) 各自隨機 — 涵蓋 CU / xbar 位址的所有位元
    r.a = 26'(s * SLOT + off); r.b = r.a + 26'(al); r.c = r.b + 26'(al);
    r.a[25:LB] = (25 - LB + 1)'($urandom()); r.b[25:LB] = (25 - LB + 1)'($urandom()); r.c[25:LB] = (25 - LB + 1)'($urandom());
    A = new[N]; B = new[k];
    foreach (A[i]) begin A[i] = new[k]; foreach (A[i][kk]) A[i][kk] = pick(mode); end
    foreach (B[kk]) begin B[kk] = new[N]; foreach (B[kk][j]) B[kk][j] = pick(mode); end
    for (int l = 0; l < al; l++) begin                    // A line l: k = l*SPL .. ; 尾端填垃圾
      for (int t = 0; t < SPL; t++) for (int i = 0; i < N; i++)
        ln[(t * N + i) * 8 +: 8] = (l * SPL + t < k) ? A[i][l * SPL + t] : 8'($urandom());
      host_write(r.a + 26'(l), ln);
      if (l == 0 && ur(0, 3) == 0) repeat (ur(4, 40)) host_write(r.a, ln);   // bank hammer (寫): 重複寫同一 line
    end
    for (int l = 0; l < al; l++) begin
      for (int t = 0; t < SPL; t++) for (int j = 0; j < N; j++)
        ln[(t * N + j) * 8 +: 8] = (l * SPL + t < k) ? B[l * SPL + t][j] : 8'($urandom());
      host_write(r.b + 26'(l), ln);
    end
    r.C = new[N * N];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      longint acc = 0;
      for (int kk = 0; kk < k; kk++)
        acc += (r.sa ? longint'($signed(A[i][kk])) : longint'(A[i][kk])) * (r.sb ? longint'($signed(B[kk][j])) : longint'(B[kk][j]));
      r.C[i * N + j] = 32'(acc);
    end
    do id = 8'($urandom()); while (inflight.exists(id) || pend_ids.exists(id));   // 在途命令之間 id 唯一
    pend_ids[id] = 1;
  endtask

  task automatic issue(rec_t r, logic [7:0] id);
    inflight[id] = r; pend_ids.delete(id);
    cmd_valid = 1'b1; cmd_id = id; cmd_a_line = r.a; cmd_b_line = r.b; cmd_c_line = r.c; cmd_k = 16'(r.k);
    cmd_a_signed = r.sa; cmd_b_signed = r.sb;
    do begin #4; if (cmd_ready) break; @(posedge clk); #1; end while (1);
    @(posedge clk); #1;
    cmd_valid = 1'b0; cmd_id = 8'($urandom()); cmd_a_line = 26'($urandom()); cmd_k = 16'($urandom());
    n_issued++;
  endtask

  task automatic check_cpl(logic [7:0] id);
    logic [511:0] ln;
    rec_t r;
    if (!inflight.exists(id)) begin $display("ERROR: completion for unknown / duplicate id %0d", id); n_err++; return; end
    r = inflight[id];
    // bank hammer: 偶爾連續多拍讀同一個 line (同一 bank), 檢查其他主端不會被餓死 (有界等待 / round-robin)
    if (ur(0, 3) == 0) repeat (ur(4, 40)) host_read(r.c, ln);
    for (int l = 0; l < CL; l++) begin
      host_read(r.c + 26'(l), ln);
      for (int e = 0; e < 16 && l * 16 + e < N * N; e++)
        if (ln[e * 32 +: 32] !== r.C[l * 16 + e]) begin
          if (n_err < 20) $display("MISMATCH id=%0d k=%0d C[%0d] dut=%h ref=%h", id, r.k, l * 16 + e, ln[e * 32 +: 32], r.C[l * 16 + e]);
          n_err++;
        end
    end
    k_sum += r.k;
    if (r.slot >= 0) for (int i = 0; i < r.ns; i++) slot_busy[r.slot + i] = 0;
    inflight.delete(id);
    n_ok++;
  endtask

  int unsigned n_batch = 0;
  task automatic producer(int n);
    int c = 0;
    process::self().srandom(seed * 7919 + n_batch * 31 + 1);   // fork 出的 process 有各自的 RNG: 明確以 seed 播種
    while (c < n) begin
      // burst: 一次準備 1..(QD+NCU+2) 個命令再連續送出 (受 slot 數限制)
      int nb; rec_t r [$]; logic [7:0] ids [$];
      nb = ur(1, QD + NCU + 2); r.delete(); ids.delete();
      if (nb > n - c) nb = n - c;
      if (nb > NSLOT - 2) nb = NSLOT - 2;
      for (int b = 0; b < nb; b++) begin
        rec_t x; logic [7:0] id;
        int k = (ur(0, 5) == 0) ? 1 : (ur(0, 4) == 0) ? ur(KMAX / 2, KMAX) : ur(1, 4 * SPL);
        prepare(k, x, id); r.push_back(x); ids.push_back(id);
      end
      foreach (r[b]) issue(r[b], ids[b]);
      c += nb;
      repeat (ur(0, 3)) @(posedge clk);
    end
  endtask
  task automatic consumer(int n);
    int got = 0;
    process::self().srandom(seed * 7919 + n_batch * 31 + 2);
    while (got < n) begin
      if (ur(0, 99) == 0) begin               // 偶爾長時間不收完成 (多個 / 全部 CU 的完成同時待回報)
        cpl_ready = 1'b0; repeat (ur(50, 2000)) @(posedge clk); #1;
      end
      cpl_ready = (ur(0, 99) >= cpl_bp);
      #4;
      if (cpl_valid && cpl_ready) begin
        automatic logic [7:0] id = cpl_id;
        @(posedge clk); #1 cpl_ready = 1'b0;
        check_cpl(id); got++;
      end else begin
        @(posedge clk); #1;
      end
    end
    cpl_ready = 1'b0;
  endtask

  task automatic do_reset();
    rst_n = 1'b0; cmd_valid = 1'b0; cpl_ready = 1'b0; h_rd_req = 1'b0; h_wr_req = 1'b0;
    repeat (ur(1, 3)) @(posedge clk);
    #1 rst_n = 1'b1;
    inflight.delete(); pend_ids.delete(); foreach (slot_busy[s]) slot_busy[s] = 0; k_sum = 0; pb0 = 0; ps0 = 0;
    if (busy || perf_steps != 0) begin $display("ERROR: state not cleared by reset"); n_err++; end
  endtask

  // abort burst: 送出 QD+NCU 個隨機長命令 (K 為隨機 16-bit, 位址隨機; 不比對結果) 填滿佇列後 reset —
  // 檢查佇列 / CU 執行中被 reset 能完全清除, 也讓命令欄位的所有位元經過佇列
  task automatic abort_burst();
    for (int c = 0; c < QD + NCU; c++) begin
      rec_t r; logic [7:0] id;
      r.k = ur(256, 65535); r.sa = 1'($urandom()); r.sb = 1'($urandom()); r.slot = -1; r.ns = 0;
      r.a = 26'($urandom()); r.b = 26'($urandom()); r.c = 26'($urandom());
      do id = 8'($urandom()); while (inflight.exists(id));
      issue(r, id);                       // inflight 記錄讓「只寫 C 區」檢查接受其寫入
    end
    repeat (ur(0, 20)) @(posedge clk);
    #1 do_reset(); n_rst++;
  endtask

  // round-robin 折返 (每個槽 j = 1..NCU-1): CU 0..j-1 執行長命令, 短命令連續落在 CU j 並各自完成 —
  // 完成輸出在「上一次選中者就是同一個 CU」時必須折返選到它
  task automatic directed_rr();
    for (int j = 1; j < NCU; j++)
      fork
        begin
          rec_t xs [$]; logic [7:0] is [$];
          for (int c = 0; c < j; c++) begin rec_t x; logic [7:0] id; prepare(KMAX, x, id); xs.push_back(x); is.push_back(id); end
          foreach (xs[c]) issue(xs[c], is[c]);  // 先備妥資料再連續送出: CU 0..j-1 同時忙碌
          for (int c = 0; c < 3; c++) begin
            rec_t x; logic [7:0] id; int got0 = n_ok;
            prepare(1, x, id); issue(x, id);
            wait (n_ok > got0);                  // 前一個短命令完成後才送下一個
          end
        end
        begin cpl_bp = 0; consumer(j + 3); end
      join
  endtask

  task automatic perf_write();
    // 隨機寫入效能計數器 (含接近 2^48 的值: 驗證進位鏈與環繞)
    perf_wr = 1'b1;
    case (ur(0, 3))
      0: begin perf_wdata_busy = 48'hFFFF_FFFF_FFFF - 48'(ur(0, 50)); perf_wdata_steps = 48'hFFFF_FFFF_FFFF - 48'(ur(0, 50)); end
      1: begin perf_wdata_busy = '0; perf_wdata_steps = '0; end    // 清除
      default: begin perf_wdata_busy = {16'($urandom()), 32'($urandom())}; perf_wdata_steps = {16'($urandom()), 32'($urandom())}; end
    endcase
    @(posedge clk); #1 perf_wr = 1'b0;
    if (perf_busy !== perf_wdata_busy || perf_steps !== perf_wdata_steps) begin $display("ERROR: perf counter write"); n_err++; end
    pb0 = perf_busy; ps0 = perf_steps; k_sum = 0;
  endtask

  task automatic batch(int n, int rst_at);
    rst_fired = 0; n_batch++;
    if (ur(0, 1)) perf_write();
    fork
      begin fork producer(n); consumer(n); join end
      begin if (rst_at > 0) begin repeat (rst_at) @(posedge clk); #1 rst_fired = 1; end else wait (0); end
    join_any
    disable fork;
    if (rst_fired) begin
      do_reset(); n_rst++;
    end else begin
      // 正常結束: 全部完成, 不再忙碌; perf_steps 等於自上次 reset 起所有命令 K 的總和
      repeat (2) @(posedge clk);
      if (busy) begin $display("ERROR: busy after all completions"); n_err++; end
      if (perf_steps != 48'(ps0 + k_sum)) begin $display("ERROR: perf_steps %0d != base + sum K %0d", perf_steps, ps0 + k_sum); n_err++; end
      if (48'(perf_busy - 48'(pb0)) < 48'(k_sum / NCU)) begin $display("ERROR: perf_busy increment too small"); n_err++; end
    end
  endtask

  // 長命令: K >= 32768, A / B 從隨機 line 開始一路讀下去 (超過 L2 容量即折返); 先以後門填隨機資料並計算參考值。
  // C 區寫回發生在全部讀取之後 (輸出階段), 因此即使 C 與 A / B 的折返範圍重疊也不影響參考值。
  function automatic logic [7:0] bd8(longint unsigned line, int off);
    longint unsigned l = (line + off / 64) % LINES;
    return dut.u_l2.mem[l % NB][(l / NB) % DEPTH][(off % 64) * 8 +: 8];
  endfunction
  task automatic long_run();
    rec_t r; logic [511:0] ln;
    int k = ur(32768, 65535);
    for (int b = 0; b < NB; b++) for (int d = 0; d < DEPTH; d++) dut.u_l2.mem[b][d] = {16{$urandom()}};
    r.k = k; r.sa = 1'($urandom()); r.sb = 1'($urandom()); r.slot = 0; r.ns = 1;
    r.a = 26'($urandom()); r.b = 26'($urandom()); r.c = 26'($urandom());
    r.C = new[N * N];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      longint acc = 0;
      for (int kk = 0; kk < k; kk++) begin
        logic [7:0] av = bd8(r.a, kk * N + i), bv = bd8(r.b, kk * N + j);
        acc += (r.sa ? longint'($signed(av)) : longint'(av)) * (r.sb ? longint'($signed(bv)) : longint'(bv));
      end
      r.C[i * N + j] = 32'(acc);
    end
    slot_busy[0] = 1;
    issue(r, 8'($urandom()));
    consumer(1);
  endtask

  task automatic perf_run();
    rec_t r [$]; logic [7:0] ids [$];
    longint ks = 0, b0, s0;
    real util;
    int ns = NSLOT / NCU;                                 // 每個命令佔 NSLOT/NCU 個 slot: K 可到 4000
    int k = ((ns * SLOT - CL - 2) / 2) * SPL;
    int pk;
    if (k > 4000) k = 4000;
    if ($value$plusargs("perf_k=%d", pk) && pk < k) k = pk;   // +perf_k=K: 量測較短 K 的利用率
    for (int c = 0; c < NCU; c++) begin rec_t x; logic [7:0] id; prepare(k, x, id, ns); r.push_back(x); ids.push_back(id); ks += k; end
    b0 = perf_busy; s0 = perf_steps;
    fork
      foreach (r[c]) issue(r[c], ids[c]);
      consumer(r.size());
    join
    util = 100.0 * real'(perf_steps - s0) / (real'(perf_busy - b0) * NCU);
    $display("PERF: N=%0d NCU=%0d NB=%0d K=%0d cmds=%0d busy_cycles=%0d steps=%0d MAC/cycle=%0.1f (peak %0d) utilization=%0.1f%%",
             N, NCU, NB, k, r.size(), perf_busy - b0, perf_steps - s0,
             real'(perf_steps - s0) * N * N / real'(perf_busy - b0), N * N * NCU, util);
    if (perf_steps - s0 != ks) begin $display("ERROR: perf_steps %0d != sum K %0d", perf_steps - s0, ks); n_err++; end
    if (util < real'(min_util)) begin $display("ERROR: utilization %0.1f%% < required %0d%%", util, min_util); n_err++; end
  endtask

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cmds=%d", ncmds)) ncmds = 200;
    void'($value$plusargs("min_util=%d", min_util));
    // 在 Verilator 5.052 中, 含 fork ... disable fork 的 process 不受 $urandom(seed) 播種影響, 改用 process::srandom
    process::self().srandom(seed);
    repeat (3) @(posedge clk); #1 rst_n = 1'b1;
    if ($test$plusargs("perf")) perf_run();
    else if ($test$plusargs("long")) long_run();
    else begin
      if (NCU > 1) directed_rr();
      for (int done_cmds = 0; done_cmds < ncmds; ) begin
        int n;
        n = ur(1, 3 * NCU + 8);
        test_en = (ur(0, 3) == 0);
        cpl_bp = (ur(0, 2) == 0) ? 0 : ur(10, 80);
        if (ur(0, 7) == 0) abort_burst();
        batch(n, (ur(0, 9) == 0) ? ur(1, 400) : 0);
        done_cmds += n;
        if (n_err > 30) break;
      end
    end
    $display("NPUCORE: N=%0d NCU=%0d NB=%0d seed=%0d issued=%0d completed=%0d resets=%0d max_q=%0d host_cu_overlap=%0d errors=%0d",
             N, NCU, NB, seed, n_issued, n_ok, n_rst, max_q, n_ovl, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
