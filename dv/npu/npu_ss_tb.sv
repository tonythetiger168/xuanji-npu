// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss — CRV TB (每個產品組態各跑一次, 參數由 -G 指定)
//   每個 batch 產生一段隨機命令程式 (GEMM / DMA_LD / DMA_ST / NOP, 含 K = 0、line 數 0、位址環繞),
//   以「依序執行」的參考模型 (L2 + 外部記憶體) 求得預期結果。產生器追蹤自上一個等待旗標 (wt) 以來
//   所有命令的讀 / 寫範圍: 新命令與其衝突時必設 wt (另有隨機 wt) — 因此 DUT 的並行執行必須等同依序執行。
//   外部記憶體模型: 讀 / 寫 grant 隨機 (與 req 無關)、讀回應依序且延遲隨機 (0..L 拍後、可間斷)。
//   batch 結束 (所有完成都已收到且 busy = 0) 時比對: 整個 L2 (後門) 與外部記憶體所有被寫過的 line、
//   每個 id 恰好完成一次、wt 命令的完成必在所有先前命令之後、perf_steps = 起始值 + 所有 GEMM 的 K。
//   每個 batch 隨機: 各埠 grant / 回壓機率、test_en、效能計數器起始值; 偶爾在 batch 中途 reset
//   (之後以 DUT 狀態重新同步參考模型)。
// plusargs: +seed=N  +batches=N (預設 30)  +cmds=N (每個 batch 的命令數上限, 預設 60)  +no_big (不跑長 DMA)
//           +perf_dma=L: DMA 吞吐量量測 (外部埠全速, 讀取延遲固定 L 拍)
//=============================================================================
module npu_ss_tb #(parameter int N = 8, parameter int NCU = 4, parameter int NB = 4, parameter int DEPTH = 1024,
                    parameter int OUT = 32);
  import orca_pkg::*;
  localparam int QD = 8;                                  // 與 orca_npu_ss 預設一致
  localparam int LINES = NB * DEPTH, SPL = 64 / N, CL = (4 * N * N + 63) / 64;
  localparam int LB = $clog2(LINES);

  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready;
  npu_tile_cmd_t cmd = '0;
  logic cpl_valid, cpl_ready = 1'b0, irq, busy;
  logic [7:0] cpl_id;
  logic m_rd_req, m_rd_gnt = 1'b0, m_rd_rvalid = 1'b0, m_wr_req, m_wr_gnt = 1'b0;
  logic [31:0] m_rd_addr, m_wr_addr;
  logic [511:0] m_rd_rdata = '0, m_wr_data;
  logic perf_wr = 1'b0;
  logic [47:0] perf_wdata_busy = '0, perf_wdata_steps = '0, perf_busy, perf_steps;
  always #5 clk = ~clk;

  orca_npu_ss #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(QD), .OUT(OUT)) dut (.*);

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic bit rnd(int pct); return ur(0, 99) < pct; endfunction

  // ---------------- 外部記憶體 (DUT 側) 與參考模型 ----------------
  int unsigned salt;
  logic [511:0] emem [int unsigned];
  logic [511:0] rext [int unsigned];
  logic [511:0] rl2 [LINES];
  function automatic logic [511:0] einit(int unsigned a);
    logic [511:0] d;
    for (int w = 0; w < 16; w++) d[w * 32 +: 32] = (a * 32'd2654435761) ^ (32'(w) * 32'd40503 + salt) ^ ((a >> 7) * 32'd97);
    return d;
  endfunction
  function automatic logic [511:0] erd(int unsigned a); return emem.exists(a) ? emem[a] : einit(a); endfunction
  function automatic logic [511:0] rerd(int unsigned a); return rext.exists(a) ? rext[a] : einit(a); endfunction
  function automatic int l2i(logic [25:0] l); return int'(l) % LINES; endfunction
  function automatic logic [511:0] l2_bd(int i); return dut.u_core.u_l2.mem[i % NB][(i / NB) % DEPTH]; endfunction

  // 參考模型: 依序執行一個命令
  function automatic void ref_exec(npu_tile_cmd_t c);
    case (c.op)
      NPU_OP_DMA_LD: for (int i = 0; i < int'(c.k); i++) rl2[l2i(c.a + 26'(i))] = rerd(c.ext + 32'(i));
      NPU_OP_DMA_ST: for (int i = 0; i < int'(c.k); i++) rext[c.ext + 32'(i)] = rl2[l2i(c.a + 26'(i))];
      NPU_OP_GEMM: if (c.k != 0) begin
        logic [31:0] C [];
        C = new[N * N];
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
          longint acc = 0;
          for (int kk = 0; kk < int'(c.k); kk++) begin
            logic [7:0] av = rl2[l2i(c.a + 26'(kk / SPL))][((kk % SPL) * N + i) * 8 +: 8];
            logic [7:0] bv = rl2[l2i(c.b + 26'(kk / SPL))][((kk % SPL) * N + j) * 8 +: 8];
            acc += (c.sa ? longint'($signed(av)) : longint'(av)) * (c.sb ? longint'($signed(bv)) : longint'(bv));
          end
          C[i * N + j] = 32'(acc);
        end
        for (int l = 0; l < CL; l++) for (int e = 0; e < 16; e++) rl2[l2i(c.c + 26'(l))][e * 32 +: 32] = C[l * 16 + e];
      end
      default: ;
    endcase
  endfunction

  // ---------------- 命令產生器 (衝突時設 wt) ----------------
  bit acc_l2r [LINES], acc_l2w [LINES];
  bit acc_er [int unsigned], acc_ew [int unsigned];
  logic [25:0] hot_l2 [8];
  int unsigned hot_ext [6];
  function automatic void acc_clear();
    foreach (acc_l2r[i]) begin acc_l2r[i] = 0; acc_l2w[i] = 0; end
    acc_er.delete(); acc_ew.delete();
  endfunction
  // 命令的讀 / 寫範圍
  typedef struct { int l2r [$]; int l2w [$]; int unsigned er [$]; int unsigned ew [$]; } fp_t;
  function automatic fp_t footprint(npu_tile_cmd_t c);
    fp_t f;
    case (c.op)
      NPU_OP_DMA_LD: for (int i = 0; i < int'(c.k); i++) begin f.l2w.push_back(l2i(c.a + 26'(i))); f.er.push_back(c.ext + 32'(i)); end
      NPU_OP_DMA_ST: for (int i = 0; i < int'(c.k); i++) begin f.l2r.push_back(l2i(c.a + 26'(i))); f.ew.push_back(c.ext + 32'(i)); end
      NPU_OP_GEMM: if (c.k != 0) begin
        for (int l = 0; l < (int'(c.k) + SPL - 1) / SPL; l++) begin f.l2r.push_back(l2i(c.a + 26'(l))); f.l2r.push_back(l2i(c.b + 26'(l))); end
        for (int l = 0; l < CL; l++) f.l2w.push_back(l2i(c.c + 26'(l)));
      end
      default: ;
    endcase
    return f;
  endfunction
  function automatic bit conflicts(fp_t f);
    foreach (f.l2w[i]) if (acc_l2r[f.l2w[i]] || acc_l2w[f.l2w[i]]) return 1;
    foreach (f.l2r[i]) if (acc_l2w[f.l2r[i]]) return 1;
    foreach (f.ew[i])  if (acc_er.exists(f.ew[i]) || acc_ew.exists(f.ew[i])) return 1;
    foreach (f.er[i])  if (acc_ew.exists(f.er[i])) return 1;
    return 0;
  endfunction
  function automatic void acc_add(fp_t f);
    foreach (f.l2w[i]) acc_l2w[f.l2w[i]] = 1;
    foreach (f.l2r[i]) acc_l2r[f.l2r[i]] = 1;
    foreach (f.ew[i])  acc_ew[f.ew[i]] = 1;
    foreach (f.er[i])  acc_er[f.er[i]] = 1;
  endfunction
  int p_hot = 50;
  function automatic logic [25:0] pick_l2(int span);
    logic [25:0] l = rnd(p_hot) ? hot_l2[ur(0, 7)] + 26'(ur(0, 3)) : 26'(ur(0, LINES - 1));
    if (rnd(8)) l = 26'(LINES - ur(1, span + 1));        // 跨越 L2 容量折返
    l[25:LB] = (26 - LB)'($urandom());                   // 高位元 (xbar 只解碼低 LB bit)
    if (rnd(3)) l = 26'h3FF_FFFF - 26'(ur(0, span));     // 跨越 26 bit line 位址環繞
    return l;
  endfunction
  function automatic int unsigned pick_ext();
    case (ur(0, 9))
      0:       return 32'hFFFF_FFFF - 32'(ur(0, 20));   // 外部位址環繞
      1, 2, 3: return $urandom();
      default: return hot_ext[ur(0, 5)] + 32'(ur(0, 4));
    endcase
  endfunction
  function automatic logic [15:0] pick_n();
    case (ur(0, 9))
      0:       return 16'd0;
      1:       return 16'd1;
      2, 3, 4: return 16'(ur(2, OUT));
      5, 6, 7: return 16'(ur(OUT + 1, 4 * OUT));
      default: return 16'(ur(4 * OUT + 1, 6 * OUT));
    endcase
  endfunction
  function automatic logic [15:0] pick_k();
    case (ur(0, 11))
      0:       return 16'd0;
      1:       return 16'd1;
      2:       return 16'(SPL);
      3, 4, 5: return 16'(ur(2, 3 * SPL));
      6, 7, 8: return 16'(ur(1, 64));
      default: return 16'(ur(65, 260));
    endcase
  endfunction
  function automatic bit disjoint(int a0, int an, int b0, int bn);   // L2 範圍 (模 LINES) 不重疊
    for (int i = 0; i < an; i++) for (int j = 0; j < bn; j++) if ((a0 + i) % LINES == (b0 + j) % LINES) return 0;
    return 1;
  endfunction
  // mode: 0 = 混合; 1 = GEMM / NOP 突發 (填滿 npu_core 佇列, 各來源的完成同時待回報)
  function automatic npu_tile_cmd_t gen_cmd(logic [7:0] id, int p_wt, int mode);
    npu_tile_cmd_t c;
    fp_t f;
    c = '0; c.id = id; c.ext = $urandom(); c.a = 26'($urandom()); c.b = 26'($urandom()); c.c = 26'($urandom());
    c.k = 16'($urandom()); c.sa = 1'($urandom()); c.sb = 1'($urandom());
    case (mode == 1 ? (rnd(65) ? 0 : rnd(70) ? 19 : 7) : ur(0, 19))
      0, 1, 2, 3, 4, 5, 6: begin
        int al;
        c.op = NPU_OP_GEMM; c.k = pick_k(); al = (int'(c.k) + SPL - 1) / SPL;
        c.a = pick_l2(al); c.b = rnd(10) ? c.a : pick_l2(al);
        do c.c = pick_l2(CL);
        while (c.k != 0 && !(disjoint(l2i(c.c), CL, l2i(c.a), al) && disjoint(l2i(c.c), CL, l2i(c.b), al)));
      end
      7, 8, 9, 10, 11, 12: begin c.op = NPU_OP_DMA_LD; c.k = (mode == 1) ? 16'(ur(0, 3)) : pick_n(); c.a = pick_l2(int'(c.k)); c.ext = pick_ext(); end
      13, 14, 15, 16, 17:  begin c.op = NPU_OP_DMA_ST; c.k = pick_n(); c.a = pick_l2(int'(c.k)); c.ext = pick_ext(); end
      default: c.op = NPU_OP_NOP;
    endcase
    f = footprint(c);
    c.wt = rnd(p_wt);
    if (conflicts(f)) c.wt = 1'b1;
    if (c.wt) acc_clear();
    acc_add(f);
    return c;
  endfunction

  // 長 DMA (line 數接近 2^16, L2 範圍折返多次): 涵蓋 DMA 計數器的所有位元
  function automatic npu_tile_cmd_t big_dma(logic [7:0] id, bit st);
    npu_tile_cmd_t c = '0;
    c.id = id; c.op = st ? NPU_OP_DMA_ST : NPU_OP_DMA_LD; c.wt = 1'b1;
    c.k = st ? 16'(ur(40000, 65534)) : 16'hFFFF; c.a = 26'($urandom()); c.ext = $urandom();
    acc_clear(); acc_add(footprint(c));
    return c;
  endfunction

  // ---------------- 驅動 / 監看 (posedge 取樣, 非阻塞驅動) ----------------
  npu_tile_cmd_t prog [$];
  int pidx = 0, ncpl = 0, n_err = 0, n_rst = 0, n_batch = 0, n_cmds = 0;
  int cpl_pos [int];                     // id -> 完成順序
  int prog_pos [int];                    // id -> 程式中的位置
  int p_cmd = 80, p_cpl = 80, p_rgnt = 80, p_wgnt = 80, p_rv = 80, rlat = 4;
  typedef struct { logic [511:0] d; longint t; } rsp_t;
  rsp_t rq [$];
  longint cyc = 0, last_t = 0;
  int fix_lat = -1;                      // +perf_dma: 固定讀取延遲
  longint cpl_cyc [int];                 // id -> 完成拍
  bit in_batch = 0;

  always @(posedge clk) begin
    cyc++;
    if (!rst_n) begin
      cmd_valid <= 1'b0; cpl_ready <= 1'b0; m_rd_gnt <= 1'b0; m_wr_gnt <= 1'b0; m_rd_rvalid <= 1'b0;
      rq.delete();
    end else begin
      // 外部讀: grant → 依序回應 (延遲 0..rlat, 回應之間可間斷)
      if (m_rd_req && m_rd_gnt) begin
        automatic longint t = cyc + longint'(fix_lat >= 0 ? fix_lat : ur(0, rlat));
        if (t <= last_t) t = last_t + 1;
        last_t = t;
        rq.push_back('{erd(m_rd_addr), t});
      end
      if (rq.size() != 0 && rq[0].t <= cyc && rnd(p_rv)) begin m_rd_rvalid <= 1'b1; m_rd_rdata <= rq.pop_front().d; end
      else begin m_rd_rvalid <= 1'b0; m_rd_rdata <= {16{$urandom()}}; end
      m_rd_gnt <= rnd(p_rgnt);
      // 外部寫: grant 拍完成
      if (m_wr_req && m_wr_gnt) emem[m_wr_addr] = m_wr_data;
      m_wr_gnt <= rnd(p_wgnt);
      // 命令
      if (cmd_valid && cmd_ready) pidx++;
      if (cmd_valid && !cmd_ready) ;                       // 保持
      else if (in_batch && pidx < prog.size() && rnd(p_cmd)) begin cmd_valid <= 1'b1; cmd <= prog[pidx]; end
      else begin
        automatic npu_tile_cmd_t j = '0;
        j.id = 8'($urandom()); j.op = npu_op_e'($urandom()); j.k = 16'($urandom()); j.a = 26'($urandom());
        cmd_valid <= 1'b0; cmd <= j;
      end
      // 完成
      if (cpl_valid && cpl_ready) begin
        if (!prog_pos.exists(int'(cpl_id)) || cpl_pos.exists(int'(cpl_id)) || prog_pos[int'(cpl_id)] >= pidx) begin
          if (n_err < 20) $display("ERROR: completion for unknown / duplicate / unissued id %0d", cpl_id);
          n_err++;
        end else begin cpl_pos[int'(cpl_id)] = ncpl; cpl_cyc[int'(cpl_id)] = cyc; end
        ncpl++;
      end
      cpl_ready <= rnd(p_cpl);
      if (irq !== cpl_valid) begin $display("ERROR: irq != cpl_valid"); n_err++; end
    end
  end

  // watchdog
  longint last_prog = 0;
  int ncpl_q = 0, pidx_q = 0;
  always @(posedge clk) begin
    if (ncpl != ncpl_q || pidx != pidx_q || !in_batch || (m_rd_req && m_rd_gnt) || (m_wr_req && m_wr_gnt)) begin
      ncpl_q = ncpl; pidx_q = pidx; last_prog = cyc;
    end
    else if (cyc - last_prog > 200_000) begin
      $display("ERROR: watchdog - no progress for 200k cycles (issued %0d / %0d, completed %0d)", pidx, prog.size(), ncpl);
      $display("TB FAIL"); $finish;
    end
  end

  // ---------------- batch ----------------
  task automatic resync();                // reset 後: 參考模型改用 DUT 的實際狀態
    for (int i = 0; i < LINES; i++) rl2[i] = l2_bd(i);
    rext = emem;
  endtask

  task automatic check_state();
    int bad = 0;
    for (int i = 0; i < LINES; i++)
      if (l2_bd(i) !== rl2[i]) begin
        if (bad < 5) $display("MISMATCH L2 line %0d (batch %0d)\n  dut=%h\n  ref=%h", i, n_batch, l2_bd(i), rl2[i]);
        bad++;
      end
    foreach (rext[a]) if (erd(a) !== rext[a]) begin
      if (bad < 10) $display("MISMATCH ext line %h (batch %0d)", a, n_batch);
      bad++;
    end
    foreach (emem[a]) if (!rext.exists(a) && emem[a] !== einit(a)) begin
      if (bad < 10) $display("MISMATCH ext line %h written by DUT only (batch %0d)", a, n_batch);
      bad++;
    end
    n_err += bad;
  endtask

  task automatic batch(int ncmd, bit big = 0);
    logic [7:0] ids [256];
    longint ksum = 0;
    longint unsigned ps0;
    int rst_at, p_wt, mode;
    n_batch++;
    // 各埠壓力
    p_cmd  = (ur(0, 3) == 0) ? 100 : ur(20, 95);
    p_cpl  = (ur(0, 3) == 0) ? ur(3, 15) : ur(50, 100);
    p_rgnt = (ur(0, 3) == 0) ? ur(5, 30) : ur(60, 100);
    p_wgnt = (ur(0, 3) == 0) ? ur(5, 30) : ur(60, 100);
    p_rv   = ur(30, 100);
    rlat   = (ur(0, 2) == 0) ? 0 : (ur(0, 1) == 0) ? ur(1, 12) : ur(OUT, 3 * OUT);   // 長延遲: 在途額度用盡
    p_wt   = ur(0, 3) * 10;
    mode   = (ur(0, 3) == 0) ? 1 : 0;
    p_hot  = (mode == 1) ? 0 : 50;
    if (mode == 1) p_wt = 0;
    test_en = rnd(15);
    if (big) begin p_rgnt = 100; p_wgnt = 100; p_rv = 100; end   // 長 DMA: 外部埠全速
    for (int i = 0; i < 8; i++) hot_l2[i] = 26'(ur(0, LINES - 1));
    for (int i = 0; i < 6; i++) hot_ext[i] = $urandom();
    // 效能計數器起始值 (含接近 2^48 的值)
    @(posedge clk);
    perf_wr <= 1'b1;
    perf_wdata_busy <= {16'($urandom()), 32'($urandom())};
    perf_wdata_steps <= rnd(30) ? 48'hFFFF_FFFF_FFFF - 48'(ur(0, 3000)) : {16'($urandom()), 32'($urandom())};
    @(posedge clk);
    perf_wr <= 1'b0;
    @(posedge clk);
    ps0 = perf_steps;
    // 程式 (id 在 batch 內唯一)
    foreach (ids[i]) ids[i] = 8'(i);
    ids.shuffle();
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    acc_clear();
    for (int i = 0; i < ncmd; i++) begin
      automatic npu_tile_cmd_t c = (big && i == 1) ? big_dma(ids[i], 0) : (big && i == 2) ? big_dma(ids[i], 1) :
                                  gen_cmd(ids[i], p_wt, mode);
      prog.push_back(c); prog_pos[int'(c.id)] = i;
      ref_exec(c);
      if (c.op == NPU_OP_GEMM) ksum += longint'(c.k);
    end
    rst_at = (ur(0, 9) == 0 && !big) ? ur(10, 3000) : -1;
    in_batch = 1;
    for (int w = 0; ncpl != ncmd && !(rst_at >= 0 && w >= rst_at); w++) @(posedge clk);
    in_batch = 0;
    if (ncpl != ncmd) begin                // 中途 reset: 丟棄在途命令, 參考模型改用 DUT 狀態
      @(posedge clk); rst_n <= 1'b0;
      repeat (ur(1, 4)) @(posedge clk);
      rst_n <= 1'b1;
      @(posedge clk);
      resync();
      n_rst++;
      return;
    end
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);        // 所有完成都已收到: busy 須在數拍內落下
    repeat (3) @(posedge clk);
    if (busy) begin $display("ERROR: busy after all completions"); $display("TB FAIL"); $finish; end
    check_state();
    // wt 命令的完成必在所有先前命令之後
    foreach (prog[i]) if (prog[i].wt)
      for (int j = 0; j < i; j++)
        if (cpl_pos[int'(prog[j].id)] > cpl_pos[int'(prog[i].id)]) begin
          if (n_err < 20) $display("ERROR: wt command #%0d (id %0d) completed before earlier command #%0d", i, prog[i].id, j);
          n_err++;
        end
    if (longint'(perf_steps) != longint'(48'(ps0 + longint'(ksum)))) begin
      $display("ERROR: perf_steps %0d != %0d + %0d", perf_steps, ps0, ksum); n_err++;
    end
    n_cmds += ncmd;
  endtask

  // +perf_dma: 外部埠全速、固定讀取延遲 L 時的 DMA 吞吐量 (line/拍); LD 2048 line 後 (wt) ST 2048 line
  task automatic perf_dma();
    npu_tile_cmd_t c;
    longint t0, t_ld, t_st;
    p_cmd = 100; p_cpl = 100; p_rgnt = 100; p_wgnt = 100; p_rv = 100; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); cpl_cyc.delete(); pidx = 0; ncpl = 0;
    c = '0; c.op = NPU_OP_DMA_LD; c.id = 8'd1; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h1000_0000;
    prog.push_back(c); prog_pos[1] = 0; ref_exec(c);
    c = '0; c.op = NPU_OP_DMA_ST; c.id = 8'd2; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h2000_0000; c.wt = 1'b1;
    prog.push_back(c); prog_pos[2] = 1; ref_exec(c);
    @(posedge clk); t0 = cyc; in_batch = 1;
    while (ncpl != 2) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    t_ld = cpl_cyc[1] - t0; t_st = cpl_cyc[2] - cpl_cyc[1];
    $display("PERFDMA: OUT=%0d read_latency=%0d LD %0d lines %0d cycles (%.3f line/cycle)  ST %0d lines %0d cycles (%.3f line/cycle)",
             OUT, fix_lat, 2048, t_ld, 2048.0 / real'(t_ld), 2048, t_st, 2048.0 / real'(t_st));
  endtask

  // directed: 完成輸出回壓期間填滿所有槽 (npu_core 佇列 + 每個 CU + 其輸出暫存 + tile 輸出暫存, 再加 DMA 與 NOP),
  //           oc 達到上限 CQD + NCU + 3 (界限是緊的); 之後放開回壓並照常比對
  task automatic fill_all();
    localparam int K = 200, AL = (K + SPL - 1) / SPL, STRIDE = 2 * AL + CL;
    localparam int NG = 4 + NCU + 2;     // npu_core 最多 4 + NCU + 1 個, 另一個已在 tile 輸出暫存
    npu_tile_cmd_t c;
    longint unsigned ps0 = perf_steps;
    n_batch++;
    p_cmd = 100; p_cpl = 0; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    for (int i = 0; i < NG + 2; i++) begin
      c = '0; c.id = 8'(i);
      if (i < NG) begin
        c.op = NPU_OP_GEMM; c.k = 16'(K); c.sa = 1'($urandom()); c.sb = 1'($urandom());
        c.a = 26'(i * STRIDE); c.b = c.a + 26'(AL); c.c = c.b + 26'(AL);
      end else if (i == NG) begin
        c.op = NPU_OP_DMA_LD; c.k = 16'd2; c.a = 26'(NG * STRIDE); c.ext = $urandom();
      end else c.op = NPU_OP_NOP;
      prog.push_back(c); prog_pos[i] = i; ref_exec(c);
    end
    in_batch = 1;
    // 等到所有槽都滿, 且 DMA / NOP 的完成都在待回報 (放開回壓後三個來源同時競爭)
    for (int w = 0; w < 20000 && !(int'(dut.oc) == 4 + NCU + 3 && dut.dma_cv && dut.nop_cv); w++) @(posedge clk);
    if (int'(dut.oc) != 4 + NCU + 3) begin $display("ERROR: fill_all did not reach oc = %0d (oc = %0d)", 4 + NCU + 3, dut.oc); n_err++; end
    p_cpl = 80;
    while (ncpl != NG + 2) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    if (longint'(perf_steps) != longint'(48'(ps0 + longint'(NG * K)))) begin $display("ERROR: perf_steps (fill_all) %0d != %0d + %0d", perf_steps, ps0, NG * K); n_err++; end
    n_cmds += NG + 2;
  endtask

  int unsigned seed, nb, nc;
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("batches=%d", nb)) nb = 30;
    if (!$value$plusargs("cmds=%d", nc)) nc = 60;
    void'($urandom(seed));
    salt = $urandom();
    for (int b = 0; b < NB; b++) for (int d = 0; d < DEPTH; d++) dut.u_core.u_l2.mem[b][d] = {16{$urandom()}};
    for (int i = 0; i < LINES; i++) rl2[i] = l2_bd(i);
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("perf_dma=%d", fix_lat)) perf_dma();
    else for (int b = 0; b < int'(nb); b++) if (b == 3) fill_all(); else batch((b % 5 == 0) ? 1 + ur(0, 3) : ur(nc / 2, nc), b == 1 && !$test$plusargs("no_big"));
    $display("NPUSS: N=%0d NCU=%0d NB=%0d seed=%0d batches=%0d cmds=%0d resets=%0d errors=%0d",
             N, NCU, NB, seed, n_batch, n_cmds, n_rst, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
