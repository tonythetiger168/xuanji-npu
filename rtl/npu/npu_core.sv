// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 子系統 (v6.3.12)
`include "orca_pkg.sv"

// npu_core: NCU 個 GEMM 計算單元 (各含 N×N npu_mma) + NB bank 的共用 L2 + 命令佇列 / 分派 / 完成佇列。
//
// 產品組態 (見 docs/npu.md; rtl/legacy/ai/npu_cfg_pkg.sv):
//   Xuanji-X1-Nano  N=4,  NCU=1, NB=2   16   MAC/拍
//   Xuanji-X1       N=8,  NCU=4, NB=4   256  MAC/拍
//   Xuanji-X1-Pro   N=16, NCU=8, NB=16  2048 MAC/拍
//
// host 介面:
//   命令: cmd_valid / cmd_ready + {id, A / B / C line 位址, K, 有號性, 累加} (語意見 npu_gemm_cu), 先進 QD 項佇列,
//         依序分派: 優先給編號最小的完全閒置 CU, 沒有時給編號最小的可接受命令 CU (輸入端閒置, 前一個命令
//         還在輸出 / 寫回: 命令重疊, v6.3.12); 不同命令可同時在不同 CU 執行, 完成順序不保證。
//         命令的 id 當作 tag 隨命令進入 CU, 完成時由 CU 帶回 (一個 CU 最多持有 3 個, 依序完成)。
//   完成: cpl_valid / cpl_ready + cpl_id (輸出暫存, 等待 ready 期間保持); 每個 CU 一個待回報槽,
//         CU 的 done 保持到槽空 (done_ready); 多個槽同時待回報時 round-robin 進入輸出暫存。
//   L2:   host 為 xbar 的最後一個主端 (載入 A / B、讀回 C); host 不得在命令執行期間存取其 A / B / C 區域。
//   效能計數: perf_busy (任一 CU 忙碌的拍數)、perf_steps (所有 CU 餵入的 k 步總和; MAC 數 = steps × N²);
//            perf_wr 時兩者載入 perf_wdata_* (該拍不計數)。
// 低功耗: 每個 CU 一個 ICG — 閒置 (且沒有分派 / 完成脈衝) 時整個 CU 停止時脈; CU 內 npu_mma 另有列級閘控。
module npu_core
  import orca_pkg::*;
#(
  parameter int N     = 8,
  parameter int NCU   = 4,
  parameter int NB    = 4,
  parameter int DEPTH = 1024,          // 每個 bank 的 line 數
  parameter int QD    = 4,             // 命令佇列深度 (2 的冪)
  parameter int LS_IDLE = 64,          // L2 bank 閒置幾拍後進入 SRAM light sleep (0 = 不睡; 見 npu_l2_xbar; 32 時 Pro 的利用率掉到 92.9%)
  parameter int LS_WAKE = 2            // light sleep 喚醒拍數
)(
  input  logic clk, rst_n,
  input  logic test_en,
  // 命令
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [7:0]  cmd_id,
  input  logic [25:0] cmd_a_line, cmd_b_line, cmd_c_line,
  input  logic [15:0] cmd_k,
  input  logic        cmd_a_signed, cmd_b_signed,
  input  logic        cmd_acc,           // v6.3.14: C += A × B (寫回前讀舊的 C 相加; 見 npu_gemm_cu)
  // 完成
  output logic        cpl_valid,
  input  logic        cpl_ready,
  output logic [7:0]  cpl_id,
  // host L2 埠
  input  logic         h_rd_req,
  input  logic [25:0]  h_rd_line,
  output logic         h_rd_gnt,
  output logic [511:0] h_rd_data,
  input  logic         h_wr_req,
  input  logic [25:0]  h_wr_line,
  input  logic [511:0] h_wr_data,
  output logic         h_wr_gnt,
  // 狀態 / 效能
  output logic         busy,
  input  logic         perf_wr,          // 寫入效能計數器 (save / restore 或清除)
  input  logic [47:0]  perf_wdata_busy, perf_wdata_steps,
  output logic [47:0]  perf_busy,
  output logic [47:0]  perf_steps
);
  localparam int RP = (N == 32) ? 2 : 1;      // 每個 CU 的 L2 讀取埠數 (與 npu_gemm_cu 相同公式)
  // xbar 主端: 讀 [0, NCU) = 各 CU 的埠 0, NCU = host, [NCU+1, NCU+1+NCU) = 各 CU 的埠 1 (RP = 2);
  //           寫 [0, NCU) = 各 CU, NCU = host
  localparam int NM = NCU * RP + 1;
  localparam int NW = NCU + 1;
  localparam int QW = $clog2(QD);

  typedef struct packed {
    logic [7:0]  id;
    logic [25:0] a, b, c;
    logic [15:0] k;
    logic        sa, sb;
    logic        acc;
  } cmd_t;

  // ---------------- 命令佇列 ----------------
  cmd_t [QD-1:0] q;
  logic [QW-1:0] q_rd, q_wr;
  logic [QW:0]   q_cnt;
  assign cmd_ready = (q_cnt != (QW+1)'(QD));
  wire q_push = cmd_valid && cmd_ready;
  wire q_head_v = (q_cnt != '0);
  cmd_t head;
  assign head = q[q_rd];

  // ---------------- 分派 ----------------
  // 隊首命令分派給編號最小的空閒 CU (取最低位: free & -free)
  logic [NCU-1:0] cu_ready, cu_busy, cu_done, cu_dack, cu_feed, cu_v, cu_free, cu_idle, cu_cand, cpl_pend, first_free;
  logic [NCU-1:0][7:0] cu_dtag, cpl_tag;
  logic           sel_v;
  // 可接受命令的 CU (cu_free) 中優先選完全閒置者 (cu_idle), 讓工作分散到各 CU 而不是全部重疊在 CU 0
  assign cu_free    = cu_ready;
  assign cu_idle    = cu_ready & ~cu_busy;
  assign cu_cand    = (cu_idle != '0) ? cu_idle : cu_free;
  assign first_free = cu_cand & (~cu_cand + 1'b1);
  assign sel_v      = (cu_free != '0);
  assign cu_dack    = ~cpl_pend;               // 待回報槽空時接受 CU 的完成
  assign cu_v       = q_head_v ? first_free : '0;
  wire q_pop = q_head_v && sel_v;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= '0; q_rd <= '0; q_wr <= '0; q_cnt <= '0;
    end else begin
      if (q_push) begin
        q[q_wr] <= '{id: cmd_id, a: cmd_a_line, b: cmd_b_line, c: cmd_c_line, k: cmd_k, sa: cmd_a_signed, sb: cmd_b_signed, acc: cmd_acc};
        q_wr <= q_wr + 1'b1;
      end
      if (q_pop) q_rd <= q_rd + 1'b1;
      q_cnt <= q_cnt + (QW+1)'(q_push) - (QW+1)'(q_pop);
    end
  end

  // ---------------- 計算單元 ----------------
  logic [NM-1:0]        rd_req, rd_gnt;
  logic [NM-1:0][25:0]  rd_line;
  logic [NM-1:0][511:0] rd_data;
  logic [NW-1:0]        wr_req, wr_gnt;
  logic [NW-1:0][25:0]  wr_line;
  logic [NW-1:0][511:0] wr_data;
  logic [NCU-1:0]       cu_en, cu_rd;             // cu_rd: CU 任一讀取埠有請求
  for (genvar i = 0; i < NCU; i++) begin : g_cu
    // CU 時脈致能: 持有命令 (含 done 等待握手) 或本拍分派 (CU 在輸入端閒置時取樣命令)
    assign cu_en[i] = cu_busy[i] || cu_v[i];
    logic gclk;
    clock_gate u_icg (.clk(clk), .en(cu_en[i] || !rst_n), .test_en(test_en), .clk_out(gclk));
    logic [RP-1:0]        c_rq, c_rg;
    logic [RP-1:0][25:0]  c_rl;
    logic [RP-1:0][511:0] c_rd;
    for (genvar p = 0; p < RP; p++) begin : g_p
      localparam int M = (p == 0) ? i : NCU + 1 + i;   // 此讀取埠對應的 xbar 主端
      assign rd_req[M] = c_rq[p]; assign rd_line[M] = c_rl[p];
      assign c_rg[p] = rd_gnt[M]; assign c_rd[p] = rd_data[M];
    end
    assign cu_rd[i] = |c_rq;
    npu_gemm_cu #(.N(N)) u_cu (
      .clk(gclk), .rst_n(rst_n), .test_en(test_en),
      .cmd_valid(cu_v[i]), .cmd_ready(cu_ready[i]),
      .cmd_a_line(head.a), .cmd_b_line(head.b), .cmd_c_line(head.c), .cmd_k(head.k),
      .cmd_a_signed(head.sa), .cmd_b_signed(head.sb), .cmd_acc(head.acc), .cmd_tag(head.id),
      .done(cu_done[i]), .done_ready(cu_dack[i]), .done_tag(cu_dtag[i]), .busy(cu_busy[i]), .feeding(cu_feed[i]),
      .rd_req(c_rq), .rd_line(c_rl), .rd_gnt(c_rg), .rd_data(c_rd),
      .wr_req(wr_req[i]), .wr_line(wr_line[i]), .wr_data(wr_data[i]), .wr_gnt(wr_gnt[i]));
  end

  // host = 主端 NCU
  assign rd_req[NCU] = h_rd_req;  assign rd_line[NCU] = h_rd_line;
  assign h_rd_gnt = rd_gnt[NCU];  assign h_rd_data = rd_data[NCU];
  assign wr_req[NCU] = h_wr_req;  assign wr_line[NCU] = h_wr_line;  assign wr_data[NCU] = h_wr_data;
  assign h_wr_gnt = wr_gnt[NCU];

  logic [NB-1:0] l2_ls;                          // 各 bank 的 SRAM light sleep (觀察用; TB 量測睡眠比例)
  npu_l2_xbar #(.NM(NM), .NW(NW), .NB(NB), .DEPTH(DEPTH), .LS_IDLE(LS_IDLE), .LS_WAKE(LS_WAKE)) u_l2 (
    .clk(clk), .rst_n(rst_n),
    .rd_req(rd_req), .rd_line(rd_line), .rd_gnt(rd_gnt), .rd_data(rd_data),
    .wr_req(wr_req), .wr_line(wr_line), .wr_data(wr_data), .wr_gnt(wr_gnt), .bank_ls(l2_ls));

  // ---------------- 完成佇列 ----------------
  // 輸出暫存: cpl_valid 期間 cpl_id 保持 (等待 cpl_ready); 空出時從待回報槽 round-robin 取一個 (one-hot)
  logic [NCU-1:0] cpl_oh;                        // 本拍選中的待回報槽
  wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);
  if (NCU == 1) begin : g_cpl1
    assign cpl_oh = cpl_pend;
  end else begin : g_cpln
    logic [NCU-1:0] cpl_last;                    // 上一次選中者 (one-hot), round-robin 從其下一位開始
    logic [NCU-1:1]   le_hi;                     // 槽 1..NCU-1 中位置 <= cpl_last 者 (槽 0 恆 <= cpl_last, 不放入)
    logic [NCU-1:1]   above;                     // 比 cpl_last 高位的待回報槽
    logic [2*NCU-2:0] dbl, msk;
    always_comb begin
      // 優先找比 cpl_last 高位的待回報槽, 沒有再從最低位開始 (取最低位: x & -x)
      le_hi = (NCU-1)'((NCU'(cpl_last << 1) - 1'b1) >> 1);
      above = cpl_pend[NCU-1:1] & ~le_hi;
      msk     = {cpl_pend, above};
      dbl    = msk & (~msk + 1'b1);
      cpl_oh = {dbl[NCU-2:0], 1'b0} | dbl[2*NCU-2:NCU-1];
    end
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) cpl_last <= NCU'(1) << (NCU - 1);
      else if (cpl_load) cpl_last <= cpl_oh;
  end
  logic [7:0] cpl_tag_sel;
  always_comb begin
    cpl_tag_sel = '0;
    for (int i = 0; i < NCU; i++) cpl_tag_sel |= cpl_oh[i] ? cpl_tag[i] : 8'h00;
  end

  // 待回報槽的 tag: 只在完成握手拍更新 (每個 CU 一個 always_ff; 寫在迴圈裡時 Verilator 的 expression coverage 不計數)
  for (genvar i = 0; i < NCU; i++) begin : g_ctag
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) cpl_tag[i] <= '0;
      else if (cu_done[i] && !cpl_pend[i]) cpl_tag[i] <= cu_dtag[i];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cpl_pend <= '0; cpl_valid <= 1'b0; cpl_id <= '0;
      perf_busy <= '0; perf_steps <= '0;
    end else begin
      // 完成握手 = done && 待回報槽空 (cu_dack = ~cpl_pend; 直接寫 ~cpl_pend, 不出現 cpl_pend = 0 且 dack = 0 的組合)
      cpl_pend <= (cpl_pend & ~(cpl_load ? cpl_oh : '0)) | (cu_done & ~cpl_pend);
      if (cpl_load) begin cpl_valid <= 1'b1; cpl_id <= cpl_tag_sel; end
      else if (cpl_ready) cpl_valid <= 1'b0;
      if (perf_wr) begin
        perf_busy <= perf_wdata_busy; perf_steps <= perf_wdata_steps;
      end else begin
        if (cu_busy != '0) perf_busy <= perf_busy + 1'b1;
        perf_steps <= perf_steps + 48'($countones(cu_feed));
      end
    end
  end
  assign busy = (|cu_busy) || (|cpl_pend) || cpl_valid || q_head_v;

`ifdef FORMAL
  // ---------------- 形式驗證 (formal/npu_core: CU / xbar / ICG 為 stubs.sv 的替身; 模擬時不編入) ----------------
  // CU 的介面契約在此以 assume 表達 (stub 的輸出全部任意); 檢查器從埠記錄每個 CU 持有的 tag 佇列 f_ct / f_cn
  // (分派握手 push、完成握手 pop), 不引用 CU 內部。
  // host 契約: id 不與尚未完成的命令重複; 檢查器用 FIW bit id 空間 (未完成最多 QD + 4·NCU + 1 個 < 2^FIW;
  // 空間越大證明越難 — 計數與兩兩不同是 pigeonhole 型的推理)
  localparam int FIW = $clog2(QD + 4 * NCU + 2);
  initial assume (!rst_n);
  logic [7:0]  f_acc, f_out;
  logic [(1<<FIW)-1:0] f_os;               // 已接受、尚未輸出的 id (由埠記錄)
  logic [NCU-1:0][2:0][7:0] f_ct;          // CU i 持有的命令 tag (依受理順序)
  logic [NCU-1:0][1:0]      f_cn;          // CU i 持有的命令數 (0..3)
  // (Yosys 不正確處理函式內讀取的模組訊號: f_os 以引數傳入)
  function automatic logic f_ok(logic [(1<<FIW)-1:0] os, logic [7:0] id); return (id >> FIW) == '0 && os[FIW'(id)]; endfunction
  always @(*) if (rst_n && cmd_valid) assume ((cmd_id >> FIW) == '0 && !f_os[FIW'(cmd_id)]);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_acc <= '0; f_out <= '0; f_os <= '0; f_ct <= '0; f_cn <= '0; end
    else begin
      f_acc <= f_acc + 8'(q_push);
      f_out <= f_out + 8'(cpl_valid && cpl_ready);
      f_os  <= (f_os & ~((cpl_valid && cpl_ready) ? (1<<FIW)'(1) << FIW'(cpl_id) : '0))
                     | (q_push ? (1<<FIW)'(1) << FIW'(cmd_id) : '0);
      for (int i = 0; i < NCU; i++) begin
        automatic logic [2:0][7:0] t = f_ct[i];
        automatic logic [1:0] n = f_cn[i];
        if (cu_done[i] && cu_dack[i]) begin t = {8'h00, t[2:1]}; n = n - 1'b1; end
        if (cu_v[i] && cu_ready[i]) begin t[n] = head.id; n = n + 1'b1; end
        f_ct[i] <= t; f_cn[i] <= n;
      end
    end
  // CU 介面契約 (npu_gemm_cu, 由其 SVA 在模擬中檢查): 最多持有 3 個命令; 閒置時可接受; busy = 持有命令;
  // done 依受理順序帶回 tag 且保持到 done_ready
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  always @(*) if (rst_n) for (int i = 0; i < NCU; i++) begin
    assume (cu_busy[i] == (f_cn[i] != 2'd0));
    if (f_cn[i] == 2'd0) begin assume (cu_ready[i]); end
    if (cu_ready[i]) begin assume (f_cn[i] != 2'd3); end
    if (cu_done[i]) begin assume (f_cn[i] != 2'd0 && cu_dtag[i] == f_ct[i][0]); end
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) for (int i = 0; i < NCU; i++) begin
    if ($past(cu_done[i] && !cu_dack[i])) begin assume (cu_done[i] && cu_dtag[i] == $past(cu_dtag[i])); end
  end
  // 未完成項目: 佇列中 + CU 持有 + 待回報槽 + 輸出暫存
  logic [7:0] f_live;
  always @(*) begin
    f_live = 8'(q_cnt) + 8'(cpl_valid);
    for (int i = 0; i < NCU; i++) f_live = f_live + 8'(f_cn[i]) + 8'(cpl_pend[i]);
  end
  always @(*) if (rst_n) begin
    // 命令守恆 (接受 - 輸出 = 未完成) 與上限
    assert (8'(f_acc - f_out) == f_live);
    assert (int'(f_live) <= QD + 4 * NCU + 1);
    assert (q_cnt <= (QW+1)'(QD));
    assert (QW'(q_wr - q_rd) == QW'(q_cnt));
    assert ($countones(f_os) == int'(f_live));
    // 輸出的 id 必須是未完成的命令 (不捏造、不重複)
    if (cpl_valid) begin assert (f_ok(f_os, cpl_id)); end
    // 所有未完成項目的 id 都在 f_os 中且兩兩不同 (歸納輔助; 與上面的計數合起來 = 不遺失)
    for (int s = 0; s < QD; s++)
      if (s < int'(q_cnt)) begin
        // 索引先存進 QW 位元變數才會模 QD 環繞 (寫在索引運算式裡, 轉換後會被 32 位元語境放寬而越界)
        automatic logic [QW-1:0] ix = q_rd + QW'(unsigned'(s));
        assert (f_ok(f_os, q[ix].id));
        if (cpl_valid) begin assert (q[ix].id != cpl_id); end
        for (int t = s + 1; t < QD; t++)
          if (t < int'(q_cnt)) begin
            automatic logic [QW-1:0] jx = q_rd + QW'(unsigned'(t));
            assert (q[ix].id != q[jx].id);
          end
        for (int i = 0; i < NCU; i++) begin
          if (cpl_pend[i]) begin assert (q[ix].id != cpl_tag[i]); end
          for (int a = 0; a < 3; a++) if (a < int'(f_cn[i])) begin assert (q[ix].id != f_ct[i][a]); end
        end
      end
    for (int i = 0; i < NCU; i++) begin
      if (cpl_pend[i]) begin
        assert (f_ok(f_os, cpl_tag[i]));
        if (cpl_valid) begin assert (cpl_tag[i] != cpl_id); end
        for (int j = 0; j < NCU; j++) begin
          if (j > i && cpl_pend[j]) begin assert (cpl_tag[i] != cpl_tag[j]); end
          for (int a = 0; a < 3; a++) if (a < int'(f_cn[j])) begin assert (cpl_tag[i] != f_ct[j][a]); end
        end
      end
      for (int a = 0; a < 3; a++)
        if (a < int'(f_cn[i])) begin
          assert (f_ok(f_os, f_ct[i][a]));
          if (cpl_valid) begin assert (f_ct[i][a] != cpl_id); end
          for (int b = a + 1; b < 3; b++) if (b < int'(f_cn[i])) begin assert (f_ct[i][a] != f_ct[i][b]); end
          for (int j = i + 1; j < NCU; j++)
            for (int b = 0; b < 3; b++) if (b < int'(f_cn[j])) begin assert (f_ct[i][a] != f_ct[j][b]); end
        end
    end
    // 分派: 最多一個 CU、只給可接受命令的 CU、有完全閒置者時給其中編號最小者, 否則給可接受者中編號最小者;
    // 有命令且有可接受的 CU 必分派
    assert ($onehot0(cu_v));
    for (int i = 0; i < NCU; i++)
      if (cu_v[i]) begin
        assert (cu_ready[i] && q_head_v);
        if ((cu_ready & ~cu_busy) != '0) begin
          assert (!cu_busy[i]);
          for (int j = 0; j < i; j++) begin assert (!(cu_ready[j] && !cu_busy[j])); end
        end else begin
          for (int j = 0; j < i; j++) begin assert (!cu_ready[j]); end
        end
      end
    for (int i = 0; i < NCU; i++)
      if (q_head_v && cu_ready[i]) begin assert (cu_v != '0); end
    // 完成: CU 的完成在待回報槽空時被接受; 有待回報且輸出暫存空時必載入; busy / cmd_ready 正確
    assert (cu_dack == ~cpl_pend);
    if (cpl_pend != '0 && !cpl_valid) begin assert (cpl_load); end
    assert (busy == (q_cnt != '0 || cpl_valid || cpl_pend != '0 || cu_busy != '0));
    assert (cmd_ready == (q_cnt != (QW+1)'(QD)));
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(cpl_valid && !cpl_ready)) begin assert (cpl_valid && cpl_id == $past(cpl_id)); end
    // 待回報槽的 tag = CU 完成握手時帶回的 tag
    for (int i = 0; i < NCU; i++)
      if ($past(cu_done[i] && cu_dack[i])) begin assert (cpl_pend[i] && cpl_tag[i] == $past(cu_dtag[i])); end
  end
  // 完成仲裁公平: 某槽待回報期間, 輸出暫存載入其他槽的次數 <= NCU - 1 (round-robin)
  logic [NCU-1:0][3:0] f_sk;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) f_sk <= '0;
    else for (int i = 0; i < NCU; i++) f_sk[i] <= cpl_pend[i] ? f_sk[i] + 4'(cpl_load) : 4'h0;
  always @(*) if (rst_n)
    for (int i = 0; i < NCU; i++) if (cpl_pend[i]) begin assert (int'(f_sk[i]) <= NCU - 1); end
  // 歸納輔助: 已被略過的次數 + (上次選中者到槽 i 的循環距離, 1..NCU) <= NCU
  if (NCU > 1) begin : g_f_rr
    always @(*) if (rst_n) begin
      automatic int li = 0;
      assert ($onehot(g_cpln.cpl_last));
      for (int k = 0; k < NCU; k++) if (g_cpln.cpl_last[k]) li = k;
      for (int i = 0; i < NCU; i++)
        if (cpl_pend[i]) begin
          automatic int d = (i - li + NCU) % NCU;
          assert (int'(f_sk[i]) + (d == 0 ? NCU : d) <= NCU);
        end
    end
  end
  always @(*) if (rst_n) begin
    cover (int'(f_live) == QD + 4 * NCU + 1);                // 上限可達 (緊)
    cover (cpl_pend[0] && int'(f_sk[0]) == NCU - 1);           // 最長等待可達
    cover (cpl_pend == '1 && cpl_valid && !cpl_ready);
    cover (cmd_valid && !cmd_ready);
    cover ((cu_v & cu_busy) != '0);                            // 分派給仍在輸出的 CU
    cover (cu_done[0] && !cu_dack[0]);                         // 完成等待待回報槽
  end
`endif
endmodule : npu_core
