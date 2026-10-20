// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 子系統 (v6.3.8)
`include "orca_pkg.sv"

// npu_core: NCU 個 GEMM 計算單元 (各含 N×N npu_mma) + NB bank 的共用 L2 + 命令佇列 / 分派 / 完成佇列。
//
// 產品組態 (見 docs/npu.md; rtl/legacy/ai/npu_cfg_pkg.sv):
//   Xuanji-X1-Nano  N=4,  NCU=1, NB=2   16   MAC/拍
//   Xuanji-X1       N=8,  NCU=4, NB=4   256  MAC/拍
//   Xuanji-X1-Pro   N=16, NCU=8, NB=16  2048 MAC/拍
//
// host 介面:
//   命令: cmd_valid / cmd_ready + {id, A / B / C line 位址, K, 有號性} (語意見 npu_gemm_cu), 先進 QD 項佇列,
//         依序分派給編號最小的閒置 CU; 不同命令可同時在不同 CU 執行, 完成順序不保證。
//   完成: cpl_valid / cpl_ready + cpl_id (輸出暫存, 等待 ready 期間保持); 每個 CU 一個待回報槽,
//         多個 CU 同時完成時 round-robin 進入輸出暫存。CU 的完成移入輸出暫存前不會再分派新命令給它。
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
  parameter int QD    = 4              // 命令佇列深度 (2 的冪)
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
  localparam int NM = NCU + 1;
  localparam int QW = $clog2(QD);

  typedef struct packed {
    logic [7:0]  id;
    logic [25:0] a, b, c;
    logic [15:0] k;
    logic        sa, sb;
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
  logic [NCU-1:0] cu_ready, cu_done, cu_feed, cu_v, cu_free, cpl_pend, first_free;
  logic [NCU-1:0][7:0] cu_tag;
  logic           sel_v;
  // done 脈衝當拍也不可分派: 待回報槽在下一拍才設定, 若同拍分派, 新命令的 tag 會蓋掉尚未回報的 id
  assign cu_free    = cu_ready & ~cpl_pend & ~cu_done;
  assign first_free = cu_free & (~cu_free + 1'b1);
  assign sel_v      = (cu_free != '0);
  assign cu_v       = q_head_v ? first_free : '0;
  wire q_pop = q_head_v && sel_v;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= '0; q_rd <= '0; q_wr <= '0; q_cnt <= '0;
    end else begin
      if (q_push) begin
        q[q_wr] <= '{id: cmd_id, a: cmd_a_line, b: cmd_b_line, c: cmd_c_line, k: cmd_k, sa: cmd_a_signed, sb: cmd_b_signed};
        q_wr <= q_wr + 1'b1;
      end
      if (q_pop) q_rd <= q_rd + 1'b1;
      q_cnt <= q_cnt + (QW+1)'(q_push) - (QW+1)'(q_pop);
    end
  end

  // ---------------- 計算單元 ----------------
  logic [NM-1:0]        rd_req, rd_gnt, wr_req, wr_gnt;
  logic [NM-1:0][25:0]  rd_line, wr_line;
  logic [NM-1:0][511:0] rd_data, wr_data;
  logic [NCU-1:0]       cu_en;
  for (genvar i = 0; i < NCU; i++) begin : g_cu
    // CU 時脈致能: 忙碌、本拍分派 (CU 在 idle 取樣命令)、或 done 脈衝尚待清除
    assign cu_en[i] = !cu_ready[i] || cu_v[i] || cu_done[i];
    logic gclk;
    clock_gate u_icg (.clk(clk), .en(cu_en[i]), .test_en(test_en), .clk_out(gclk));
    npu_gemm_cu #(.N(N)) u_cu (
      .clk(gclk), .rst_n(rst_n), .test_en(test_en),
      .cmd_valid(cu_v[i]), .cmd_ready(cu_ready[i]),
      .cmd_a_line(head.a), .cmd_b_line(head.b), .cmd_c_line(head.c), .cmd_k(head.k),
      .cmd_a_signed(head.sa), .cmd_b_signed(head.sb),
      .done(cu_done[i]), .feeding(cu_feed[i]),
      .rd_req(rd_req[i]), .rd_line(rd_line[i]), .rd_gnt(rd_gnt[i]), .rd_data(rd_data[i]),
      .wr_req(wr_req[i]), .wr_line(wr_line[i]), .wr_data(wr_data[i]), .wr_gnt(wr_gnt[i]));
  end

  // host = 主端 NCU
  assign rd_req[NCU] = h_rd_req;  assign rd_line[NCU] = h_rd_line;
  assign h_rd_gnt = rd_gnt[NCU];  assign h_rd_data = rd_data[NCU];
  assign wr_req[NCU] = h_wr_req;  assign wr_line[NCU] = h_wr_line;  assign wr_data[NCU] = h_wr_data;
  assign h_wr_gnt = wr_gnt[NCU];

  npu_l2_xbar #(.NM(NM), .NB(NB), .DEPTH(DEPTH)) u_l2 (
    .clk(clk), .rst_n(rst_n),
    .rd_req(rd_req), .rd_line(rd_line), .rd_gnt(rd_gnt), .rd_data(rd_data),
    .wr_req(wr_req), .wr_line(wr_line), .wr_data(wr_data), .wr_gnt(wr_gnt));

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
    for (int i = 0; i < NCU; i++) cpl_tag_sel |= cpl_oh[i] ? cu_tag[i] : 8'h00;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cpl_pend <= '0; cu_tag <= '0; cpl_valid <= 1'b0; cpl_id <= '0;
      perf_busy <= '0; perf_steps <= '0;
    end else begin
      for (int i = 0; i < NCU; i++) if (cu_v[i]) cu_tag[i] <= head.id;
      cpl_pend <= (cpl_pend & ~(cpl_load ? cpl_oh : '0)) | cu_done;
      if (cpl_load) begin cpl_valid <= 1'b1; cpl_id <= cpl_tag_sel; end
      else if (cpl_ready) cpl_valid <= 1'b0;
      if (perf_wr) begin
        perf_busy <= perf_wdata_busy; perf_steps <= perf_wdata_steps;
      end else begin
        if (!(&cu_ready)) perf_busy <= perf_busy + 1'b1;
        perf_steps <= perf_steps + 48'($countones(cu_feed));
      end
    end
  end
  assign busy = !(&cu_ready) || (|cpl_pend) || cpl_valid || q_head_v;
endmodule : npu_core
