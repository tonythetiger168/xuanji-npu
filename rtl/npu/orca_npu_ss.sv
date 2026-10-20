// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU Subsystem (v6.3.9: 由舊 AI tile 以 npu_core 重建)
`include "orca_pkg.sv"

// orca_npu_ss: 命令佇列 + 分派 (等待旗標) + npu_core (GEMM) + npu_ss_dma (外部記憶體 <-> L2) + 完成合併。
//
// 命令 (npu_tile_cmd_t, cmd_valid / cmd_ready): 依序進 QD 項佇列, 依序分派 —
//   GEMM → npu_core (可多個同時在不同 CU 執行); DMA_LD / DMA_ST → DMA (一次一個);
//   NOP 與 K = 0 的 GEMM → 直接完成 (C 不變)。DMA 的 line 數 k = 0 時也立即完成。
//   wt = 1 的命令在先前所有命令「完成」(結果已寫入 L2 / 外部記憶體, 且其完成已移入輸出暫存) 後才分派;
//   wt = 0 的命令可與先前的命令同時執行 — 軟體須保證兩者存取的 L2 / 外部區域不重疊, 否則以 wt 分隔。
//   隊首等待時後面的命令也等待 (依序分派)。
// 完成 (cpl_valid / cpl_ready + cpl_id): 輸出暫存, 等待 ready 期間保持; 來源 (GEMM / DMA / NOP)
//   round-robin 進入。完成順序不保證與命令順序相同 (wt 只約束分派)。irq = cpl_valid (level)。
// 外部記憶體埠: 見 npu_ss_dma (64 B line, 讀回應依序且無回壓)。
// 效能計數: npu_core 的 perf_busy / perf_steps 直通。
// 低功耗: DMA 一個 ICG (閒置停止時脈); npu_core 內每個 CU 一個 ICG。
module orca_npu_ss
  import orca_pkg::*;
#(
  parameter int TILE_ID = 0,
  parameter int N     = 8,              // 產品組態 (見 rtl/npu/npu_products.sv; 預設 Xuanji-X1)
  parameter int NCU   = 4,
  parameter int NB    = 4,
  parameter int DEPTH = 1024,
  parameter int QD    = 8,              // tile 命令佇列深度 (2 的冪)
  parameter int OUT   = 32              // DMA 在途上限 (見 npu_products.sv: Nano 16 / X1 32 / Pro 64)
)(
  input  logic clk, rst_n,
  input  logic test_en,
  // 命令
  input  logic          cmd_valid,
  output logic          cmd_ready,
  input  npu_tile_cmd_t cmd,
  // 完成
  output logic        cpl_valid,
  input  logic        cpl_ready,
  output logic [7:0]  cpl_id,
  output logic        irq,
  output logic        busy,
  // 外部記憶體
  output logic         m_rd_req,
  output logic [31:0]  m_rd_addr,
  input  logic         m_rd_gnt,
  input  logic         m_rd_rvalid,
  input  logic [511:0] m_rd_rdata,
  output logic         m_wr_req,
  output logic [31:0]  m_wr_addr,
  output logic [511:0] m_wr_data,
  input  logic         m_wr_gnt,
  // 效能計數
  input  logic         perf_wr,
  input  logic [47:0]  perf_wdata_busy, perf_wdata_steps,
  output logic [47:0]  perf_busy, perf_steps
);
  localparam int QW = $clog2(QD);
  if ((1 << QW) != QD || QD < 2) begin : g_bad
    $error("orca_npu_ss: QD 須為 2 的冪且 >= 2");
  end

  // ---------------- 命令佇列 ----------------
  npu_tile_cmd_t [QD-1:0] q;
  logic [QW-1:0] q_rd, q_wr;
  logic [QW:0]   q_cnt;
  assign cmd_ready = (q_cnt != (QW+1)'(QD));
  wire q_push = cmd_valid && cmd_ready;
  wire q_hv   = (q_cnt != '0);
  npu_tile_cmd_t head;
  assign head = q[q_rd];

  // ---------------- 分派 ----------------
  // oc: 已分派、完成尚未移入輸出暫存的命令數。上限: npu_core 佇列 + 每個 CU 一個 (執行中或待回報, CU 有待回報的完成時
  //     不再分派) + npu_core 輸出暫存 + DMA + NOP (SVA a_oc_bound)
  localparam int CQD = 4;                // npu_core 命令佇列深度
  localparam int OC_MAX = CQD + NCU + 1 + 2;
  localparam int OCW = $clog2(OC_MAX + 1);
  logic [OCW-1:0] oc;
  logic       dma_cv, nop_cv;            // DMA / NOP 完成槽
  logic [7:0] dma_id, nop_id;
  logic       c_cmd_ready, d_cmd_ready;
  wire go     = q_hv && (!head.wt || oc == '0);
  // K = 0 的 GEMM 不送 npu_core (CU 要求 K >= 1), 與 NOP 相同只回報完成 (C 不變)
  wire h_nop  = head.op == NPU_OP_NOP || (head.op == NPU_OP_GEMM && head.k == '0);
  wire g_core = go && head.op == NPU_OP_GEMM && !h_nop;
  wire g_dma  = go && (head.op == NPU_OP_DMA_LD || head.op == NPU_OP_DMA_ST) && !dma_cv;
  wire g_nop  = go && h_nop && !nop_cv;
  wire d_take = g_dma && d_cmd_ready;
  wire q_pop  = (g_core && c_cmd_ready) || d_take || g_nop;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= '0; q_rd <= '0; q_wr <= '0; q_cnt <= '0;
    end else begin
      if (q_push) begin q[q_wr] <= cmd; q_wr <= q_wr + 1'b1; end
      if (q_pop) q_rd <= q_rd + 1'b1;
      q_cnt <= q_cnt + (QW+1)'(q_push) - (QW+1)'(q_pop);
    end
  end

  // ---------------- npu_core ----------------
  logic         c_cpl_valid, c_cpl_ready, c_busy;
  logic [7:0]   c_cpl_id;
  logic         h_rd_req, h_rd_gnt, h_wr_req, h_wr_gnt;
  logic [25:0]  h_rd_line, h_wr_line;
  logic [511:0] h_rd_data, h_wr_data;

  npu_core #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(CQD)) u_core (
    .clk(clk), .rst_n(rst_n), .test_en(test_en),
    .cmd_valid(g_core), .cmd_ready(c_cmd_ready), .cmd_id(head.id),
    .cmd_a_line(head.a), .cmd_b_line(head.b), .cmd_c_line(head.c), .cmd_k(head.k),
    .cmd_a_signed(head.sa), .cmd_b_signed(head.sb),
    .cpl_valid(c_cpl_valid), .cpl_ready(c_cpl_ready), .cpl_id(c_cpl_id),
    .h_rd_req(h_rd_req), .h_rd_line(h_rd_line), .h_rd_gnt(h_rd_gnt), .h_rd_data(h_rd_data),
    .h_wr_req(h_wr_req), .h_wr_line(h_wr_line), .h_wr_data(h_wr_data), .h_wr_gnt(h_wr_gnt),
    .busy(c_busy),
    .perf_wr(perf_wr), .perf_wdata_busy(perf_wdata_busy), .perf_wdata_steps(perf_wdata_steps),
    .perf_busy(perf_busy), .perf_steps(perf_steps));

  // ---------------- DMA ----------------
  logic d_done, d_busy, d_en, d_gclk;
  assign d_en = d_busy || g_dma;          // 閒置且沒有分派時停止 DMA 時脈 (命令在 idle 取樣)
  clock_gate u_dma_icg (.clk(clk), .en(d_en), .test_en(test_en), .clk_out(d_gclk));
  npu_ss_dma #(.OUT(OUT)) u_dma (
    .clk(d_gclk), .rst_n(rst_n),
    .cmd_valid(g_dma), .cmd_ready(d_cmd_ready), .cmd_st(head.op == NPU_OP_DMA_ST),
    .cmd_ext(head.ext), .cmd_l2(head.a), .cmd_n(head.k),
    .done(d_done), .busy(d_busy),
    .h_rd_req(h_rd_req), .h_rd_line(h_rd_line), .h_rd_gnt(h_rd_gnt), .h_rd_data(h_rd_data),
    .h_wr_req(h_wr_req), .h_wr_line(h_wr_line), .h_wr_data(h_wr_data), .h_wr_gnt(h_wr_gnt),
    .m_rd_req(m_rd_req), .m_rd_addr(m_rd_addr), .m_rd_gnt(m_rd_gnt),
    .m_rd_rvalid(m_rd_rvalid), .m_rd_rdata(m_rd_rdata),
    .m_wr_req(m_wr_req), .m_wr_addr(m_wr_addr), .m_wr_data(m_wr_data), .m_wr_gnt(m_wr_gnt));

  // ---------------- 完成合併 ----------------
  // 來源 0 = npu_core, 1 = DMA, 2 = NOP; 輸出暫存空出 (或本拍被接受) 時從上次選中者的下一個開始找
  logic [2:0] src_v, sel;
  logic [1:0] last;
  assign src_v = {nop_cv, dma_cv, c_cpl_valid};
  wire load = (src_v != '0) && (!cpl_valid || cpl_ready);
  always_comb begin
    sel = '0;
    for (int i = 2; i >= 0; i--) begin                  // 由遠到近, 最後寫入者 = 最近的有效來源
      automatic int s = (int'(last) + 1 + i) % 3;
      if (src_v[s]) sel = 3'(1 << s);
    end
  end
  assign c_cpl_ready = load && sel[0];
  assign irq  = cpl_valid;
  assign busy = q_hv || (oc != '0) || cpl_valid || c_busy || d_busy;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      oc <= '0; dma_cv <= 1'b0; nop_cv <= 1'b0; dma_id <= '0; nop_id <= '0;
      cpl_valid <= 1'b0; cpl_id <= '0; last <= 2'd2;
    end else begin
      oc <= oc + OCW'(q_pop) - OCW'(load);
      if (d_take) dma_id <= head.id;
      if (d_done) dma_cv <= 1'b1;
      else if (load && sel[1]) dma_cv <= 1'b0;
      if (g_nop) begin nop_cv <= 1'b1; nop_id <= head.id; end
      else if (load && sel[2]) nop_cv <= 1'b0;
      if (load) begin
        cpl_valid <= 1'b1;
        cpl_id    <= sel[0] ? c_cpl_id : sel[1] ? dma_id : nop_id;
        last      <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : 2'd2;
      end else if (cpl_ready) cpl_valid <= 1'b0;
    end
  end
endmodule : orca_npu_ss
