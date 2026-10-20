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
`ifdef FORMAL
  // ---------------- 形式驗證 (formal/orca_npu_ss: npu_core / DMA 以遵守介面契約的替身取代; 模擬時不編入) ----------------
  initial assume (!rst_n);
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  // 命令守恆: 接受 = 佇列中 + 已分派; 已分派 = 未完成 (oc) + 已載入輸出; 已載入 = 已輸出 + 輸出暫存中
  logic [7:0] f_acc, f_disp, f_load, f_out, f_cin, f_cout;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_acc <= 0; f_disp <= 0; f_load <= 0; f_out <= 0; f_cin <= 0; f_cout <= 0; end
    else begin
      f_acc  <= f_acc  + 8'(q_push);
      f_disp <= f_disp + 8'(q_pop);
      f_load <= f_load + 8'(load);
      f_out  <= f_out  + 8'(cpl_valid && cpl_ready);
      f_cin  <= f_cin  + 8'(g_core && c_cmd_ready);
      f_cout <= f_cout + 8'(c_cpl_valid && c_cpl_ready);
    end
  // 分派到 DMA / NOP 的命令 id
  logic [7:0] f_dma_id, f_nop_id;
  always @(posedge clk) begin
    if (d_take) f_dma_id <= head.id;
    if (g_nop)  f_nop_id <= head.id;
  end
  wire [7:0] f_core_pend = f_cin - f_cout;
  // npu_core 的介面契約 (stub 內部也成立; 以 tile 自己的計數重述, 讓歸納不從不可達狀態出發):
  //   容量 CQD + NCU + 1 滿時不收命令; 只在有未回報命令時回報完成
  always @(*) if (rst_n) begin
    assume (!c_cmd_ready || f_core_pend < 8'(CQD + NCU + 1));
    assume (!c_cpl_valid || f_core_pend != 0);
  end
  always @(*) if (rst_n) begin
    assert (f_core_pend <= 8'(CQD + NCU + 1));
    assert (q_cnt <= (QW+1)'(QD));
    assert (cmd_ready == (q_cnt != (QW+1)'(QD)));
    assert (8'(f_acc - f_disp) == 8'(q_cnt));
    assert (8'(f_disp - f_load) == 8'(oc));
    assert (8'(f_load - f_out) == 8'(cpl_valid));
    // 未完成數 = npu_core 內 + DMA (執行中或完成待回報) + NOP 完成待回報; 不超過上限
    assert (8'(oc) == f_core_pend + 8'(d_busy || dma_cv) + 8'(nop_cv));
    assert (!(d_busy && dma_cv));
    // DMA / NOP 完成槽的 id 在整個佔用期間等於分派時的 id (不被後來的命令覆蓋)
    if (d_busy || dma_cv) assert (dma_id == f_dma_id);
    if (nop_cv) assert (nop_id == f_nop_id);
    assert (int'(oc) <= OC_MAX);
    assert (last != 2'd3);                                          // round-robin 指標只有 3 個合法值
    // 分派
    assert ($onehot0({g_core, g_dma, g_nop}));
    if (q_pop && head.wt) assert (oc == '0);
    if (g_core) assert (head.k != '0);
    if (q_hv && (!head.wt || oc == '0) &&
        ((head.op == NPU_OP_GEMM && !h_nop && c_cmd_ready) ||
         ((head.op == NPU_OP_DMA_LD || head.op == NPU_OP_DMA_ST) && !dma_cv && d_cmd_ready) || (h_nop && !nop_cv)))
      assert (q_pop);
    // 完成合併: 選一個有效來源, 確切 round-robin, 輸出暫存空時必載入; id 來自對應來源
    if (load) begin
      assert ($onehot(sel) && (sel & ~src_v) == '0);
      if (sel[1]) assert (dma_id == f_dma_id);
      if (sel[2]) assert (nop_id == f_nop_id);
    end
    if (src_v != '0 && !cpl_valid) assert (load);
    assert (irq == cpl_valid);
    if (!busy) assert (!q_hv && oc == '0 && !cpl_valid && !c_busy && !d_busy);
    // DMA 時脈閘控: 關閉時 DMA 必閒置且本拍沒有分派
    if (!test_en && !d_en) assert (!d_busy && !g_dma);
  end
  // 確切 round-robin: 上次選中者之後第一個有效來源
  // 上次選中者: 由檢查器自己從 sel 記錄 (不用 RTL 的 last, 否則與被檢查的邏輯循環)
  logic [1:0] f_last;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) f_last <= 2'd2;
    else if (load) f_last <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : 2'd2;
  always @(*) if (rst_n) assert (f_last != 2'd3 && last == f_last);   // RTL 的指標必須追蹤實際選中者
  wire [1:0] f_s0 = (f_last == 2'd2) ? 2'd0 : f_last + 2'd1;      // 上次選中者的下一個
  wire [1:0] f_s1 = (f_s0 == 2'd2) ? 2'd0 : f_s0 + 2'd1;
  always @(*) if (rst_n && load) begin
    if (src_v[f_s0]) begin assert (sel[f_s0]); end
    else if (src_v[f_s1]) begin assert (sel[f_s1]); end
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(cpl_valid && !cpl_ready)) assert (cpl_valid && cpl_id == $past(cpl_id));
    if ($past(load && sel[0])) assert (cpl_id == $past(c_cpl_id));
  end
  always @(*) if (rst_n) begin
    cover (int'(oc) == OC_MAX);
    cover (src_v == 3'b111 && cpl_valid && !cpl_ready);
    cover (q_hv && head.wt && oc != '0);
    cover (cmd_valid && !cmd_ready);
  end
`endif
endmodule : orca_npu_ss
