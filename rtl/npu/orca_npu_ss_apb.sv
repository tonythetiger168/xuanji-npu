// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU Subsystem + APB4 設定暫存器 (v6.3.10)。暫存器表見 docs/apb.md。
`include "orca_pkg.sv"

// orca_npu_ss_apb: orca_npu_ss + npu_ss_regs 的純接線 (SoC 以此接 CPU 的 APB 與外部記憶體)。
// 只有接線: 由整合測試 dv/npu/orca_npu_ss_apb_tb.sv 驗證 (只經 APB 下命令、DMA 搬移資料、取回完成)。
module orca_npu_ss_apb
  import orca_pkg::*;
#(
  parameter int TILE_ID = 0,
  parameter int N     = 8,
  parameter int NCU   = 4,
  parameter int NB    = 4,
  parameter int DEPTH = 1024,
  parameter int QD    = 8,
  parameter int OUT   = 32,
  parameter int CQ    = 4
)(
  input  logic        clk, rst_n,
  input  logic        test_en,
  input  logic        clk_aon, por_n,      // 常開域時脈 / 上電 reset (retention 影子暫存器)
  input  logic        ret_save, ret_restore,
  input  logic        psel, penable, pwrite,
  input  logic [11:0] paddr,
  input  logic [31:0] pwdata,
  input  logic [3:0]  pstrb,
  output logic        pready,
  output logic [31:0] prdata,
  output logic        pslverr,
  output logic        irq,
  output logic        busy,              // 子系統忙碌或 APB 命令 FIFO 非空 (= STATUS.BUSY; PMU 用)
  output logic         m_rd_req,
  output logic [31:0]  m_rd_addr,
  input  logic         m_rd_gnt,
  input  logic         m_rd_rvalid,
  input  logic [511:0] m_rd_rdata,
  output logic         m_wr_req,
  output logic [31:0]  m_wr_addr,
  output logic [511:0] m_wr_data,
  input  logic         m_wr_gnt
);
  logic          cmd_valid, cmd_ready, cpl_valid, cpl_ready, ss_busy, perf_wr, vec_err;
  npu_tile_cmd_t cmd;
  logic [7:0]    cpl_id;
  logic [47:0]   perf_busy, perf_steps;
  npu_ss_regs #(.TILE_ID(TILE_ID), .N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(QD), .OUT(OUT), .CQ(CQ)) u_regs (
    .clk, .rst_n, .test_en, .clk_aon, .por_n, .ret_save, .ret_restore, .psel, .penable, .pwrite, .paddr, .pwdata, .pstrb, .pready, .prdata, .pslverr, .irq,
    .cmd_valid, .cmd_ready, .cmd, .cpl_valid, .cpl_ready, .cpl_id, .ss_busy, .vec_err,
    .perf_wr, .perf_busy, .perf_steps);
  assign busy = ss_busy || cmd_valid;
  orca_npu_ss #(.TILE_ID(TILE_ID), .N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(QD), .OUT(OUT)) u_ss (
    .clk, .rst_n, .test_en, .cmd_valid, .cmd_ready, .cmd, .cpl_valid, .cpl_ready, .cpl_id,
    .irq(), .busy(ss_busy), .vec_err,
    .m_rd_req, .m_rd_addr, .m_rd_gnt, .m_rd_rvalid, .m_rd_rdata, .m_wr_req, .m_wr_addr, .m_wr_data, .m_wr_gnt,
    .perf_wr, .perf_wdata_busy('0), .perf_wdata_steps('0), .perf_busy, .perf_steps);   // perf_wr = 清除
endmodule : orca_npu_ss_apb
