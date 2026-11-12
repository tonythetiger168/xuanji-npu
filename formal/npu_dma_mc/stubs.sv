// SPDX-License-Identifier: Apache-2.0
// npu_dma_mc 形式驗證用的替身: npu_dma 通道只遵守介面契約 (請求 / 位址 / 資料保持到 grant; 只在 idle 接受命令;
// 完成前在途的外部讀取都已回來 — 由 npu_dma_mc 的 FORMAL 區塊以 assume 約束), 其餘輸出任意。
// npu_dma 本身另外形式證明 (formal/npu_dma); npu_rr_lock 用真的 (另外形式證明)。clock_gate: 直通。
`include "orca_pkg.sv"
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule

module npu_dma #(parameter int OUT = 8, DW = 512) (
  input  logic clk, rst_n,
  input  logic cmd_valid, output logic cmd_ready, input logic cmd_st, input logic [1:0] cmd_fmt,
  input  logic [31:0] cmd_ext, input logic [25:0] cmd_l2, input logic [15:0] cmd_n,
  input  logic [15:0] cmd_cols, input logic [25:0] cmd_stride, input logic cmd_ind, input logic [25:0] cmd_list,
  output logic done, busy,
  output logic h_rd_req, output logic [25:0] h_rd_line, input logic h_rd_gnt, input logic [DW-1:0] h_rd_data,
  output logic h_wr_req, output logic [25:0] h_wr_line, output logic [DW-1:0] h_wr_data, input logic h_wr_gnt,
  output logic m_rd_req, output logic [31:0] m_rd_addr, input logic m_rd_gnt, input logic m_rd_rvalid,
  input  logic [DW-1:0] m_rd_rdata,
  output logic m_wr_req, output logic [31:0] m_wr_addr, output logic [DW-1:0] m_wr_data, input logic m_wr_gnt
);
  (* anyseq *) logic done_any, hr_any, hw_any, mr_any, mw_any;
  (* anyseq *) logic [25:0] hrl_any, hwl_any;
  (* anyseq *) logic [31:0] mra_any, mwa_any;
  (* anyseq *) logic [DW-1:0] hwd_any, mwd_any;
  logic act;
  // 等待 grant 的請求保持 (位址 / 資料不變)
  logic hr_p, hw_p, mr_p, mw_p;
  logic [25:0] hrl_q, hwl_q; logic [31:0] mra_q, mwa_q; logic [DW-1:0] hwd_q, mwd_q;
  assign cmd_ready = !act; assign busy = act; assign done = act && done_any && !h_rd_req && !h_wr_req && !m_rd_req && !m_wr_req;
  assign h_rd_req = act && (hr_p || hr_any); assign h_rd_line = hr_p ? hrl_q : hrl_any;
  assign h_wr_req = act && (hw_p || hw_any); assign h_wr_line = hw_p ? hwl_q : hwl_any; assign h_wr_data = hw_p ? hwd_q : hwd_any;
  assign m_rd_req = act && (mr_p || mr_any); assign m_rd_addr = mr_p ? mra_q : mra_any;
  assign m_wr_req = act && (mw_p || mw_any); assign m_wr_addr = mw_p ? mwa_q : mwa_any; assign m_wr_data = mw_p ? mwd_q : mwd_any;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      act <= 0; hr_p <= 0; hw_p <= 0; mr_p <= 0; mw_p <= 0;
      hrl_q <= 0; hwl_q <= 0; mra_q <= 0; mwa_q <= 0; hwd_q <= 0; mwd_q <= 0;
    end else begin
      if (cmd_valid && cmd_ready) act <= 1;
      else if (done) act <= 0;
      hr_p <= h_rd_req && !h_rd_gnt; hrl_q <= h_rd_line;
      hw_p <= h_wr_req && !h_wr_gnt; hwl_q <= h_wr_line; hwd_q <= h_wr_data;
      mr_p <= m_rd_req && !m_rd_gnt; mra_q <= m_rd_addr;
      mw_p <= m_wr_req && !m_wr_gnt; mwa_q <= m_wr_addr; mwd_q <= m_wr_data;
    end
endmodule
