// SPDX-License-Identifier: Apache-2.0
// orca_npu_ss 形式驗證用的子模組替身 (只遵守介面契約, 其餘輸出任意):
//   npu_core: 已接受未回報的命令數 p <= CQD + NCU + 1 (真實容量); cpl_valid 只在 p > 0 時出現,
//             等待 cpl_ready 期間保持 (id 不變); cmd_ready 任意但 p 滿時為 0; busy = (p != 0)。
//             npu_core 本身由 run_npu_cov / run_npu_mutation 驗證。
//   npu_ss_dma: 只在 idle 接受命令; done 在忙碌中任意一拍出現, 之後回到 idle (DMA 本身已形式證明)。
//   clock_gate: 直通 (閘控安全以 orca_npu_ss 內的組合性質檢查 en)。
`include "orca_pkg.sv"
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule

module npu_core #(parameter int N = 8, NCU = 4, NB = 4, DEPTH = 1024, QD = 4) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready, input logic [7:0] cmd_id,
  input  logic [25:0] cmd_a_line, cmd_b_line, cmd_c_line, input logic [15:0] cmd_k,
  input  logic cmd_a_signed, cmd_b_signed,
  output logic cpl_valid, input logic cpl_ready, output logic [7:0] cpl_id,
  input  logic h_rd_req, input logic [25:0] h_rd_line, output logic h_rd_gnt, output logic [511:0] h_rd_data,
  input  logic h_wr_req, input logic [25:0] h_wr_line, input logic [511:0] h_wr_data, output logic h_wr_gnt,
  output logic busy,
  input  logic perf_wr, input logic [47:0] perf_wdata_busy, perf_wdata_steps,
  output logic [47:0] perf_busy, perf_steps
);
  localparam int CAP = QD + NCU + 1;
  (* anyseq *) logic rdy_any, cv_any, rg_any, wg_any;
  (* anyseq *) logic [7:0] id_any;
  logic [7:0] p;
  assign cmd_ready = rdy_any && (p < 8'(CAP));
  assign busy = (p != 0);
  assign h_rd_gnt = rg_any; assign h_wr_gnt = wg_any; assign h_rd_data = '0;
  assign perf_busy = '0; assign perf_steps = '0;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin p <= 0; cpl_valid <= 0; cpl_id <= 0; end
    else begin
      p <= p + 8'(cmd_valid && cmd_ready) - 8'(cpl_valid && cpl_ready);
      if (!cpl_valid || cpl_ready) begin                               // 等待 cpl_ready 期間保持
        // 下一個完成: 只在還有 (扣掉本拍被接受的那個) 未回報命令時
        cpl_valid <= cv_any && (p - 8'(cpl_valid && cpl_ready) != 0);
        cpl_id <= id_any;
      end
    end
endmodule

module npu_ss_dma #(parameter int OUT = 8, DW = 512) (
  input  logic clk, rst_n,
  input  logic cmd_valid, output logic cmd_ready, input logic cmd_st,
  input  logic [31:0] cmd_ext, input logic [25:0] cmd_l2, input logic [15:0] cmd_n,
  output logic done, busy,
  output logic h_rd_req, output logic [25:0] h_rd_line, input logic h_rd_gnt, input logic [DW-1:0] h_rd_data,
  output logic h_wr_req, output logic [25:0] h_wr_line, output logic [DW-1:0] h_wr_data, input logic h_wr_gnt,
  output logic m_rd_req, output logic [31:0] m_rd_addr, input logic m_rd_gnt, input logic m_rd_rvalid,
  input  logic [DW-1:0] m_rd_rdata,
  output logic m_wr_req, output logic [31:0] m_wr_addr, output logic [DW-1:0] m_wr_data, input logic m_wr_gnt
);
  (* anyseq *) logic done_any;
  logic act;
  assign cmd_ready = !act; assign busy = act; assign done = act && done_any;
  assign h_rd_req = 0; assign h_rd_line = 0; assign h_wr_req = 0; assign h_wr_line = 0; assign h_wr_data = 0;
  assign m_rd_req = 0; assign m_rd_addr = 0; assign m_wr_req = 0; assign m_wr_addr = 0; assign m_wr_data = 0;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) act <= 0;
    else if (cmd_valid && cmd_ready) act <= 1;
    else if (done) act <= 0;
endmodule
