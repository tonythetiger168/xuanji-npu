// SPDX-License-Identifier: Apache-2.0
// orca_npu_ss 形式驗證用的子模組替身 (只遵守介面契約, 其餘輸出任意):
//   npu_core: 已接受未回報的命令數 p <= CQD + 4·NCU + 1 (真實容量: 每個 CU 最多 3 個命令 + 1 個待回報); cpl_valid 只在 p > 0 時出現,
//             等待 cpl_ready 期間保持 (id 不變); cmd_ready 任意但 p 滿時為 0; busy = (p != 0)。
//             npu_core 本身由 run_npu_cov / run_npu_mutation 驗證。
//   npu_dma_mc (v6.3.13): 最多 NDC 個未回報命令; 執行中的命令任意一拍完成, 進入待回報 (cpl_valid, id 任意)。
//             npu_dma_mc / npu_dma 本身另外形式證明。
//   npu_vec_eng: 同 DMA (只在 idle 接受; done 在忙碌中任意一拍; derr 任意); L2 埠不發請求 (仲裁器另外形式證明)。
//   npu_llm_eng (v6.3.13): 同 VEC (只在 idle 接受; done 一拍, 在忙碌中任意一拍; busy 含 done 那一拍); L2 埠不發請求。
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
  localparam int CAP = QD + 4 * NCU + 1;
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

module npu_dma_mc #(parameter int NDC = 2, OUT = 8, DW = 512) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready, input logic [7:0] cmd_id, input logic cmd_st, input logic [1:0] cmd_fmt,
  input  logic [31:0] cmd_ext, input logic [25:0] cmd_l2, input logic [15:0] cmd_n,
  input  logic [15:0] cmd_cols, input logic [25:0] cmd_stride, input logic cmd_ind, input logic [25:0] cmd_list,
  output logic cpl_valid, input logic cpl_ready, output logic [7:0] cpl_id,
  output logic busy,
  output logic h_rd_req, output logic [25:0] h_rd_line, input logic h_rd_gnt, input logic [DW-1:0] h_rd_data,
  output logic h_wr_req, output logic [25:0] h_wr_line, output logic [DW-1:0] h_wr_data, input logic h_wr_gnt,
  output logic m_rd_req, output logic [31:0] m_rd_addr, input logic m_rd_gnt, input logic m_rd_rvalid,
  input  logic [DW-1:0] m_rd_rdata,
  output logic m_wr_req, output logic [31:0] m_wr_addr, output logic [DW-1:0] m_wr_data, input logic m_wr_gnt
);
  // 未回報命令 p <= NDC; 其中 q 個已完成待回報 (cpl_valid = q > 0, id 任意); 其餘執行中 (busy)
  (* anyseq *) logic rdy_any, done_any;
  (* anyseq *) logic [7:0] id_any;
  logic [7:0] p, q;
  assign cmd_ready = rdy_any && (p < 8'(NDC));
  assign cpl_valid = (q != 0); assign cpl_id = id_any;
  assign busy = (p != q);
  assign h_rd_req = 0; assign h_rd_line = 0; assign h_wr_req = 0; assign h_wr_line = 0; assign h_wr_data = 0;
  assign m_rd_req = 0; assign m_rd_addr = 0; assign m_wr_req = 0; assign m_wr_addr = 0; assign m_wr_data = 0;
  wire fin = done_any && (p != q);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin p <= 0; q <= 0; end
    else begin
      p <= p + 8'(cmd_valid && cmd_ready) - 8'(cpl_valid && cpl_ready);
      q <= q + 8'(fin) - 8'(cpl_valid && cpl_ready);
    end
endmodule

module npu_vec_eng #(parameter int NV = 8) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready, input logic [25:0] cmd_line, input logic [15:0] cmd_n,
  output logic done, derr, busy,
  output logic rd_req, output logic [25:0] rd_line, input logic rd_gnt, input logic [511:0] rd_data,
  output logic wr_req, output logic [25:0] wr_line, output logic [511:0] wr_data, input logic wr_gnt
);
  (* anyseq *) logic fin_any, err_any;
  logic act;
  assign cmd_ready = !act && !done; assign busy = act || done; assign derr = err_any;
  assign rd_req = 0; assign rd_line = 0; assign wr_req = 0; assign wr_line = 0; assign wr_data = 0;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin act <= 0; done <= 0; end
    else begin
      done <= act && fin_any;
      if (cmd_valid && cmd_ready) act <= 1;
      else if (act && fin_any) act <= 0;
    end
endmodule

module npu_llm_eng #(parameter int NP = 64, NSEQ = 8, NSLOT = 16, BS = 16, RW = 3, NEXP = 16, PW = 14, TW = 12) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready, input logic cmd_op,
  input  logic [25:0] cmd_a, cmd_b, cmd_c, input logic [31:0] cmd_ext, input logic [15:0] cmd_k,
  output logic done, busy,
  output logic rd_req, output logic [25:0] rd_line, input logic rd_gnt, input logic [511:0] rd_data,
  output logic wr_req, output logic [25:0] wr_line, output logic [511:0] wr_data, input logic wr_gnt
);
  (* anyseq *) logic fin_any;
  logic act;
  assign cmd_ready = !act; assign busy = act || done;
  assign rd_req = 0; assign rd_line = 0; assign wr_req = 0; assign wr_line = 0; assign wr_data = 0;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin act <= 0; done <= 0; end
    else begin
      done <= act && fin_any;
      if (cmd_valid && cmd_ready) act <= 1;
      else if (act && fin_any) act <= 0;
    end
endmodule
