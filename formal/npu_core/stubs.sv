// SPDX-License-Identifier: Apache-2.0
// npu_core 形式驗證用的子模組替身 (只遵守介面契約):
//   npu_gemm_cu: 輸出任意; 介面契約 (最多持有 3 個命令、閒置時可接受、busy = 持有命令、done 依受理順序帶回 tag 並
//                保持到 done_ready) 由 npu_core.sv 的 FORMAL 區塊 assume。CU 本身由 NPU flow 驗證。
//   npu_l2_xbar: grant 任意但只給有請求者 (已另外形式證明); 資料任意。
//   clock_gate:  直通 (閘控安全由模擬 SVA 與 orca_npu_ss 的證明涵蓋)。
`include "npu_pkg.sv"
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule

module npu_gemm_cu #(parameter int N = 8, localparam int RP = (N == 32) ? 2 : 1) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready,
  input  logic [25:0] cmd_a_line, cmd_b_line, cmd_c_line, input logic [15:0] cmd_k,
  input  logic cmd_a_signed, cmd_b_signed,
  input  logic cmd_acc,
  input  logic [7:0] cmd_tag,
  output logic done, input logic done_ready, output logic [7:0] done_tag,
  output logic busy, feeding,
  output logic [RP-1:0] rd_req, output logic [RP-1:0][25:0] rd_line, input logic [RP-1:0] rd_gnt,
  input  logic [RP-1:0][511:0] rd_data,
  output logic wr_req, output logic [25:0] wr_line, output logic [511:0] wr_data, input logic wr_gnt
);
  // 輸出全部任意; 契約 (持有數 / 依序完成 / done 保持) 由 npu_core 的 FORMAL 區塊以 assume 表達
  (* anyseq *) logic r_any, d_any, b_any, f_any;
  (* anyseq *) logic [7:0] t_any;
  assign cmd_ready = r_any; assign done = d_any; assign done_tag = t_any; assign busy = b_any; assign feeding = f_any;
  assign rd_req = 1'b0; assign rd_line = '0; assign wr_req = 1'b0; assign wr_line = '0; assign wr_data = '0;
endmodule

module npu_l2_xbar #(parameter int NM = 5, NW = NM, NB = 4, DEPTH = 1024, LDW = 512, LS_IDLE = 64, LS_WAKE = 2) (
  input  logic clk, rst_n,
  input  logic [NM-1:0] rd_req, input logic [NM-1:0][25:0] rd_line,
  output logic [NM-1:0] rd_gnt, output logic [NM-1:0][LDW-1:0] rd_data,
  input  logic [NW-1:0] wr_req, input logic [NW-1:0][25:0] wr_line, input logic [NW-1:0][LDW-1:0] wr_data,
  output logic [NW-1:0] wr_gnt,
  output logic [NB-1:0] bank_ls
);
  (* anyseq *) logic [NM-1:0] rg;
  (* anyseq *) logic [NB-1:0] ls;
  assign bank_ls = ls;
  (* anyseq *) logic [NW-1:0] wg;
  assign rd_gnt = rd_req & rg; assign wr_gnt = wr_req & wg; assign rd_data = '0;
endmodule
