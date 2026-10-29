// SPDX-License-Identifier: Apache-2.0
// npu_core 形式驗證用的子模組替身 (只遵守介面契約):
//   npu_gemm_cu: idle 時 cmd_ready = 1 並接受命令; 忙碌任意拍數後回到 idle, 回到 idle 那拍 done 為一拍脈衝
//                (與 RTL 相同: st <= G_IDLE 與 done <= 1 同拍); feeding 只在忙碌時。CU 本身由 NPU flow 驗證。
//   npu_l2_xbar: grant 任意但只給有請求者 (已另外形式證明); 資料任意。
//   clock_gate:  直通 (閘控安全由模擬 SVA 與 orca_npu_ss 的證明涵蓋)。
`include "orca_pkg.sv"
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule

module npu_gemm_cu #(parameter int N = 8, localparam int RP = (N == 32) ? 2 : 1) (
  input  logic clk, rst_n, test_en,
  input  logic cmd_valid, output logic cmd_ready,
  input  logic [25:0] cmd_a_line, cmd_b_line, cmd_c_line, input logic [15:0] cmd_k,
  input  logic cmd_a_signed, cmd_b_signed,
  output logic done, feeding,
  output logic [RP-1:0] rd_req, output logic [RP-1:0][25:0] rd_line, input logic [RP-1:0] rd_gnt,
  input  logic [RP-1:0][511:0] rd_data,
  output logic wr_req, output logic [25:0] wr_line, output logic [511:0] wr_data, input logic wr_gnt
);
  (* anyseq *) logic fin_any, feed_any;
  logic act;
  assign cmd_ready = !act; assign feeding = act && feed_any;
  assign rd_req = 1'b0; assign rd_line = '0; assign wr_req = 1'b0; assign wr_line = '0; assign wr_data = '0;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin act <= 1'b0; done <= 1'b0; end
    else begin
      done <= 1'b0;
      if (!act && cmd_valid) act <= 1'b1;
      else if (act && fin_any) begin act <= 1'b0; done <= 1'b1; end
    end
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
