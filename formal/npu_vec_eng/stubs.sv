// SPDX-License-Identifier: Apache-2.0
// npu_vec_eng 形式驗證用的替身: npu_vec 只遵守介面契約 — LD 發讀取請求 (line = 指令的 line) 直到 grant、ST 發寫入請求
// 直到 grant、其他指令一拍完成; err 任意 (npu_vec 本身已形式證明)。clock_gate 直通。
`include "npu_vec_pkg.sv"
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule
module npu_vec import npu_vec_pkg::*; #(parameter int NV = 8) (
  input  logic clk, rst_n, test_en,
  input  logic ins_valid, output logic ins_ready, input npu_vec_ins_t ins, output logic err, input logic err_clr,
  output logic rd_req, output logic [25:0] rd_line, input logic rd_gnt, input logic [511:0] rd_data,
  output logic wr_req, output logic [25:0] wr_line, output logic [511:0] wr_data, input logic wr_gnt
);
  (* anyseq *) logic e_any;
  (* anyseq *) logic [511:0] d_any;
  wire ld = ins_valid && ins.op == VOP_LD, st = ins_valid && ins.op == VOP_ST;
  assign rd_req = ld; assign rd_line = ins.line; assign wr_req = st; assign wr_line = ins.line; assign wr_data = d_any;
  assign ins_ready = ld ? rd_gnt : st ? wr_gnt : 1'b1;
  assign err = e_any;
endmodule
