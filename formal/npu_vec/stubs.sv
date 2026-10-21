// SPDX-License-Identifier: Apache-2.0
// npu_vec 形式驗證用的替身: 浮點運算元件輸出任意值 (資料路徑由模擬的位元一致比對驗證; 這裡只證明控制:
// L2 交握、ready、err、暫存器檔的寫入紀律, 與資料值無關)。clock_gate 直通 (閘控由模擬 SVA 與 mutation 涵蓋)。
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule
module fp_add #(parameter int E = 8, parameter int M = 23) (input logic [E+M:0] a, b, output logic [E+M:0] y);
  (* anyseq *) logic [E+M:0] v; assign y = v;
endmodule
module fp_mul #(parameter int E = 8, parameter int M = 23) (input logic [E+M:0] a, b, output logic [E+M:0] y);
  (* anyseq *) logic [E+M:0] v; assign y = v;
endmodule
module fp_max (input logic [31:0] a, b, output logic [31:0] y);
  (* anyseq *) logic [31:0] v; assign y = v;
endmodule
module sfu (input logic [2:0] op, input logic [31:0] x, output logic [31:0] y);
  (* anyseq *) logic [31:0] v; assign y = v;
endmodule
module fp_widen #(parameter int E = 8, parameter int MI = 7, parameter int MO = 23) (input logic [E+MI:0] a, output logic [E+MO:0] y);
  (* anyseq *) logic [E+MO:0] v; assign y = v;
endmodule
module fp_narrow #(parameter int E = 8, parameter int MI = 23, parameter int MO = 7) (input logic [E+MI:0] a, output logic [E+MO:0] y);
  (* anyseq *) logic [E+MO:0] v; assign y = v;
endmodule
module fp_small_dec #(parameter int E = 4, parameter int M = 3, parameter int BIAS = 7, parameter int SP = 1) (
  input logic [E+M:0] x, output logic [8+M:0] y);
  (* anyseq *) logic [8+M:0] v; assign y = v;
endmodule
module fp_mx_scale #(parameter int M = 3) (input logic [8+M:0] x, input logic [7:0] scale, output logic [8+M:0] y);
  (* anyseq *) logic [8+M:0] v; assign y = v;
endmodule
module fp_from_int (input logic [31:0] a, output logic [31:0] y);
  (* anyseq *) logic [31:0] v; assign y = v;
endmodule
