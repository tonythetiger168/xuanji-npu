// SPDX-License-Identifier: Apache-2.0
// npu_vec 的 SVA: L2 交握 (只有 LD / ST 發請求、讀寫不同時、ready 跟隨 grant)、暫存器檔只在寫入該暫存器的拍改變
// (clock gating 安全)、err 黏著且只因非法指令設定; 功能 coverage (每個 op / 格式 / part / SFU 函式 / 等待)。
// 以 bind 掛到 DUT, .* 連接同名訊號。以未閘控的 clk 取樣。
module npu_vec_sva
  import npu_vec_pkg::*;
#(parameter int NV = 8) (
  input logic clk, rst_n, test_en,
  input logic ins_valid, ins_ready, err, err_clr,
  input npu_vec_ins_t ins,
  input logic rd_req, rd_gnt, wr_req, wr_gnt,
  input logic illegal, fire, we,
  input logic [NV-1:0][VL-1:0][31:0] vr
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  // TB 端的約定: 指令在 ready 之前保持
  m_hold: assume property (ins_valid && !ins_ready |=> ins_valid && $stable(ins));

  a_rd_ld:   assert property (rd_req |-> ins_valid && ins.op == VOP_LD && !illegal);
  a_wr_st:   assert property (wr_req |-> ins_valid && ins.op == VOP_ST && !illegal);
  a_rd_wr:   assert property (!(rd_req && wr_req));
  a_rdy_rd:  assert property (rd_req |-> ins_ready == rd_gnt);
  a_rdy_wr:  assert property (wr_req |-> ins_ready == wr_gnt);
  a_rdy_alu: assert property (ins_valid && !rd_req && !wr_req |-> ins_ready);
  a_err_set: assert property (fire && illegal |=> err);
  a_err_stk: assert property (err && !err_clr |=> err);
  a_err_clr: assert property (err_clr && !(fire && illegal) |=> !err);
  a_err_why: assert property (!err ##1 err |-> $past(fire && illegal));
  c_err_clr: cover property (err && err_clr);
  a_we:      assert property (we |-> fire && !illegal && ins.op != VOP_NOP && ins.op != VOP_ST);
  for (genvar r = 0; r < NV; r++) begin : g_r
    a_reg_stable: assert property (!(we && ins.vd == 3'(r)) |=> $stable(vr[r]));
  end

  c_rd_wait: cover property (rd_req && !rd_gnt);
  c_wr_wait: cover property (wr_req && !wr_gnt);
  c_back:    cover property (fire ##1 fire ##1 fire);          // 連續發射
  c_err:     cover property (fire && illegal && err);          // 已有 err 時再遇到非法指令
  c_te_we:   cover property (test_en && we);

  covergroup cg_ins @(posedge clk iff (rst_n && fire));
    cp_op:   coverpoint ins.op iff (!illegal) { bins op[] = {[VOP_NOP:VOP_RMAX]}; }
    cp_ld:   coverpoint ins.fn iff (!illegal && ins.op == VOP_LD) { bins f[] = {[0:6]}; }
    cp_st:   coverpoint ins.fn iff (!illegal && ins.op == VOP_ST) { bins f[] = {[0:1]}; }
    cp_sfu:  coverpoint ins.fn iff (!illegal && ins.op == VOP_SFU) { bins f[] = {[0:5]}; }
    // 每種寬度的每個 part (32 bit 格式不看 part)
    cp_p16:  coverpoint ins.part[0]   iff (!illegal && ins.op == VOP_LD && ins.fn inside {1, 2}) { bins p[] = {[0:1]}; }
    cp_p8:   coverpoint ins.part[1:0] iff (!illegal && ins.op == VOP_LD && ins.fn inside {4, 5}) { bins p[] = {[0:3]}; }
    cp_p4:   coverpoint ins.part      iff (!illegal && ins.op == VOP_LD && ins.fn == 6) { bins p[] = {[0:7]}; }
    cp_scale: coverpoint ins.imm[7:0] iff (!illegal && ins.op == VOP_LD && ins.fn >= 4) {
      bins zero = {0}; bins low = {[1:126]}; bins one = {127}; bins high = {[128:254]}; bins nan = {255};
    }
    cp_vd:   coverpoint ins.vd iff (we) { bins r[] = {[0:NV-1]}; }
    cp_ill:  coverpoint {ins.op > VOP_RMAX, ins.op == VOP_LD, ins.op == VOP_ST, ins.op == VOP_SFU} iff (illegal) {
      bins op = {4'b1000}; bins ld = {4'b0100}; bins st = {4'b0010}; bins sfu = {4'b0001};
    }
  endgroup
  cg_ins u_cg = new;
endmodule

bind npu_vec npu_vec_sva #(.NV(NV)) u_sva (.*);
