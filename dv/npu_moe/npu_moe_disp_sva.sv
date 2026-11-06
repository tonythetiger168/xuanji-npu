// SPDX-License-Identifier: Apache-2.0
// npu_moe_disp 的 SVA: 命令只在閒置時接受、輸入只在 P0 / P1 接受、輸出 / 數量等待時保持、drop 時 pos = 0、
// err ⇒ drop、done 只在最後一筆數量之後一拍、閘控安全; coverage: 兩趟模式、容量滿、輸出回壓下送輸入、PFX 期間。
// 以 bind 掛到 DUT, 以未閘控的 clk 取樣。
module npu_moe_disp_sva #(parameter int PW = 9, EW = 3, TW = 8, WW = 16) (
  input logic clk, rst_n, en,
  input logic cmd_valid, cmd_ready, in_valid, in_ready, in_fire, in_last, out_valid, out_ready, out_drop, out_err,
  input logic cnt_valid, cnt_ready, cnt_fire, done, busy, dl, full, bad,
  input logic [2:0] st,
  input logic [PW-1:0] out_pos,
  input logic [EW-1:0] out_exp, cnt_exp,
  input logic [TW-1:0] out_tok,
  input logic [WW-1:0] out_w,
  input logic [PW:0] cnt_n, cnt_off
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  localparam logic [2:0] S_IDLE = 3'd0, S_P0 = 3'd2, S_PFX = 3'd3, S_P1 = 3'd4, S_CNT = 3'd5;

  a_cmd_idle:  assert property (cmd_ready == (st == S_IDLE));
  a_busy:      assert property (busy == (st != S_IDLE));
  a_in_state:  assert property (in_ready |-> st == S_P0 || st == S_P1);
  a_p0_dl:     assert property (st == S_P0 |-> dl);
  a_out_hold:  assert property (out_valid && !out_ready |=> out_valid && $stable({out_pos, out_exp, out_tok, out_w, out_drop, out_err}));
  a_cnt_hold:  assert property (cnt_valid && !cnt_ready |=> cnt_valid && $stable({cnt_exp, cnt_n, cnt_off}));
  a_drop_pos:  assert property (out_valid && out_drop |-> out_pos == '0);
  a_err_drop:  assert property (out_valid && out_err |-> out_drop && out_exp == '0);
  a_dl_nodrop: assert property (out_valid && dl && !out_err |-> !out_drop);
  a_done:      assert property (done |-> $past(cnt_fire) && st == S_IDLE);
  a_done_1:    assert property (done |=> !done);
  a_quiet:     assert property (!en |-> !in_fire && !cnt_fire && st == S_IDLE);

  c_two_pass:  cover property (st == S_P0 && in_fire && in_last ##1 st == S_PFX);
  c_full:      cover property (in_fire && st == S_P1 && full && !bad);
  c_bp_in:     cover property (out_valid && out_ready && in_fire);       // 輸出被取走的同拍接受下一個輸入
  c_bp_stall:  cover property (out_valid && !out_ready && in_valid);
  c_cnt_stall: cover property (cnt_valid && !cnt_ready);
  c_bad_p0:    cover property (st == S_P0 && in_fire && bad);
endmodule

bind npu_moe_disp npu_moe_disp_sva #(.PW(PW), .EW(EW), .TW(TW), .WW(WW)) u_sva (.*);
