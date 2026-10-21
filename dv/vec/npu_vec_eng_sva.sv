// SPDX-License-Identifier: Apache-2.0
// npu_vec_eng 的 SVA: 命令只在閒置時接受、每個命令恰好一個 done、L2 請求保持到 grant (位址 / 資料不變)、
// 取指與 npu_vec 的讀取不衝突、閘控安全 (時脈關閉時沒有活動); coverage: 取指與 LD 競爭、預取、各種程式長度。
// 以 bind 掛到 DUT, .* 連接同名訊號 (以未閘控的 clk 取樣)。
module npu_vec_eng_sva (
  input logic clk, rst_n, en,
  input logic cmd_valid, cmd_ready, done, busy, act,
  input logic rd_req, rd_gnt, wr_req, wr_gnt,
  input logic [25:0] rd_line, wr_line,
  input logic [511:0] wr_data,
  input logic f_sel, f_want, v_rd_req, fire, ln_end,
  input logic [1:0] fv,
  input logic [15:0] cmd_n, ni
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  m_cmd_n:    assume property (cmd_valid |-> cmd_n != '0);

  a_idle:     assert property (cmd_valid && cmd_ready |-> !act);
  a_done:     assert property (done |-> $past(act) && !act);
  a_one_done: assert property (done |=> !done);
  a_rd_hold:  assert property (rd_req && !rd_gnt |=> rd_req && $stable(rd_line));
  a_wr_hold:  assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  a_quiet:    assert property (!en |-> !rd_req && !wr_req && !done);
  a_fetch:    assert property (f_sel |-> act && f_want);
  a_busy:     assert property (act |-> busy);

  c_contend:  cover property (f_want && v_rd_req);                  // 取指與 LD 同時想用讀取埠
  c_pf:       cover property (fire && fv == 2'b11);                 // 執行時下一個 line 已預取
  c_fetch_wait: cover property (f_sel && !rd_gnt ##1 v_rd_req);     // 取指等待中 npu_vec 也要讀
  c_starve:   cover property (act && ni != '0 && fv == 2'b00);      // 指令用完, 等取指
  c_n1:       cover property (cmd_valid && cmd_ready && cmd_n == 16'd1);
  c_n4:       cover property (cmd_valid && cmd_ready && cmd_n == 16'd4);
  c_n5:       cover property (cmd_valid && cmd_ready && cmd_n == 16'd5);
endmodule

bind npu_vec_eng npu_vec_eng_sva u_sva (.*);
