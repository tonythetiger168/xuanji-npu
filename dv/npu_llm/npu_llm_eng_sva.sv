// SPDX-License-Identifier: Apache-2.0
// npu_llm_eng 的 SVA: L2 請求 (讀: 位址; 寫: 位址與資料) 保持到 grant、讀寫不同時、命令只在閒置時接受、
// done 單拍且只在命令之後、閘控安全; coverage: 讀 / 寫等待 grant、MAP 跨多個 line、MOE 兩趟、讀 − 改 − 寫。
module npu_llm_eng_sva (
  input logic clk, rst_n, en,
  input logic cmd_valid, cmd_ready, done, busy, op,
  input logic rd_req, rd_gnt, wr_req, wr_gnt,
  input logic [25:0] rd_line, wr_line,
  input logic [511:0] wr_data,
  input logic [4:0] st,
  input logic kv_bad, p_cmd_ready, d_cmd_ready, d_in_ready, d_out_valid, d_cnt_valid,
  input logic [2:0] sub
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  a_rd_hold: assert property (rd_req && !rd_gnt |=> rd_req && $stable(rd_line));
  a_wr_hold: assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  a_rw_excl: assert property (!(rd_req && wr_req));
  a_idle:    assert property (cmd_ready == (st == 5'd0) && (st == 5'd0 || busy));
  a_done_1:  assert property (done |=> !done);
  a_done:    assert property (done |-> $past(busy));
  a_quiet:   assert property (!en |-> !rd_req && !wr_req && !done);
  // 引擎依序使用子模組, 子模組在被使用時必定就緒 (RTL 依此省略等待)
  localparam logic [4:0] S_KCMD = 5'd3, S_OCMD = 5'd10, S_OFEED = 5'd12, S_OOUT = 5'd13, S_OCNT = 5'd18;
  a_pager_rdy: assert property (st == S_KCMD && !kv_bad && sub != 3'd4 |-> p_cmd_ready);
  a_disp_rdy:  assert property (st == S_OCMD |-> d_cmd_ready);
  c_in_wait:   cover property (st == S_OFEED && !d_in_ready);         // 等 dispatch 清除計數
  a_out_v:     assert property (st == S_OOUT |-> d_out_valid);
  a_cnt_v:     assert property (st == S_OCNT |-> d_cnt_valid);

  c_rd_wait: cover property (rd_req && !rd_gnt);
  c_wr_wait: cover property (wr_req && !wr_gnt);
  c_kv_done: cover property (done && !op);
  c_moe_done: cover property (done && op);
endmodule

bind npu_llm_eng npu_llm_eng_sva u_sva (.*);
