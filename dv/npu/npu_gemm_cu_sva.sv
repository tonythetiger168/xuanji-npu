// SPDX-License-Identifier: Apache-2.0
// npu_gemm_cu 的 SVA: L2 主介面協定、預取緩衝不變式、命令 / 完成語意與功能 coverage。
// 以 bind 掛到 DUT, .* 連接同名訊號。
module npu_gemm_cu_sva #(parameter int N = 8) (
  input logic clk, rst_n,
  input logic cmd_valid, cmd_ready, done,
  input logic [15:0] cmd_k,
  input logic rd_req, rd_gnt, wr_req, wr_gnt,
  input logic [25:0] rd_line, wr_line,
  input logic [511:0] wr_data,
  input logic [1:0] st,
  input logic [15:0] k_left,
  input logic [$clog2((65535 + 64 / N - 1) / (64 / N) + 1)-1:0] f_left,
  input logic f_b,
  input logic [$clog2(64 / N)-1:0] step,
  input logic [1:0] cnt,
  input logic feed, pop, push,
  input logic mma_busy, mma_in_valid, mma_in_ready, mma_out_valid, mma_out_ready
);
  localparam int SPL = 64 / N;
  localparam logic [1:0] G_IDLE = 2'd0, G_OUT = 2'd1, G_WR = 2'd2, G_RUN = 2'd3;
  default clocking cb @(posedge clk); endclocking
  // clk 可能是閘控時脈: 非同步 reset 若落在兩個取樣沿之間, 取樣時 rst_n 已回到 1, 但 $past 跨過了 reset。
  // rst_seen 在 reset 時非同步設定、下一個取樣沿清除, 讓跨過 reset 的嘗試被停用。
  logic rst_seen;
  always_ff @(posedge clk or negedge rst_n) rst_seen <= !rst_n ? 1'b1 : 1'b0;
  default disable iff (!rst_n || rst_seen);

  // 環境假設: K >= 1
  m_k_nonzero:   assume property (cmd_valid && cmd_ready |-> cmd_k != 0);

  a_ready_idle:  assert property (cmd_ready == (st == G_IDLE));
  a_mma_idle:    assert property (st == G_IDLE |-> !mma_busy);           // 新命令開始時 npu_mma 必在 idle
  // 等待 grant 時請求與位址 / 資料保持不變
  a_rd_hold:     assert property (rd_req && !rd_gnt |=> rd_req && $stable(rd_line));
  a_wr_hold:     assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  a_no_rd_wr:    assert property (!(rd_req && wr_req));
  a_gnt_req:     assert property ((rd_gnt |-> rd_req) and (wr_gnt |-> wr_req));   // 環境 (xbar) 只 grant 有請求者
  // 緩衝: 不溢位、不在空時取用; 餵資料時 npu_mma 必就緒
  a_cnt_max:     assert property (cnt <= 2'd2);
  a_no_overflow: assert property (cnt == 2'd2 |-> !(rd_req && !f_b));
  a_feed_ready:  assert property (mma_in_valid |-> mma_in_ready);
  a_pop_nonempty: assert property (pop |-> cnt != 2'd0);
  // line 帳: 剩餘 k 步 (從緩衝頭的 step 起) 所需的 line 數 = 未讀 line 對數 + 緩衝內完整項數
  a_line_acct:   assert property (st == G_RUN |-> (32'(k_left) + 32'(step) + SPL - 1) / SPL == 32'(f_left) + 32'(cnt));
  a_k_left:      assert property (st == G_RUN |-> k_left != 0);
  // done 只在最後一次寫回 grant 的下一拍出現, 且之後回到 idle
  a_done_after_wr: assert property (done |-> $past(st == G_WR && wr_gnt && !mma_busy) && st == G_IDLE);

  // ---------------- cover / covergroup ----------------
  c_rd_stall:    cover property (rd_req && !rd_gnt);
  c_wr_stall:    cover property (wr_req && !wr_gnt);
  c_starve:      cover property (st == G_RUN && cnt == 2'd0 && !f_b);    // 緩衝空 (grant 不足造成空拍)
  c_buf_full:    cover property (st == G_RUN && cnt == 2'd2);
  c_push_pop:    cover property (push && pop);
  c_back_to_back: cover property (done ##1 cmd_valid && cmd_ready);
  c_reset_busy:  cover property (@(posedge clk) disable iff (1'b0) rst_n && st != G_IDLE ##1 !rst_n);

  logic [15:0] k_q;
  logic [$clog2(SPL)-1:0] k_mod;               // K mod SPL
  assign k_mod = k_q[$clog2(SPL)-1:0];
  always_ff @(posedge clk or negedge rst_n) if (!rst_n) k_q <= '0; else if (cmd_valid && cmd_ready) k_q <= cmd_k;
  covergroup cg_cmd @(posedge clk iff (rst_n && done));
    // K 對齊 line 邊界 (SPL 步) 與否、單一 line、多 line
    cp_k: coverpoint k_q { bins one = {1}; bins part_line = {[2:SPL-1]}; bins one_line = {SPL}; bins line_p1 = {SPL+1};
                           bins multi = {[SPL+2:$]}; }
    cp_k_mod: coverpoint k_mod { bins aligned = {0}; bins other = {[1:SPL-1]}; }
  endgroup
  covergroup cg_buf @(posedge clk iff (rst_n && st == G_RUN));
    cp_cnt:  coverpoint cnt { bins c[] = {0, 1, 2}; }      // 1 / 2 時本拍必餵入 (feed = RUN && cnt != 0)
    cp_rd:   coverpoint {rd_req, rd_gnt, f_b} { bins a_wait = {3'b100}; bins a_go = {3'b110}; bins b_wait = {3'b101}; bins b_go = {3'b111}; }
  endgroup
  covergroup cg_wr @(posedge clk iff (rst_n && wr_req));
    cp_wr: coverpoint {wr_gnt, mma_busy} { bins stall = {2'b00, 2'b01}; bins mid_line = {2'b11}; bins last_line = {2'b10}; }
  endgroup
  cg_cmd u_cg_cmd = new;
  cg_buf u_cg_buf = new;
  cg_wr  u_cg_wr  = new;
endmodule

bind npu_gemm_cu npu_gemm_cu_sva #(.N(N)) u_sva (.*);
