// SPDX-License-Identifier: Apache-2.0
// npu_gemm_cu 的 SVA: L2 主介面協定、命令 / 完成語意與功能 coverage。以 bind 掛到 DUT, .* 連接同名訊號。
module npu_gemm_cu_sva #(parameter int N = 8) (
  input logic clk, rst_n,
  input logic cmd_valid, cmd_ready, done,
  input logic [15:0] cmd_k,
  input logic rd_req, rd_gnt, wr_req, wr_gnt,
  input logic [25:0] rd_line, wr_line,
  input logic [511:0] wr_data,
  input logic [2:0] st,
  input logic [15:0] k_left,
  input logic mma_busy, mma_in_valid, mma_in_ready, mma_out_valid, mma_out_ready
);
  localparam logic [2:0] G_IDLE = 3'd0, G_RD_A = 3'd1, G_RD_B = 3'd2, G_FEED = 3'd3, G_OUT = 3'd4, G_WR = 3'd5;
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  // 環境假設: K >= 1
  m_k_nonzero:   assume property (cmd_valid && cmd_ready |-> cmd_k != 0);

  a_state_legal: assert property (st inside {G_IDLE, G_RD_A, G_RD_B, G_FEED, G_OUT, G_WR});
  a_ready_idle:  assert property (cmd_ready == (st == G_IDLE));
  a_mma_idle:    assert property (st == G_IDLE |-> !mma_busy);           // 新命令開始時 npu_mma 必在 idle
  // 等待 grant 時請求與位址 / 資料保持不變
  a_rd_hold:     assert property (rd_req && !rd_gnt |=> rd_req && $stable(rd_line));
  a_wr_hold:     assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  a_no_rd_wr:    assert property (!(rd_req && wr_req));
  // done 只在最後一次寫回 grant 的下一拍出現, 且之後回到 idle
  a_done_after_wr: assert property (done |-> $past(st == G_WR && wr_gnt && !mma_busy) && st == G_IDLE);
  // 餵資料時 npu_mma 一定在 RUN (in_ready), 不會有被拒絕的 in_valid
  a_feed_ready:  assert property (mma_in_valid |-> mma_in_ready);
  a_k_left:      assert property (st == G_FEED |-> k_left != 0);

  // ---------------- cover / covergroup ----------------
  c_rd_stall:    cover property (rd_req && !rd_gnt);
  c_wr_stall:    cover property (wr_req && !wr_gnt);
  c_back_to_back: cover property (done ##1 cmd_valid && cmd_ready);
  c_reset_busy:  cover property (@(posedge clk) disable iff (1'b0) rst_n && st != G_IDLE ##1 !rst_n);

  localparam int SPL = 64 / N;                 // 每個 line 的 k 步數
  logic [15:0] k_q;
  logic [$clog2(SPL)-1:0] k_mod;               // K mod SPL
  assign k_mod = k_q[$clog2(SPL)-1:0];
  always_ff @(posedge clk or negedge rst_n) if (!rst_n) k_q <= '0; else if (cmd_valid && cmd_ready) k_q <= cmd_k;
  covergroup cg_cmd @(posedge clk iff (rst_n && done));
    // K 對齊 line 邊界 (64/N 步) 與否、單一 line、多 line
    cp_k: coverpoint k_q { bins one = {1}; bins part_line = {[2:SPL-1]}; bins one_line = {SPL}; bins line_p1 = {SPL+1};
                           bins multi = {[SPL+2:$]}; }
    cp_k_mod: coverpoint k_mod { bins aligned = {0}; bins other = {[1:SPL-1]}; }
  endgroup
  covergroup cg_stall @(posedge clk iff rst_n);
    cp_rd: coverpoint {st, rd_gnt} iff (rd_req) { bins a_wait = {4'b0010}; bins a_go = {4'b0011};     // {G_RD_A, gnt}
                                                 bins b_wait = {4'b0100}; bins b_go = {4'b0101}; }    // {G_RD_B, gnt}
    cp_wr: coverpoint {wr_gnt, mma_busy} iff (wr_req) { bins stall = {2'b00, 2'b01}; bins mid_line = {2'b11}; bins last_line = {2'b10}; }
  endgroup
  cg_cmd   u_cg_cmd   = new;
  cg_stall u_cg_stall = new;
endmodule

bind npu_gemm_cu npu_gemm_cu_sva #(.N(N)) u_sva (.*);
