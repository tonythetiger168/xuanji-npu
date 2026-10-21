// SPDX-License-Identifier: Apache-2.0
// npu_gemm_cu 的 SVA: L2 主介面協定、預取緩衝不變式、命令 / 完成語意與功能 coverage。
// 以 bind 掛到 DUT, .* 連接同名訊號。
module npu_gemm_cu_sva #(parameter int N = 8,
  localparam int RP = (N == 32) ? 2 : 1, localparam int PF = (RP == 2) ? 4 : 2, localparam int PW = $clog2(PF)) (
  input logic clk, rst_n,
  input logic cmd_valid, cmd_ready, done,
  input logic [15:0] cmd_k,
  input logic [RP-1:0] rd_req, rd_gnt,
  input logic wr_req, wr_gnt,
  input logic [RP-1:0][25:0] rd_line,
  input logic [25:0] wr_line,
  input logic [511:0] wr_data,
  input logic [1:0] st,
  input logic [15:0] k_left,
  input logic [$clog2((65535 + 64 / N - 1) / (64 / N) + 1)-1:0] fa, fb,
  input logic [$clog2(64 / N)-1:0] step,
  input logic [PW:0] a_cnt, b_cnt,
  input logic have, a_req, b_req, a_gnt, b_gnt,
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
  // 等待 grant 時請求與位址 / 資料保持不變 (每個讀取埠)
  for (genvar p = 0; p < RP; p++) begin : g_port
    a_rd_hold:   assert property (rd_req[p] && !rd_gnt[p] |=> rd_req[p] && $stable(rd_line[p]));
    a_gnt_req:   assert property (rd_gnt[p] |-> rd_req[p]);                 // 環境 (xbar) 只 grant 有請求者
  end
  a_wr_hold:     assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  a_no_rd_wr:    assert property (!(rd_req != '0 && wr_req));
  a_wgnt_req:    assert property (wr_gnt |-> wr_req);
  a_trk_gnt:     assert property ((a_gnt |-> a_req) and (b_gnt |-> b_req));
  // 緩衝: 不溢位、不在空時取用; 餵資料時 npu_mma 必就緒
  a_cnt_max:     assert property (a_cnt <= (PW+1)'(PF) && b_cnt <= (PW+1)'(PF));
  a_no_overflow: assert property ((a_cnt == (PW+1)'(PF) |-> !a_req) and (b_cnt == (PW+1)'(PF) |-> !b_req));
  a_feed_ready:  assert property (mma_in_valid |-> mma_in_ready);
  a_pop_nonempty: assert property (pop |-> have);
  a_have:        assert property (have == (a_cnt != 0 && b_cnt != 0));
  // line 帳: 剩餘 k 步 (從緩衝頭的 step 起) 所需的 line 數 = 該軌未讀 line 數 + 該軌緩衝內項數 (A / B 各自成立)
  a_line_acct:   assert property (st == G_RUN |-> (32'(k_left) + 32'(step) + SPL - 1) / SPL == 32'(fa) + 32'(a_cnt)
                                                && (32'(k_left) + 32'(step) + SPL - 1) / SPL == 32'(fb) + 32'(b_cnt));
  if (RP == 1) begin : g_share
    // 共用埠: A、B 交替, A 最多領先 B 一項
    a_alt:       assert property (a_cnt == b_cnt || a_cnt == b_cnt + 1'b1);
  end
  a_k_left:      assert property (st == G_RUN |-> k_left != 0);
  // done 只在最後一次寫回 grant 的下一拍出現, 且之後回到 idle
  a_done_after_wr: assert property (done |-> $past(st == G_WR && wr_gnt && !mma_busy) && st == G_IDLE);

  // ---------------- cover / covergroup ----------------
  c_rd_stall:    cover property (rd_req != '0 && rd_gnt != rd_req);
  c_wr_stall:    cover property (wr_req && !wr_gnt);
  c_starve:      cover property (st == G_RUN && !have);                    // 緩衝空 (grant 不足造成空拍)
  c_buf_full:    cover property (st == G_RUN && a_cnt == (PW+1)'(PF) && b_cnt == (PW+1)'(PF));
  if (RP == 2) begin : g_dual
    c_both_gnt:  cover property (a_gnt && b_gnt);                           // 兩埠同拍讀
    c_a_ahead:   cover property (st == G_RUN && a_cnt > b_cnt + 1'b1);            // A 領先 B 超過一項
    c_b_ahead:   cover property (st == G_RUN && b_cnt > a_cnt);
  end
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
    cp_ca:   coverpoint a_cnt { bins c[] = {[0:PF]}; }
    cp_cb:   coverpoint b_cnt { bins c[] = {[0:PF]}; }
    cp_a:    coverpoint {a_req, a_gnt} { bins stall = {2'b10}; bins go = {2'b11}; }
    cp_b:    coverpoint {b_req, b_gnt} { bins stall = {2'b10}; bins go = {2'b11}; }
  endgroup
  covergroup cg_wr @(posedge clk iff (rst_n && wr_req));
    cp_wr: coverpoint {wr_gnt, mma_busy} { bins stall = {2'b00, 2'b01}; bins mid_line = {2'b11}; bins last_line = {2'b10}; }
  endgroup
  cg_cmd u_cg_cmd = new;
  cg_buf u_cg_buf = new;
  cg_wr  u_cg_wr  = new;
endmodule

bind npu_gemm_cu npu_gemm_cu_sva #(.N(N)) u_sva (.*);
