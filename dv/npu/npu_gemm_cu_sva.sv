// SPDX-License-Identifier: Apache-2.0
// npu_gemm_cu 的 SVA: L2 主介面協定、預取緩衝不變式、命令 / 完成語意與功能 coverage。
// 以 bind 掛到 DUT, .* 連接同名訊號。
// v6.3.12 命令重疊: 完成順序 / tag 由檢查器從命令埠自己記錄 (f_tag 佇列), 不引用 DUT 的 tag 暫存。
module npu_gemm_cu_sva #(parameter int N = 8,
  localparam int RP = (N == 32) ? 2 : 1, localparam int PF = (RP == 2) ? 4 : 2, localparam int PW = $clog2(PF)) (
  input logic clk, rst_n,
  input logic cmd_valid, cmd_ready, done, done_ready, busy,
  input logic [7:0] cmd_tag, done_tag,
  input logic [15:0] cmd_k,
  input logic [RP-1:0] rd_req, rd_gnt,
  input logic wr_req, wr_gnt,
  input logic [RP-1:0][25:0] rd_line,
  input logic [25:0] wr_line,
  input logic [511:0] wr_data,
  input logic fst,
  input logic [2:0] wst,
  input logic pend, lr, take, feed_last,
  input logic cmd_acc, x_w, w_rq, w_sel, w_gnt, lk_f, f_rq0,
  input logic [25:0] c_ptr,
  input logic [15:0] k_left,
  input logic [$clog2((65535 + 64 / N - 1) / (64 / N) + 1)-1:0] fa, fb,
  input logic [$clog2(64 / N)-1:0] step,
  input logic [PW:0] a_cnt, b_cnt,
  input logic have, a_req, b_req, a_gnt, b_gnt,
  input logic feed, pop, push,
  input logic mma_busy, mma_in_idle, mma_in_valid, mma_in_last, mma_in_ready, mma_out_valid, mma_out_ready
);
  localparam int SPL = 64 / N;
  localparam logic F_IDLE = 1'b0, F_RUN = 1'b1;
  localparam logic [2:0] W_IDLE = 3'd0, W_OUT = 3'd1, W_WR = 3'd2, W_DONE = 3'd3, W_RD = 3'd4;
  default clocking cb @(posedge clk); endclocking
  // clk 可能是閘控時脈: 非同步 reset 若落在兩個取樣沿之間, 取樣時 rst_n 已回到 1, 但 $past 跨過了 reset。
  // rst_seen 在 reset 時非同步設定、下一個取樣沿清除, 讓跨過 reset 的嘗試被停用。
  logic rst_seen;
  always_ff @(posedge clk or negedge rst_n) rst_seen <= !rst_n ? 1'b1 : 1'b0;
  default disable iff (!rst_n || rst_seen);

  // 環境假設: K >= 1
  m_k_nonzero:   assume property (cmd_valid && cmd_ready |-> cmd_k != 0);

  a_ready_idle:  assert property (cmd_ready == (fst == F_IDLE));
  a_mma_idle:    assert property (fst == F_IDLE |-> mma_in_idle);        // 新命令開始時 npu_mma 輸入端必在 idle
  // 等待輸出的 tile 只在寫回器可取時移出; 最後一步餵入時等待槽必空 (npu_mma 輸出端閒置 → 已被取走)
  a_pend_free:   assert property (feed_last |-> !pend);
  a_take:        assert property (take == (pend && (wst == W_IDLE || (wst == W_DONE && done_ready))));
  a_out_ready:   assert property (mma_out_ready == (wst == W_OUT));
  // 寫回器閒置時 npu_mma 不輸出 (輸出的 tile 必已被寫回器取走)
  a_out_owned:   assert property (mma_out_valid |-> wst != W_IDLE);

  // ---- 完成順序 (埠層級模型: 已接受、尚未完成握手的 tag 佇列, 最多 3 個) ----
  logic [2:0][7:0] f_tag;
  logic [1:0]      f_n;
  wire f_acc = cmd_valid && cmd_ready;
  wire f_cpl = done && done_ready;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_tag <= '0; f_n <= '0; end
    else begin
      automatic logic [2:0][7:0] t = f_tag;
      automatic logic [1:0] n = f_n;
      if (f_cpl) begin t = {8'h00, t[2:1]}; n = n - 1'b1; end
      if (f_acc) begin t[n] = cmd_tag; n = n + 1'b1; end
      f_tag <= t; f_n <= n;
    end
  a_n_max:       assert property (f_n <= 2'd3);
  a_acc_room:    assert property (f_acc |-> f_n != 2'd3 || f_cpl);
  a_busy_cnt:    assert property (busy == (f_n != 2'd0));
  a_done_order:  assert property (done |-> f_n != 2'd0 && done_tag == f_tag[0]);
  a_done_hold:   assert property (done && !done_ready |=> done && $stable(done_tag));
  // 等待 grant 時請求與位址 / 資料保持不變 (每個讀取埠)
  for (genvar p = 0; p < RP; p++) begin : g_port
    a_rd_hold:   assert property (rd_req[p] && !rd_gnt[p] |=> rd_req[p] && $stable(rd_line[p]));
    a_gnt_req:   assert property (rd_gnt[p] |-> rd_req[p]);                 // 環境 (xbar) 只 grant 有請求者
  end
  a_wr_hold:     assert property (wr_req && !wr_gnt |=> wr_req && $stable(wr_line) && $stable(wr_data));
  // 命令重疊: 讀下一個 tile 與寫回前一個可同拍 (xbar 的讀 / 寫是不同主端)
  a_wgnt_req:    assert property (wr_gnt |-> wr_req);
  a_trk_gnt:     assert property ((a_gnt |-> a_req) and (b_gnt |-> b_req));
  // 緩衝: 不溢位、不在空時取用; 餵資料時 npu_mma 必就緒
  a_cnt_max:     assert property (a_cnt <= (PW+1)'(PF) && b_cnt <= (PW+1)'(PF));
  a_no_overflow: assert property ((a_cnt == (PW+1)'(PF) |-> !a_req) and (b_cnt == (PW+1)'(PF) |-> !b_req));
  // npu_mma 在輸入 RUN 時就緒, 除了最後一步要等前一個 tile 輸出完
  a_feed_ready:  assert property (mma_in_valid && !mma_in_last |-> mma_in_ready);
  a_pop_nonempty: assert property (pop |-> have);
  a_have:        assert property (have == (a_cnt != 0 && b_cnt != 0));
  // line 帳: 剩餘 k 步 (從緩衝頭的 step 起) 所需的 line 數 = 該軌未讀 line 數 + 該軌緩衝內項數 (A / B 各自成立)
  a_line_acct:   assert property (fst == F_RUN |-> (32'(k_left) + 32'(step) + SPL - 1) / SPL == 32'(fa) + 32'(a_cnt)
                                                && (32'(k_left) + 32'(step) + SPL - 1) / SPL == 32'(fb) + 32'(b_cnt));
  if (RP == 1) begin : g_share
    // 共用埠: A、B 交替, A 最多領先 B 一項
    a_alt:       assert property (a_cnt == b_cnt || a_cnt == b_cnt + 1'b1);
  end
  a_k_left:      assert property (fst == F_RUN |-> k_left != 0);
  // ---- 累加 (v6.3.14): 寫回器讀舊的 C ----
  // 埠 0 的佔用: 取線器已發出未拿到 grant 的請求先完成 (請求保持另由 a_rd_hold 檢查); 寫回器讀的是正要寫的 line
  a_lk_f:        assert property (lk_f |-> !w_sel);
  a_wrd_line:    assert property (w_sel |-> rd_req[0] && rd_line[0] == c_ptr);
  a_wrd_only:    assert property (w_rq |-> x_w);                            // 只有累加命令讀舊的 C
  // 每個累加寫回之前, 剛讀過同一個 line (檢查器記錄寫回器的讀取 grant)
  logic        f_rv;
  logic [25:0] f_rl;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_rv <= 1'b0; f_rl <= '0; end
    else if (w_gnt) begin f_rv <= 1'b1; f_rl <= rd_line[0]; end
    else if (wr_gnt) f_rv <= 1'b0;
  a_acc_rmw:     assert property (wr_req && x_w |-> f_rv && f_rl == wr_line);
  a_noacc_nord:  assert property (wr_req && !x_w |-> !f_rv);
  c_acc_wait_f:  cover property (w_rq && lk_f);                             // 寫回器等取線器的請求完成
  c_acc_block:   cover property (w_sel && f_rq0);                           // 取線器被寫回器擋住
  c_acc_stall:   cover property (w_sel && !rd_gnt[0]);
  c_acc_ovl:     cover property (w_gnt && fst == F_RUN);                    // 讀舊 C 與下一個命令的取線重疊
  // done 只在含最後一列的 line 寫回 grant 之後出現
  a_done_after_wr: assert property (done && !$past(done) |-> $past(wst == W_WR && wr_gnt && lr));

  // ---------------- cover / covergroup ----------------
  c_rd_stall:    cover property (rd_req != '0 && rd_gnt != rd_req);
  c_wr_stall:    cover property (wr_req && !wr_gnt);
  c_starve:      cover property (fst == F_RUN && !have);                    // 緩衝空 (grant 不足造成空拍)
  c_buf_full:    cover property (fst == F_RUN && a_cnt == (PW+1)'(PF) && b_cnt == (PW+1)'(PF));
  if (RP == 2) begin : g_dual
    c_both_gnt:  cover property (a_gnt && b_gnt);                           // 兩埠同拍讀
    c_a_ahead:   cover property (fst == F_RUN && a_cnt > b_cnt + 1'b1);            // A 領先 B 超過一項
    c_b_ahead:   cover property (fst == F_RUN && b_cnt > a_cnt);
  end
  c_push_pop:    cover property (push && pop);
  c_back_to_back: cover property (feed_last ##1 cmd_valid && cmd_ready);   // 前一個命令餵完的下一拍接受新命令
  c_ovl_wr:      cover property (feed && wst == W_WR);                      // 餵入與寫回重疊
  c_rd_wr:       cover property (rd_req != '0 && wr_req);                   // 讀取與寫回同拍
  c_three:       cover property (f_n == 2'd3);                              // 同時持有 3 個命令
  c_done_wait:   cover property (done && !done_ready);
  c_take_done:   cover property (take && wst == W_DONE);                    // 完成握手同拍取下一個 tile
  c_last_stall:  cover property (mma_in_valid && mma_in_last && !mma_in_ready);
  c_reset_busy:  cover property (@(posedge clk) disable iff (1'b0) rst_n && busy ##1 !rst_n);

  logic [15:0] k_q;
  logic [$clog2(SPL)-1:0] k_mod;               // K mod SPL
  assign k_mod = k_q[$clog2(SPL)-1:0];
  always_ff @(posedge clk or negedge rst_n) if (!rst_n) k_q <= '0; else if (cmd_valid && cmd_ready) k_q <= cmd_k;
  covergroup cg_cmd @(posedge clk iff (rst_n && feed_last));
    // K 對齊 line 邊界 (SPL 步) 與否、單一 line、多 line
    cp_k: coverpoint k_q { bins one = {1}; bins part_line = {[2:SPL-1]}; bins one_line = {SPL}; bins line_p1 = {SPL+1};
                           bins multi = {[SPL+2:$]}; }
    cp_k_mod: coverpoint k_mod { bins aligned = {0}; bins other = {[1:SPL-1]}; }
  endgroup
  covergroup cg_buf @(posedge clk iff (rst_n && fst == F_RUN));
    cp_ca:   coverpoint a_cnt { bins c[] = {[0:PF]}; }
    cp_cb:   coverpoint b_cnt { bins c[] = {[0:PF]}; }
    cp_a:    coverpoint {a_req, a_gnt} { bins stall = {2'b10}; bins go = {2'b11}; }
    cp_b:    coverpoint {b_req, b_gnt} { bins stall = {2'b10}; bins go = {2'b11}; }
  endgroup
  covergroup cg_wr @(posedge clk iff (rst_n && wr_req));
    cp_wr: coverpoint {wr_gnt, lr} { bins stall = {2'b00, 2'b01}; bins mid_line = {2'b10}; bins last_line = {2'b11}; }
    cp_ovl: coverpoint {fst, pend} { bins none = {2'b00}; bins feeding = {2'b10}; bins pending = {2'b01}; bins both = {2'b11}; }
    cp_acc: coverpoint x_w { bins plain = {1'b0}; bins acc = {1'b1}; }
  endgroup
  cg_cmd u_cg_cmd = new;
  cg_buf u_cg_buf = new;
  cg_wr  u_cg_wr  = new;
endmodule

bind npu_gemm_cu npu_gemm_cu_sva #(.N(N)) u_sva (.*);
