// SPDX-License-Identifier: Apache-2.0
// npu_mma 的 SVA: 介面協定、資料流不變式 (token 成對到達 / 排空足夠 / 閘控安全) 與功能 coverage。
// v6.3.12 tile 管線化: 輸入端 (ist) 與輸出端 (ost) 分開; 結果在 res (last token 時存入)。
// 以 bind 掛到 DUT (見檔尾), .* 連接 DUT 內部同名訊號。
module npu_mma_sva #(parameter int N = 8) (
  input logic clk, rst_n, test_en,
  input logic start, a_signed, b_signed, busy, in_idle,
  input logic in_valid, in_ready, in_last,
  input logic [N-1:0][7:0] a_vec, b_vec,
  input logic out_valid, out_ready,
  input logic [$clog2(N)-1:0] out_row,
  input logic [N-1:0][31:0] out_vec,
  input logic ist,
  input logic [1:0] ost,
  input logic go, fire_in, fire_last, sgn_a, sgn_b,
  input logic [N-1:0] a_tok_edge, b_tok_edge, row_act,
  input logic [N-1:0][N-1:0] tok_q,
  input logic [N-1:0][N-1:0][3:0] fl_q,
  input logic [N-1:0][3:0] fl_edge,
  input logic [N-1:0][N-1:0][31:0] res
);
  localparam int RW = $clog2(N);
  localparam logic I_IDLE = 1'b0, I_RUN = 1'b1;
  localparam logic [1:0] O_IDLE = 2'd0, O_DRAIN = 2'd1, O_OUT = 2'd2;

  default clocking cb @(posedge clk); endclocking
  // clk 可能是閘控時脈: 非同步 reset 若落在兩個取樣沿之間, 取樣時 rst_n 已回到 1, 但 $past 跨過了 reset。
  // rst_seen 在 reset 時非同步設定、下一個取樣沿清除, 讓跨過 reset 的嘗試被停用。
  logic rst_seen;
  always_ff @(posedge clk or negedge rst_n) rst_seen <= !rst_n ? 1'b1 : 1'b0;
  default disable iff (!rst_n || rst_seen);

  // ---------------- 協定 / 控制 ----------------
  a_busy:          assert property (busy == (ist != I_IDLE || ost != O_IDLE) && in_idle == (ist == I_IDLE));
  a_ost_legal:     assert property (ost != 2'd3);
  // 輸入端在 RUN 時就緒, 除非這一步是 last 而上一個 tile 還沒輸出完
  a_ready_in_run:  assert property (in_ready == (ist == I_RUN && !(in_last && ost != O_IDLE)));
  a_out_valid:     assert property (out_valid == (ost == O_OUT));
  a_out_hold:      assert property (out_valid && !out_ready |=> out_valid && $stable(out_row) && $stable(out_vec));
  a_out_row_seq:   assert property (out_valid && out_ready && out_row != RW'(N - 1) |=> out_valid && out_row == $past(out_row) + 1'b1);
  a_out_last_row:  assert property (out_valid && out_ready && out_row == RW'(N - 1) |=> ost == O_IDLE);
  a_out_first_row: assert property (ost != O_OUT ##1 ost == O_OUT |-> out_row == '0);
  a_start_ignored: assert property (!in_idle && start |=> $stable(sgn_a) && $stable(sgn_b));
  a_known:         assert property (!$isunknown({busy, in_ready, out_valid, out_row}));
  a_known_out:     assert property (out_valid |-> !$isunknown(out_vec));

  // ---------------- 資料流不變式 ----------------
  // 單一 token 的前提: 從左邊進入 PE[r][c] 的 token 與從上方進入的 token 相同 (A / B skew 對齊),
  // 即 PE[r][c-1] 與 PE[r-1][c] 的 token 一致 (反對角線上同時有效); 邊緣以 skew 輸出代替
  a_edge_pair:     assert property (a_tok_edge == b_tok_edge);
  for (genvar r = 0; r < N; r++) begin : g_d
    for (genvar c = 0; c < N; c++) begin : g_dc
      if (r > 0 && c > 0) begin : g_in
        a_tok_diag: assert property (tok_q[r][c-1] == tok_q[r-1][c]);
        a_fl_diag:  assert property (tok_q[r][c-1] |-> fl_q[r][c-1] == fl_q[r-1][c]);   // 同一步 k 的旗標相同
      end else if (r > 0) begin : g_l
        a_tok_diag: assert property (a_tok_edge[r] == tok_q[r-1][0]);
        a_fl_diag:  assert property (a_tok_edge[r] |-> fl_edge[r] == fl_q[r-1][0]);
      end else if (c > 0) begin : g_t
        a_tok_diag: assert property (tok_q[0][c-1] == b_tok_edge[c]);
      end
    end
  end
  // 排空長度足夠: 輸出端進入輸出時, 陣列內沒有帶 last 旗標的 token (本 tile 的最後一步已全部累加完;
  // 其他 token 屬於下一個 tile)
  for (genvar r = 0; r < N; r++) begin : g_dr
    for (genvar c = 0; c < N; c++) begin : g_dc
      a_drained: assert property (ost == O_OUT |-> !(tok_q[r][c] && fl_q[r][c][2]));
    end
  end
  // 只有 RUN 會產生 token; 輸出端只因 last 握手離開閒置; last 握手時輸出端必閒置 (res 不被覆蓋)
  a_tok_src:       assert property (fire_in |-> ist == I_RUN);
  a_last_free:     assert property (fire_last |-> ost == O_IDLE);
  a_ost_start:     assert property (ost == O_IDLE ##1 ost != O_IDLE |-> $past(fire_last));
  // 輸出期間結果暫存不變
  a_res_hold:      assert property (ost == O_OUT && $past(ost) == O_OUT |-> $stable(res));
  // 閘控安全: 列時脈關閉時, 該列沒有任何 token 要處理 (關閉時鐘不會吃掉運算)
  for (genvar r = 0; r < N; r++) begin : g_r
    a_gate_safe:   assert property (!row_act[r] |-> tok_q[r] == '0 && a_tok_edge[r] == 1'b0 &&
                                    (r == 0 ? b_tok_edge == '0 : tok_q[(r == 0) ? 0 : r - 1] == '0));
  end

  // ---------------- cover ----------------
  c_bubble:        cover property (ist == I_RUN && !in_valid);
  c_gate_all_off:  cover property (ist == I_RUN && row_act == '0 && !test_en);
  c_gate_partial:  cover property (row_act != '0 && row_act != '1);
  c_gate_all_on:   cover property (row_act == '1);
  c_start_busy:    cover property (!in_idle && start);
  c_overlap_drn:   cover property (fire_in && ost == O_DRAIN);       // 下一個 tile 在前一個排空時輸入
  c_overlap_out:   cover property (fire_in && ost == O_OUT);         // 下一個 tile 在前一個輸出時輸入
  c_last_stall:    cover property (ist == I_RUN && in_valid && in_last && ost != O_IDLE);
  c_valid_no_rdy:  cover property (in_valid && !in_ready);
  c_out_stall:     cover property (out_valid && !out_ready);
  c_k1:            cover property (go ##1 fire_in && in_last);
  c_back_to_back:  cover property (fire_last ##1 go);                 // 前一個 tile 最後一步的下一拍就 start
  c_reset_in_run:  cover property (@(posedge clk) disable iff (1'b0) ist == I_RUN ##1 !rst_n);
  c_reset_in_drn:  cover property (@(posedge clk) disable iff (1'b0) ost == O_DRAIN ##1 !rst_n);
  c_reset_in_out:  cover property (@(posedge clk) disable iff (1'b0) ost == O_OUT ##1 !rst_n);

  // ---------------- 功能 coverage ----------------
  int unsigned kcnt;   // 本 tile 已接受的 k 步數
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) kcnt <= 0;
    else if (go) kcnt <= 0;
    else if (fire_in) kcnt <= kcnt + 1;

  covergroup cg_tile @(posedge clk iff (rst_n && fire_in && in_last));
    cp_k: coverpoint kcnt + 1 {
      bins one = {1}; bins lt_n = {[2:N-1]}; bins eq_n = {N}; bins le_2n = {[N+1:2*N]};
      bins mid = {[2*N+1:63]}; bins long_k = {[64:$]};
    }
    cp_sgn: coverpoint {sgn_a, sgn_b};
    cp_te:  coverpoint test_en;
    x_k_sgn: cross cp_k, cp_sgn;
  endgroup

  // 運算元邊界值 (以第 0 / 最後一個 lane 取樣), 與有號 / 無號交叉
  covergroup cg_opnd @(posedge clk iff (rst_n && fire_in));
    cp_a0: coverpoint a_vec[0] { bins zero = {0}; bins one = {1}; bins pmax = {8'h7F}; bins nmin = {8'h80}; bins all1 = {8'hFF}; }
    cp_bl: coverpoint b_vec[N-1] { bins zero = {0}; bins one = {1}; bins pmax = {8'h7F}; bins nmin = {8'h80}; bins all1 = {8'hFF}; }
    cp_sgn: coverpoint {sgn_a, sgn_b};
    x_a: cross cp_a0, cp_sgn;
    x_b: cross cp_bl, cp_sgn;
  endgroup

  logic stall_q;   // 上一拍輸出被回壓
  always_ff @(posedge clk or negedge rst_n) stall_q <= !rst_n ? 1'b0 : (out_valid && !out_ready);

  // 結果符號 / 量級
  covergroup cg_out @(posedge clk iff (rst_n && out_valid && out_ready));
    cp_row: coverpoint out_row;
    cp_c0:  coverpoint out_vec[0] { bins zero = {0}; bins pos_s = {[1:32'h7FFF]}; bins pos_l = {[32'h8000:32'h7FFF_FFFF]};
                                    bins neg_s = {[32'hFFFF_8000:32'hFFFF_FFFF]}; bins neg_l = {[32'h8000_0000:32'hFFFF_7FFF]}; }
    cp_stall: coverpoint stall_q;
  endgroup

  cg_tile cg_t = new;
  cg_opnd cg_o = new;
  cg_out  cg_c = new;
endmodule

bind npu_mma npu_mma_sva #(.N(N)) u_sva (.*);
