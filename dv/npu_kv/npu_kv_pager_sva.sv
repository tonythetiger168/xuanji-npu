// SPDX-License-Identifier: Apache-2.0
// npu_kv_pager 的 SVA: 空閒表帳 (水位 / FIFO 不溢位、free_cnt ≤ NP)、配發只在有空閒頁時、FIFO 推入時不滿、
// 命令只在閒置且回應暫存器可用時接受、FREE_SEQ 期間沒有回應 / 不受理命令、回應等待時保持、
// 轉換結果 hit 與 oor 互斥、閘控安全 (時脈關閉時沒有狀態變化的來源)。
// coverage: FIFO 配發 / 水位配發、FIFO 滿、空閒表耗盡、參考計數達上限、FREE_SEQ 與轉換並行、回壓下送命令。
// 以 bind 掛到 DUT, 以未閘控的 clk 取樣。
module npu_kv_pager_sva #(parameter int NP = 64, PW = 6, RW = 3, DW = 7) (
  input logic clk, rst_n, en,
  input logic cmd_valid, cmd_ready, cmd_fire, rsp_valid, rsp_ready, sweep, sw_last,
  input logic do_alloc, do_share, do_unmap, push, fq_e, ev,
  input logic tr_valid, tr_rvalid, tr_hit, tr_oor,
  input logic [PW:0] hw, fq_n, free_cnt,
  input logic [2:0] rsp_st,
  input logic [DW-1:0] rsp_data,
  input logic [1:0] cmd_op
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  a_hw:        assert property (hw <= (PW+1)'(NP));
  a_fq_n:      assert property (fq_n <= hw);                         // FIFO 只放配發過、已釋放的頁
  a_free_cnt:  assert property (free_cnt <= (PW+1)'(NP));
  a_alloc_ok:  assert property (do_alloc |-> free_cnt != '0);
  a_push_room: assert property (push |-> fq_n < hw);                   // 推入時 FIFO 不會滿
  a_one_op:    assert property ($onehot0({do_alloc, do_share, do_unmap}));
  a_cmd_idle:  assert property (cmd_fire |-> !sweep);
  a_sweep_rsp: assert property (sweep |-> !rsp_valid && !cmd_ready);
  a_rsp_hold:  assert property (rsp_valid && !rsp_ready |=> rsp_valid && $stable(rsp_st) && $stable(rsp_data));
  a_err_data:  assert property (rsp_valid && rsp_st != 3'd0 |-> rsp_data == '0);
  a_tr_excl:   assert property (!(tr_hit && tr_oor));
  a_tr_valid:  assert property (!tr_rvalid |-> !tr_hit && !tr_oor);
  a_tr_lat:    assert property (tr_rvalid == $past(tr_valid));
  a_quiet:     assert property (!en |-> !cmd_fire && !sweep && !do_alloc && !do_share && !do_unmap);
  a_free_ret:  assert property (push |=> free_cnt == $past(free_cnt) + 1'b1);
  a_alloc_dec: assert property (do_alloc |=> free_cnt == $past(free_cnt) - 1'b1);

  c_alloc_fq:   cover property (do_alloc && !fq_e);                    // 從 FIFO 配發 (重複使用釋放的頁)
  c_alloc_hw:   cover property (do_alloc && fq_e);
  c_nomem:      cover property (cmd_fire && cmd_op == 2'd0 && free_cnt == '0);
  c_full_fq:    cover property (fq_n == (PW+1)'(NP));                  // 全部頁都在 FIFO
  c_unmap_keep: cover property (do_unmap && !push);                    // 共享頁: 解除一個映射後仍被引用
  c_sweep_tr:   cover property (sweep && ev && tr_valid);              // FREE_SEQ 與轉換並行
  c_sweep_last_hit: cover property (sw_last && ev);
  c_cmd_bp:     cover property (cmd_fire && rsp_valid && rsp_ready);   // 回應被取走的同拍送下一個命令
  c_rsp_stall:  cover property (rsp_valid && !rsp_ready && cmd_valid);
endmodule

bind npu_kv_pager npu_kv_pager_sva #(.NP(NP), .PW(PW), .RW(RW), .DW(DW)) u_sva (.*);
