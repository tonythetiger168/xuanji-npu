// SPDX-License-Identifier: Apache-2.0
// npu_core 的 SVA: 命令佇列 / 分派 / 完成協定、CU 時脈閘控安全與功能 coverage。以 bind 掛到 DUT。
module npu_core_sva #(parameter int NCU = 4, parameter int QD = 4) (
  input logic clk, rst_n, test_en,
  input logic cmd_valid, cmd_ready, cpl_valid, cpl_ready, busy,
  input logic [7:0] cpl_id,
  input logic h_rd_req, h_rd_gnt, h_wr_req, h_wr_gnt,
  input logic [$clog2(QD):0] q_cnt,
  input logic [NCU-1:0] cu_ready, cu_busy, cu_idle, cu_done, cu_dack, cu_v, cu_free, cpl_pend, cu_en, cu_feed,
  input logic [NCU-1:0] rd_req, wr_req           // CU 的讀取請求 (任一埠, cu_rd) / 寫入請求 (低 NCU 位)
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  a_q_bound:     assert property (q_cnt <= ($clog2(QD)+1)'(QD));
  a_cmd_ready:   assert property (cmd_ready == (q_cnt != ($clog2(QD)+1)'(QD)));
  a_disp_one:    assert property ($onehot0(cu_v));
  a_disp_free:   assert property ((cu_v & ~cu_free) == '0);
  a_idle_def:    assert property (cu_idle == (cu_ready & ~cu_busy));
  // 有完全閒置的 CU 時給其中編號最小者; 沒有時給可接受命令者中編號最小者
  a_disp_first:  assert property (cu_v != '0 && cu_idle != '0 |-> (cu_v & cu_idle) != '0 && ((cu_v - 1'b1) & cu_idle) == '0);
  a_disp_ovl:    assert property (cu_v != '0 && cu_idle == '0 |-> ((cu_v - 1'b1) & cu_free) == '0);
  a_no_idle_q:   assert property (q_cnt != 0 && cu_free != '0 |-> cu_v != '0);        // 有命令且有可接受的 CU 必分派
  a_cpl_hold:    assert property (cpl_valid && !cpl_ready |=> cpl_valid && $stable(cpl_id));
  a_dack:        assert property (cu_dack == ~cpl_pend);                                // 待回報槽空時接受完成
  a_done_busy:   assert property ((cu_done & ~cu_busy) == '0);
  a_busy:        assert property (!busy |-> cu_busy == '0 && cpl_pend == '0 && !cpl_valid && q_cnt == 0);
  // 閘控安全: CU 時脈關閉時必在 idle, 沒有分派 / done / L2 請求 (test_en 時 ICG 強制開啟, 不檢查)
  a_gate_safe:   assert property (!test_en |-> ((~cu_en) & (~cu_ready | cu_busy | cu_done | cu_v | rd_req | wr_req | cu_feed)) == '0);

  c_q_full:      cover property (cmd_valid && !cmd_ready);
  c_cpl_bp:      cover property (cpl_valid && !cpl_ready);
  c_all_busy:    cover property (cu_ready == '0);
  c_disp_ovl:    cover property ((cu_v & cu_busy) != '0);                              // 分派給前一個命令還在輸出的 CU
  c_done_wait:   cover property ((cu_done & cpl_pend) != '0);                          // CU 完成等待待回報槽
  c_gated:       cover property (!test_en && busy && cu_en != '1);
  c_host_rd_cu:  cover property (h_rd_req && rd_req != '0);
  c_host_wr_cu:  cover property (h_wr_req && rd_req != '0);
  if (NCU > 1) begin : g_multi
    c_two_done:  cover property ($countones(cu_done) >= 2);
    c_cpl_while_pend: cover property (cpl_valid && !cpl_ready && cpl_pend != '0);
  end

  covergroup cg_occ @(posedge clk iff rst_n);
    cp_busy_cu: coverpoint $countones(cu_busy) { bins n[] = {[0:NCU]}; }
    cp_feed_cu: coverpoint $countones(~cu_ready) { bins n[] = {[0:NCU]}; }
    cp_q:       coverpoint q_cnt { bins n[] = {[0:QD]}; }
    cp_pend:    coverpoint $countones(cpl_pend) { bins n[] = {[0:NCU]}; }
  endgroup
  cg_occ u_cg_occ = new;
endmodule

bind npu_core npu_core_sva #(.NCU(NCU), .QD(QD)) u_sva (
  .clk, .rst_n, .test_en, .cmd_valid, .cmd_ready, .cpl_valid, .cpl_ready, .busy, .cpl_id,
  .h_rd_req, .h_rd_gnt, .h_wr_req, .h_wr_gnt, .q_cnt, .cu_ready, .cu_busy, .cu_idle, .cu_done, .cu_dack, .cu_v, .cu_free, .cpl_pend, .cu_en, .cu_feed,
  .rd_req(cu_rd), .wr_req(wr_req[NCU-1:0]));
