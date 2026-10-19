// SPDX-License-Identifier: Apache-2.0
// npu_l2_xbar 的 SVA: grant 合法性 (每 bank 每拍最多一讀一寫、只給有請求者)、work-conserving、
// round-robin 有界等待, 以及衝突情境的 cover。以 bind 掛到 DUT, .* 連接同名訊號。
module npu_l2_xbar_sva #(parameter int NM = 5, parameter int NB = 4) (
  input logic clk, rst_n,
  input logic [NM-1:0] rd_req, rd_gnt, wr_req, wr_gnt,
  input logic [NM-1:0][25:0] rd_line, wr_line
);
  localparam int BW = $clog2(NB);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  a_rd_gnt_req: assert property ((rd_gnt & ~rd_req) == '0);
  a_wr_gnt_req: assert property ((wr_gnt & ~wr_req) == '0);
  for (genvar b = 0; b < NB; b++) begin : g_b
    logic [NM-1:0] rq, wq, rg, wg;                 // 對本 bank 的請求 / grant
    for (genvar m = 0; m < NM; m++) begin : g_m
      assign rq[m] = rd_req[m] && rd_line[m][BW-1:0] == BW'(b);
      assign wq[m] = wr_req[m] && wr_line[m][BW-1:0] == BW'(b);
      assign rg[m] = rd_gnt[m] && rd_line[m][BW-1:0] == BW'(b);
      assign wg[m] = wr_gnt[m] && wr_line[m][BW-1:0] == BW'(b);
    end
    a_one_rd:   assert property ($onehot0(rg));
    a_one_wr:   assert property ($onehot0(wg));
    a_wc_rd:    assert property (rq != '0 |-> rg != '0);     // work-conserving
    a_wc_wr:    assert property (wq != '0 |-> wg != '0);
    c_rd_conflict: cover property ($countones(rq) >= 2);
    c_wr_conflict: cover property ($countones(wq) >= 2);
    c_rw_same:     cover property (rg != '0 && wg != '0);     // 同 bank 同拍一讀一寫
  end
  // 有界等待: 持續請求的主端 NM 拍內必得到 grant (主端在 grant 前保持請求)
  for (genvar m = 0; m < NM; m++) begin : g_w
    a_rd_bounded: assert property (rd_req[m] |-> ##[0:NM-1] rd_gnt[m]);
    a_wr_bounded: assert property (wr_req[m] |-> ##[0:NM-1] wr_gnt[m]);
    c_rd_wait:    cover property (rd_req[m] && !rd_gnt[m]);
    c_wr_wait:    cover property (wr_req[m] && !wr_gnt[m]);
  end
  // 平行度: 同拍的讀 / 寫 grant 數 (多數主端同時存取不同 bank)。
  // 註: 「所有 CU 同拍讀」在此設計中不會發生 — 預取下每個 CU 每 SPL 拍只讀 2 次, 且分派一拍一個而錯開。
  localparam int MAXG = (NM < NB) ? NM : NB;
  covergroup cg_par @(posedge clk iff rst_n);
    cp_rd: coverpoint $countones(rd_gnt) { bins n0 = {0}; bins n1 = {1}; bins multi = {[2:MAXG]}; }
    cp_wr: coverpoint $countones(wr_gnt) { bins n0 = {0}; bins n1 = {1}; bins multi = {[2:MAXG]}; }
  endgroup
  cg_par u_cg_par = new;
endmodule

bind npu_l2_xbar npu_l2_xbar_sva #(.NM(NM), .NB(NB)) u_sva (.*);
