// SPDX-License-Identifier: Apache-2.0
// npu_l2_xbar 的 SVA: grant 合法性 (每 bank 每拍最多一讀一寫、只給有請求者)、work-conserving、
// round-robin 有界等待, 以及衝突情境的 cover。以 bind 掛到 DUT, .* 連接同名訊號。
module npu_l2_xbar_sva
  import orca_pkg::*;
#(parameter int NM = 5, parameter int NW = NM, parameter int NB = 4, parameter int DEPTH = 1024) (
  input logic clk, rst_n,
  input logic [NM-1:0] rd_req, rd_gnt,
  input logic [NW-1:0] wr_req, wr_gnt,
  input logic [NM-1:0][25:0] rd_line,
  input logic [NW-1:0][25:0] wr_line
);
  localparam int BW = $clog2(NB), DW = $clog2(DEPTH);
  function automatic logic [BW-1:0] bk(logic [25:0] l); return BW'(npu_l2_bank(l, BW, DW)); endfunction
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  a_rd_gnt_req: assert property ((rd_gnt & ~rd_req) == '0);
  a_wr_gnt_req: assert property ((wr_gnt & ~wr_req) == '0);
  for (genvar b = 0; b < NB; b++) begin : g_b
    logic [NM-1:0] rq, rg;                         // 對本 bank 的讀請求 / grant
    logic [NW-1:0] wq, wg;                         // 對本 bank 的寫請求 / grant
    for (genvar m = 0; m < NM; m++) begin : g_m
      assign rq[m] = rd_req[m] && bk(rd_line[m]) == BW'(b);
      assign rg[m] = rd_gnt[m] && bk(rd_line[m]) == BW'(b);
    end
    for (genvar m = 0; m < NW; m++) begin : g_mw
      assign wq[m] = wr_req[m] && bk(wr_line[m]) == BW'(b);
      assign wg[m] = wr_gnt[m] && bk(wr_line[m]) == BW'(b);
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
    c_rd_wait:    cover property (rd_req[m] && !rd_gnt[m]);
  end
  for (genvar m = 0; m < NW; m++) begin : g_ww
    a_wr_bounded: assert property (wr_req[m] |-> ##[0:NW-1] wr_gnt[m]);
    c_wr_wait:    cover property (wr_req[m] && !wr_gnt[m]);
  end
  // round-robin 順序: 主端 m 對同一 bank 持續等待期間, 其他主端在該 bank 最多各贏一次。
  //   有界等待在 NB 大、bank 雜湊分散時很少觸發 (Max 組態固定優先權 48 拍內不會飢餓), 此檢查一拍就抓到被插隊兩次。
  //   只用埠: 檢查器自己記錄 m 等待期間誰已在 m 的 bank 拿到 grant。
  for (genvar m = 0; m < NM; m++) begin : g_ov
    logic [NM-1:0] ov, hit;                        // ov: 本次等待中已贏過的主端
    logic [BW-1:0] bm_q;                           // 上一拍 m 等待的 bank
    logic          w_q;                            // 上一拍 m 在等待
    wire  [BW-1:0] bm = bk(rd_line[m]);
    wire           cont = w_q && rd_req[m] && bm == bm_q;   // 同一 bank 的等待延續
    for (genvar j = 0; j < NM; j++) begin : g_j
      assign hit[j] = (j != m) && rd_gnt[j] && bk(rd_line[j]) == bm;
    end
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin ov <= '0; w_q <= 1'b0; bm_q <= '0; end
      else begin
        w_q  <= rd_req[m] && !rd_gnt[m];
        bm_q <= bm;
        ov   <= (rd_req[m] && !rd_gnt[m]) ? ((cont ? ov : '0) | hit) : '0;
      end
    a_rd_rr: assert property (cont |-> (hit & ov) == '0);
    c_rd_ov: cover property (cont && hit != '0);   // 等待中被別人插隊過
  end
  for (genvar m = 0; m < NW; m++) begin : g_ovw
    logic [NW-1:0] ov, hit;
    logic [BW-1:0] bm_q;
    logic          w_q;
    wire  [BW-1:0] bm = bk(wr_line[m]);
    wire           cont = w_q && wr_req[m] && bm == bm_q;
    for (genvar j = 0; j < NW; j++) begin : g_j
      assign hit[j] = (j != m) && wr_gnt[j] && bk(wr_line[j]) == bm;
    end
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin ov <= '0; w_q <= 1'b0; bm_q <= '0; end
      else begin
        w_q  <= wr_req[m] && !wr_gnt[m];
        bm_q <= bm;
        ov   <= (wr_req[m] && !wr_gnt[m]) ? ((cont ? ov : '0) | hit) : '0;
      end
    a_wr_rr: assert property (cont |-> (hit & ov) == '0);
    c_wr_ov: cover property (cont && hit != '0);
  end
  // 平行度: 同拍的讀 / 寫 grant 數 (多數主端同時存取不同 bank)。
  // 註: 「所有 CU 同拍讀」在此設計中不會發生 — 預取下每個 CU 每 SPL 拍只讀 2 次, 且分派一拍一個而錯開。
  localparam int MAXG = (NM < NB) ? NM : NB, MAXW = (NW < NB) ? NW : NB;
  covergroup cg_par @(posedge clk iff rst_n);
    cp_rd: coverpoint $countones(rd_gnt) { bins n0 = {0}; bins n1 = {1}; bins multi = {[2:MAXG]}; }
    cp_wr: coverpoint $countones(wr_gnt) { bins n0 = {0}; bins n1 = {1}; bins multi = {[2:MAXW]}; }
  endgroup
  cg_par u_cg_par = new;
endmodule

bind npu_l2_xbar npu_l2_xbar_sva #(.NM(NM), .NW(NW), .NB(NB), .DEPTH(DEPTH)) u_sva (.*);
