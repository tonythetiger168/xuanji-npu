// SPDX-License-Identifier: Apache-2.0
// npu_l2_arb2 的 SVA (bind; 只看埠): 送往下游的請求在等待 grant 期間保持 (位址 / 寫資料不變), grant 只給有請求者、
// 一次最多一個且與下游 grant 同拍; 公平: 主端 i 持續等待期間, 另一個主端最多拿到一次 grant (round-robin)。
module npu_l2_arb2_sva #(parameter int DW = 512) (
  input logic clk, rst_n,
  input logic [1:0] rd_req, rd_gnt, wr_req, wr_gnt,
  input logic m_rd_req, m_rd_gnt, m_wr_req, m_wr_gnt,
  input logic [25:0] m_rd_line, m_wr_line,
  input logic [DW-1:0] m_wr_data
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  a_rd_hold: assert property (m_rd_req && !m_rd_gnt |=> $stable(m_rd_line));
  a_wr_hold: assert property (m_wr_req && !m_wr_gnt |=> $stable(m_wr_line) && $stable(m_wr_data));
  a_rd_gnt:  assert property ((rd_gnt & ~rd_req) == '0 && $onehot0(rd_gnt) && (rd_gnt != '0) == (m_rd_req && m_rd_gnt));
  a_wr_gnt:  assert property ((wr_gnt & ~wr_req) == '0 && $onehot0(wr_gnt) && (wr_gnt != '0) == (m_wr_req && m_wr_gnt));
  for (genvar i = 0; i < 2; i++) begin : g_m
    logic rov, wov;                      // 主端 i 本次等待中, 另一個主端已拿到過一次 grant
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin rov <= 1'b0; wov <= 1'b0; end
      else begin
        rov <= (rd_req[i] && !rd_gnt[i]) && (rov || rd_gnt[1 - i]);
        wov <= (wr_req[i] && !wr_gnt[i]) && (wov || wr_gnt[1 - i]);
      end
    a_rd_fair: assert property (rov && rd_req[i] |-> !rd_gnt[1 - i]);
    a_wr_fair: assert property (wov && wr_req[i] |-> !wr_gnt[1 - i]);
    c_rd_ov:   cover property (rov && rd_gnt[i]);   // 等待中被搶先一次後拿到 grant
    c_wr_ov:   cover property (wov && wr_gnt[i]);
  end
  c_rd_both: cover property (rd_req == 2'b11);
  c_wr_both: cover property (wr_req == 2'b11);
endmodule

bind npu_l2_arb2 npu_l2_arb2_sva #(.DW(DW)) u_sva (.*);
