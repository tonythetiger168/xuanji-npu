// SPDX-License-Identifier: Apache-2.0
// npu_rr_lock 的 SVA (bind; 只看埠): 等待下游 grant 期間選擇不變, grant 只給選中者;
// cover 鎖定的必要情境 — 主端 i 單獨請求且沒拿到 grant, 下一拍另一個主端也請求, 而上次拿到 grant 的是 i
// (沒有鎖定時 round-robin 會在等待中改選另一個)。f_last 由檢查器從 gnt 埠記錄。
// coverage 以模組層級合併: 四個埠選擇器 (L2 讀 / 寫、外部讀 / 寫) 任一個命中即可。
module npu_rr_lock_sva #(parameter int NM = 2) (
  input logic clk, rst_n,
  input logic [NM-1:0] req, sel, gnt,
  input logic m_gnt, out_req
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  a_sel_hold: assert property (out_req && !m_gnt |=> $stable(sel));
  a_gnt_sel:  assert property (gnt == (m_gnt ? sel : '0) && $onehot0(sel) && (sel & ~req) == '0);
  if (NM == 2) begin : g_nm2
    logic f_last;
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) f_last <= 1'b1;
      else if (gnt != '0) f_last <= gnt[1];
    // (分開寫, 不放在 generate 迴圈: coverage 會把迴圈內的 cover 合併成一個點)
    c_lock_0: cover property (req == 2'b01 && !m_gnt ##1 req == 2'b11 && f_last == 1'b0);
    c_lock_1: cover property (req == 2'b10 && !m_gnt ##1 req == 2'b11 && f_last == 1'b1);
  end
endmodule

bind npu_rr_lock npu_rr_lock_sva #(.NM(NM)) u_sva (.*);
