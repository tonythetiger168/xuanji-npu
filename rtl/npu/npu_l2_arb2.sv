// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU Subsystem: 兩個主端共用一組 L2 讀 / 寫埠 (v6.3.10: DMA 與向量引擎共用 npu_core 的 host 埠)。
`include "npu_pkg.sv"

// npu_l2_arb2: 讀、寫各自獨立仲裁。L2 介面規則: 請求與位址 (寫入時含資料) 保持到 grant, 讀資料與 grant 同拍。
//   - 選中者在等待 grant 期間不變 (lock), 所以送往下游的請求 / 位址也保持; 下游 grant 只轉給選中者。
//   - 選擇: 只有一個請求時給它; 兩個都請求時 round-robin (上次拿到 grant 者之後的那個)。
//   - 讀資料直接廣播 (只有拿到 grant 的主端取用)。
module npu_l2_arb2 #(parameter int DW = 512) (
  input  logic clk, rst_n,
  input  logic [1:0]          rd_req,
  input  logic [1:0][25:0]    rd_line,
  output logic [1:0]          rd_gnt,
  output logic [DW-1:0]       rd_data,
  input  logic [1:0]          wr_req,
  input  logic [1:0][25:0]    wr_line,
  input  logic [1:0][DW-1:0]  wr_data,
  output logic [1:0]          wr_gnt,
  output logic                m_rd_req,
  output logic [25:0]         m_rd_line,
  input  logic                m_rd_gnt,
  input  logic [DW-1:0]       m_rd_data,
  output logic                m_wr_req,
  output logic [25:0]         m_wr_line,
  output logic [DW-1:0]       m_wr_data,
  input  logic                m_wr_gnt
);
  // 讀
  logic r_lock, r_own, r_last;
  wire  r_pick = (rd_req == 2'b11) ? !r_last : rd_req[1];   // 兩個都請求: 上次之後的那個
  wire  r_sel  = r_lock ? r_own : r_pick;
  assign m_rd_req  = |rd_req;
  assign m_rd_line = rd_line[r_sel];
  assign rd_gnt    = {m_rd_gnt && r_sel, m_rd_gnt && !r_sel} & rd_req;
  assign rd_data   = m_rd_data;
  // 寫
  logic w_lock, w_own, w_last;
  wire  w_pick = (wr_req == 2'b11) ? !w_last : wr_req[1];
  wire  w_sel  = w_lock ? w_own : w_pick;
  assign m_wr_req  = |wr_req;
  assign m_wr_line = wr_line[w_sel];
  assign m_wr_data = wr_data[w_sel];
  assign wr_gnt    = {m_wr_gnt && w_sel, m_wr_gnt && !w_sel} & wr_req;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      r_lock <= 1'b0; r_own <= 1'b0; r_last <= 1'b1; w_lock <= 1'b0; w_own <= 1'b0; w_last <= 1'b1;
    end else begin
      r_lock <= m_rd_req && !m_rd_gnt;
      r_own  <= r_sel;
      if (m_rd_req && m_rd_gnt) r_last <= r_sel;
      w_lock <= m_wr_req && !m_wr_gnt;
      w_own  <= w_sel;
      if (m_wr_req && m_wr_gnt) w_last <= w_sel;
    end
`ifdef FORMAL_TOP_npu_l2_arb2
  // ---------------- 形式驗證 (formal/npu_l2_arb2; 只在本模組是證明頂層時編入 — 作為 orca_npu_ss 的子模組時,
  // 這裡的環境假設不可約束上層; 模擬時不編入) ----------------
  // 環境: 主端請求保持到 grant (位址 / 資料不變); 下游 grant 只在有請求時。
  // 檢查器從埠記錄「上次拿到 grant 的主端」與「誰在等」, 不引用 r_last / r_own / r_lock。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    for (int i = 0; i < 2; i++) begin
      if ($past(rd_req[i] && !rd_gnt[i])) begin assume (rd_req[i] && rd_line[i] == $past(rd_line[i])); end
      if ($past(wr_req[i] && !wr_gnt[i])) begin assume (wr_req[i] && wr_line[i] == $past(wr_line[i]) && wr_data[i] == $past(wr_data[i])); end
    end
  end
  always @(*) begin
    if (!m_rd_req) begin assume (!m_rd_gnt); end
    if (!m_wr_req) begin assume (!m_wr_gnt); end
  end
  logic fr_last, fw_last;                                   // 上次拿到 grant 者 (從 grant 埠記錄)
  logic [1:0] fr_w, fw_w;                                   // 等待中的 grant 次數 (被另一方搶先)
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin fr_last <= 1'b1; fw_last <= 1'b1; fr_w <= '0; fw_w <= '0; end
    else begin
      if (rd_gnt != '0) fr_last <= rd_gnt[1];
      if (wr_gnt != '0) fw_last <= wr_gnt[1];
      for (int i = 0; i < 2; i++) begin
        if (rd_req[i] && !rd_gnt[i] && rd_gnt[1-i]) fr_w[i] <= 1'b1; else if (!rd_req[i] || rd_gnt[i]) fr_w[i] <= 1'b0;
        if (wr_req[i] && !wr_gnt[i] && wr_gnt[1-i]) fw_w[i] <= 1'b1; else if (!wr_req[i] || wr_gnt[i]) fw_w[i] <= 1'b0;
      end
    end
  always @(*) if (rst_n) begin
    // grant 只給一個、只給有請求者、只在下游 grant 時
    assert (!(rd_gnt[0] && rd_gnt[1]) && !(wr_gnt[0] && wr_gnt[1]));
    assert ((rd_gnt & ~rd_req) == '0 && (wr_gnt & ~wr_req) == '0);
    assert ((rd_gnt != '0) == (m_rd_gnt && m_rd_req) && (wr_gnt != '0) == (m_wr_gnt && m_wr_req));
    assert (m_rd_req == (rd_req != '0) && m_wr_req == (wr_req != '0));
    // 下游看到的位址 / 資料屬於拿到 grant 的主端
    if (rd_gnt[0]) begin assert (m_rd_line == rd_line[0]); end
    if (rd_gnt[1]) begin assert (m_rd_line == rd_line[1]); end
    if (wr_gnt[0]) begin assert (m_wr_line == wr_line[0] && m_wr_data == wr_data[0]); end
    if (wr_gnt[1]) begin assert (m_wr_line == wr_line[1] && m_wr_data == wr_data[1]); end
    assert (rd_data == m_rd_data);
    // 公平: 被另一方搶先過一次的主端, 下一次 grant 必定給它 (兩方輪流, 不會連續被插隊)
    for (int i = 0; i < 2; i++) begin
      if (fr_w[i] && rd_gnt != '0) begin assert (rd_gnt[i]); end
      if (fw_w[i] && wr_gnt != '0) begin assert (wr_gnt[i]); end
    end
    // 歸納強化: RTL 的狀態與檢查器記錄一致 (只在斷言中引用)
    assert (r_last == fr_last && w_last == fw_last);
    if (r_lock) begin assert (rd_req[r_own]); end                  // 鎖定的選中者仍在請求 (等待 grant)
    if (w_lock) begin assert (wr_req[w_own]); end
    for (int i = 0; i < 2; i++) begin                         // 被插隊者: 上次 grant 是另一方, 鎖定的不會是另一方
      if (fr_w[i]) begin assert (r_last == 1'(1 - i) && (!r_lock || r_own == 1'(i))); end
      if (fw_w[i]) begin assert (w_last == 1'(1 - i) && (!w_lock || w_own == 1'(i))); end
    end
    if (fr_w[0] || fr_w[1]) begin assert (rd_req == 2'b11 || (fr_w[0] ? rd_req[0] : rd_req[1])); end
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    // 下游請求等待中時, 位址 (與寫入資料) 保持
    if ($past(m_rd_req && !m_rd_gnt)) begin assert (m_rd_req && m_rd_line == $past(m_rd_line)); end
    if ($past(m_wr_req && !m_wr_gnt)) begin assert (m_wr_req && m_wr_line == $past(m_wr_line) && m_wr_data == $past(m_wr_data)); end
    cover ($past(rd_gnt[1]) && rd_req == 2'b11 && rd_gnt[0]);   // 連續輪流
  end
  always @(*) if (rst_n) begin
    cover (rd_req == 2'b11 && rd_gnt[1]);
    cover (wr_req == 2'b11 && wr_gnt[0]);
  end
`endif
endmodule : npu_l2_arb2
