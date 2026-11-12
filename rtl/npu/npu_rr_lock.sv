// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU: NM 個主端共用一個「請求保持到 grant」的下游埠的選擇器 (v6.3.13, 多通道 DMA 使用; 可供其他主端共用)。

// npu_rr_lock: 只產生選擇 (one-hot), 資料 / 位址的多工由使用者以 sel 做。
//   - 主端規則: 請求 (與其位址 / 資料) 保持到拿到 grant。
//   - 選中者在等待下游 grant 期間不變 (lock), 所以送往下游的請求 / 位址也保持。
//   - 選擇: round-robin, 從上次拿到 grant 者的下一個開始找 (每個請求者最多等 NM − 1 次別人的 grant)。
//   - out_req = 任一請求; 下游 grant 只轉給選中者: gnt[i] = m_gnt && sel[i]。
module npu_rr_lock #(parameter int NM = 2) (
  input  logic          clk, rst_n,
  input  logic [NM-1:0] req,
  input  logic          m_gnt,          // 下游 grant (只在 out_req 時有效)
  output logic          out_req,
  output logic [NM-1:0] sel,            // one-hot (out_req = 0 時全 0)
  output logic [NM-1:0] gnt
);
  localparam int IW = (NM > 1) ? $clog2(NM) : 1;
  logic          lock;
  logic [NM-1:0] own;
  logic [IW-1:0] last;
  logic [NM-1:0] pick;
  always_comb begin
    pick = '0;
    for (int i = NM - 1; i >= 0; i--) begin            // 由遠到近, 最後寫入者 = 上次之後最近的請求者
      automatic int s = (int'(last) + 1 + i) % NM;
      if (req[s]) pick = NM'(1) << s;
    end
  end
  assign out_req = |req;
  assign sel     = lock ? own : pick;
  assign gnt     = m_gnt ? sel : '0;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      lock <= 1'b0; own <= '0; last <= IW'(NM - 1);
    end else begin
      lock <= out_req && !m_gnt;
      own  <= sel;
      if (out_req && m_gnt)
        for (int i = 0; i < NM; i++) if (sel[i]) last <= IW'(i);
    end

`ifdef FORMAL_TOP_npu_rr_lock
  // ---------------- 形式驗證 (formal/npu_rr_lock; 只在本模組是證明頂層時編入) ----------------
  // 環境: 主端請求保持到 grant; 下游 grant 只在有請求時。檢查器從埠記錄上次的得主與每個主端被搶先的次數。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  always @(posedge clk) if (f_past && rst_n && $past(rst_n))
    for (int i = 0; i < NM; i++) if ($past(req[i] && !gnt[i])) assume (req[i]);
  always @(*) if (!out_req) assume (!m_gnt);
  logic [IW-1:0] f_last;
  logic [NM-1:0][IW:0] f_d;                                // 從上次得主往後到 i 的距離 (1..NM)
  logic [NM-1:0]       f_first;                            // 上次得主之後第一個請求者
  always_comb
    for (int i = 0; i < NM; i++) f_d[i] = (IW+1)'((i + 2 * NM - 1 - int'(f_last)) % NM + 1);
  always_comb
    for (int i = 0; i < NM; i++) begin
      f_first[i] = req[i];
      for (int j = 0; j < NM; j++) if (req[j] && f_d[j] < f_d[i]) f_first[i] = 1'b0;
    end
  logic [NM-1:0][IW:0] f_w;                                // 等待中被別人拿走 grant 的次數
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_last <= IW'(NM - 1); f_w <= '0; end
    else begin
      for (int i = 0; i < NM; i++) begin
        if (gnt[i]) f_last <= IW'(i);
        f_w[i] <= (!req[i] || gnt[i]) ? '0 : (gnt != '0) ? f_w[i] + 1'b1 : f_w[i];
      end
    end
  always @(*) if (rst_n) begin
    assert ($onehot0(sel) && ((sel & ~req) == '0) && (out_req == (req != '0)) && (out_req == (sel != '0)));
    assert (gnt == (m_gnt ? sel : '0));
    for (int i = 0; i < NM; i++) assert (32'(f_w[i]) <= NM - 1);
    // 沒有鎖定時: 選中者 = 上次得主之後第一個請求者
    if (!lock) assert (sel == f_first);
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(out_req && !m_gnt)) assert (sel == $past(sel));   // 等待 grant 時保持選擇
  end
`ifndef FORMAL_NO_HELPER
  // 輔助不變式 (歸納): 內部狀態與檢查器一致
  always @(*) if (rst_n) begin
    assert (last == f_last && 32'(last) < NM);
    if (lock) assert ($onehot(own) && ((own & ~req) == '0));
    // 等待者 i: 已被搶先次數 + 從上次得主到 i 的距離 (1..NM) <= NM; 鎖定中的得主在 i 之前
    for (int i = 0; i < NM; i++) if (f_w[i] != '0) begin
      assert (req[i] && 32'(f_w[i]) + 32'(f_d[i]) <= NM);
      if (lock && !own[i])
        for (int j = 0; j < NM; j++) if (own[j]) assert (f_d[j] < f_d[i]);
    end
  end
`endif
  always @(posedge clk) if (rst_n) begin
    for (int i = 0; i < NM; i++) cover (f_w[i] == (IW+1)'(NM - 1) && gnt[i]);
    cover (lock && req == '1);
  end
`endif
endmodule
