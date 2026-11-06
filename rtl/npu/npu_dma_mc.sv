// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU 多通道 DMA (v6.3.13): NDC 個 npu_dma 通道共用 L2 host 埠與外部記憶體埠。
`include "orca_pkg.sv"

// npu_dma_mc: 讓多個 DMA 命令同時執行 (例: 載入下一層權重的同時 gather KV / 寫回結果)。
//   命令: cmd_valid / cmd_ready; 有空閒通道 (閒置且完成槽空) 時接受, 交給編號最小的空閒通道。
//   完成: cpl_valid / cpl_ready + cpl_id; 每個通道一個完成槽, round-robin 輸出 (完成順序不保證與命令順序相同)。
//   通道之間沒有順序保證: 軟體須保證同時執行的 DMA 存取的區域不重疊 (orca_npu_ss 的 wt 規則)。
//   埠共用: L2 讀 / L2 寫 / 外部讀 / 外部寫 各一個 npu_rr_lock (round-robin, 等待 grant 時鎖定)。
//   外部讀回應依請求順序回來 (無回壓): 以 tag FIFO (深度 NDC × OUT) 記錄每個讀取屬於哪個通道, 回應送給 FIFO 頭的通道。
//   每個通道一個 ICG (閒置且沒有分派時停止時脈)。
module npu_dma_mc
  import orca_pkg::*;
#(
  parameter int NDC = 2,                // 通道數
  parameter int OUT = 8,                // 每個通道的在途上限 (2 的冪)
  parameter int DW  = 512
)(
  input  logic clk, rst_n, test_en,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [7:0]  cmd_id,
  input  logic        cmd_st,
  input  logic [1:0]  cmd_fmt,
  input  logic [31:0] cmd_ext,
  input  logic [25:0] cmd_l2,
  input  logic [15:0] cmd_n,
  input  logic [15:0] cmd_cols,
  input  logic [25:0] cmd_stride,
  input  logic        cmd_ind,
  input  logic [25:0] cmd_list,
  output logic        cpl_valid,
  input  logic        cpl_ready,
  output logic [7:0]  cpl_id,
  output logic        busy,             // 任一通道執行中
  // NPU L2 (經 orca_npu_ss 的 host 埠仲裁)
  output logic          h_rd_req,
  output logic [25:0]   h_rd_line,
  input  logic          h_rd_gnt,
  input  logic [DW-1:0] h_rd_data,
  output logic          h_wr_req,
  output logic [25:0]   h_wr_line,
  output logic [DW-1:0] h_wr_data,
  input  logic          h_wr_gnt,
  // 外部記憶體
  output logic          m_rd_req,
  output logic [31:0]   m_rd_addr,
  input  logic          m_rd_gnt,
  input  logic          m_rd_rvalid,
  input  logic [DW-1:0] m_rd_rdata,
  output logic          m_wr_req,
  output logic [31:0]   m_wr_addr,
  output logic [DW-1:0] m_wr_data,
  input  logic          m_wr_gnt
);
  localparam int CW = (NDC > 1) ? $clog2(NDC) : 1;
  localparam int TD = NDC * OUT;        // tag FIFO 深度 = 所有通道在途上限的總和
  localparam int TW = $clog2(TD);
  if (NDC < 1 || (1 << TW) != TD) begin : g_bad
    $error("npu_dma_mc: NDC >= 1 且 NDC × OUT 須為 2 的冪");
  end

  // ---------------- 通道 ----------------
  logic [NDC-1:0] c_ready, c_done, c_busy, c_take, cv;
  logic [NDC-1:0][7:0] c_id;
  logic [NDC-1:0] c_hrd_req, c_hrd_gnt, c_hwr_req, c_hwr_gnt, c_mrd_req, c_mrd_gnt, c_mrd_rv, c_mwr_req, c_mwr_gnt;
  logic [NDC-1:0][25:0]   c_hrd_line, c_hwr_line;
  logic [NDC-1:0][DW-1:0] c_hwr_data, c_mwr_data;
  logic [NDC-1:0][31:0]   c_mrd_addr, c_mwr_addr;

  // 空閒通道 (閒置且完成槽空) 中編號最小者接受命令
  wire [NDC-1:0] c_free = c_ready & ~cv;
  always_comb begin
    c_take = '0;
    for (int i = NDC - 1; i >= 0; i--) if (c_free[i]) c_take = NDC'(1) << i;
    if (!cmd_valid) c_take = '0;
  end
  assign cmd_ready = |c_free;
  assign busy      = |c_busy;

  for (genvar i = 0; i < NDC; i++) begin : g_ch
    logic gclk;
    clock_gate u_icg (.clk(clk), .en(c_busy[i] || c_take[i] || !rst_n), .test_en(test_en), .clk_out(gclk));
    npu_dma #(.OUT(OUT), .DW(DW)) u_dma (
      .clk(gclk), .rst_n(rst_n),
      .cmd_valid(c_take[i]), .cmd_ready(c_ready[i]), .cmd_st(cmd_st), .cmd_fmt(cmd_fmt),
      .cmd_ext(cmd_ext), .cmd_l2(cmd_l2), .cmd_n(cmd_n), .cmd_cols(cmd_cols), .cmd_stride(cmd_stride),
      .cmd_ind(cmd_ind), .cmd_list(cmd_list),
      .done(c_done[i]), .busy(c_busy[i]),
      .h_rd_req(c_hrd_req[i]), .h_rd_line(c_hrd_line[i]), .h_rd_gnt(c_hrd_gnt[i]), .h_rd_data(h_rd_data),
      .h_wr_req(c_hwr_req[i]), .h_wr_line(c_hwr_line[i]), .h_wr_data(c_hwr_data[i]), .h_wr_gnt(c_hwr_gnt[i]),
      .m_rd_req(c_mrd_req[i]), .m_rd_addr(c_mrd_addr[i]), .m_rd_gnt(c_mrd_gnt[i]),
      .m_rd_rvalid(c_mrd_rv[i]), .m_rd_rdata(m_rd_rdata),
      .m_wr_req(c_mwr_req[i]), .m_wr_addr(c_mwr_addr[i]), .m_wr_data(c_mwr_data[i]), .m_wr_gnt(c_mwr_gnt[i]));
  end

  // ---------------- 埠共用 ----------------
  logic [NDC-1:0] s_hrd, s_hwr, s_mrd, s_mwr;
  npu_rr_lock #(.NM(NDC)) u_a_hrd (.clk, .rst_n, .req(c_hrd_req), .m_gnt(h_rd_gnt), .out_req(h_rd_req), .sel(s_hrd), .gnt(c_hrd_gnt));
  npu_rr_lock #(.NM(NDC)) u_a_hwr (.clk, .rst_n, .req(c_hwr_req), .m_gnt(h_wr_gnt), .out_req(h_wr_req), .sel(s_hwr), .gnt(c_hwr_gnt));
  npu_rr_lock #(.NM(NDC)) u_a_mrd (.clk, .rst_n, .req(c_mrd_req), .m_gnt(m_rd_gnt), .out_req(m_rd_req), .sel(s_mrd), .gnt(c_mrd_gnt));
  npu_rr_lock #(.NM(NDC)) u_a_mwr (.clk, .rst_n, .req(c_mwr_req), .m_gnt(m_wr_gnt), .out_req(m_wr_req), .sel(s_mwr), .gnt(c_mwr_gnt));
  always_comb begin                     // one-hot AND-OR 多工
    h_rd_line = '0; h_wr_line = '0; h_wr_data = '0; m_rd_addr = '0; m_wr_addr = '0; m_wr_data = '0;
    for (int i = 0; i < NDC; i++) begin
      if (s_hrd[i]) h_rd_line = c_hrd_line[i];
      if (s_hwr[i]) begin h_wr_line = c_hwr_line[i]; h_wr_data = c_hwr_data[i]; end
      if (s_mrd[i]) m_rd_addr = c_mrd_addr[i];
      if (s_mwr[i]) begin m_wr_addr = c_mwr_addr[i]; m_wr_data = c_mwr_data[i]; end
    end
  end

  // 外部讀回應的通道 tag (依請求順序)
  logic [CW-1:0] tg [TD];
  logic [TW-1:0] t_rd, t_wr;
  logic [TW:0]   t_cnt;
  logic [CW-1:0] t_in;
  always_comb begin
    t_in = '0;
    for (int i = 0; i < NDC; i++) if (s_mrd[i]) t_in = CW'(i);
  end
  wire t_push = m_rd_req && m_rd_gnt;
  wire [CW-1:0] t_head = tg[t_rd];
  always_comb begin
    c_mrd_rv = '0;
    if (m_rd_rvalid) c_mrd_rv[t_head] = 1'b1;
  end
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin t_rd <= '0; t_wr <= '0; t_cnt <= '0; end
    else begin
      if (t_push) t_wr <= t_wr + 1'b1;
      if (m_rd_rvalid) t_rd <= t_rd + 1'b1;
      t_cnt <= t_cnt + (TW+1)'(t_push) - (TW+1)'(m_rd_rvalid);
    end
  always_ff @(posedge clk) if (t_push) tg[t_wr] <= t_in;

  // ---------------- 完成槽 + round-robin 輸出 ----------------
  logic [NDC-1:0] c_sel;
  logic [CW-1:0]  c_last;
  always_comb begin
    c_sel = '0;
    for (int i = NDC - 1; i >= 0; i--) begin
      automatic int s = (int'(c_last) + 1 + i) % NDC;
      if (cv[s]) c_sel = NDC'(1) << s;
    end
  end
  assign cpl_valid = |cv;
  always_comb begin
    cpl_id = '0;
    for (int i = 0; i < NDC; i++) if (c_sel[i]) cpl_id = c_id[i];
  end
  wire c_pop = cpl_valid && cpl_ready;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      cv <= '0; c_id <= '0; c_last <= CW'(NDC - 1);
    end else begin
      for (int i = 0; i < NDC; i++) begin
        if (c_take[i]) c_id[i] <= cmd_id;
        if (c_done[i]) cv[i] <= 1'b1;
        else if (c_pop && c_sel[i]) cv[i] <= 1'b0;
        if (c_pop && c_sel[i]) c_last <= CW'(i);
      end
    end
`ifdef FORMAL_TOP_npu_dma_mc
  // ---------------- 形式驗證 (formal/npu_dma_mc: 通道以 formal/npu_dma_mc/stubs.sv 的替身取代) ----------------
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  // 環境: 下游 grant 只在有請求時; 外部讀回應只在有在途讀取時 (gnt 之後至少 1 拍)
  logic [7:0] f_g, f_r;                                   // 外部讀 grant / 回應總數
  logic [NDC-1:0][7:0] f_o;                               // 每個通道的在途外部讀取 (從埠計算)
  always @(*) begin
    if (!h_rd_req) assume (!h_rd_gnt);
    if (!h_wr_req) assume (!h_wr_gnt);
    if (!m_rd_req) assume (!m_rd_gnt);
    if (!m_wr_req) assume (!m_wr_gnt);
    if (rst_n) begin
      assume (!m_rd_rvalid || f_g != f_r);
      assume (!t_push || f_g != 8'hFF);                     // 計數器不環繞 (足夠長於證明深度)
      for (int i = 0; i < NDC; i++) begin
        assume (!c_done[i] || f_o[i] == '0);                // 通道契約: 完成前讀取都已回來
        assume (!c_mrd_req[i] || f_o[i] < 8'(OUT));        // 通道契約: 在途 <= OUT
        assume (!c_mrd_req[i] || c_busy[i]);
      end
    end
  end
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_g <= '0; f_r <= '0; f_o <= '0; end
    else begin
      f_g <= f_g + 8'(t_push); f_r <= f_r + 8'(m_rd_rvalid);
      for (int i = 0; i < NDC; i++) f_o[i] <= f_o[i] + 8'(c_mrd_gnt[i]) - 8'(c_mrd_rv[i]);
    end
  // 任選第 f_k 個外部讀取: 它的回應必須送回發出它的通道
  (* anyconst *) logic [7:0] f_k;
  logic f_kv; logic [CW-1:0] f_kch;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_kv <= 1'b0; f_kch <= '0; end
    else if (t_push && f_g == f_k) begin f_kv <= 1'b1; f_kch <= t_in; end
  // 每個通道的命令 id 與完成槽
  logic [NDC-1:0][7:0] f_id;
  always @(posedge clk) for (int i = 0; i < NDC; i++) if (c_take[i]) f_id[i] <= cmd_id;
  wire [7:0] f_koff = f_k - f_r;
  // 命令守恆: 接受 − 回報 = 執行中或完成待回報的通道數 (完成不遺失、不重複)
  logic [7:0] f_acc, f_pop;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_acc <= '0; f_pop <= '0; end
    else begin f_acc <= f_acc + 8'(cmd_valid && cmd_ready); f_pop <= f_pop + 8'(cpl_valid && cpl_ready); end
  always @(*) if (rst_n) begin
    // 命令: 交給空閒通道中編號最小者
    assert (cmd_ready == ((c_ready & ~cv) != '0) && $onehot0(c_take));
    for (int i = 0; i < NDC; i++) if (c_take[i]) begin
      assert (cmd_valid && c_ready[i] && !cv[i]);
      for (int j = 0; j < i; j++) assert (!(c_ready[j] && !cv[j]));
    end
    if (cmd_valid && cmd_ready) assert (c_take != '0);
    // 埠: grant 只給一個請求者, 下游的位址 / 資料屬於拿到 grant 者
`ifndef FORMAL_NO_HELPER   // 輔助不變式: 埠選擇器的選擇 (npu_rr_lock 另外證明)
    assert ($onehot0(s_hrd) && $onehot0(s_hwr) && $onehot0(s_mrd) && $onehot0(s_mwr));
    assert ((s_hrd & ~c_hrd_req) == '0 && (s_hwr & ~c_hwr_req) == '0 && (s_mrd & ~c_mrd_req) == '0 && (s_mwr & ~c_mwr_req) == '0);
    assert ((s_hrd != '0) == h_rd_req && (s_hwr != '0) == h_wr_req && (s_mrd != '0) == m_rd_req && (s_mwr != '0) == m_wr_req);
`endif
    assert ($onehot0(c_hrd_gnt) && $onehot0(c_hwr_gnt) && $onehot0(c_mrd_gnt) && $onehot0(c_mwr_gnt));
    assert ((c_hrd_gnt & ~c_hrd_req) == '0 && (c_hwr_gnt & ~c_hwr_req) == '0 &&
            (c_mrd_gnt & ~c_mrd_req) == '0 && (c_mwr_gnt & ~c_mwr_req) == '0);
    assert (h_rd_req == (c_hrd_req != '0) && h_wr_req == (c_hwr_req != '0) &&
            m_rd_req == (c_mrd_req != '0) && m_wr_req == (c_mwr_req != '0));
    assert ((c_hrd_gnt != '0) == h_rd_gnt && (c_hwr_gnt != '0) == h_wr_gnt &&
            (c_mrd_gnt != '0) == m_rd_gnt && (c_mwr_gnt != '0) == m_wr_gnt);
    for (int i = 0; i < NDC; i++) begin
      if (c_hrd_gnt[i]) assert (h_rd_line == c_hrd_line[i]);
      if (c_hwr_gnt[i]) assert (h_wr_line == c_hwr_line[i] && h_wr_data == c_hwr_data[i]);
      if (c_mrd_gnt[i]) assert (m_rd_addr == c_mrd_addr[i]);
      if (c_mwr_gnt[i]) assert (m_wr_addr == c_mwr_addr[i] && m_wr_data == c_mwr_data[i]);
    end
    // 外部讀回應: 只送給一個有在途讀取的通道; 第 f_k 個回應送回發出第 f_k 個讀取的通道
    assert ($onehot0(c_mrd_rv) && ((c_mrd_rv != '0) == m_rd_rvalid));
    for (int i = 0; i < NDC; i++) if (c_mrd_rv[i]) assert (f_o[i] != '0);
    if (m_rd_rvalid && f_r == f_k) assert (f_kv && c_mrd_rv[f_kch]);
    // 完成: cpl_id = 該通道接受的命令 id; 完成槽滿時通道不接受命令
    assert (cpl_valid == (cv != '0) && $onehot0(c_sel) && ((c_sel & ~cv) == '0) && ((cv != '0) == (c_sel != '0)));
    for (int i = 0; i < NDC; i++) begin
      if (c_sel[i]) assert (cpl_id == f_id[i]);
      if (cv[i] || c_busy[i]) assert (c_id[i] == f_id[i]);
      assert (!(cv[i] && c_busy[i]));
    end
    assert (busy == (c_busy != '0));
    begin
      automatic logic [7:0] f_occ = '0;
      for (int i = 0; i < NDC; i++) f_occ = f_occ + 8'(c_busy[i] || cv[i]);
      assert (f_acc - f_pop == f_occ);
    end
`ifndef FORMAL_NO_HELPER
    // 輔助不變式: tag FIFO = 依序的在途讀取
    assert (32'(t_cnt) == 32'(f_g - f_r) && t_cnt <= (TW+1)'(TD) && TW'(t_wr - t_rd) == TW'(t_cnt));
    if (f_kv) assert (f_g > f_k);
    if (!f_kv) assert (f_g <= f_k);
    if (f_kv && f_koff < 8'(t_cnt) && f_r <= f_k) assert (tg[t_rd + TW'(f_koff)] == f_kch);
    // 每個通道的在途讀取數 = tag FIFO 中屬於它的項數
    for (int i = 0; i < NDC; i++) begin
      automatic logic [8:0] f_cn = '0;
      assert (f_o[i] <= 8'(OUT));
      for (int j = 0; j < TD; j++) if (j < int'(t_cnt) && tg[t_rd + TW'(j)] == CW'(i)) f_cn = f_cn + 1'b1;
      assert (f_cn == 9'(f_o[i]));
    end
`endif
  end
  always @(posedge clk) if (rst_n) begin
    cover (cv == '1 && cpl_valid && !cpl_ready);
    cover (c_busy == '1 && m_rd_rvalid && t_cnt > (TW+1)'(2));
    cover ((c_mrd_req == '1) && m_rd_gnt);
  end
`endif
endmodule
