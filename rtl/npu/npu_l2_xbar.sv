// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU banked L2 + crossbar (v6.3.8)
`include "orca_pkg.sv"

// npu_l2_xbar: NM 個讀取主端 / NW 個寫入主端 (計算單元 + host) 共用 NB 個 L2 bank
//   (line 交錯 + XOR 摺疊雜湊: bank = orca_pkg::npu_l2_bank, 列 = line[BW +: DW])。
//   NW 預設 = NM; 雙讀取埠的 CU (N = 32) 讓讀取主端多於寫入主端。
//   每個 bank 每拍 1 讀 + 1 寫 (1R1W SRAM 行為模型, 組合式讀取), 讀 / 寫各自 round-robin 仲裁。
//   主端介面與 npu_gemm_cu 相同: req 保持到 gnt; 讀資料與 gnt 同拍有效; 寫在 gnt 拍完成。
//   同一 line 同拍讀寫: 讀到舊值 (寫在時脈沿生效)。
//   容量 NB × DEPTH 個 64 B line; line 位址的 bank 列 = (line >> log2 NB) mod DEPTH (超出容量的位址環繞)。
// 低功耗: bank 陣列只在寫入 grant 時寫; 無請求的 bank 讀埠不動作 (行為模型以組合讀表示,
//         實體 SRAM 的 CE 取自該 bank 的讀 grant)。
//   SRAM light sleep (v6.3.11, LS_IDLE > 0): bank 連續 LS_IDLE 拍沒有任何讀 / 寫請求 → 進入 light sleep (bank_ls = 1,
//   接 SRAM 的 LS 腳: 保留內容、不可存取); 有請求指向睡眠中的 bank 時喚醒, 喚醒後 LS_WAKE 拍內不給 grant
//   (SRAM 的喚醒時間; 主端的 req 本來就保持到 gnt)。LS_IDLE = 0 時不睡 (與 v6.3.10 相同)。
module npu_l2_xbar
  import orca_pkg::*;
#(
  parameter int NM    = 5,             // 讀取主端數
  parameter int NW    = NM,            // 寫入主端數 (<= NM)
  parameter int NB    = 4,
  parameter int DEPTH = 1024,
  parameter int LDW   = 512,           // line 資料寬度 (形式驗證以較小值證明控制與資料搬移)
  parameter int LS_IDLE = 64,          // 閒置幾拍後進入 light sleep (0 = 不睡)
  parameter int LS_WAKE = 2            // light sleep 喚醒拍數 (>= 1)
)(
  input  logic clk, rst_n,
  input  logic [NM-1:0]          rd_req,
  input  logic [NM-1:0][25:0]    rd_line,
  output logic [NM-1:0]          rd_gnt,
  output logic [NM-1:0][LDW-1:0] rd_data,
  input  logic [NW-1:0]          wr_req,
  input  logic [NW-1:0][25:0]    wr_line,
  input  logic [NW-1:0][LDW-1:0] wr_data,
  output logic [NW-1:0]          wr_gnt,
  output logic [NB-1:0]          bank_ls        // 該 bank 的 SRAM 在 light sleep (喚醒中也算: 不可存取)
);
  localparam int BW = $clog2(NB);
  localparam int DW = $clog2(DEPTH);
  localparam int MW = $clog2(NM);
  localparam int WW = $clog2(NW);
  if (NB < 2 || (1 << BW) != NB || (1 << DW) != DEPTH || NM < 2 || NW < 2 || NW > NM) begin : g_bad
    $error("npu_l2_xbar: NB / DEPTH 須為 2 的冪 (NB >= 2), 2 <= NW <= NM");
  end

  logic [LDW-1:0] mem [NB][DEPTH];
  logic [NB-1:0][MW-1:0] rd_rr, rd_bm;           // 每個 bank 上一次得到讀 grant 的主端 / 本拍得到讀 grant 的主端
  logic [NB-1:0][WW-1:0] wr_rr, wr_bm;           // 同上 (寫)
  logic [NB-1:0]         rd_bv, wr_bv;           // bank 本拍有讀 / 寫 grant
  logic [NB-1:0]         rd_rq, wr_rq;           // bank 本拍有讀 / 寫請求 (不論是否醒著)
  logic [NB-1:0]         b_rdy;                  // bank 醒著 (可存取)

  // line → (bank, 列): orca_pkg::npu_l2_bank (XOR 摺疊雜湊) / npu_l2_row
  function automatic logic [BW-1:0] bank_of(logic [25:0] l); return BW'(npu_l2_bank(l, BW, DW)); endfunction
  function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)); endfunction

  // round-robin: 本 bank 的請求向量右旋 rr+1 位後取最低位 (等同從 rr+1 開始找第一個請求者)
  function automatic void arb(input logic [NM-1:0] req, input logic [NM-1:0][25:0] line,
                              input int b, input logic [MW-1:0] rr, output logic v, output logic [MW-1:0] m);
    logic [NM-1:0] rq;
    logic [2*NM-1:0] rot;
    int f;
    for (int i = 0; i < NM; i++) rq[i] = req[i] && bank_of(line[i]) == BW'(b);
    rot = {rq, rq} >> ((int'(rr) + 1) % NM);
    f = 0;
    for (int i = NM - 1; i >= 0; i--) if (rot[i]) f = i;
    v = (rq != '0);
    m = MW'((int'(rr) + 1 + f) % NM);
  endfunction
  function automatic void arbw(input logic [NW-1:0] req, input logic [NW-1:0][25:0] line,
                               input int b, input logic [WW-1:0] rr, output logic v, output logic [WW-1:0] m);
    logic [NW-1:0] rq;
    logic [2*NW-1:0] rot;
    int f;
    for (int i = 0; i < NW; i++) rq[i] = req[i] && bank_of(line[i]) == BW'(b);
    rot = {rq, rq} >> ((int'(rr) + 1) % NW);
    f = 0;
    for (int i = NW - 1; i >= 0; i--) if (rot[i]) f = i;
    v = (rq != '0);
    m = WW'((int'(rr) + 1 + f) % NW);
  endfunction

  always_comb begin
    for (int b = 0; b < NB; b++) begin
      arb(rd_req, rd_line, b, rd_rr[b], rd_rq[b], rd_bm[b]);
      arbw(wr_req, wr_line, b, wr_rr[b], wr_rq[b], wr_bm[b]);
      rd_bv[b] = rd_rq[b] && b_rdy[b];
      wr_bv[b] = wr_rq[b] && b_rdy[b];
    end
  end
  for (genvar m = 0; m < NM; m++) begin : g_m
    wire [BW-1:0] rb = bank_of(rd_line[m]);
    assign rd_gnt[m]  = rd_req[m] && rd_bv[rb] && rd_bm[rb] == MW'(m);
    assign rd_data[m] = mem[rb][row_of(rd_line[m])];
  end
  for (genvar m = 0; m < NW; m++) begin : g_w
    wire [BW-1:0] wb = bank_of(wr_line[m]);
    assign wr_gnt[m]  = wr_req[m] && wr_bv[wb] && wr_bm[wb] == WW'(m);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_rr <= '0; wr_rr <= '0;   // SRAM 陣列不重置
    end else begin
      for (int b = 0; b < NB; b++) begin
        if (rd_bv[b]) rd_rr[b] <= rd_bm[b];
        if (wr_bv[b]) wr_rr[b] <= wr_bm[b];
      end
    end
  end
  for (genvar b = 0; b < NB; b++) begin : g_bank
    always_ff @(posedge clk)
      if (wr_bv[b]) mem[b][row_of(wr_line[wr_bm[b]])] <= wr_data[wr_bm[b]];
  end
  // ---- SRAM light sleep (每個 bank) ----
  if (LS_IDLE > 0) begin : g_ls
    localparam int IW = (LS_IDLE > 1) ? $clog2(LS_IDLE) : 1, KW = $clog2(LS_WAKE + 1);   // idle 只數到 LS_IDLE - 1
    logic [NB-1:0]         ls;
    logic [NB-1:0][IW-1:0] idle;
    logic [NB-1:0][KW-1:0] wk;
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin ls <= '0; idle <= '0; wk <= '0; end
      else for (int b = 0; b < NB; b++) begin
        if (ls[b]) begin
          if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end
        end else if (wk[b] != '0) wk[b] <= wk[b] - 1'b1;
        else if (rd_rq[b] || wr_rq[b]) idle[b] <= '0;
        else if (idle[b] == IW'(LS_IDLE - 1)) begin ls[b] <= 1'b1; idle[b] <= '0; end
        else idle[b] <= idle[b] + 1'b1;
      end
    for (genvar b = 0; b < NB; b++) begin : g_b
      assign b_rdy[b]   = !ls[b] && wk[b] == '0;
      assign bank_ls[b] = !b_rdy[b];
    end
  end else begin : g_nols
    assign b_rdy = '1;
    assign bank_ls = '0;
  end
`ifdef FORMAL
  // ---------------- 形式驗證 (formal/npu_l2_xbar; 模擬時不編入) ----------------
  // 參考值一律由檢查器從埠記錄 (grant 誰、寫了什麼), 不引用 rd_rr / rd_bm 等被檢查的內部狀態
  localparam int LW = BW + DW;                          // 有效 line 位址位元 (更高位元模數環繞)
  initial assume (!rst_n);
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  // 主端介面契約: 請求等待 grant 期間保持 (位址 / 資料不變)
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    for (int m = 0; m < NM; m++)
      if ($past(rd_req[m] && !rd_gnt[m])) begin assume (rd_req[m] && rd_line[m] == $past(rd_line[m])); end
    for (int m = 0; m < NW; m++)
      if ($past(wr_req[m] && !wr_gnt[m])) begin
        assume (wr_req[m] && wr_line[m] == $past(wr_line[m]) && wr_data[m] == $past(wr_data[m]));
      end
  end
  // 每個 bank 上次得到讀 / 寫 grant 的主端 (由 gnt 埠記錄)
  logic [NB-1:0][MW-1:0] f_rl;
  logic [NB-1:0][WW-1:0] f_wl;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_rl <= '0; f_wl <= '0; end
    else begin
      for (int m = 0; m < NM; m++) if (rd_gnt[m]) f_rl[bank_of(rd_line[m])] <= MW'(m);
      for (int m = 0; m < NW; m++) if (wr_gnt[m]) f_wl[bank_of(wr_line[m])] <= WW'(m);
    end
  always @(*) if (rst_n) begin
    for (int b = 0; b < NB; b++) begin
      assert (int'(f_rl[b]) < NM && int'(f_wl[b]) < NW);
      assert (rd_rr[b] == f_rl[b] && wr_rr[b] == f_wl[b]);   // 歸納輔助
    end
    for (int m = 0; m < NM; m++) begin
      if (rd_gnt[m]) begin assert (rd_req[m]); end
      for (int n = m + 1; n < NM; n++)                       // 同一 bank 每拍最多一個讀 grant
        if (rd_gnt[m] && rd_gnt[n]) begin assert (bank_of(rd_line[m]) != bank_of(rd_line[n])); end
    end
    for (int m = 0; m < NW; m++) begin
      if (wr_gnt[m]) begin assert (wr_req[m]); end
      for (int n = m + 1; n < NW; n++)                       // 同一 bank 每拍最多一個寫 grant
        if (wr_gnt[m] && wr_gnt[n]) begin assert (bank_of(wr_line[m]) != bank_of(wr_line[n])); end
    end
  end
  // 確切 round-robin + work-conserving: bank 有請求者時, grant 給上次得主之後的第一個請求者
  function automatic logic [MW-1:0] f_first(input logic [NM-1:0] rq, input logic [MW-1:0] last);
    logic [MW-1:0] w;
    w = '0;
    for (int k = NM; k >= 1; k--) if (rq[(int'(last) + k) % NM]) w = MW'((int'(last) + k) % NM);
    return w;
  endfunction
  function automatic logic [WW-1:0] f_firstw(input logic [NW-1:0] rq, input logic [WW-1:0] last);
    logic [WW-1:0] w;
    w = '0;
    for (int k = NW; k >= 1; k--) if (rq[(int'(last) + k) % NW]) w = WW'((int'(last) + k) % NW);
    return w;
  endfunction
  // light sleep 的參考模型 (檢查器自己從請求埠計算: 閒置拍數、睡眠、喚醒倒數)
  logic [NB-1:0]       f_ls;
  logic [NB-1:0][7:0]  f_idle, f_wk;
  logic [NB-1:0]       f_brq;
  always @(*) for (int b = 0; b < NB; b++) begin
    f_brq[b] = 1'b0;
    for (int m = 0; m < NM; m++) if (rd_req[m] && bank_of(rd_line[m]) == BW'(b)) f_brq[b] = 1'b1;
    for (int m = 0; m < NW; m++) if (wr_req[m] && bank_of(wr_line[m]) == BW'(b)) f_brq[b] = 1'b1;
  end
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_ls <= '0; f_idle <= '0; f_wk <= '0; end
    else if (LS_IDLE > 0) for (int b = 0; b < NB; b++) begin
      if (f_ls[b]) begin if (f_brq[b]) begin f_ls[b] <= 1'b0; f_wk[b] <= 8'(LS_WAKE); end end
      else if (f_wk[b] != 0) f_wk[b] <= f_wk[b] - 1'b1;
      else if (f_brq[b]) f_idle[b] <= 0;
      else if (f_idle[b] == 8'(LS_IDLE - 1)) begin f_ls[b] <= 1'b1; f_idle[b] <= 0; end
      else f_idle[b] <= f_idle[b] + 1'b1;
    end
  always @(*) if (rst_n)
    for (int b = 0; b < NB; b++) begin
      logic [NM-1:0] rq;
      logic [NW-1:0] wq;
      logic          awake;
      awake = !f_ls[b] && f_wk[b] == 0;
      for (int m = 0; m < NM; m++) rq[m] = rd_req[m] && bank_of(rd_line[m]) == BW'(b);
      for (int m = 0; m < NW; m++) wq[m] = wr_req[m] && bank_of(wr_line[m]) == BW'(b);
      assert (bank_ls[b] == !awake);
      if (awake && rq != '0) begin assert (rd_gnt[f_first(rq, f_rl[b])]); end
      if (awake && wq != '0) begin assert (wr_gnt[f_firstw(wq, f_wl[b])]); end
      for (int m = 0; m < NM; m++) if (!awake && rq[m]) begin assert (!rd_gnt[m]); end   // 睡眠 / 喚醒中不可存取
      for (int m = 0; m < NW; m++) if (!awake && wq[m]) begin assert (!wr_gnt[m]); end
      if (LS_IDLE > 0) begin assert (int'(f_idle[b]) < LS_IDLE && int'(f_wk[b]) <= LS_WAKE && !(f_ls[b] && f_wk[b] != 0)); end
      else begin assert (!f_ls[b] && f_wk[b] == 0 && f_idle[b] == 0); end
    end
  // 歸納輔助: RTL 的睡眠計數器與參考模型一致 (只幫助歸納; 被抓到的錯誤由上面的 bank_ls / grant 性質判定)
`ifndef FORMAL_NO_HELPER
  if (LS_IDLE > 0) begin : g_fls
    always @(*) if (rst_n) for (int b = 0; b < NB; b++) begin
      assert (g_ls.ls[b] == f_ls[b] && 8'(g_ls.idle[b]) == f_idle[b] && 8'(g_ls.wk[b]) == f_wk[b]);
    end
  end
`endif
  // 等待上限: 保持中的請求在 NM - 1 (寫: NW - 1) 拍內得到 grant, 加上 light sleep 喚醒的 LS_WAKE + 1 拍
  logic [NM-1:0][3:0] f_rw;
  logic [NW-1:0][3:0] f_ww;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_rw <= '0; f_ww <= '0; end
    else begin
      for (int m = 0; m < NM; m++) f_rw[m] <= (rd_req[m] && !rd_gnt[m]) ? f_rw[m] + 1'b1 : '0;
      for (int m = 0; m < NW; m++) f_ww[m] <= (wr_req[m] && !wr_gnt[m]) ? f_ww[m] + 1'b1 : '0;
    end
  always @(*) if (rst_n) begin
    for (int m = 0; m < NM; m++) begin assert (int'(f_rw[m]) <= NM - 1 + ((LS_IDLE > 0) ? LS_WAKE + 1 : 0)); end
    for (int m = 0; m < NW; m++) begin assert (int'(f_ww[m]) <= NW - 1 + ((LS_IDLE > 0) ? LS_WAKE + 1 : 0)); end
  end
  // 資料: 任選一個 line (anyconst, 只比有效位元), 由寫入埠記錄最後寫入的資料; 讀到該 line 必須相同
  (* anyconst *) logic [LW-1:0] f_l;
  logic         f_have;
  logic [LDW-1:0] f_dat;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) f_have <= 1'b0;
    else for (int m = 0; m < NW; m++)
      if (wr_gnt[m] && wr_line[m][LW-1:0] == f_l) begin f_have <= 1'b1; f_dat <= wr_data[m]; end
  always @(*) if (rst_n) begin
`ifndef FORMAL_NO_HELPER
    if (f_have) begin assert (mem[bank_of(26'(f_l))][row_of(26'(f_l))] == f_dat); end   // 歸納輔助
`endif
    for (int m = 0; m < NM; m++)
      if (f_have && rd_gnt[m] && rd_line[m][LW-1:0] == f_l) begin assert (rd_data[m] == f_dat); end
  end
  always @(*) if (rst_n) begin
    cover (rd_req == '1 && rd_gnt != rd_req);                // 讀衝突
    cover (wr_gnt[0] && wr_gnt[NW-1]);                        // 不同 bank 同拍寫
    cover (f_have && rd_gnt[NM-1] && rd_line[NM-1][LW-1:0] == f_l); // 讀回寫入值 (讀取專用主端)
    cover (f_rw[0] == 4'(NM - 1));                            // 最長等待
    cover (f_ls[0] && f_brq[0]);                              // 睡眠中的 bank 被喚醒
  end
`endif
endmodule : npu_l2_xbar
