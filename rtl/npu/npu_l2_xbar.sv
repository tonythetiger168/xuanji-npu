// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU banked L2 + crossbar (v6.3.8)
`include "orca_pkg.sv"

// npu_l2_xbar: NM 個主端 (計算單元 + host) 共用 NB 個 L2 bank (line 交錯: bank = line[log2 NB-1:0])。
//   每個 bank 每拍 1 讀 + 1 寫 (1R1W SRAM 行為模型, 組合式讀取), 讀 / 寫各自 round-robin 仲裁。
//   主端介面與 npu_gemm_cu 相同: req 保持到 gnt; 讀資料與 gnt 同拍有效; 寫在 gnt 拍完成。
//   同一 line 同拍讀寫: 讀到舊值 (寫在時脈沿生效)。
//   容量 NB × DEPTH 個 64 B line; line 位址的 bank 列 = (line >> log2 NB) mod DEPTH。
// 低功耗: bank 陣列只在寫入 grant 時寫; 無請求的 bank 讀埠不動作 (行為模型以組合讀表示,
//         實體 SRAM 的 CE 取自該 bank 的讀 grant)。
module npu_l2_xbar
  import orca_pkg::*;
#(
  parameter int NM    = 5,
  parameter int NB    = 4,
  parameter int DEPTH = 1024,
  parameter int LDW   = 512            // line 資料寬度 (形式驗證以較小值證明控制與資料搬移)
)(
  input  logic clk, rst_n,
  input  logic [NM-1:0]          rd_req,
  input  logic [NM-1:0][25:0]    rd_line,
  output logic [NM-1:0]          rd_gnt,
  output logic [NM-1:0][LDW-1:0] rd_data,
  input  logic [NM-1:0]          wr_req,
  input  logic [NM-1:0][25:0]    wr_line,
  input  logic [NM-1:0][LDW-1:0] wr_data,
  output logic [NM-1:0]          wr_gnt
);
  localparam int BW = $clog2(NB);
  localparam int DW = $clog2(DEPTH);
  localparam int MW = $clog2(NM);
  if (NB < 2 || (1 << BW) != NB || (1 << DW) != DEPTH || NM < 2) begin : g_bad
    $error("npu_l2_xbar: NB / DEPTH 須為 2 的冪 (NB >= 2), NM >= 2");
  end

  logic [LDW-1:0] mem [NB][DEPTH];
  logic [NB-1:0][MW-1:0] rd_rr, wr_rr;           // 每個 bank 上一次得到 grant 的主端
  logic [NB-1:0]         rd_bv, wr_bv;           // bank 本拍有讀 / 寫 grant
  logic [NB-1:0][MW-1:0] rd_bm, wr_bm;           // 得到 grant 的主端

  function automatic logic [BW-1:0] bank_of(logic [25:0] l); return l[BW-1:0]; endfunction
  function automatic logic [DW-1:0] row_of(logic [25:0] l);  return l[BW +: DW]; endfunction

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

  always_comb begin
    for (int b = 0; b < NB; b++) begin
      arb(rd_req, rd_line, b, rd_rr[b], rd_bv[b], rd_bm[b]);
      arb(wr_req, wr_line, b, wr_rr[b], wr_bv[b], wr_bm[b]);
    end
  end
  for (genvar m = 0; m < NM; m++) begin : g_m
    wire [BW-1:0] rb = bank_of(rd_line[m]);
    wire [BW-1:0] wb = bank_of(wr_line[m]);
    assign rd_gnt[m]  = rd_req[m] && rd_bv[rb] && rd_bm[rb] == MW'(m);
    assign wr_gnt[m]  = wr_req[m] && wr_bv[wb] && wr_bm[wb] == MW'(m);
    assign rd_data[m] = mem[rb][row_of(rd_line[m])];
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
`ifdef FORMAL
  // ---------------- 形式驗證 (formal/npu_l2_xbar; 模擬時不編入) ----------------
  // 參考值一律由檢查器從埠記錄 (grant 誰、寫了什麼), 不引用 rd_rr / rd_bm 等被檢查的內部狀態
  localparam int LW = BW + DW;                          // 有效 line 位址位元 (更高位元模數環繞)
  initial assume (!rst_n);
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  // 主端介面契約: 請求等待 grant 期間保持 (位址 / 資料不變)
  always @(posedge clk) if (f_past && rst_n && $past(rst_n))
    for (int m = 0; m < NM; m++) begin
      if ($past(rd_req[m] && !rd_gnt[m])) begin assume (rd_req[m] && rd_line[m] == $past(rd_line[m])); end
      if ($past(wr_req[m] && !wr_gnt[m])) begin
        assume (wr_req[m] && wr_line[m] == $past(wr_line[m]) && wr_data[m] == $past(wr_data[m]));
      end
    end
  // 每個 bank 上次得到讀 / 寫 grant 的主端 (由 gnt 埠記錄)
  logic [NB-1:0][MW-1:0] f_rl, f_wl;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_rl <= '0; f_wl <= '0; end
    else for (int m = 0; m < NM; m++) begin
      if (rd_gnt[m]) f_rl[bank_of(rd_line[m])] <= MW'(m);
      if (wr_gnt[m]) f_wl[bank_of(wr_line[m])] <= MW'(m);
    end
  always @(*) if (rst_n) begin
    for (int b = 0; b < NB; b++) begin
      assert (int'(f_rl[b]) < NM && int'(f_wl[b]) < NM);
      assert (rd_rr[b] == f_rl[b] && wr_rr[b] == f_wl[b]);   // 歸納輔助
    end
    for (int m = 0; m < NM; m++) begin
      if (rd_gnt[m]) begin assert (rd_req[m]); end
      if (wr_gnt[m]) begin assert (wr_req[m]); end
      for (int n = m + 1; n < NM; n++) begin
        // 同一 bank 每拍最多一個讀 grant、一個寫 grant
        if (rd_gnt[m] && rd_gnt[n]) begin assert (bank_of(rd_line[m]) != bank_of(rd_line[n])); end
        if (wr_gnt[m] && wr_gnt[n]) begin assert (bank_of(wr_line[m]) != bank_of(wr_line[n])); end
      end
    end
  end
  // 確切 round-robin + work-conserving: bank 有請求者時, grant 給上次得主之後的第一個請求者
  function automatic logic [MW-1:0] f_first(input logic [NM-1:0] rq, input logic [MW-1:0] last);
    logic [MW-1:0] w;
    w = '0;
    for (int k = NM; k >= 1; k--) if (rq[(int'(last) + k) % NM]) w = MW'((int'(last) + k) % NM);
    return w;
  endfunction
  always @(*) if (rst_n)
    for (int b = 0; b < NB; b++) begin
      logic [NM-1:0] rq, wq;
      for (int m = 0; m < NM; m++) begin
        rq[m] = rd_req[m] && bank_of(rd_line[m]) == BW'(b);
        wq[m] = wr_req[m] && bank_of(wr_line[m]) == BW'(b);
      end
      if (rq != '0) begin assert (rd_gnt[f_first(rq, f_rl[b])]); end
      if (wq != '0) begin assert (wr_gnt[f_first(wq, f_wl[b])]); end
    end
  // 等待上限: 保持中的請求在 NM - 1 拍內得到 grant
  logic [NM-1:0][3:0] f_rw, f_ww;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_rw <= '0; f_ww <= '0; end
    else for (int m = 0; m < NM; m++) begin
      f_rw[m] <= (rd_req[m] && !rd_gnt[m]) ? f_rw[m] + 1'b1 : '0;
      f_ww[m] <= (wr_req[m] && !wr_gnt[m]) ? f_ww[m] + 1'b1 : '0;
    end
  always @(*) if (rst_n)
    for (int m = 0; m < NM; m++) begin
      assert (int'(f_rw[m]) <= NM - 1 && int'(f_ww[m]) <= NM - 1);
    end
  // 資料: 任選一個 line (anyconst, 只比有效位元), 由寫入埠記錄最後寫入的資料; 讀到該 line 必須相同
  (* anyconst *) logic [LW-1:0] f_l;
  logic         f_have;
  logic [LDW-1:0] f_dat;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) f_have <= 1'b0;
    else for (int m = 0; m < NM; m++)
      if (wr_gnt[m] && wr_line[m][LW-1:0] == f_l) begin f_have <= 1'b1; f_dat <= wr_data[m]; end
  always @(*) if (rst_n) begin
    if (f_have) begin assert (mem[f_l[BW-1:0]][f_l[LW-1:BW]] == f_dat); end   // 歸納輔助
    for (int m = 0; m < NM; m++)
      if (f_have && rd_gnt[m] && rd_line[m][LW-1:0] == f_l) begin assert (rd_data[m] == f_dat); end
  end
  always @(*) if (rst_n) begin
    cover (rd_req == '1 && rd_gnt != rd_req);                // 讀衝突
    cover (wr_gnt[0] && wr_gnt[NM-1]);                        // 不同 bank 同拍寫
    cover (f_have && rd_gnt[1] && rd_line[1][LW-1:0] == f_l); // 讀回寫入值
    cover (f_rw[0] == 4'(NM - 1));                            // 最長等待
  end
`endif
endmodule : npu_l2_xbar
