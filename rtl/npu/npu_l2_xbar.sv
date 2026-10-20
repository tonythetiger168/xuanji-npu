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
  parameter int DEPTH = 1024
)(
  input  logic clk, rst_n,
  input  logic [NM-1:0]          rd_req,
  input  logic [NM-1:0][25:0]    rd_line,
  output logic [NM-1:0]          rd_gnt,
  output logic [NM-1:0][511:0]   rd_data,
  input  logic [NM-1:0]          wr_req,
  input  logic [NM-1:0][25:0]    wr_line,
  input  logic [NM-1:0][511:0]   wr_data,
  output logic [NM-1:0]          wr_gnt
);
  localparam int BW = $clog2(NB);
  localparam int DW = $clog2(DEPTH);
  localparam int MW = $clog2(NM);
  if (NB < 2 || (1 << BW) != NB || (1 << DW) != DEPTH || NM < 2) begin : g_bad
    $error("npu_l2_xbar: NB / DEPTH 須為 2 的冪 (NB >= 2), NM >= 2");
  end

  logic [511:0] mem [NB][DEPTH];
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
endmodule : npu_l2_xbar
