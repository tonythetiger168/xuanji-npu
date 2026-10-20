// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU Subsystem DMA (v6.3.9)
`include "orca_pkg.sv"

// npu_ss_dma: 外部記憶體 <-> NPU L2 的 64 B line 搬移, 一次一個命令。
//   LD (st=0): 讀外部 ext..ext+n-1 → 寫 L2 l2..l2+n-1;  ST (st=1): 讀 L2 → 寫外部。n = 0 時立即完成。
//   命令: cmd_valid / cmd_ready (只在 idle 接受); done 為完成脈衝 (最後一個寫入 grant 的下一拍; n = 0 時接受後下一拍)。
//   外部讀埠: m_rd_req 保持到 m_rd_gnt; 讀資料依請求順序以 m_rd_rvalid 回傳 (gnt 之後至少 1 拍, 無回壓)。
//   外部寫埠: m_wr_req 保持到 m_wr_gnt, 寫在 gnt 拍完成。L2 埠語意同 npu_core host 埠 (讀資料與 gnt 同拍)。
//   在途上限 OUT: 已發出讀取但尚未寫出的 line 數 <= OUT, 由 OUT 項 FIFO 承接 (讀回應不需回壓)。
//   位址: 外部 line 位址 32 bit、L2 line 位址 26 bit, 皆以模數環繞。
module npu_ss_dma
  import orca_pkg::*;
#(
  parameter int OUT = 8                 // 在途上限 / FIFO 深度 (2 的冪)
)(
  input  logic clk, rst_n,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic        cmd_st,
  input  logic [31:0] cmd_ext,
  input  logic [25:0] cmd_l2,
  input  logic [15:0] cmd_n,
  output logic        done,
  output logic        busy,
  // NPU L2 (npu_core host 埠)
  output logic         h_rd_req,
  output logic [25:0]  h_rd_line,
  input  logic         h_rd_gnt,
  input  logic [511:0] h_rd_data,
  output logic         h_wr_req,
  output logic [25:0]  h_wr_line,
  output logic [511:0] h_wr_data,
  input  logic         h_wr_gnt,
  // 外部記憶體
  output logic         m_rd_req,
  output logic [31:0]  m_rd_addr,
  input  logic         m_rd_gnt,
  input  logic         m_rd_rvalid,
  input  logic [511:0] m_rd_rdata,
  output logic         m_wr_req,
  output logic [31:0]  m_wr_addr,
  output logic [511:0] m_wr_data,
  input  logic         m_wr_gnt
);
  localparam int OW = $clog2(OUT);
  if ((1 << OW) != OUT || OUT < 2) begin : g_bad
    $error("npu_ss_dma: OUT 須為 2 的冪且 >= 2");
  end

  logic        act, st;
  logic [31:0] ext;
  logic [25:0] l2;
  logic [15:0] n, iss, wr;                // iss: 已發出讀取的 line 數; wr: 已寫出的 line 數
  logic [511:0] fifo [OUT];
  logic [OW-1:0] f_rd, f_wr;
  logic [OW:0]   f_cnt;

  // 在途 (已讀未寫) line 數 = iss - wr <= OUT (SVA a_outst), 只需低 log2(OUT)+1 bit 的差
  wire [OW:0] outst = iss[OW:0] - wr[OW:0];
  wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT));
  wire f_nempty = (f_cnt != '0);

  assign cmd_ready = !act;
  assign busy      = act;
  assign done      = act && (wr == n);

  // 讀端: LD 讀外部, ST 讀 L2
  assign m_rd_req  = can_iss && !st;
  assign m_rd_addr = ext + 32'(iss);
  assign h_rd_req  = can_iss && st;
  assign h_rd_line = l2 + 26'(iss);
  // 寫端: FIFO 頭 → LD 寫 L2, ST 寫外部
  assign h_wr_req  = act && f_nempty && !st;
  assign h_wr_line = l2 + 26'(wr);
  assign h_wr_data = fifo[f_rd];
  assign m_wr_req  = act && f_nempty && st;
  assign m_wr_addr = ext + 32'(wr);
  assign m_wr_data = fifo[f_rd];

  // grant 只在本端有請求時有效 (外部埠的 gnt 可能與 req 無關)
  wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);
  wire push   = st ? (h_rd_req && h_rd_gnt) : m_rd_rvalid;
  wire pop    = st ? (m_wr_req && m_wr_gnt) : (h_wr_req && h_wr_gnt);
  wire [511:0] push_d = st ? h_rd_data : m_rd_rdata;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act <= 1'b0; st <= 1'b0; ext <= '0; l2 <= '0; n <= '0; iss <= '0; wr <= '0;
      f_rd <= '0; f_wr <= '0; f_cnt <= '0;
    end else begin
      if (cmd_valid && cmd_ready) begin
        act <= 1'b1; st <= cmd_st; ext <= cmd_ext; l2 <= cmd_l2; n <= cmd_n; iss <= '0; wr <= '0;
      end else if (done) begin
        act <= 1'b0;
      end else begin
        if (iss_ok) iss <= iss + 1'b1;
        if (pop)    wr  <= wr + 1'b1;
      end
      if (push) f_wr <= f_wr + 1'b1;
      if (pop)  f_rd <= f_rd + 1'b1;
      f_cnt <= f_cnt + (OW+1)'(push) - (OW+1)'(pop);
    end
  end
  // FIFO 資料陣列不重置; 只在 push 時寫入
  always_ff @(posedge clk) if (push) fifo[f_wr] <= push_d;
endmodule : npu_ss_dma
