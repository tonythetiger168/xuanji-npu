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
//   2D (v6.3.11, #33): cmd_cols != 0 時外部端為 2D — 每列 cmd_cols 個 line, 列與列相隔 cmd_stride 個 line;
//   第 i 個 line 的外部位址 = ext + (i / cols) × stride + i % cols, L2 端仍連續 (l2 + i)。cmd_cols = 0 為 1D
//   (外部連續, 與 v6.3.10 相同)。n 不必是 cols 的倍數 (最後一列不滿)。LD 時外部是讀端, ST 時是寫端。
module npu_ss_dma
  import orca_pkg::*;
#(
  parameter int OUT = 8,                // 在途上限 / FIFO 深度 (2 的冪)
  parameter int DW  = 512               // line 資料寬度 (DMA 只搬移資料, 形式驗證以較窄的 DW 證明)
)(
  input  logic clk, rst_n,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic        cmd_st,
  input  logic [31:0] cmd_ext,
  input  logic [25:0] cmd_l2,
  input  logic [15:0] cmd_n,
  input  logic [15:0] cmd_cols,         // 2D: 每列 line 數 (0 = 1D)
  input  logic [25:0] cmd_stride,       // 2D: 列距 (line)
  output logic        done,
  output logic        busy,
  // NPU L2 (npu_core host 埠)
  output logic         h_rd_req,
  output logic [25:0]  h_rd_line,
  input  logic         h_rd_gnt,
  input  logic [DW-1:0] h_rd_data,
  output logic         h_wr_req,
  output logic [25:0]  h_wr_line,
  output logic [DW-1:0] h_wr_data,
  input  logic         h_wr_gnt,
  // 外部記憶體
  output logic         m_rd_req,
  output logic [31:0]  m_rd_addr,
  input  logic         m_rd_gnt,
  input  logic         m_rd_rvalid,
  input  logic [DW-1:0] m_rd_rdata,
  output logic         m_wr_req,
  output logic [31:0]  m_wr_addr,
  output logic [DW-1:0] m_wr_data,
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
  logic [15:0] cols, col;                 // 2D: 每列 line 數 / 外部端目前列內位置
  logic [25:0] stride;                    // 2D: 列距
  logic [31:0] erow;                      // 2D: 外部端目前列的起點
  logic [DW-1:0] fifo [OUT];
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
  wire [31:0] e_addr = erow + 32'(col);   // 外部端 (LD: 讀, ST: 寫) 目前的 line 位址
  assign m_rd_addr = e_addr;
  assign h_rd_req  = can_iss && st;
  assign h_rd_line = l2 + 26'(iss);
  // 寫端: FIFO 頭 → LD 寫 L2, ST 寫外部
  assign h_wr_req  = act && f_nempty && !st;
  assign h_wr_line = l2 + 26'(wr);
  assign h_wr_data = fifo[f_rd];
  assign m_wr_req  = act && f_nempty && st;
  assign m_wr_addr = e_addr;
  assign m_wr_data = fifo[f_rd];

  // grant 只在本端有請求時有效 (外部埠的 gnt 可能與 req 無關)
  wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);
  wire push   = st ? (h_rd_req && h_rd_gnt) : m_rd_rvalid;
  wire pop    = st ? (m_wr_req && m_wr_gnt) : (h_wr_req && h_wr_gnt);
  wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;
  wire e_step = st ? pop : iss_ok;        // 外部端前進一個 line
  wire e_wrap = (cols != '0) && (col == cols - 1'b1);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act <= 1'b0; st <= 1'b0; ext <= '0; l2 <= '0; n <= '0; iss <= '0; wr <= '0;
      cols <= '0; col <= '0; stride <= '0; erow <= '0;
      f_rd <= '0; f_wr <= '0; f_cnt <= '0;
    end else begin
      if (cmd_valid && cmd_ready) begin
        act <= 1'b1; st <= cmd_st; ext <= cmd_ext; l2 <= cmd_l2; n <= cmd_n; iss <= '0; wr <= '0;
        cols <= cmd_cols; stride <= cmd_stride; col <= '0; erow <= cmd_ext;
      end else if (done) begin
        act <= 1'b0;
      end else begin
        if (iss_ok) iss <= iss + 1'b1;
        if (pop)    wr  <= wr + 1'b1;
        if (e_step) begin
          if (e_wrap) begin col <= '0; erow <= erow + 32'(stride); end
          else col <= col + 1'b1;
        end
      end
      if (push) f_wr <= f_wr + 1'b1;
      if (pop)  f_rd <= f_rd + 1'b1;
      f_cnt <= f_cnt + (OW+1)'(push) - (OW+1)'(pop);
    end
  end
  // FIFO 資料陣列不重置; 只在 push 時寫入
  always_ff @(posedge clk) if (push) fifo[f_wr] <= push_d;

`ifdef FORMAL
  // ---------------- 形式驗證 (SymbiYosys, formal/npu_ss_dma; 模擬時不編入) ----------------
  // 環境假設: 外部讀回應只在 LD 進行中、且有已 grant 未回應的讀時出現。
  // 性質以 k-induction 證明; 以下的不變式同時是歸納的強化條件。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  wire [15:0] f_in = st ? iss : 16'(wr + 16'(f_cnt));     // 本命令已進入 FIFO 的 line 數
  always @(*) if (rst_n) assume (!m_rd_rvalid || (act && !st && f_in != iss));
  // 資料完整: 任選一個 line index (anyconst), 從輸入埠記錄其資料 (不用內部的 push / push_d, 否則檢查會與
  // 被檢查的邏輯循環), 寫出時必須相同
  (* anyconst *) logic [15:0] f_idx;
  logic f_have;
  logic [DW-1:0] f_dat;
  wire          f_port_in  = st ? (h_rd_req && h_rd_gnt) : m_rd_rvalid;
  wire [DW-1:0] f_port_dat = st ? h_rd_data : m_rd_rdata;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) f_have <= 1'b0;
    else if (cmd_valid && cmd_ready) f_have <= 1'b0;
    else if (act && f_port_in && f_in == f_idx) begin f_have <= 1'b1; f_dat <= f_port_dat; end
  wire [15:0] f_off = f_idx - wr;
  always @(*) if (rst_n) begin
    if (!act) assert (f_cnt == '0 && !h_rd_req && !h_wr_req && !m_rd_req && !m_wr_req);
    if (act) begin
      assert (wr <= iss && iss <= n);
      assert (iss - wr <= 16'(OUT));
      assert (st ? (16'(f_cnt) == iss - wr) : (16'(f_cnt) <= iss - wr));
      assert (done == (wr == n));
      assert (f_have == (f_in > f_idx));
      // 被追蹤的 line 在 FIFO 中時, 位於 f_rd + (idx - wr)
      if (f_have && f_idx >= wr) assert (fifo[f_rd + OW'(f_off)] == f_dat);
      // 寫出第 idx 個 line 時資料正確
      if (wr == f_idx && (st ? m_wr_req : h_wr_req)) assert (f_have && (st ? m_wr_data : h_wr_data) == f_dat);
      // 有額度就發讀
      if (iss != n && iss - wr < 16'(OUT)) assert (st ? h_rd_req : m_rd_req);
    end
    assert (OW'(f_wr - f_rd) == OW'(f_cnt) && f_cnt <= (OW+1)'(OUT));
  end
  // 請求保持 / done 脈衝
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(m_rd_req && !m_rd_gnt)) assert (m_rd_req && m_rd_addr == $past(m_rd_addr));
    if ($past(m_wr_req && !m_wr_gnt)) assert (m_wr_req && m_wr_addr == $past(m_wr_addr) && m_wr_data == $past(m_wr_data));
    if ($past(h_rd_req && !h_rd_gnt)) assert (h_rd_req && h_rd_line == $past(h_rd_line));
    if ($past(h_wr_req && !h_wr_gnt)) assert (h_wr_req && h_wr_line == $past(h_wr_line) && h_wr_data == $past(h_wr_data));
    if ($past(done)) assert (!done);
  end
  // 位址: 檢查器從命令埠記下參數, 從外部端的握手 (LD: 讀 grant; ST: 寫 grant) 自己推進 2D 位置, 從 L2 端的握手
  // 計數; 送出的位址必須相符
  logic [31:0] f_ext0, f_erow;
  logic [25:0] f_stride;
  logic [15:0] f_cols, f_ecol, f_lcnt;
  logic [25:0] f_l20;
  logic        f_st;
  wire f_ehs = f_st ? (m_wr_req && m_wr_gnt) : (m_rd_req && m_rd_gnt);
  wire f_lhs = f_st ? (h_rd_req && h_rd_gnt) : (h_wr_req && h_wr_gnt);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_ecol <= '0; f_erow <= '0; f_lcnt <= '0; f_cols <= '0; f_stride <= '0; f_ext0 <= '0; f_l20 <= '0; f_st <= 1'b0; end
    else if (cmd_valid && cmd_ready) begin
      f_ext0 <= cmd_ext; f_erow <= cmd_ext; f_ecol <= '0; f_cols <= cmd_cols; f_stride <= cmd_stride;
      f_l20 <= cmd_l2; f_lcnt <= '0; f_st <= cmd_st;
    end else begin
      if (f_ehs) begin
        if (f_cols != '0 && f_ecol == f_cols - 1'b1) begin f_ecol <= '0; f_erow <= f_erow + 32'(f_stride); end
        else f_ecol <= f_ecol + 1'b1;
      end
      if (f_lhs) f_lcnt <= f_lcnt + 1'b1;
    end
  always @(*) if (rst_n && act) begin
    assert (f_st == st);
    if (m_rd_req) begin assert (m_rd_addr == f_erow + 32'(f_ecol)); end
    if (m_wr_req) begin assert (m_wr_addr == f_erow + 32'(f_ecol)); end
    if (h_rd_req) begin assert (h_rd_line == f_l20 + 26'(f_lcnt)); end
    if (h_wr_req) begin assert (h_wr_line == f_l20 + 26'(f_lcnt)); end
`ifndef FORMAL_NO_HELPER
    // 歸納輔助 (引用內部狀態)
    assert (cols == f_cols && stride == f_stride && col == f_ecol && erow == f_erow && l2 == f_l20);
    assert (f_lcnt == (st ? iss : wr) && (f_cols == '0 || f_ecol < f_cols));
`endif
  end
  always @(*) if (rst_n) begin
    cover (act && !st && done && n == 16'd6 && cols == 16'd2 && stride == 26'd5);
    cover (act && st && done && n == 16'd5 && cols == 16'd2);
    cover (act && !st && done && n == 16'd6);
    cover (act && st && done && n == 16'd6);
    cover (act && iss - wr == 16'(OUT) && iss != n);
  end
`endif
endmodule : npu_ss_dma
