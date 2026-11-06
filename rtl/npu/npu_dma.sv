// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU DMA 引擎 (v6.3.13)
`include "orca_pkg.sv"

// npu_dma: 外部記憶體 <-> NPU L2 的 64 B line 搬移, 一次一個命令; 由 npu_ss_dma (v6.3.11) 加上載入時解碼。
//   npu_ss_dma 保持不變 (SIS 子系統 orca_sis_ss 使用); orca_npu_ss 從 v6.3.13 起改用本模組。
//   LD (st=0): 讀外部 ext..ext+n-1 → 寫 L2;  ST (st=1): 讀 L2 l2..l2+n-1 → 寫外部。n = 0 時立即完成。
//   命令: cmd_valid / cmd_ready (只在 idle 接受); done 為完成脈衝 (最後一個寫入 grant 的下一拍; n = 0 時接受後下一拍)。
//   外部讀埠: m_rd_req 保持到 m_rd_gnt; 讀資料依請求順序以 m_rd_rvalid 回傳 (gnt 之後至少 1 拍, 無回壓)。
//   外部寫埠: m_wr_req 保持到 m_wr_gnt, 寫在 gnt 拍完成。L2 埠語意同 npu_core host 埠 (讀資料與 gnt 同拍)。
//   在途上限 OUT: 已發出讀取但尚未寫出的 line 數 <= OUT, 由 OUT 項 FIFO 承接 (讀回應不需回壓)。
//   位址: 外部 line 位址 32 bit、L2 line 位址 26 bit, 皆以模數環繞。
//   2D: cmd_cols != 0 時外部端為 2D — 每列 cmd_cols 個 line, 列與列相隔 cmd_stride 個 line;
//   第 i 個 line 的外部位址 = ext + (i / cols) × stride + i % cols。cmd_cols = 0 為 1D。
//
// 載入時解碼 (v6.3.13, cmd_fmt; 只對 LD 有效, ST 一律原樣搬移):
//   0 COPY   原樣搬移, 外部 1 line → L2 1 line
//   1 INT4   有號 4 bit → INT8 (符號延伸)
//   2 UINT4  無號 4 bit → INT8 (補 0)
//   3 MXFP4  OCP MX FP4 (E2M1) 元素 × 2 → INT8: ±{0, 1, 2, 3, 4, 6, 8, 12} (E2M1 的值都是 0.5 的倍數, ×2 後精確;
//            −0 → 0); 區塊比例 (E8M0) 不在這裡套用 (由 GEMM 之後的反量化或 MX 區塊比例處理)
//   解碼模式下外部 1 line (128 個 4 bit 元素) → L2 2 line (128 個 INT8): 第 2i 個元素 = byte i 的低 4 bit
//   (OCP MX 的打包順序), 外部 line 的前半 (元素 0..63) 寫到 L2 的 l2 + 2j, 後半寫到 l2 + 2j + 1 (j = 外部 line 序號)。
//   n 仍是外部 line 數 (L2 寫入 2n 個 line)。外部頻寬需求減半: L2 寫入 1 line/拍時外部只需 0.5 line/拍。
module npu_dma
  import orca_pkg::*;
#(
  parameter int OUT = 8,                // 在途上限 / FIFO 深度 (2 的冪)
  parameter int DW  = 512               // line 資料寬度 (形式驗證以較窄的 DW 證明; 須為 16 的倍數)
)(
  input  logic clk, rst_n,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic        cmd_st,
  input  logic [1:0]  cmd_fmt,          // 載入時解碼 (LD): 0 COPY / 1 INT4 / 2 UINT4 / 3 MXFP4
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
  localparam int NE = DW / 8;           // 每個 L2 line 的 byte 數 (解碼時 = 半個外部 line 的 4 bit 元素數)
  if ((1 << OW) != OUT || OUT < 2) begin : g_bad
    $error("npu_dma: OUT 須為 2 的冪且 >= 2");
  end
  if (DW % 16 != 0) begin : g_bad_dw
    $error("npu_dma: DW 須為 16 的倍數");
  end

  // 4 bit → INT8 解碼 (fmt 1..3); E2M1 × 2 的大小以查表 (碼 0..7 → 0, 1, 2, 3, 4, 6, 8, 12)
  localparam logic [7:0] MX_TAB [8] = '{8'd0, 8'd1, 8'd2, 8'd3, 8'd4, 8'd6, 8'd8, 8'd12};
  function automatic logic [7:0] dec4(logic [1:0] f, logic [3:0] x);
    logic [7:0] m = MX_TAB[x[2:0]];
    return (f == 2'd1) ? {{4{x[3]}}, x} :
           (f == 2'd2) ? {4'd0, x} :
           (x[3] ? 8'(-m) : m);               // MXFP4 (−0 → 0)
  endfunction

  logic        act, st, dec;              // dec: 本命令為解碼 LD (fmt != 0 且 LD)
  logic [1:0]  fmt;
  logic        half;                      // 解碼: FIFO 頭的外部 line 目前寫第幾半
  logic [25:0] l2;
  logic [15:0] n, iss, wr;                // iss: 已發出讀取的 line 數; wr: 已寫出 (解碼時: 兩半都寫完) 的外部 line 數
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

  // FIFO 頭 → L2 寫入資料: COPY 原樣; 解碼時取前半 / 後半的 NE 個 4 bit 元素
  logic [DW-1:0] f_head, w_dec;
  assign f_head = fifo[f_rd];
  wire [DW/2-1:0] f_sel = half ? f_head[DW-1:DW/2] : f_head[DW/2-1:0];
  for (genvar e = 0; e < NE; e++) begin : g_dec      // generate (Verilator 不計 always 內 for 迴圈的 coverage)
    assign w_dec[e*8 +: 8] = dec4(fmt, f_sel[e*4 +: 4]);
  end

  // 讀端: LD 讀外部, ST 讀 L2
  assign m_rd_req  = can_iss && !st;
  wire [31:0] e_addr = erow + 32'(col);   // 外部端 (LD: 讀, ST: 寫) 目前的 line 位址
  assign m_rd_addr = e_addr;
  assign h_rd_req  = can_iss && st;
  assign h_rd_line = l2 + 26'(iss);
  // 寫端: FIFO 頭 → LD 寫 L2, ST 寫外部
  assign h_wr_req  = act && f_nempty && !st;
  assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(wr));
  assign h_wr_data = dec ? w_dec : f_head;
  assign m_wr_req  = act && f_nempty && st;
  assign m_wr_addr = e_addr;
  assign m_wr_data = f_head;

  // grant 只在本端有請求時有效 (外部埠的 gnt 可能與 req 無關)
  wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);
  wire push   = st ? (h_rd_req && h_rd_gnt) : m_rd_rvalid;
  wire w_fire = st ? (m_wr_req && m_wr_gnt) : (h_wr_req && h_wr_gnt);
  wire pop    = w_fire && (!dec || half);  // 解碼: 後半寫完才釋放 FIFO 頭
  wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;
  wire e_step = st ? pop : iss_ok;        // 外部端前進一個 line
  wire e_wrap = (cols != '0) && (col == cols - 1'b1);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act <= 1'b0; st <= 1'b0; dec <= 1'b0; fmt <= '0; half <= 1'b0; l2 <= '0; n <= '0; iss <= '0; wr <= '0;
      cols <= '0; col <= '0; stride <= '0; erow <= '0;
      f_rd <= '0; f_wr <= '0; f_cnt <= '0;
    end else begin
      if (cmd_valid && cmd_ready) begin
        act <= 1'b1; st <= cmd_st; l2 <= cmd_l2; n <= cmd_n; iss <= '0; wr <= '0;
        fmt <= cmd_fmt; dec <= !cmd_st && (cmd_fmt != 2'd0); half <= 1'b0;
        cols <= cmd_cols; stride <= cmd_stride; col <= '0; erow <= cmd_ext;
      end else if (done) begin
        act <= 1'b0;
      end else begin
        if (iss_ok) iss <= iss + 1'b1;
        if (pop)    wr  <= wr + 1'b1;
        if (dec && w_fire) half <= ~half;
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
  // ---------------- 形式驗證 (SymbiYosys, formal/npu_dma; 模擬時不編入) ----------------
  // 環境假設: 外部讀回應只在 LD 進行中、且有已 grant 未回應的讀時出現。
  // 性質以 k-induction 證明; 以下的不變式同時是歸納的強化條件。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  wire [15:0] f_in = st ? iss : 16'(wr + 16'(f_cnt));     // 本命令已進入 FIFO 的 line 數
  always @(*) if (rst_n) assume (!m_rd_rvalid || (act && !st && f_in != iss));
  // 檢查器自己的解碼參考 (與 RTL 的 dec4 分開寫: 以 E2M1 的定義 2^(e-1) × (1 + m/2) × 2 計算)
  function automatic logic [7:0] f_ref4(logic [1:0] f, logic [3:0] x);
    logic [7:0] m;
    if (x[2:1] == 2'd0) m = {7'd0, x[0]};                       // 次正規: 0 / 0.5 → 0 / 1
    else m = 8'((2 + x[0]) << (x[2:1] - 1));                    // 2^(e-1) × (2 + m)
    if (f == 2'd1) return 8'($signed(x));
    if (f == 2'd2) return 8'(x);
    return x[3] ? 8'(-m) : m;
  endfunction
  function automatic logic [DW-1:0] f_dec(logic [1:0] f, logic [DW-1:0] d, logic h);
    for (int e = 0; e < NE; e++) f_dec[e*8 +: 8] = f_ref4(f, d[(h ? DW/2 : 0) + e*4 +: 4]);
  endfunction
  // 資料完整: 任選一個 line index (anyconst), 從輸入埠記錄其資料 (不用內部的 push / push_d, 否則檢查會與
  // 被檢查的邏輯循環), 寫出時必須相同 (解碼時為檢查器自己解碼的前半 / 後半)
  (* anyconst *) logic [15:0] f_idx;
  logic f_have;
  logic [DW-1:0] f_dat;
  logic [1:0]    f_fmt;
  logic          f_dec_m;
  wire          f_port_in  = st ? (h_rd_req && h_rd_gnt) : m_rd_rvalid;
  wire [DW-1:0] f_port_dat = st ? h_rd_data : m_rd_rdata;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_have <= 1'b0; f_fmt <= '0; f_dec_m <= 1'b0; end
    else if (cmd_valid && cmd_ready) begin f_have <= 1'b0; f_fmt <= cmd_fmt; f_dec_m <= !cmd_st && cmd_fmt != 2'd0; end
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
      assert (fmt == f_fmt && dec == f_dec_m && (!dec || !st));
      assert (!half || (dec && f_nempty));
      // 被追蹤的 line 在 FIFO 中時, 位於 f_rd + (idx - wr)
      if (f_have && f_idx >= wr) assert (fifo[f_rd + OW'(f_off)] == f_dat);
      // 寫出第 idx 個 line 時資料正確 (解碼: 兩半各自為檢查器解碼的結果, L2 位址 l2 + 2 idx + 半)
      if (wr == f_idx && (st ? m_wr_req : h_wr_req)) begin
        assert (f_have);
        if (st) begin assert (m_wr_data == f_dat); end
        else if (f_dec_m) begin assert (h_wr_data == f_dec(f_fmt, f_dat, half)); end
        else begin assert (h_wr_data == f_dat); end
      end
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
  // 計數; 送出的位址必須相符 (解碼 LD: L2 端每個外部 line 寫 2 個 line, 位址 = l2 + L2 握手數)
  logic [31:0] f_ext0, f_erow;
  logic [25:0] f_stride;
  logic [15:0] f_cols, f_ecol;
  logic [16:0] f_lcnt;
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
    assert (f_lcnt == (st ? 17'(iss) : dec ? {wr, half} : 17'(wr)) && (f_cols == '0 || f_ecol < f_cols));
`endif
  end
  always @(*) if (rst_n) begin
    cover (act && !st && done && n == 16'd6 && cols == 16'd2 && stride == 26'd5);
    cover (act && st && done && n == 16'd5 && cols == 16'd2);
    cover (act && !st && done && n == 16'd6);
    cover (act && st && done && n == 16'd6);
    cover (act && iss - wr == 16'(OUT) && iss != n);
    cover (act && dec && fmt == 2'd3 && done && n == 16'd5);
    cover (act && dec && fmt == 2'd1 && iss - wr == 16'(OUT) && iss != n);
  end
`endif
endmodule : npu_dma
