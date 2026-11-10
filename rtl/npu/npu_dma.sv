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
//
// 間接 (v6.3.13, cmd_ind; gather / scatter, 供 paged KV 與 MoE 使用):
//   位址表在 L2 line cmd_list 起: 每項 u32 (小端), 每個 line EPL = 16 項, 第 i 項在 line list + i/16 的 byte 4 × (i % 16)。
//   cmd_n = 項數; 每項搬 cw = (cmd_cols = 0 ? 1 : cmd_cols) 個 line; 第 i 項的值 E: 外部端為 ext + E .. ext + E + cw − 1,
//   L2 端仍連續 (l2 + i × cw; 解碼時 × 2)。LD = 從外部 gather, ST = scatter 到外部。cmd_stride 不使用。
//   總 line 數 k × cw 須 <= 65535 (計數器為 16 bit; 較長的 gather 由軟體切成多個命令)。
//   E = 0xFFFF_FFFF 為略過: LD 寫 0 到該項的 L2 line (不讀外部; 為保持 FIFO 順序, 等先前的外部讀取都回來才進入),
//   ST 不寫外部 (L2 照讀)。位址表經 host 讀埠讀取 (2 個 line 緩衝, 預取下一個; ST 時與資料讀取共用, 位址表優先)。
module npu_dma
  import orca_pkg::*;
#(
  parameter int OUT = 8,                // 在途上限 / FIFO 深度 (2 的冪)
  parameter int DW  = 512               // line 資料寬度 (形式驗證以較窄的 DW 證明; 須為 32 的倍數且 >= 64)
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
  input  logic        cmd_ind,          // 間接 (v6.3.13): 外部位址取自 L2 的位址表 (見檔頭)
  input  logic [25:0] cmd_list,         // 間接: 位址表的 L2 line 位址
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
  if (DW % 32 != 0 || DW < 64) begin : g_bad_dw
    $error("npu_dma: DW 須為 32 的倍數且 >= 64 (位址表每 line 至少 2 項)");
  end

  // 4 bit → INT8 解碼 (fmt 1..3); E2M1 × 2 的大小以查表 (碼 0..7 → 0, 1, 2, 3, 4, 6, 8, 12)
  localparam logic [7:0] MX_TAB [8] = '{8'd0, 8'd1, 8'd2, 8'd3, 8'd4, 8'd6, 8'd8, 8'd12};
  function automatic logic [7:0] dec4(logic [1:0] f, logic [3:0] x);
    logic [7:0] m = MX_TAB[x[2:0]];
    return (f == 2'd1) ? {{4{x[3]}}, x} :
           (f == 2'd2) ? {4'd0, x} :
           (x[3] ? 8'(-m) : m);               // MXFP4 (−0 → 0)
  endfunction

  localparam int EPL = DW / 32;          // 每個位址表 line 的項數 (DW = 512: 16)
  localparam int EW  = $clog2(EPL);
  localparam int LW  = $clog2((65535 + EPL - 1) / EPL + 1);   // 位址表 line 計數寬度 (DW = 512: 13 bit)
  logic        act, st, dec, ind;         // dec: 本命令為解碼 LD (fmt != 0 且 LD); ind: 間接
  logic [1:0]  fmt;
  logic        half;                      // 解碼: FIFO 頭的外部 line 目前寫第幾半
  logic [25:0] l2;
  logic [31:0] ext;
  logic [15:0] n, iss, wr;                // 資料 line 數 (間接: 項數 × 每項 line 數); iss: 已發出讀取; wr: 已寫出 (解碼時: 兩半都寫完)
  logic [15:0] cols, col;                 // 2D: 每列 line 數 / 外部端目前列內位置; 間接: 每項 line 數 (0 = 1) / 項內位置
  logic [25:0] stride;                    // 2D: 列距
  logic [31:0] erow;                      // 2D: 外部端目前列的起點
  logic [DW-1:0] fifo [OUT];
  logic [OW-1:0] f_rd, f_wr;
  logic [OW:0]   f_cnt;
  // 間接: 位址表 2 個 line 的緩衝 (預取下一個 line); 外部端目前的項 = lb[lh] 的第 ent % EPL 項
  logic [25:0]   list;
  logic [LW-1:0] nl, lf;                  // 位址表 line 數 = ceil(項數 / EPL) / 已讀取數
  logic [1:0][DW-1:0] lb;
  logic [1:0]    lbv;
  logic          lh;
  wire           lo = ~lh;               // 另一項 (以 1 bit 訊號索引; ~lh 直接放進索引運算式時, sv2v 會在 32 bit 語境展開)
  logic [EW-1:0] ent;                     // 外部端目前的項在 line 內的位置
  logic          hp, hp_l;                // host 讀埠: 上一拍有請求未 grant / 當時選的是位址表讀取 (請求保持)

  // 在途 (已讀未寫) line 數 = iss - wr <= OUT (SVA a_outst), 只需低 log2(OUT)+1 bit 的差
  wire [OW:0] outst = iss[OW:0] - wr[OW:0];
  wire f_nempty = (f_cnt != '0);
  // 間接: 目前的項 (外部端: LD 在發出讀取時、ST 在寫出時前進); 0xFFFF_FFFF = 略過
  wire        e_v    = !ind || lbv[lh];
  wire [31:0] e_val  = lb[lh][32'(ent) * 32 +: 32];
  wire        e_skip = ind && (e_val == 32'hFFFF_FFFF);
  // LD 略過的 line 不讀外部, 以 0 進入 FIFO; 為了保持 FIFO 順序, 只在沒有尚未回應的外部讀取時進入
  wire [OW:0] pend_rd = outst - f_cnt;    // 已發出、尚未回應的外部讀取數 (LD)
  wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT)) && (st || e_v);
  wire ld_skip = can_iss && !st && e_skip && (pend_rd == '0);

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

  // 讀端: LD 讀外部, ST 讀 L2; host 讀埠另做位址表讀取 (優先; 等待 grant 時保持同一種請求)
  wire l_need  = act && ind && (lf != nl) && (lbv != 2'b11);
  wire sel_l   = hp ? hp_l : l_need;
  wire d_hreq  = can_iss && st;
  assign m_rd_req  = can_iss && !st && !e_skip;
  wire [31:0] e_addr = ind ? (ext + e_val + 32'(col)) : (erow + 32'(col));   // 外部端目前的 line 位址
  assign m_rd_addr = e_addr;
  assign h_rd_req  = sel_l || d_hreq;
  assign h_rd_line = sel_l ? (list + 26'(lf)) : (l2 + 26'(iss));
  // 寫端: FIFO 頭 → LD 寫 L2, ST 寫外部 (間接 ST 的略過項: 不寫外部, 直接釋放)
  assign h_wr_req  = act && f_nempty && !st;
  assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(wr));
  assign h_wr_data = dec ? w_dec : f_head;
  wire   st_wv     = act && f_nempty && st && e_v;
  assign m_wr_req  = st_wv && !e_skip;
  assign m_wr_addr = e_addr;
  assign m_wr_data = f_head;

  // grant 只在本端有請求時有效 (外部埠的 gnt 可能與 req 無關)
  wire l_fire = sel_l && h_rd_gnt;
  wire iss_ok = st ? (d_hreq && !sel_l && h_rd_gnt) : ((m_rd_req && m_rd_gnt) || ld_skip);
  wire push   = st ? (d_hreq && !sel_l && h_rd_gnt) : (m_rd_rvalid || ld_skip);
  wire w_fire = st ? ((m_wr_req && m_wr_gnt) || (st_wv && e_skip)) : (h_wr_req && h_wr_gnt);
  wire pop    = w_fire && (!dec || half);  // 解碼: 後半寫完才釋放 FIFO 頭
  wire [DW-1:0] push_d = st ? h_rd_data : ld_skip ? '0 : m_rd_rdata;
  wire e_step = st ? pop : iss_ok;        // 外部端前進一個 line
  wire e_wrap = ind ? (col == ((cols == '0) ? 16'd0 : cols - 1'b1)) : ((cols != '0) && (col == cols - 1'b1));
  wire e_next = e_step && e_wrap;         // 間接: 本拍用完目前的項
  wire l_free = ind && e_next && (ent == EW'(EPL - 1));   // 間接: 用完 lb[lh] 的最後一項

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act <= 1'b0; st <= 1'b0; dec <= 1'b0; ind <= 1'b0; fmt <= '0; half <= 1'b0; l2 <= '0; ext <= '0;
      n <= '0; iss <= '0; wr <= '0; cols <= '0; col <= '0; stride <= '0; erow <= '0;
      f_rd <= '0; f_wr <= '0; f_cnt <= '0;
      list <= '0; nl <= '0; lf <= '0; lb <= '0; lbv <= '0; lh <= 1'b0; ent <= '0; hp <= 1'b0; hp_l <= 1'b0;
    end else begin
      hp <= h_rd_req && !h_rd_gnt; hp_l <= sel_l;
      if (cmd_valid && cmd_ready) begin
        act <= 1'b1; st <= cmd_st; l2 <= cmd_l2; ext <= cmd_ext; iss <= '0; wr <= '0;
        // 間接: 資料 line 數 = 項數 × 每項 line 數 (cols = 0 視為 1)
`ifndef FORMAL
        n <= cmd_ind ? cmd_n * ((cmd_cols == '0) ? 16'd1 : cmd_cols) : cmd_n;   // 間接: 軟體保證 k × cw <= 65535
`else
        // 形式驗證: 間接命令假設 cw <= 2 (見 FORMAL 區塊), 乘法改寫為移位 (z3 對 32 bit 乘法器的歸納很慢)
        n <= (cmd_ind && cmd_cols == 16'd2) ? {cmd_n[14:0], 1'b0} : cmd_n;
`endif
        fmt <= cmd_fmt; dec <= !cmd_st && (cmd_fmt != 2'd0); half <= 1'b0; ind <= cmd_ind;
        cols <= cmd_cols; stride <= cmd_stride; col <= '0; erow <= cmd_ext;
        list <= cmd_list; nl <= cmd_ind ? LW'((32'(cmd_n) + EPL - 1) / EPL) : '0; lf <= '0; lbv <= '0; lh <= 1'b0; ent <= '0;
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
        // 位址表緩衝: 讀到的 line 放進空的那一項 (lh 空時放 lh, 否則放另一項); 用完一個 line 時釋放並換到另一項
        if (l_fire) begin
          lb[lbv[lh] ? lo : lh] <= h_rd_data; lf <= lf + 1'b1;
        end
        lbv <= (lbv | (l_fire ? (lbv[lh] ? (2'b01 << lo) : (2'b01 << lh)) : 2'b00))
               & ~(l_free ? (2'b01 << lh) : 2'b00);
        if (l_free) lh <= lo;
        if (e_next) ent <= ent + 1'b1;
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
  wire [15:0] f_in = st ? iss : (wr + 16'(f_cnt));     // 本命令已進入 FIFO 的 line 數
  always @(*) if (rst_n) assume (!m_rd_rvalid || (act && !st && f_in != iss));
  // 間接命令每項最多 2 個 line (cw 為 1 / 2: 讓「line 數 = 項數 × cw」的歸納不需要乘法器; 任意 cw 由 CRV 驗證)
  always @(*) if (rst_n && cmd_valid && cmd_ind) assume (cmd_cols <= 16'd2 && (cmd_cols != 16'd2 || !cmd_n[15]));   // k × cw <= 65535
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
  // 資料完整: 任選一個 line index (anyconst), 從輸入記錄其資料, 寫出時必須相同 (解碼時為檢查器自己解碼的前半 / 後半)。
  // 輸入 = ST: host 讀埠的資料讀取 grant (間接時排除位址表讀取, 以 sel_l 區分); LD: 外部讀回應, 或略過項 (ld_skip, 資料為 0)
  (* anyconst *) logic [15:0] f_idx;
  logic f_have;
  logic [DW-1:0] f_dat;
  logic [1:0]    f_fmt;
  logic          f_dec_m, f_ind;
  wire          f_port_in  = st ? (h_rd_req && h_rd_gnt && !sel_l) : (m_rd_rvalid || ld_skip);
  wire [DW-1:0] f_port_dat = st ? h_rd_data : ld_skip ? '0 : m_rd_rdata;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_have <= 1'b0; f_fmt <= '0; f_dec_m <= 1'b0; f_ind <= 1'b0; end
    else if (cmd_valid && cmd_ready) begin f_have <= 1'b0; f_fmt <= cmd_fmt; f_dec_m <= !cmd_st && cmd_fmt != 2'd0; f_ind <= cmd_ind; end
    else if (act && f_port_in && f_in == f_idx) begin f_have <= 1'b1; f_dat <= f_port_dat; end
  wire [15:0] f_off = f_idx - wr;
  always @(*) if (rst_n) begin
    if (!act) begin
      assert (f_cnt == '0); assert (!h_wr_req && !m_rd_req && !m_wr_req);
      assert (!hp); assert (!h_rd_req);
    end
    if (act) begin
      assert (wr <= iss && iss <= n);
      assert (iss - wr <= 16'(OUT));
      assert (st ? (16'(f_cnt) == iss - wr) : (16'(f_cnt) <= iss - wr));
      assert (done == (wr == n));
      assert (f_have == (f_in > f_idx));
      assert (fmt == f_fmt && dec == f_dec_m && (!dec || !st) && ind == f_ind);
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
      // 有額度就發讀 (非間接; 間接時還要等位址表)
      if (!ind && iss != n && iss - wr < 16'(OUT)) assert (st ? h_rd_req : m_rd_req);
      // LD 略過項進入 FIFO 時沒有尚未回應的外部讀取 (FIFO 依序)
      if (ld_skip) assert (16'(f_cnt) == iss - wr && !m_rd_req);
      if (!ind) assert (!sel_l && lbv == '0);
      if (wr == n) begin assert (!d_hreq); assert (!l_need); assert (!hp); end
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
  // 位址: 檢查器從命令埠記下參數, 自己推進外部端的 2D / 間接位置, 從 L2 端的握手計數; 送出的位址必須相符。
  // 外部端前進: 非間接 = 埠的握手 (LD: 讀 grant; ST: 寫 grant); 間接另含略過項 (LD: ld_skip; ST: 略過的釋放)。
  // 間接: 檢查器從位址表讀取 (L2 讀埠 grant 且 line = list + 已讀數) 記下任選一項 f_e 的值, 外部端在該項時
  // 位址必須 = ext + 值 + 項內位置, 且值不是略過。
  logic [31:0] f_ext0, f_erow;
  logic [25:0] f_stride, f_list;
  logic [15:0] f_cols, f_ecol, f_eent;
  logic [LW-1:0] f_lrd;
  logic [16:0] f_lcnt;
  logic [25:0] f_l20;
  logic        f_st;
  (* anyconst *) logic [15:0] f_e;
  logic [15:0] f_k;
  logic        f_cw2;
  logic        f_ehave;
  logic [31:0] f_eval;
  wire f_ehs = f_st ? ((m_wr_req && m_wr_gnt) || (f_ind && st_wv && e_skip)) : ((m_rd_req && m_rd_gnt) || (f_ind && ld_skip));
  wire f_lhs = f_st ? (h_rd_req && h_rd_gnt && !sel_l) : (h_wr_req && h_wr_gnt);
  wire f_lrd_fire = h_rd_req && h_rd_gnt && sel_l;
  wire f_ewrap = f_ind ? (f_ecol == ((f_cols == '0) ? 16'd0 : f_cols - 1'b1)) : (f_cols != '0 && f_ecol == f_cols - 1'b1);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      f_ecol <= '0; f_erow <= '0; f_lcnt <= '0; f_cols <= '0; f_stride <= '0; f_ext0 <= '0; f_l20 <= '0; f_st <= 1'b0;
      f_list <= '0; f_eent <= '0; f_lrd <= '0; f_ehave <= 1'b0; f_eval <= '0; f_k <= '0; f_cw2 <= 1'b0;
    end else if (cmd_valid && cmd_ready) begin
      f_ext0 <= cmd_ext; f_erow <= cmd_ext; f_ecol <= '0; f_cols <= cmd_cols; f_stride <= cmd_stride;
      f_l20 <= cmd_l2; f_lcnt <= '0; f_st <= cmd_st; f_list <= cmd_list; f_eent <= '0; f_lrd <= '0; f_ehave <= 1'b0;
      f_k <= cmd_n; f_cw2 <= (cmd_cols == 16'd2);
    end else begin
      if (f_ehs) begin
        if (f_ewrap) begin f_ecol <= '0; f_erow <= f_erow + 32'(f_stride); f_eent <= f_eent + 1'b1; end
        else f_ecol <= f_ecol + 1'b1;
      end
      if (f_lhs) f_lcnt <= f_lcnt + 1'b1;
      if (f_lrd_fire) begin
        f_lrd <= f_lrd + 1'b1;
        if (32'(f_lrd) == 32'(f_e / EPL)) begin f_ehave <= 1'b1; f_eval <= h_rd_data[32'(f_e % EPL) * 32 +: 32]; end
      end
    end
  always @(*) if (rst_n && act) begin
    assert (f_st == st);
    if (m_rd_req && !f_ind) begin assert (m_rd_addr == f_erow + 32'(f_ecol)); end
    if (m_wr_req && !f_ind) begin assert (m_wr_addr == f_erow + 32'(f_ecol)); end
    if (h_rd_req && !sel_l) begin assert (h_rd_line == f_l20 + 26'(f_lcnt)); end
    if (h_rd_req && sel_l) begin assert (f_ind && h_rd_line == f_list + 26'(f_lrd) && f_lrd < nl); end
    if (h_wr_req) begin assert (h_wr_line == f_l20 + 26'(f_lcnt)); end
    // 間接: 外部端在被追蹤的項時, 位址表 line 已讀到, 位址 = ext + 值 + 項內位置; 略過項不存取外部
    if (f_ind && f_eent == f_e && (m_rd_req || m_wr_req)) begin
      assert (f_ehave && f_eval != 32'hFFFF_FFFF);
      if (m_rd_req) begin assert (m_rd_addr == f_ext0 + f_eval + 32'(f_ecol)); end
      if (m_wr_req) begin assert (m_wr_addr == f_ext0 + f_eval + 32'(f_ecol)); end
    end
`ifndef FORMAL_NO_HELPER
    // 歸納輔助 (引用內部狀態)
    assert (cols == f_cols && stride == f_stride && col == f_ecol && erow == f_erow && l2 == f_l20 && ext == f_ext0);
    assert (f_lcnt == (st ? 17'(iss) : dec ? {wr, half} : 17'(wr)) && (f_cols == '0 || f_ecol < f_cols || (f_ind && f_ecol == '0)));
    assert (list == f_list && lf == f_lrd && lf <= nl && (ind || nl == '0));
    assert (!hp || (hp_l ? l_need : d_hreq));
    assert (EW'(f_eent) == ent);
    if (ind) begin
      // 外部端已用的 line 數 = 項數 × cw + 項內位置; 項數到 k 時位址表已全部讀完
      assert ((st ? wr : iss) == (f_cw2 ? {f_eent[14:0], 1'b0} : f_eent) + f_ecol);
      assert (n == (f_cw2 ? {f_k[14:0], 1'b0} : f_k) && (!f_cw2 || !f_k[15]) && f_eent <= f_k && (f_eent != f_k || f_ecol == '0));
      assert (f_ecol <= 16'(f_cw2) && 32'(nl) == (32'(f_k) + EPL - 1) / EPL && f_cw2 == (cols == 16'd2) && cols <= 16'd2);
      // 位址表緩衝: 已讀 line = 已用完的 line + 緩衝中的 line; 緩衝頭 = 外部端目前的 line
      assert (32'(lf) == 32'(f_eent / EPL) + 32'(lbv[0]) + 32'(lbv[1]));
      assert (ent == '0 || lbv[lh]);   // 用了一部分的位址表 line 仍有效
      if (lbv[lh] && f_ehave && f_eent / EPL == f_e / EPL) assert (lb[lh][32'(f_e % EPL) * 32 +: 32] == f_eval);
      if (lbv == 2'b10 || lbv == 2'b01) assert (lbv[lh] || 32'(f_eent / EPL) == 32'(lf));
      if (f_ehave) assert (32'(f_lrd) > 32'(f_e / EPL));
      if (!f_ehave) assert (32'(f_lrd) <= 32'(f_e / EPL));
      // 第二項 (~lh) 有效時存的是下一個 line
      if (lbv[lh] && lbv[lo] && f_ehave && f_eent / EPL + 1 == f_e / EPL) assert (lb[lo][32'(f_e % EPL) * 32 +: 32] == f_eval);
    end else begin
      assert (f_eent == '0 || !f_ind);
    end
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
    cover (act && ind && !st && done && n == 16'd6 && cols == 16'd2 && lf == LW'(2));
    cover (act && ind && st && done && n == 16'd4 && cols == 16'd2 && lf == LW'(1));
    cover (act && ind && ld_skip && iss != '0);
    cover (act && ind && st_wv && e_skip);
  end
`endif
endmodule : npu_dma
