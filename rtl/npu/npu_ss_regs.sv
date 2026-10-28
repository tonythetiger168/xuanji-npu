// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU Subsystem 的 APB4 設定暫存器 (v6.3.11: 共用表頭, 與 IMS 統一)。暫存器表與 APB 慣例見 docs/apb.md。
`include "orca_pkg.sv"

// npu_ss_regs: SoC CPU 經 APB4 對 orca_npu_ss 下命令、取完成、讀狀態 / 效能計數、設定中斷。
//   - APB4 從端, 零等待 (PREADY 恆 1); 讀資料在存取階段 (PSEL && PENABLE) 由位址組合產生。
//   - 共用表頭 (docs/apb.md「共用表頭」): 0x000 ID、0x004 CFG0、0x008 CFG1、0x010 IRQ_STATUS ([31] BUS_ERR)、
//     0x014 IRQ_ENABLE; 表頭的保留字 (0x00C、0x018–0x03C) 讀回 0、寫入忽略、不報錯。本區塊暫存器從 0x040 起。
//   - 未定義的位址: PSLVERR = 1, 讀回 0, 設定 IRQ_STATUS.BUS_ERR。已定義暫存器的保留位元讀回 0、寫入忽略;
//     寫入唯讀暫存器忽略 (不報錯)。RW 暫存器依 PSTRB 逐位元組寫入。
//   - 命令: CPU 先寫 CMD_CTL / CMD_EXT / CMD_A / CMD_B / CMD_C (暫存區, 寫後保留), 再寫 CMD_PUSH, 暫存區內容進入
//     CQ 項的命令 FIFO, 由 FIFO 頭依 valid / ready 送往 orca_npu_ss。FIFO 滿時 CMD_PUSH 丟棄並設定 CMD_OVF。
//   - 完成: 讀 CPL_POP 時若有完成 (cpl_valid), 回傳 {VALID=1, id} 並在同一拍取走 (cpl_ready = 1); CPL_PEEK 不取走。
//   - 中斷: irq = |(IRQ_STATUS & IRQ_ENABLE); IRQ_STATUS.CPL 是 cpl_valid 的位準, 其餘 (CMD_OVF / BUS_ERR / VEC_ERR)
//     為黏著位元 (寫 1 清除)。
//   - 效能計數: 讀 *_LO 時把同一計數器的 [47:32] 存入影子暫存器, 之後讀 *_HI 得到同一時刻的值 (48 bit 一致讀取)。
//   APB 與 NPU Subsystem 在同一時脈域 (跨時脈域橋接不在本模組)。
module npu_ss_regs
  import orca_pkg::*;
#(
  parameter int TILE_ID = 0,
  parameter int N     = 8,             // 只用來產生 CFG 唯讀暫存器 (與 orca_npu_ss 的參數相同)
  parameter int NCU   = 4,
  parameter int NB    = 4,
  parameter int DEPTH = 1024,
  parameter int QD    = 8,
  parameter int OUT   = 32,
  parameter int CQ    = 4              // APB 命令 FIFO 深度 (2 的冪, <= 8)
)(
  input  logic        clk, rst_n,
  input  logic        test_en,         // DFT: 強制開啟所有 ICG (功能不變)
  // retention (v6.3.11): IRQ_EN 斷電後保留。clk_aon / por_n 為常開域時脈與上電 reset; ret_save / ret_restore 來自 PMU
  input  logic        clk_aon, por_n,
  input  logic        ret_save, ret_restore,
  // APB4 從端
  input  logic        psel, penable, pwrite,
  input  logic [11:0] paddr,           // 4 KiB 視窗內的位元組位址 (位元 [1:0] 忽略)
  input  logic [31:0] pwdata,
  input  logic [3:0]  pstrb,
  output logic        pready,
  output logic [31:0] prdata,
  output logic        pslverr,
  output logic        irq,
  // 往 orca_npu_ss
  output logic          cmd_valid,
  input  logic          cmd_ready,
  output npu_tile_cmd_t cmd,
  input  logic          cpl_valid,
  output logic          cpl_ready,
  input  logic [7:0]    cpl_id,
  input  logic          ss_busy,
  input  logic          vec_err,               // 一拍: VEC 命令遇到非法指令 (orca_npu_ss)
  output logic          perf_wr,               // 清除效能計數 (orca_npu_ss 的 perf_wdata_* 由上層接 0)
  input  logic [47:0]   perf_busy, perf_steps
);
  // ---------------- 位址 (docs/apb.md) ----------------
  // word 位址 (paddr[11:2]); 表頭 0x000..0x00F (位元組 0x000..0x03C)
  localparam logic [9:0] A_ID = 10'h000, A_CFG0 = 10'h001, A_CFG1 = 10'h002, A_IRQ_ST = 10'h004, A_IRQ_EN = 10'h005,
                         A_STATUS = 10'h010,
                         A_CMD_CTL = 10'h018, A_CMD_EXT = 10'h019, A_CMD_A = 10'h01A, A_CMD_B = 10'h01B,
                         A_CMD_C = 10'h01C, A_CMD_PUSH = 10'h01D, A_CPL_POP = 10'h020, A_CPL_PEEK = 10'h021,
                         A_PERF_CTRL = 10'h024, A_PB_LO = 10'h025, A_PB_HI = 10'h026, A_PS_LO = 10'h027,
                         A_PS_HI = 10'h028;
  localparam logic [31:0] ID_VAL = 32'h584E_0200;          // 'X' 'N', 版本 2.0 (v6.3.11 共用表頭: 位址與 v1 不相容)
  localparam int CW = $clog2(CQ);

  wire [9:0] wa  = paddr[11:2];
  wire       acc = psel && penable;                         // 存取階段 (PREADY 恆 1: 這拍完成)
  wire       wr  = acc && pwrite;
  wire       rd  = acc && !pwrite;
  // 位址有定義 (以集合比較寫: 原本的 unique case 經 sv2v 轉成 (* full_case, parallel_case *) 後,
  // Yosys 的形式模型對已列出的位址給出 hit = 0, 與模擬不一致)
  wire       hdr = (wa[9:4] == 6'd0);                       // 表頭 (含保留字: 讀 0、寫入忽略、不報錯)
  wire       hit = hdr || wa inside {A_STATUS, A_CMD_CTL, A_CMD_EXT, A_CMD_A, A_CMD_B,
                              A_CMD_C, A_CMD_PUSH, A_CPL_POP, A_CPL_PEEK, A_PERF_CTRL, A_PB_LO, A_PB_HI, A_PS_LO, A_PS_HI};
  assign pready  = 1'b1;
  assign pslverr = acc && !hit;

  // 位元組寫入遮罩
  wire [31:0] bm = {{8{pstrb[3]}}, {8{pstrb[2]}}, {8{pstrb[1]}}, {8{pstrb[0]}}};
  function automatic logic [31:0] merge(logic [31:0] old, logic [31:0] nw, logic [31:0] m);
    return (old & ~m) | (nw & m);
  endfunction
  // IRQ 位元: [31] BUS_ERR (共用表頭), [2] VEC_ERR, [1] CMD_OVF, [0] CPL; 內部以 4 bit 存放
  function automatic logic [31:0] ien_unpack(logic [3:0] v);
    return {v[3], 28'd0, v[2:0]};
  endfunction
  function automatic logic [3:0] ien_pack(logic [31:0] w);
    return {w[31], w[2:0]};
  endfunction

  // ---------------- 命令暫存區 ----------------
  logic [31:0] r_ctl, r_ext;
  logic [25:0] r_a, r_b, r_c;
  logic [3:0]  r_ien;                                      // IRQ_ENABLE 的 [31] / [2] / [1] / [0] (BUS_ERR / VEC_ERR / CMD_OVF / CPL)
  // 低功耗: 暫存區 / 中斷致能只在 APB 寫入拍收到時脈 (ICG); 資料致能仍為 wr, test_en 強制開時脈時功能不變
  // (reset 期間 ICG 也打開: 模擬開始時 rst_n 已為低、沒有 negedge 時, 閘控的 flop 仍在時脈沿上 reset)
  // retention: IRQ_EN 的影子暫存器在常開域; 上電後 ret_restore 拍載回 (之後同拍的 APB 寫入優先)
  logic [3:0] ien_ret;
  ret_shadow #(.W(4)) u_ret_ien (.clk_aon, .por_n, .save(ret_save), .d(r_ien), .q(ien_ret));
  logic gclk_w;
  clock_gate u_icg_w (.clk(clk), .en(wr || ret_restore || !rst_n), .test_en(test_en), .clk_out(gclk_w));
  always_ff @(posedge gclk_w or negedge rst_n)
    if (!rst_n) begin
      r_ctl <= '0; r_ext <= '0; r_a <= '0; r_b <= '0; r_c <= '0; r_ien <= '0;
    end else begin
      if (ret_restore) r_ien <= ien_ret;
      if (wr) begin
      if (wa == A_CMD_CTL) r_ctl <= merge(r_ctl, pwdata, bm) & 32'hFFFF_FF3F;   // 保留位元 [7:6] 為 0
      if (wa == A_CMD_EXT) r_ext <= merge(r_ext, pwdata, bm);
      if (wa == A_CMD_A)   r_a   <= 26'(merge({6'd0, r_a}, pwdata, bm));
      if (wa == A_CMD_B)   r_b   <= 26'(merge({6'd0, r_b}, pwdata, bm));
      if (wa == A_CMD_C)   r_c   <= 26'(merge({6'd0, r_c}, pwdata, bm));
      if (wa == A_IRQ_EN)  r_ien <= ien_pack(merge(ien_unpack(r_ien), pwdata, bm));
      end
    end
  npu_tile_cmd_t stg;
  always_comb begin
    stg.op  = npu_op_e'({r_ctl[5], r_ctl[1:0]});   // OP[2] 在 bit 5 (舊的 2 bit OP 版面不變)
    stg.wt  = r_ctl[2];
    stg.sa  = r_ctl[3];
    stg.sb  = r_ctl[4];
    stg.id  = r_ctl[15:8];
    stg.k   = r_ctl[31:16];
    stg.ext = r_ext;
    stg.a   = r_a;
    stg.b   = r_b;
    stg.c   = r_c;
  end

  // ---------------- 命令 FIFO ----------------
  npu_tile_cmd_t q [CQ];
  logic [CW:0]   q_wr, q_rd;                                // 多一位: 區分滿 / 空
  wire  [CW:0]   q_n    = q_wr - q_rd;
  wire           q_full = (q_n == (CW+1)'(CQ));
  wire           push_req = wr && wa == A_CMD_PUSH;
  wire           push   = push_req && !q_full;
  wire           pop    = cmd_valid && cmd_ready;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin q_wr <= '0; q_rd <= '0; end
    else begin
      if (push) q_wr <= q_wr + 1'b1;
      if (pop)  q_rd <= q_rd + 1'b1;
    end
  logic gclk_q;                                             // FIFO 資料: 只在 push 拍收到時脈
  clock_gate u_icg_q (.clk(clk), .en(push || !rst_n), .test_en(test_en), .clk_out(gclk_q));
  always_ff @(posedge gclk_q)
    if (push) q[q_wr[CW-1:0]] <= stg;
  assign cmd_valid = (q_n != '0);
  assign cmd       = q[q_rd[CW-1:0]];

  // ---------------- 完成 / 中斷 ----------------
  assign cpl_ready = rd && wa == A_CPL_POP;                 // 沒有完成時讀 CPL_POP 不影響 (cpl_valid = 0)
  logic ovf, berr, verr;                                    // 黏著: 命令 FIFO 溢位、未定義位址、VEC 非法指令
  wire  w1c = wr && wa == A_IRQ_ST;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin ovf <= 1'b0; berr <= 1'b0; verr <= 1'b0; end
    else begin
      ovf  <= (ovf  && !(w1c && pstrb[0] && pwdata[1]))  || (push_req && q_full);
      verr <= (verr && !(w1c && pstrb[0] && pwdata[2]))  || vec_err;
      berr <= (berr && !(w1c && pstrb[3] && pwdata[31])) || pslverr;
    end
  wire [3:0] ist = {berr, verr, ovf, cpl_valid};
  assign irq = |(ist & r_ien);

  // ---------------- 效能計數 ----------------
  assign perf_wr          = wr && wa == A_PERF_CTRL && pstrb[0] && pwdata[0];
  logic [15:0] sh_b, sh_s;                                  // 讀 *_LO 時存下的高位
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin sh_b <= '0; sh_s <= '0; end
    else begin
      if (rd && wa == A_PB_LO) sh_b <= perf_busy[47:32];
      if (rd && wa == A_PS_LO) sh_s <= perf_steps[47:32];
    end

  // ---------------- 讀資料 ----------------
  always_comb begin
    prdata = '0;
    if (rd)
      unique case (wa)
        A_ID:       prdata = ID_VAL;
        A_CFG0:     prdata = {8'($clog2(DEPTH)), 8'(NB), 8'(NCU), 8'(N)};
        A_CFG1:     prdata = {8'(CQ), 8'(TILE_ID), 8'(OUT), 8'(QD)};
        A_STATUS:   prdata = {20'd0, 4'(CQ - int'(q_n)), 4'd0, !cmd_valid, q_full, cpl_valid, ss_busy || cmd_valid};
        A_IRQ_ST:   prdata = ien_unpack(ist);
        A_IRQ_EN:   prdata = ien_unpack(r_ien);
        A_CMD_CTL:  prdata = r_ctl;
        A_CMD_EXT:  prdata = r_ext;
        A_CMD_A:    prdata = {6'd0, r_a};
        A_CMD_B:    prdata = {6'd0, r_b};
        A_CMD_C:    prdata = {6'd0, r_c};
        A_CPL_POP,
        A_CPL_PEEK: prdata = {cpl_valid, 23'd0, cpl_valid ? cpl_id : 8'd0};
        A_PB_LO:    prdata = perf_busy[31:0];
        A_PB_HI:    prdata = {16'd0, sh_b};
        A_PS_LO:    prdata = perf_steps[31:0];
        A_PS_HI:    prdata = {16'd0, sh_s};
        default:    prdata = '0;                            // 表頭保留字、CMD_PUSH / PERF_CTRL (唯寫) 與未定義位址
      endcase
  end
`ifdef FORMAL
  // ---------------- 形式驗證 (SymbiYosys, formal/npu_ss_regs; 模擬時不編入) ----------------
  // 檢查器只看埠: 自己解碼位址、自己維護暫存區 / FIFO 數量 / 黏著位元 / 中斷致能的模型, 不引用 RTL 的
  // hit / push / q_n / ovf / berr / r_*。FIFO 資料以任選序號 (anyconst) 追蹤: 第 f_idx 個被接受的命令送出時必須等於
  // 被接受當時的暫存區內容 (由 APB 寫入記錄)。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n && !por_n);                     // 上電: 域 reset 與上電 reset 同時有效
  always @(*) assert (ien_ret == f_ien_sh);                    // retention 影子 (常開域, 與域 reset 無關)
  // 環境: APB 主端協定; orca_npu_ss 端的完成在 cpl_ready 之前保持
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(psel && !penable)) begin assume (psel && penable && paddr == $past(paddr) && pwrite == $past(pwrite)
                                                && pwdata == $past(pwdata) && pstrb == $past(pstrb)); end
    if ($past(cpl_valid && !cpl_ready)) begin assume (cpl_valid && cpl_id == $past(cpl_id)); end
  end
  always @(*) begin
    if (psel && !pwrite) begin assume (pstrb == '0); end
  end
  wire       f_acc = psel && penable, f_wr = f_acc && pwrite, f_rd = f_acc && !pwrite;
  wire [9:0] f_wa  = paddr[11:2];
  wire       f_def = (f_wa < 10'h010) || f_wa inside {10'h010, 10'h018, 10'h019, 10'h01A,
                                   10'h01B, 10'h01C, 10'h01D, 10'h020, 10'h021, 10'h024, 10'h025, 10'h026, 10'h027, 10'h028};
  wire [31:0] f_ienw = {f_ien[3], 28'd0, f_ien[2:0]};
  wire [31:0] f_bm = {{8{pstrb[3]}}, {8{pstrb[2]}}, {8{pstrb[1]}}, {8{pstrb[0]}}};
  logic [31:0] f_ctl, f_ext, f_a, f_b, f_c;
  logic [3:0]  f_ien, f_ien_sh;                            // f_ien_sh: retention 影子 (常開域, por_n)
  logic        f_ovf, f_berr, f_verr;
  always @(posedge clk_aon or negedge por_n)
    if (!por_n) f_ien_sh <= '0;
    else if (ret_save) f_ien_sh <= f_ien;
  logic [CW:0] f_cnt;                                       // 已接受、尚未送出的命令數
  logic [7:0]  f_acc_n, f_out_n;                            // 已接受 / 已送出的命令序號
  (* anyconst *) logic [7:0] f_idx;
  logic          f_have;
  npu_tile_cmd_t f_cmd;
  wire  f_pushreq = f_wr && f_wa == 10'h01D;
  wire  f_full    = (f_cnt == (CW+1)'(CQ));
  wire  f_pop     = cmd_valid && cmd_ready;
  wire  f_push    = f_pushreq && !f_full;
  npu_tile_cmd_t f_stg;
  always @(*) begin
    f_stg.op = npu_op_e'({f_ctl[5], f_ctl[1:0]}); f_stg.wt = f_ctl[2]; f_stg.sa = f_ctl[3]; f_stg.sb = f_ctl[4];
    f_stg.id = f_ctl[15:8]; f_stg.k = f_ctl[31:16]; f_stg.ext = f_ext;
    f_stg.a = f_a[25:0]; f_stg.b = f_b[25:0]; f_stg.c = f_c[25:0];
  end
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      f_ctl <= '0; f_ext <= '0; f_a <= '0; f_b <= '0; f_c <= '0; f_ien <= '0; f_ovf <= 1'b0; f_berr <= 1'b0; f_verr <= 1'b0;
      f_cnt <= '0; f_acc_n <= '0; f_out_n <= '0; f_have <= 1'b0;
    end else begin
      if (f_wr && f_wa == 10'h018) f_ctl <= ((f_ctl & ~f_bm) | (pwdata & f_bm)) & 32'hFFFF_FF3F;
      if (f_wr && f_wa == 10'h019) f_ext <= (f_ext & ~f_bm) | (pwdata & f_bm);
      if (f_wr && f_wa == 10'h01A) f_a <= ((f_a & ~f_bm) | (pwdata & f_bm)) & 32'h03FF_FFFF;
      if (f_wr && f_wa == 10'h01B) f_b <= ((f_b & ~f_bm) | (pwdata & f_bm)) & 32'h03FF_FFFF;
      if (f_wr && f_wa == 10'h01C) f_c <= ((f_c & ~f_bm) | (pwdata & f_bm)) & 32'h03FF_FFFF;
      if (f_wr && f_wa == 10'h005) begin
        f_ien[0] <= pstrb[0] ? pwdata[0] : f_ien[0]; f_ien[1] <= pstrb[0] ? pwdata[1] : f_ien[1];
        f_ien[2] <= pstrb[0] ? pwdata[2] : f_ien[2]; f_ien[3] <= pstrb[3] ? pwdata[31] : f_ien[3];
      end else if (ret_restore)    f_ien <= f_ien_sh;
      f_ovf  <= (f_ovf && !(f_wr && f_wa == 10'h004 && pstrb[0] && pwdata[1])) || (f_pushreq && f_full);
      f_berr <= (f_berr && !(f_wr && f_wa == 10'h004 && pstrb[3] && pwdata[31])) || (f_acc && !f_def);
      f_verr <= (f_verr && !(f_wr && f_wa == 10'h004 && pstrb[0] && pwdata[2])) || vec_err;
      f_cnt  <= f_cnt + (CW+1)'(f_push) - (CW+1)'(f_pop);
      if (f_push) begin
        f_acc_n <= f_acc_n + 1'b1;
        if (f_acc_n == f_idx) begin f_have <= 1'b1; f_cmd <= f_stg; end
      end
      if (f_pop) f_out_n <= f_out_n + 1'b1;
    end
  // 序號不回繞 (只看前 255 個命令)
  always @(*) if (rst_n) begin assume (f_acc_n != 8'hFF); end
  always @(*) if (rst_n) begin
    assert (pready);
    assert (pslverr == (f_acc && !f_def));
    assert (cpl_ready == (f_rd && f_wa == 10'h020));
    assert (cmd_valid == (f_cnt != '0));
    assert (irq == |({f_berr, f_verr, f_ovf, cpl_valid} & f_ien));
    assert (perf_wr == (f_wr && f_wa == 10'h024 && pstrb[0] && pwdata[0]));
    if (f_rd && f_wa == 10'h018) begin assert (prdata == f_ctl); end
    if (f_rd && f_wa == 10'h005) begin assert (prdata == f_ienw); end
    if (f_rd && f_wa == 10'h004) begin assert (prdata == {f_berr, 28'd0, f_verr, f_ovf, cpl_valid}); end
    if (f_rd && f_wa == 10'h000) begin assert (prdata == 32'h584E_0200); end
    if (f_rd && f_wa < 10'h010 && !(f_wa inside {10'h000, 10'h001, 10'h002, 10'h004, 10'h005})) begin assert (prdata == '0); end
    if (!f_rd || !f_def) begin assert (prdata == '0); end
    // FIFO 資料: 被追蹤的命令送出時內容正確; 送出順序 = 接受順序
    if (f_pop && f_out_n == f_idx) begin assert (f_have && cmd == f_cmd); end
    // 歸納強化: 數量與序號一致、與 RTL 指標一致 (只在斷言中引用, 不作為參考)
    assert (f_out_n <= f_acc_n);                           // 序號不回繞
    assert (f_acc_n - f_out_n == 8'(f_cnt));
    assert (f_cnt <= (CW+1)'(CQ));
    assert (q_n == f_cnt);
    assert (q_rd == (CW+1)'(f_out_n));
    assert (r_ctl == f_ctl && r_ext == f_ext && {6'd0, r_a} == f_a && {6'd0, r_b} == f_b && {6'd0, r_c} == f_c);
    assert (r_ien == f_ien && ovf == f_ovf && berr == f_berr && verr == f_verr);
    assert (f_have == (f_acc_n > f_idx));
    if (f_have && f_out_n <= f_idx) begin assert (q[(CW)'(q_rd + (CW+1)'(f_idx - f_out_n))] == f_cmd); end
  end
  always @(posedge clk) if (f_past && rst_n) begin
    cover (f_pushreq && f_full);
    cover (f_pop && f_out_n == f_idx && f_idx == 8'd5);
    cover (irq && f_ovf);
  end
`endif
endmodule : npu_ss_regs
