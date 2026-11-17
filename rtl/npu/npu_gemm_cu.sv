// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU GEMM 計算單元 (v6.3.12)
`include "npu_pkg.sv"

// npu_gemm_cu: 一個 GEMM 命令 = 從 L2 讀 A / B, 經 npu_mma 計算, 把 C 寫回 L2。
//
// L2 版面 (64 B line, SPL = 64/N 個 k 步 / line, RPL = max(1, 16/N) 列 / C line; N = 32 時一列 C 佔 CPR = 2 個 line):
//   A: a_line*64 + k*N + i     = A[i][k]   (第 k 欄連續 N byte)
//   B: b_line*64 + k*N + j     = B[k][j]   (第 k 列連續 N byte)
//   C: c_line*64 + (i*N + j)*4 = C[i][j]   (INT32 列優先; 溢位時模 2^32 環繞, 不飽和)
//   位址以 64 B line 為單位 (L2 位元組位址 = {line, 6'b0})。
// L2 介面: 讀為組合式 (rd_data 與 rd_gnt 同拍有效), 寫在 wr_gnt 拍完成; 等待 grant 時請求保持。
//
// v6.3.8: 預取 + 雙緩衝。取線器 (fetcher) 與餵入器 (feeder) 解耦: PF 項 {A line, B line} 緩衝,
//   A / B 各自一條取線軌 (各自的剩餘 line 數、寫入指標與已填項數), 餵入器每拍從緩衝頭取一個 k 步
//   (該項 A 與 B 都已到齊才可取)。緩衝暫時為空 (grant 不足) 時 in_valid=0, npu_mma 支援空拍。
//   RP 個 L2 讀取埠: N <= 16 時 RP = 1, A / B 輪流共用 (每 SPL 步 2 次讀取, SPL >= 4, 有餘裕), PF = 2;
//   N = 32 時每 2 步就要 1 個 A line + 1 個 B line, 單一埠每拍都得拿到 grant 才不掉速, 任何 bank 衝突都損失
//   頻寬 → RP = 2 (A 埠 + B 埠, 各自每 2 拍讀 1 次) 且 PF = 4 吸收衝突。
// v6.3.12: 命令重疊。輸入端 (取線器 + 餵入器, fst) 與輸出端 (寫回器, wst) 分開: 最後一個 k 步餵入後輸入端
//   立刻回到閒置並可接受下一個命令, 寫回器同時把前一個 tile 的 C 寫回 (npu_mma tile 管線化)。
//   命令的 {C line, tag} 依序經過三段: 輸入端 (c_cmd) → 等待輸出 (c_pend, 最後一步餵入時移入) → 寫回器 (c_ptr),
//   所以一個 CU 最多同時持有 3 個命令。完成: done (保持到 done_ready) + done_tag, 依接受順序。
// v6.3.14: 累加模式 (cmd_acc = 1): C += A × B。寫回器在寫每個 C line 之前先讀出 L2 中舊的 line (W_RD), 逐 32 bit
//   相加 (模 2^32) 後寫回; 用途: K 切塊 (同一個 C 分多個命令累加)、殘差相加。讀舊 C 走讀取埠 0 (與取線器共用):
//   寫回器優先, 但已發出、尚未拿到 grant 的請求一律保持到 grant (L2 介面規則), 兩邊都不會被撤回。
//   軟體規則不變: 一個命令的 C 區不可與在途命令的 A / B / C 區重疊 (同一個 C 的累加命令須等前一個完成, wt = 1)。
// 低功耗: 緩衝只在讀取 grant 拍更新; npu_mma 列級 clock gating (test_en 強制開啟)。
module npu_gemm_cu
  import npu_pkg::*;
#(parameter int N = 8,
  localparam int RP = (N == 32) ? 2 : 1     // L2 讀取埠數 (npu_core 依同一公式接線)
)(
  input  logic clk, rst_n,
  input  logic test_en,
  // 命令
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [25:0] cmd_a_line, cmd_b_line, cmd_c_line,   // 64 B line 位址
  input  logic [15:0] cmd_k,                 // K 步數, 須 >= 1 (SVA 假設)
  input  logic        cmd_a_signed, cmd_b_signed,
  input  logic        cmd_acc,               // 1: C += A × B (寫回前讀舊的 C)
  input  logic [7:0]  cmd_tag,               // 命令標籤, 完成時由 done_tag 帶回
  output logic        done,                  // 一個命令的 C 已全部寫回 (保持到 done_ready)
  input  logic        done_ready,
  output logic [7:0]  done_tag,
  output logic        busy,                  // 持有任一命令 (輸入端 / 等待輸出 / 寫回 / 等待完成握手)
  output logic        feeding,               // 本拍餵入一個 k 步 (效能計數: N*N 個 MAC)
  // L2 主介面 (讀取埠 p: RP = 1 時 A / B 共用埠 0; RP = 2 時埠 0 = A, 埠 1 = B)
  output logic [RP-1:0]         rd_req,
  output logic [RP-1:0][25:0]   rd_line,
  input  logic [RP-1:0]         rd_gnt,
  input  logic [RP-1:0][511:0]  rd_data,
  output logic         wr_req,
  output logic [25:0]  wr_line,
  output logic [511:0] wr_data,
  input  logic         wr_gnt
);
  localparam int SPL = 64 / N;                   // 每個 line 的 k 步數
  localparam int RPL = (N >= 16) ? 1 : 16 / N;   // 每個 C line 的列數
  localparam int CPR = (N > 16) ? 2 : 1;         // 每列 C 佔的 line 數 (N = 32: 2)
  localparam int SW  = $clog2(SPL);
  localparam int RW  = $clog2(N);
  localparam int PF  = (RP == 2) ? 4 : 2;        // 預取緩衝項數
  localparam int PW  = $clog2(PF);
  if (N != 4 && N != 8 && N != 16 && N != 32) begin : g_bad_n
    $error("npu_gemm_cu: N 須為 4 / 8 / 16 / 32");
  end

  // 輸入端: F_IDLE / F_RUN; 寫回器: W_OUT 收集一個 line → (累加: W_RD 讀舊的 line) → W_WR 寫回
  typedef enum logic { F_IDLE = 1'b0, F_RUN = 1'b1 } fst_t;
  typedef enum logic [2:0] { W_IDLE = 3'd0, W_OUT = 3'd1, W_WR = 3'd2, W_DONE = 3'd3, W_RD = 3'd4 } wst_t;
  fst_t fst;
  wst_t wst;
  logic [25:0] a_ptr, b_ptr, c_ptr;
  logic [25:0] c_cmd, c_pend;                // 輸入端 / 等待輸出的 C line
  logic [7:0]  t_cmd, t_pend, t_w;           // 輸入端 / 等待輸出 / 寫回器的 tag
  logic        x_cmd, x_pend, x_w;           // 輸入端 / 等待輸出 / 寫回器的累加旗標
  logic [511:0] o_line;                      // 累加: 讀出的舊 C line (只在 W_RD 的 grant 拍更新)
  logic        w_rq, w_sel, w_gnt, lk_f;     // 寫回器的讀取請求 / 佔用埠 0 / grant; 取線器在埠 0 上未拿到 grant 的請求
  logic        pend;                         // 有一個已餵完、尚未開始寫回的 tile
  logic        lr;                           // 寫回器目前的 line 含最後一列
  logic [15:0] k_left;                       // 尚未餵入的 k 步數
  localparam int FW = $clog2((65535 + SPL - 1) / SPL + 1);
  logic [FW-1:0] fa, fb;                     // A / B 尚未讀取的 line 數 (最多 ceil(65535/SPL))
  logic [SW-1:0] step;                       // 緩衝頭 line 內的 k 步
  logic [PF-1:0][511:0] buf_a, buf_b;        // PF 項緩衝
  logic [PW-1:0] wa, wb, rptr;               // A / B 寫入指標、讀取 (餵入) 指標
  logic [PW:0]   a_cnt, b_cnt;                     // A / B 已填入、尚未取用的項數 (0..PF)
  logic          have;                       // 緩衝頭的 A 與 B 都已到齊
  logic          a_req, b_req, a_gnt, b_gnt; // 各取線軌的請求 / grant
  logic [511:0]  a_dat, b_dat;               // 各取線軌讀到的 line
  logic [511:0] c_line, c_line_n;
  logic [511:0] wr_d;                        // 本次寫回的 line

  // ---------------- npu_mma ----------------
  logic mma_start, mma_busy, mma_in_idle, mma_in_valid, mma_in_ready, mma_in_last, mma_out_valid, mma_out_ready;
  logic [N-1:0][7:0] a_vec, b_vec;
  logic [RW-1:0] mma_out_row;
  logic [N-1:0][31:0] mma_out_vec;

  assign mma_start     = (fst == F_IDLE) && cmd_valid;
  assign have          = (a_cnt != '0) && (b_cnt != '0);
  assign mma_in_valid  = (fst == F_RUN) && have;
  assign mma_in_last   = (k_left == 16'd1);
  assign a_vec         = buf_a[rptr][step * N * 8 +: N * 8];
  assign b_vec         = buf_b[rptr][step * N * 8 +: N * 8];
  assign mma_out_ready = (wst == W_OUT);

  npu_mma #(.N(N)) u_mma (
    .clk(clk), .rst_n(rst_n), .test_en(test_en),
    .start(mma_start), .a_signed(cmd_a_signed), .b_signed(cmd_b_signed), .busy(mma_busy), .in_idle(mma_in_idle),
    .in_valid(mma_in_valid), .in_ready(mma_in_ready), .in_last(mma_in_last),
    .a_vec(a_vec), .b_vec(b_vec),
    .out_valid(mma_out_valid), .out_ready(mma_out_ready), .out_row(mma_out_row), .out_vec(mma_out_vec));

  // ---------------- L2 存取 ----------------
  assign cmd_ready = (fst == F_IDLE);
  assign busy      = (fst != F_IDLE) || pend || (wst != W_IDLE);
  assign done      = (wst == W_DONE);
  assign done_tag  = t_w;
  // 取線軌請求: 還有 line 要讀且該軌緩衝有空槽
  wire run = (fst == F_RUN);
  // 埠 0: 寫回器的讀取 (累加) 優先, 但取線器已發出而未拿到 grant 的請求 (lk_f) 先完成。寫回器的請求自然保持:
  //   它在 W_RD 期間一直請求, 而它佔用埠時取線器沒有發出請求, lk_f 不會出現。
  logic        f_rq0;                        // 取線器在埠 0 的請求
  logic [25:0] f_ln0;
  wire         g0f = rd_gnt[0] && !w_sel;    // 埠 0 的 grant 屬於取線器
  assign w_rq  = (wst == W_RD);
  assign w_sel = w_rq && !lk_f;
  assign w_gnt = rd_gnt[0] && w_sel;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) lk_f <= 1'b0;
    else        lk_f <= f_rq0 && !w_sel && !rd_gnt[0];
  if (RP == 2) begin : g_rp2
    // A 埠 (埠 0, 與寫回器的讀取共用) / B 埠 (埠 1) 各自獨立
    assign a_req = run && (fa != '0) && (a_cnt != (PW+1)'(PF));
    assign b_req = run && (fb != '0) && (b_cnt != (PW+1)'(PF));
    assign f_rq0 = a_req; assign f_ln0 = a_ptr;
    assign rd_req  = {b_req, w_sel || a_req};
    assign rd_line = {b_ptr, w_sel ? c_ptr : a_ptr};
    assign a_gnt = g0f; assign b_gnt = rd_gnt[1];
    assign a_dat = rd_data[0]; assign b_dat = rd_data[1];
  end else begin : g_rp1
    // 單一埠: A、B 輪流 (f_b); 讀 B 時對應的 A 已在緩衝 (b_cnt < a_cnt <= PF), 不會溢位
    logic f_b;                               // 0: 下一次讀 A, 1: 下一次讀 B
    assign a_req = run && !f_b && (fa != '0) && (a_cnt != (PW+1)'(PF));
    assign b_req = run && f_b && (fb != '0);
    assign f_rq0 = a_req || b_req; assign f_ln0 = f_b ? b_ptr : a_ptr;
    assign rd_req  = w_sel || f_rq0;
    assign rd_line = w_sel ? c_ptr : f_ln0;
    assign a_gnt = g0f && !f_b; assign b_gnt = g0f && f_b;
    assign a_dat = rd_data[0]; assign b_dat = rd_data[0];
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) f_b <= 1'b0;
      else if (mma_start) f_b <= 1'b0;
      else if (g0f) f_b <= ~f_b;
  end
  assign wr_req    = (wst == W_WR);
  assign wr_line   = c_ptr;
  // 累加: 逐 32 bit 相加 (模 2^32)
  logic [15:0][31:0] wr_sum;
  for (genvar e = 0; e < 16; e++) begin : g_sum
    assign wr_sum[e] = wr_d[32 * e +: 32] + o_line[32 * e +: 32];
  end
  assign wr_data   = x_w ? wr_sum : wr_d;

  // npu_mma 在輸入 RUN 時就緒, 除了最後一步要等前一個 tile 輸出完 (a_feed_ready)
  wire feed = mma_in_valid && mma_in_ready;
  wire feed_last = feed && mma_in_last;
  assign feeding = feed;
  wire pop  = feed && (step == SW'(SPL - 1) || mma_in_last);
  wire push = b_gnt;                         // 一項的 B 到齊 (RP = 1 時即完成一項)
  wire out_fire = mma_out_valid && mma_out_ready;
  // 寫回器取得下一個 tile: 有等待輸出的 tile 且寫回器閒置或本拍完成握手
  wire take = pend && (wst == W_IDLE || (wst == W_DONE && done_ready));

  // C line 組裝: 每列寫入 line 中的位置; 湊滿 RPL 列後寫回
  wst_t o_nxt;                               // W_OUT 輸出握手後的狀態 (湊滿一個 C line: 累加先讀舊值, 否則寫回)
  wst_t w_nxt;                               // W_WR 拿到 grant 後的狀態
  if (CPR > 1) begin : g_cw
    // N = 32: 一列 C = 2 個 line; 前半存在 c_line, 後半存在 c_hi, 依序寫回 (c_hi 只有此組態才有)
    logic [511:0] c_hi;
    logic         c_half;                    // 0: 寫前半, 1: 寫後半
    // mma_out_vec 是 [N-1:0][31:0] 二維 packed: 索引選的是元素 (每個 32 bit), 不是位元
    assign c_line_n = mma_out_vec[N/2-1:0];
    assign o_nxt    = x_w ? W_RD : W_WR;
    // 前半寫完: 寫後半 (累加時先讀後半的舊值); 後半寫完: 下一列或完成
    assign w_nxt    = c_half ? (lr ? W_DONE : W_OUT) : (x_w ? W_RD : W_WR);
    assign wr_d     = c_half ? c_hi : c_line;
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin c_hi <= '0; c_half <= 1'b0; end
      else if (out_fire) begin c_hi <= mma_out_vec[N-1:N/2]; c_half <= 1'b0; end   // 只在輸出拍更新
      else if (wst == W_WR && wr_gnt) c_half <= ~c_half;
  end else if (RPL == 1) begin : g_c1
    assign c_line_n = mma_out_vec;
    assign o_nxt    = x_w ? W_RD : W_WR;
    assign w_nxt    = lr ? W_DONE : W_OUT;
    assign wr_d     = c_line;
  end else begin : g_cn
    localparam int CW = $clog2(RPL);
    wire [CW-1:0] slot = mma_out_row[CW-1:0];
    always_comb begin
      c_line_n = c_line;
      c_line_n[slot * N * 32 +: N * 32] = mma_out_vec;
    end
    assign o_nxt   = (slot != CW'(RPL - 1)) ? W_OUT : x_w ? W_RD : W_WR;
    assign w_nxt   = lr ? W_DONE : W_OUT;
    assign wr_d     = c_line;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fst <= F_IDLE; wst <= W_IDLE; a_ptr <= '0; b_ptr <= '0; c_ptr <= '0; k_left <= '0; fa <= '0; fb <= '0;
      step <= '0; buf_a <= '0; buf_b <= '0; wa <= '0; wb <= '0; rptr <= '0; a_cnt <= '0; b_cnt <= '0;
      c_line <= '0; c_cmd <= '0; c_pend <= '0; t_cmd <= '0; t_pend <= '0; t_w <= '0; pend <= 1'b0; lr <= 1'b0;
      x_cmd <= 1'b0; x_pend <= 1'b0; x_w <= 1'b0; o_line <= '0;
    end else begin
      // 取線器 (緩衝只在 grant 拍寫入)
      if (a_gnt) begin buf_a[wa] <= a_dat; wa <= wa + 1'b1; a_ptr <= a_ptr + 1'b1; fa <= fa - 1'b1; end
      if (b_gnt) begin buf_b[wb] <= b_dat; wb <= wb + 1'b1; b_ptr <= b_ptr + 1'b1; fb <= fb - 1'b1; end
      // 餵入器
      if (feed) begin
        k_left <= k_left - 1'b1;
        step   <= pop ? '0 : step + 1'b1;
      end
      if (pop) rptr <= rptr + 1'b1;
      a_cnt <= a_cnt + (PW+1)'(a_gnt) - (PW+1)'(pop);
      b_cnt <= b_cnt + (PW+1)'(b_gnt) - (PW+1)'(pop);
      // 輸入端
      unique case (fst)
        F_IDLE: if (cmd_valid) begin
          a_ptr <= cmd_a_line; b_ptr <= cmd_b_line; c_cmd <= cmd_c_line; t_cmd <= cmd_tag; x_cmd <= cmd_acc;
          k_left <= cmd_k; fa <= FW'((32'(cmd_k) + SPL - 1) / SPL); fb <= FW'((32'(cmd_k) + SPL - 1) / SPL);
          step <= '0; wa <= '0; wb <= '0; rptr <= '0; a_cnt <= '0; b_cnt <= '0;
          fst <= F_RUN;
        end
        F_RUN: if (feed_last) fst <= F_IDLE;
      endcase
      // 等待輸出: 最後一步餵入時移入 (此時 npu_mma 輸出端閒置 → 前一個 tile 已被寫回器取走, pend = 0)
      if (feed_last) begin c_pend <= c_cmd; t_pend <= t_cmd; x_pend <= x_cmd; pend <= 1'b1; end
      else if (take) pend <= 1'b0;
      // 寫回器
      // 取下一個 tile 只發生在 W_IDLE / W_DONE (wst 轉移寫在各自的 case 項, 狀態機的弧與實際可達的轉移一致)
      if (take) begin c_ptr <= c_pend; t_w <= t_pend; x_w <= x_pend; end
      unique case (wst)
        W_IDLE: if (take) wst <= W_OUT;
        W_OUT: if (out_fire) begin
          c_line <= c_line_n; lr <= (mma_out_row == RW'(N - 1));
          wst <= o_nxt;
        end
        W_RD: if (w_gnt) begin o_line <= rd_data[0]; wst <= W_WR; end
        W_WR: if (wr_gnt) begin c_ptr <= c_ptr + 1'b1; wst <= w_nxt; end
        W_DONE: if (take) wst <= W_OUT; else if (done_ready) wst <= W_IDLE;
      endcase
    end
  end
endmodule : npu_gemm_cu
