// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU GEMM 計算單元 (v6.3.8)
`include "orca_pkg.sv"

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
// 低功耗: 緩衝只在讀取 grant 拍更新; npu_mma 列級 clock gating (test_en 強制開啟)。
module npu_gemm_cu
  import orca_pkg::*;
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
  output logic        done,                  // 一拍脈衝: C 已全部寫回
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

  // 編碼: G_OUT / G_WR 兩個位元都不同 (out_nxt 只在兩者間切換)
  typedef enum logic [1:0] { G_IDLE = 2'd0, G_OUT = 2'd1, G_WR = 2'd2, G_RUN = 2'd3 } gst_t;
  gst_t st;
  logic [25:0] a_ptr, b_ptr, c_ptr;
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
  logic         wr_end;                      // 本拍的寫回 grant 完成一整列 (N <= 16: 每次寫回; N = 32: 寫後半)

  // ---------------- npu_mma ----------------
  logic mma_start, mma_busy, mma_in_valid, mma_in_ready, mma_in_last, mma_out_valid, mma_out_ready;
  logic [N-1:0][7:0] a_vec, b_vec;
  logic [RW-1:0] mma_out_row;
  logic [N-1:0][31:0] mma_out_vec;

  assign mma_start     = (st == G_IDLE) && cmd_valid;
  assign have          = (a_cnt != '0) && (b_cnt != '0);
  assign mma_in_valid  = (st == G_RUN) && have;
  assign mma_in_last   = (k_left == 16'd1);
  assign a_vec         = buf_a[rptr][step * N * 8 +: N * 8];
  assign b_vec         = buf_b[rptr][step * N * 8 +: N * 8];
  assign mma_out_ready = (st == G_OUT);

  npu_mma #(.N(N)) u_mma (
    .clk(clk), .rst_n(rst_n), .test_en(test_en),
    .start(mma_start), .a_signed(cmd_a_signed), .b_signed(cmd_b_signed), .busy(mma_busy),
    .in_valid(mma_in_valid), .in_ready(mma_in_ready), .in_last(mma_in_last),
    .a_vec(a_vec), .b_vec(b_vec),
    .out_valid(mma_out_valid), .out_ready(mma_out_ready), .out_row(mma_out_row), .out_vec(mma_out_vec));

  // ---------------- L2 存取 ----------------
  assign cmd_ready = (st == G_IDLE);
  // 取線軌請求: 還有 line 要讀且該軌緩衝有空槽
  wire run = (st == G_RUN);
  if (RP == 2) begin : g_rp2
    // A 埠 / B 埠各自獨立
    assign a_req = run && (fa != '0) && (a_cnt != (PW+1)'(PF));
    assign b_req = run && (fb != '0) && (b_cnt != (PW+1)'(PF));
    assign rd_req  = {b_req, a_req};
    assign rd_line = {b_ptr, a_ptr};
    assign a_gnt = rd_gnt[0]; assign b_gnt = rd_gnt[1];
    assign a_dat = rd_data[0]; assign b_dat = rd_data[1];
  end else begin : g_rp1
    // 單一埠: A、B 輪流 (f_b); 讀 B 時對應的 A 已在緩衝 (b_cnt < a_cnt <= PF), 不會溢位
    logic f_b;                               // 0: 下一次讀 A, 1: 下一次讀 B
    assign a_req = run && !f_b && (fa != '0) && (a_cnt != (PW+1)'(PF));
    assign b_req = run && f_b && (fb != '0);
    assign rd_req  = a_req || b_req;
    assign rd_line = f_b ? b_ptr : a_ptr;
    assign a_gnt = rd_gnt[0] && !f_b; assign b_gnt = rd_gnt[0] && f_b;
    assign a_dat = rd_data[0]; assign b_dat = rd_data[0];
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) f_b <= 1'b0;
      else if (st == G_IDLE && cmd_valid) f_b <= 1'b0;
      else if (rd_gnt[0]) f_b <= ~f_b;
  end
  assign wr_req    = (st == G_WR);
  assign wr_line   = c_ptr;
  assign wr_data   = wr_d;

  wire feed = mma_in_valid;                  // npu_mma 在 RUN 恆 in_ready (SVA a_feed_ready)
  assign feeding = feed;
  wire pop  = feed && (step == SW'(SPL - 1) || mma_in_last);
  wire push = b_gnt;                         // 一項的 B 到齊 (RP = 1 時即完成一項)
  wire out_fire = mma_out_valid && mma_out_ready;

  // C line 組裝: 每列寫入 line 中的位置; 湊滿 RPL 列後寫回
  gst_t out_nxt;
  if (CPR > 1) begin : g_cw
    // N = 32: 一列 C = 2 個 line; 前半存在 c_line, 後半存在 c_hi, 依序寫回 (c_hi 只有此組態才有)
    logic [511:0] c_hi;
    logic         c_half;                    // 0: 寫前半, 1: 寫後半
    // mma_out_vec 是 [N-1:0][31:0] 二維 packed: 索引選的是元素 (每個 32 bit), 不是位元
    assign c_line_n = mma_out_vec[N/2-1:0];
    assign out_nxt  = G_WR;
    assign wr_d     = c_half ? c_hi : c_line;
    assign wr_end   = wr_gnt && c_half;
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) begin c_hi <= '0; c_half <= 1'b0; end
      else if (out_fire) begin c_hi <= mma_out_vec[N-1:N/2]; c_half <= 1'b0; end   // 只在輸出拍更新
      else if (st == G_WR && wr_gnt) c_half <= ~c_half;
  end else if (RPL == 1) begin : g_c1
    assign c_line_n = mma_out_vec;
    assign out_nxt  = G_WR;
    assign wr_d     = c_line;
    assign wr_end   = wr_gnt;
  end else begin : g_cn
    localparam int CW = $clog2(RPL);
    wire [CW-1:0] slot = mma_out_row[CW-1:0];
    always_comb begin
      c_line_n = c_line;
      c_line_n[slot * N * 32 +: N * 32] = mma_out_vec;
    end
    assign out_nxt = (slot == CW'(RPL - 1)) ? G_WR : G_OUT;
    assign wr_d     = c_line;
    assign wr_end   = wr_gnt;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= G_IDLE; a_ptr <= '0; b_ptr <= '0; c_ptr <= '0; k_left <= '0; fa <= '0; fb <= '0;
      step <= '0; buf_a <= '0; buf_b <= '0; wa <= '0; wb <= '0; rptr <= '0; a_cnt <= '0; b_cnt <= '0;
      c_line <= '0; done <= 1'b0;
    end else begin
      done <= 1'b0;
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
      unique case (st)
        G_IDLE: if (cmd_valid) begin
          a_ptr <= cmd_a_line; b_ptr <= cmd_b_line; c_ptr <= cmd_c_line;
          k_left <= cmd_k; fa <= FW'((32'(cmd_k) + SPL - 1) / SPL); fb <= FW'((32'(cmd_k) + SPL - 1) / SPL);
          step <= '0; wa <= '0; wb <= '0; rptr <= '0; a_cnt <= '0; b_cnt <= '0;
          st <= G_RUN;
        end
        G_RUN: if (feed && mma_in_last) st <= G_OUT;
        G_OUT: if (out_fire) begin c_line <= c_line_n; st <= out_nxt; end
        G_WR: begin
          if (wr_gnt) c_ptr <= c_ptr + 1'b1;
          if (wr_end) begin                                         // N = 32: 前半寫完留在 G_WR 寫後半
            if (!mma_busy) begin st <= G_IDLE; done <= 1'b1; end    // 最後一列已輸出, npu_mma 回到 idle
            else st <= G_OUT;
          end
        end
      endcase
    end
  end
endmodule : npu_gemm_cu
