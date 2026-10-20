// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 矩陣乘法引擎 (v6.3.7)
`include "orca_pkg.sv"

// npu_mma: N×N output-stationary systolic INT8 GEMM tile, C[N][N] (INT32) = A[N][K] × B[K][N]
//
// 取代原 npu_pe / npu_systolic 的運算核心 (原設計在 weight-stationary 模式下權重從未載入,
// 乘積恆為 0; activation / 權重沒有 skew, 部分和輸入被忽略 — 無法算出 GEMM)。
//
// 資料流 (每步 k 一次握手):
//   a_vec[i] = A[i][k] 由左側進入第 i 列 (延遲 i 拍), 向右傳遞;
//   b_vec[j] = B[k][j] 由上方進入第 j 欄 (延遲 j 拍), 向下傳遞;
//   PE[i][j] 在 t = k + i + j 同時收到 A[i][k] 與 B[k][j] (各帶一個 valid token), 累加乘積。
//   輸入空拍 (in_valid=0) 不產生 token, 不會被累加。
// 流程: start (清累加器, 設定 a_signed / b_signed) → K 次輸入握手 (最後一次 in_last=1)
//       → 排空 2N-1 拍 → 依序輸出 N 列 (out_valid / out_ready, out_row = 列號)。
// 低功耗: 每列 PE 經 ICG (rtl/common/clock_gate.sv), 只有該列有 token 在流動 (或 start 清除)
//         時才有時脈; 暫存器另有相同的資料致能, 功能不依賴 clock gating (test_en 強制開啟時結果相同)。
module npu_mma
  import orca_pkg::*;
#(parameter int N = 8)
(
  input  logic clk, rst_n,
  input  logic test_en,                        // DFT: 強制開啟所有 ICG
  // 控制
  input  logic start,                          // 只在 busy=0 時有效
  input  logic a_signed, b_signed,             // 本 tile 的 A / B 是否為有號 INT8
  output logic busy,
  // 輸入串流: A 的一欄 + B 的一列
  input  logic             in_valid,
  output logic             in_ready,
  input  logic             in_last,
  input  logic [N-1:0][7:0] a_vec,
  input  logic [N-1:0][7:0] b_vec,
  // 輸出: C 的一列
  output logic              out_valid,
  input  logic              out_ready,
  output logic [$clog2(N)-1:0] out_row,
  output logic [N-1:0][31:0]   out_vec
);
  localparam int RW = $clog2(N);
  localparam int DRAIN = 2 * N - 1;            // 最後一個 token 從注入到 PE[N-1][N-1] 累加完成
  localparam int DW = $clog2(DRAIN + 1);

  typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_OUT } st_t;
  st_t st, st_n;
  logic sgn_a, sgn_b;
  logic [DW-1:0] drain_cnt;

  assign busy      = (st != S_IDLE);
  assign in_ready  = (st == S_RUN);
  assign out_valid = (st == S_OUT);
  wire   fire_in   = in_valid && in_ready;
  wire   clr       = (st == S_IDLE) && start;   // 新 tile: 清除累加器

  // ---------------- 控制 FSM ----------------
  always_comb begin
    st_n = st;
    unique case (st)
      S_IDLE:  if (start) st_n = S_RUN;
      S_RUN:   if (fire_in && in_last) st_n = S_DRAIN;
      S_DRAIN: if (drain_cnt == DW'(DRAIN - 1)) st_n = S_OUT;
      S_OUT:   if (out_ready && out_row == RW'(N - 1)) st_n = S_IDLE;
    endcase
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; sgn_a <= 1'b0; sgn_b <= 1'b0; drain_cnt <= '0; out_row <= '0;
    end else begin
      st <= st_n;
      if (clr) begin sgn_a <= a_signed; sgn_b <= b_signed; end
      drain_cnt <= (st == S_DRAIN) ? drain_cnt + 1'b1 : '0;
      if (st == S_OUT && out_ready) out_row <= out_row + 1'b1;
      else if (st != S_OUT)         out_row <= '0;
    end
  end

  // ---------------- 輸入 skew: 第 i 列 / 第 j 欄延遲 i / j 拍 ----------------
  // a_skew[i] / b_skew[j]: 進入陣列邊緣的資料與 token
  logic [N-1:0][7:0] a_edge, b_edge;
  logic [N-1:0]      a_tok_edge, b_tok_edge;
  for (genvar i = 0; i < N; i++) begin : g_skew
    if (i == 0) begin : g_d0
      assign a_edge[0] = a_vec[0];  assign a_tok_edge[0] = fire_in;
      assign b_edge[0] = b_vec[0];  assign b_tok_edge[0] = fire_in;
    end else begin : g_dn
      logic [i-1:0][7:0] a_sr, b_sr;   // i 級延遲線
      logic [i-1:0]      t_sr;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          a_sr <= '0; b_sr <= '0; t_sr <= '0;
        end else begin
          // 只在有 token 移動時更新資料 (operand isolation: 空拍不翻轉資料位元)
          t_sr <= {t_sr, fire_in};
          for (int d = 0; d < i; d++) begin
            if ((d == 0) ? fire_in : t_sr[d-1]) begin
              a_sr[d] <= (d == 0) ? a_vec[i] : a_sr[d-1];
              b_sr[d] <= (d == 0) ? b_vec[i] : b_sr[d-1];
            end
          end
        end
      end
      assign a_edge[i] = a_sr[i-1];  assign a_tok_edge[i] = t_sr[i-1];
      assign b_edge[i] = b_sr[i-1];  assign b_tok_edge[i] = t_sr[i-1];
    end
  end

  // ---------------- PE 陣列 ----------------
  // 同一步 k 的 A[i][k] 與 B[k][j] 於 t = k + i + j 同時抵達 PE[i][j], 因此一個 token 就夠:
  // tok_q[r][c] 同時標記本拍 a_q / b_q 有效 (由左邊傳入; 上方 PE 的 token 與之相同 — SVA a_tok_diag)。
  logic [N-1:0][N-1:0][7:0]  a_q, b_q;          // PE 輸出暫存 (向右 / 向下)
  logic [N-1:0][N-1:0]       tok_q;
  logic [N-1:0][N-1:0][31:0] acc;
  logic [N-1:0]              row_act;           // 第 r 列的時脈致能
  for (genvar r = 0; r < N; r++) begin : g_row
    // 列時脈致能: 本列左邊緣有 token 進入, 或本列 token 暫存尚未清空 (否則停住的 token 會讓右側 / 下一列
    // PE 重複累加), 或 start 清除累加器。上方傳入的 token 不必另外列入: 它與 PE[r][c-1] (c=0 時為左邊緣)
    // 的 token 相同 (a_tok_diag), a_gate_safe 檢查關閉時鐘時上一列也沒有 token。
    assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);
    logic gclk;
    clock_gate u_icg (.clk(clk), .en(row_act[r] || clr), .test_en(test_en), .clk_out(gclk));
    for (genvar c = 0; c < N; c++) begin : g_pe
      logic [7:0] a_in, b_in;                   // 左鄰 / 上鄰 (陣列邊緣接 skew 輸出)
      logic       t_in;
      if (c == 0) begin : g_l  assign a_in = a_edge[r];    assign t_in = a_tok_edge[r]; end
      else        begin : g_lc assign a_in = a_q[r][c-1];  assign t_in = tok_q[r][c-1]; end
      if (r == 0) begin : g_t  assign b_in = b_edge[c];    end
      else        begin : g_tc assign b_in = b_q[r-1][c];  end
      wire signed [8:0]  ax = {sgn_a & a_in[7], a_in};
      wire signed [8:0]  bx = {sgn_b & b_in[7], b_in};
      wire signed [17:0] prod = ax * bx;
      always_ff @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
          a_q[r][c] <= '0; b_q[r][c] <= '0; tok_q[r][c] <= 1'b0; acc[r][c] <= '0;
        end else begin
          tok_q[r][c] <= t_in;
          if (t_in) begin a_q[r][c] <= a_in; b_q[r][c] <= b_in; end   // 資料只在有 token 時更新
          if (clr)       acc[r][c] <= '0;
          else if (t_in) acc[r][c] <= acc[r][c] + 32'(prod);
        end
      end
    end
  end

  assign out_vec = acc[out_row];
endmodule : npu_mma
