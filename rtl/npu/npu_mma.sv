// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 矩陣乘法引擎 (v6.3.7; v6.3.12 tile 管線化; v6.3.16 FP8 / BF16)
`include "npu_pkg.sv"

// npu_mma: N×N output-stationary systolic INT8 GEMM tile, C[N][N] (INT32) = A[N][K] × B[K][N]
// v6.3.16: 浮點格式 (start 時 fp / bf): FP8 E4M3 / E5M2 (a_signed / b_signed 選子格式) 與 BF16, 結果為 FP32,
//   每個 PE 依 k 的順序以 FP32 (fp_add, RNE) 累加精確乘積; 乘加邏輯在 npu_pe_mac。BF16 每元素兩步 (先低位元組、
//   再高位元組; 位元組流與 INT8 相同, K 計的是步數)。
//
// 取代原 npu_pe / npu_systolic 的運算核心 (原設計在 weight-stationary 模式下權重從未載入,
// 乘積恆為 0; activation / 權重沒有 skew, 部分和輸入被忽略 — 無法算出 GEMM)。
//
// 資料流 (每步 k 一次握手):
//   a_vec[i] = A[i][k] 由左側進入第 i 列 (延遲 i 拍), 向右傳遞;
//   b_vec[j] = B[k][j] 由上方進入第 j 欄 (延遲 j 拍), 向下傳遞;
//   PE[i][j] 在 t = k + i + j 同時收到 A[i][k] 與 B[k][j] (各帶一個 valid token), 累加乘積。
//   輸入空拍 (in_valid=0) 不產生 token, 不會被累加。
// 流程: start (設定 a_signed / b_signed) → K 次輸入握手 (最後一次 in_last=1)
//       → 排空 2N-1 拍 → 依序輸出 N 列 (out_valid / out_ready, out_row = 列號)。
// v6.3.12 tile 管線化 (#33): token 帶旗標 {first, last, a_signed, b_signed} 跟著資料走; PE 在 first token
//   重新開始累加 (不需要清除拍), 在 last token 把最後的和存入結果暫存 res。輸出讀 res, 所以下一個 tile 可以在
//   本 tile 排空 / 輸出期間就 start 並輸入 (in_idle = 1 即可 start)。唯一的限制: 下一個 tile 的 in_last 要等
//   本 tile 的 N 列都輸出完 (否則會覆蓋 res) — 該拍 in_ready = 0。
// 低功耗: 每列 PE 經 ICG (rtl/common/clock_gate.sv), 只有該列有 token 在流動時才有時脈; 暫存器另有相同的
//         資料致能, 功能不依賴 clock gating (test_en 強制開啟時結果相同)。
module npu_mma
  import npu_pkg::*;
#(parameter int N = 8)
(
  input  logic clk, rst_n,
  input  logic test_en,                        // DFT: 強制開啟所有 ICG
  // 控制
  input  logic start,                          // 只在 in_idle=1 時有效
  input  logic a_signed, b_signed,             // 本 tile 的 A / B 是否為有號 INT8 (FP8 時: 1 = E5M2, 0 = E4M3)
  input  logic fp, bf,                         // v6.3.16: 1 = 浮點 (FP32 累加), bf = 1: BF16 (byte-planar, 每元素兩步), 否則 FP8
  output logic busy,                           // 輸入或輸出端還有 tile
  output logic in_idle,                        // 輸入端閒置: 可以 start 下一個 tile
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

  // 輸入端: I_IDLE → (start) → I_RUN → (in_last 握手) → I_IDLE
  // 輸出端: O_IDLE → (in_last 握手) → O_DRAIN (DRAIN 拍) → O_OUT (N 列) → O_IDLE
  typedef enum logic { I_IDLE, I_RUN } ist_t;
  typedef enum logic [1:0] { O_IDLE, O_DRAIN, O_OUT } ost_t;
  ist_t ist;
  ost_t ost;
  logic sgn_a, sgn_b, first;
  logic m_fp, m_bf, hb;                        // 本 tile 的格式; hb: BF16 下一步是高位元組
  logic [DW-1:0] drain_cnt;

  assign busy      = (ist != I_IDLE) || (ost != O_IDLE);
  assign in_idle   = (ist == I_IDLE);
  // 上一個 tile 還沒輸出完時, 本 tile 的最後一步不可注入
  assign in_ready  = (ist == I_RUN) && !(in_last && ost != O_IDLE);
  assign out_valid = (ost == O_OUT);
  wire   fire_in   = in_valid && in_ready;
  wire   go        = (ist == I_IDLE) && start;
  wire   fire_last = fire_in && in_last;

  // ---------------- 控制 FSM ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ist <= I_IDLE; ost <= O_IDLE; sgn_a <= 1'b0; sgn_b <= 1'b0; first <= 1'b0; drain_cnt <= '0; out_row <= '0;
      m_fp <= 1'b0; m_bf <= 1'b0; hb <= 1'b0;
    end else begin
      if (go) begin ist <= I_RUN; sgn_a <= a_signed; sgn_b <= b_signed; first <= 1'b1; m_fp <= fp; m_bf <= fp && bf; hb <= 1'b0; end
      else if (fire_in) begin
        first <= 1'b0;
        hb <= m_bf && !hb;
        if (in_last) ist <= I_IDLE;
      end
      unique case (ost)
        O_IDLE:  if (fire_last) ost <= O_DRAIN;
        O_DRAIN: if (drain_cnt == DW'(DRAIN - 1)) ost <= O_OUT;
        O_OUT:   if (out_ready && out_row == RW'(N - 1)) ost <= O_IDLE;
        // 2 bit 編碼的第 4 個值不可達
        /* verilator coverage_off */
        default: ost <= O_IDLE;
        /* verilator coverage_on */
      endcase
      drain_cnt <= (ost == O_DRAIN) ? drain_cnt + 1'b1 : '0;
      if (ost == O_OUT && out_ready) out_row <= out_row + 1'b1;
      else if (ost != O_OUT)         out_row <= '0;
    end
  end

  // token 旗標: {first, last, a 有號, b 有號, 浮點, BF16, BF16 高位元組}; 與 token 一起經過 skew 與 PE 陣列 (沿列向右傳遞)
  typedef struct packed { logic f, l, sa, sb, fp, bf, h; } tfl_t;
  wire tfl_t fl_in = '{f: first, l: in_last, sa: sgn_a, sb: sgn_b, fp: m_fp, bf: m_bf, h: hb};

  // ---------------- 輸入 skew: 第 i 列 / 第 j 欄延遲 i / j 拍 ----------------
  // a_skew[i] / b_skew[j]: 進入陣列邊緣的資料與 token (列端另帶旗標)
  logic [N-1:0][7:0] a_edge, b_edge;
  logic [N-1:0]      a_tok_edge, b_tok_edge;
  tfl_t [N-1:0]      fl_edge;
  for (genvar i = 0; i < N; i++) begin : g_skew
    if (i == 0) begin : g_d0
      assign a_edge[0] = a_vec[0];  assign a_tok_edge[0] = fire_in;  assign fl_edge[0] = fl_in;
      assign b_edge[0] = b_vec[0];  assign b_tok_edge[0] = fire_in;
    end else begin : g_dn
      logic [i-1:0][7:0] a_sr, b_sr;   // i 級延遲線
      logic [i-1:0]      t_sr;
      tfl_t [i-1:0]      f_sr;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          a_sr <= '0; b_sr <= '0; t_sr <= '0; f_sr <= '0;
        end else begin
          // 只在有 token 移動時更新資料 (operand isolation: 空拍不翻轉資料位元)
          t_sr <= {t_sr, fire_in};
          for (int d = 0; d < i; d++) begin
            if ((d == 0) ? fire_in : t_sr[d-1]) begin
              a_sr[d] <= (d == 0) ? a_vec[i] : a_sr[d-1];
              b_sr[d] <= (d == 0) ? b_vec[i] : b_sr[d-1];
              f_sr[d] <= (d == 0) ? fl_in : f_sr[d-1];
            end
          end
        end
      end
      assign a_edge[i] = a_sr[i-1];  assign a_tok_edge[i] = t_sr[i-1];  assign fl_edge[i] = f_sr[i-1];
      assign b_edge[i] = b_sr[i-1];  assign b_tok_edge[i] = t_sr[i-1];
    end
  end

  // ---------------- PE 陣列 ----------------
  // 同一步 k 的 A[i][k] 與 B[k][j] 於 t = k + i + j 同時抵達 PE[i][j], 因此一個 token 就夠:
  // tok_q[r][c] 同時標記本拍 a_q / b_q 有效 (由左邊傳入; 上方 PE 的 token 與之相同 — SVA a_tok_diag)。
  // 旗標 fl_q 跟著 token 沿列傳遞 (同一步 k 的旗標在整個陣列相同)。
  logic [N-1:0][N-1:0][7:0]  a_q, b_q;          // PE 輸出暫存 (向右 / 向下)
  logic [N-1:0][N-1:0]       tok_q;
  tfl_t [N-1:0][N-1:0]       fl_q;
  logic [N-1:0][N-1:0][31:0] acc, res;          // 累加器 / 結果 (last token 時存入)
  logic [N-1:0]              row_act;           // 第 r 列的時脈致能
  for (genvar r = 0; r < N; r++) begin : g_row
    // 列時脈致能: 本列左邊緣有 token 進入, 或本列 token 暫存尚未清空 (否則停住的 token 會讓右側 / 下一列
    // PE 重複累加)。上方傳入的 token 不必另外列入: 它與 PE[r][c-1] (c=0 時為左邊緣) 的 token 相同
    // (a_tok_diag), a_gate_safe 檢查關閉時鐘時上一列也沒有 token。
    assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);
    logic gclk;
    clock_gate u_icg (.clk(clk), .en(row_act[r] || !rst_n), .test_en(test_en), .clk_out(gclk));
    for (genvar c = 0; c < N; c++) begin : g_pe
      logic [7:0] a_in, b_in;                   // 左鄰 / 上鄰 (陣列邊緣接 skew 輸出)
      logic       t_in;
      tfl_t       f_in;
      if (c == 0) begin : g_l  assign a_in = a_edge[r];    assign t_in = a_tok_edge[r]; assign f_in = fl_edge[r]; end
      else        begin : g_lc assign a_in = a_q[r][c-1];  assign t_in = tok_q[r][c-1]; assign f_in = fl_q[r][c-1]; end
      if (r == 0) begin : g_t  assign b_in = b_edge[c];    end
      else        begin : g_tc assign b_in = b_q[r-1][c];  end
      // BF16: 低位元組步驟 (h = 0) 只鎖存 A / B 的低位元組與 first, 下一個 (高位元組) 步驟才乘加 (更新 acc);
      // 最後一步是低位元組 (奇數步) 時結果仍取自本拍的 sum (運算元 = {0x00, 低位元組} = ±0)
      logic [7:0] a_lo, b_lo;
      logic       f_lo;
      wire        mac = !f_in.bf || f_in.h;
      logic [31:0] sum;
      npu_pe_mac u_mac (.a(a_in), .b(b_in), .a_lo(a_lo), .b_lo(b_lo), .sa(f_in.sa), .sb(f_in.sb), .fp(f_in.fp), .bf(f_in.bf),
                        .h(f_in.h), .first(f_in.h ? f_lo : f_in.f), .acc(acc[r][c]), .sum(sum));
      always_ff @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
          a_q[r][c] <= '0; b_q[r][c] <= '0; tok_q[r][c] <= 1'b0; fl_q[r][c] <= '0; acc[r][c] <= '0; res[r][c] <= '0;
          a_lo <= '0; b_lo <= '0; f_lo <= 1'b0;
        end else begin
          tok_q[r][c] <= t_in;
          if (t_in) begin                        // 資料只在有 token 時更新
            a_q[r][c] <= a_in; b_q[r][c] <= b_in; fl_q[r][c] <= f_in;
            if (mac) acc[r][c] <= sum;
            else begin a_lo <= a_in; b_lo <= b_in; f_lo <= f_in.f; end
            if (f_in.l) res[r][c] <= sum;
          end
        end
      end
    end
  end

  assign out_vec = res[out_row];
endmodule : npu_mma
