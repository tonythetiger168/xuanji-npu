// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU MoE dispatch (v6.3.13): router 的 top-k 結果 → 每個 expert 的 token 清單位置 (dispatch 表) 與各 expert 數量
//
// 一個命令處理一批 (token, expert, weight) 項目 (top-k 的輸出, 每個 token K 項, 順序任意), 為每一項算出它在
// dispatch 緩衝中的位置 pos, 依輸入順序輸出 {pos, expert, token, weight, drop}; 最後輸出每個 expert 的數量與起點。
// 上層 (DMA 間接模式) 依 pos 把 token 的 activation 搬到 expert 的連續區域 (gather), expert GEMM 之後再依同一張表
// 加權搬回 (scatter / combine)。這個單元只算位置, 不搬資料。
//
// 兩種模式 (cmd_dropless):
//   容量模式 (0, GShard / Switch): expert e 的區域 = [e × cap, (e + 1) × cap); 第 r 個 (依輸入順序) 指向 e 的項目
//       pos = e × cap + r; r ≥ cap 的項目 drop = 1 (pos = 0, 不搬)。輸入只需一趟。
//   無丟棄模式 (1, MegaBlocks 式): expert 區域緊密相接, pos = off[e] + r, off = 各 expert 數量的前綴和。
//       需要輸入兩趟 (相同內容、相同順序): 第一趟統計數量 (沒有輸出), 第二趟輸出位置。
// expert ≥ NEXP (或 ≥ cmd_nexp) 的項目: err = 1、drop = 1, 不計入任何 expert。
// 數量輸出 (cnt_*): 每個 expert 一筆 {expert, n, off}; 容量模式 n = min(數量, cap)、off = e × cap; 無丟棄模式 n = 數量。
// 一個命令的項目數上限 2^PW − 1 (超過時 ovf = 1, 之後的位置無效; 由軟體保證不發生)。
//
// 時序: 命令開始時以 NEXP 拍清除計數 (busy), 之後接受輸入; 輸出暫存一級 (in_ready = 輸出暫存可用)。
// 低功耗: 閒置時 ICG 關閉。
`include "orca_pkg.sv"

module npu_moe_disp #(
  parameter int NEXP = 16,                    // expert 數上限 (2 的冪)
  parameter int TW   = 12,                    // token 索引寬度
  parameter int PW   = 14,                    // dispatch 位置寬度 (容量模式: cap ≤ 2^(PW − log2 NEXP) − 1)
  parameter int WW   = 16,                    // weight 寬度 (原樣傳遞, 例如 BF16)
  localparam int EW = $clog2(NEXP), CW = PW - EW
)(
  input  logic clk, rst_n,
  input  logic test_en,
  // 命令
  input  logic          cmd_valid,
  output logic          cmd_ready,
  input  logic          cmd_dropless,
  input  logic [CW-1:0] cmd_cap,              // 容量模式的每 expert 容量
  input  logic [EW:0]   cmd_nexp,             // 本命令實際的 expert 數 (1..NEXP); 之外的 expert id 當錯誤
  // 輸入項目 (無丟棄模式送兩趟, 每趟以 in_last 結束)
  input  logic          in_valid,
  output logic          in_ready,
  input  logic [EW+1:0] in_exp,               // 多 2 bit: 可表達超出範圍的 id (例如 router 的 int8 / int16 id 截取)
  input  logic [TW-1:0] in_tok,
  input  logic [WW-1:0] in_w,
  input  logic          in_last,
  // 輸出 (第二趟 / 容量模式的唯一一趟)
  output logic          out_valid,
  input  logic          out_ready,
  output logic [PW-1:0] out_pos,
  output logic [EW-1:0] out_exp,
  output logic [TW-1:0] out_tok,
  output logic [WW-1:0] out_w,
  output logic          out_drop,
  output logic          out_err,
  // 各 expert 數量 (輸入結束後依 expert 順序 cmd_nexp 筆)
  output logic          cnt_valid,
  input  logic          cnt_ready,
  output logic [EW-1:0] cnt_exp,
  output logic [PW:0]   cnt_n,
  output logic [PW:0]   cnt_off,
  output logic          done,                 // 最後一筆數量被接受的那一拍 (一拍)
  output logic          ovf,                  // 本命令項目數超過 2^PW − 1 (黏著到下一個命令)
  output logic          busy
);
  typedef enum logic [2:0] { S_IDLE, S_CLR, S_P0, S_PFX, S_P1, S_CNT } st_e;
  st_e st;
  logic          dl;                          // 本命令無丟棄
  logic [CW-1:0] cap;
  logic [EW:0]   ne;
  logic [EW:0]   i;                           // CLR / PFX / CNT 的 expert 索引
  logic [PW:0]   n   [NEXP];                  // 各 expert 已見數量
  logic [PW:0]   off [NEXP];                  // 無丟棄模式的起點
  logic [PW:0]   sum, tot;

  logic gclk;
  wire  en = cmd_valid || st != S_IDLE || out_valid || done;
  clock_gate u_icg (.clk(clk), .en(en || !rst_n), .test_en(test_en), .clk_out(gclk));

  assign cmd_ready = (st == S_IDLE);
  assign busy      = (st != S_IDLE);
  wire   o_free    = !out_valid || out_ready;
  assign in_ready  = (st == S_P0) || (st == S_P1 && o_free);
  wire   in_fire   = in_valid && in_ready;

  // 本項目
  wire          bad  = in_exp >= (EW+2)'(ne);
  wire [EW-1:0] e    = in_exp[EW-1:0];
  wire [PW:0]   r    = n[e];
  wire          full = !dl && r >= (PW+1)'(cap);
  wire [PW:0]   pos  = dl ? off[e] + r : (PW+1)'(e) * (PW+1)'(cap) + r;

  wire [EW-1:0] ci = i[EW-1:0];
  assign cnt_valid = (st == S_CNT);
  assign cnt_exp   = ci;
  assign cnt_n     = (!dl && n[ci] > (PW+1)'(cap)) ? (PW+1)'(cap) : n[ci];
  assign cnt_off   = dl ? off[ci] : (PW+1)'(ci) * (PW+1)'(cap);
  wire   cnt_fire  = cnt_valid && cnt_ready;

  always_ff @(posedge gclk or negedge rst_n)
    if (!rst_n) begin
      st <= S_IDLE; dl <= 1'b0; cap <= '0; ne <= '0; i <= '0; sum <= '0; tot <= '0; ovf <= 1'b0; done <= 1'b0;
      out_valid <= 1'b0; out_pos <= '0; out_exp <= '0; out_tok <= '0; out_w <= '0; out_drop <= 1'b0; out_err <= 1'b0;
    end else begin
      done <= 1'b0;
      if (out_valid && out_ready) out_valid <= 1'b0;
      case (st)
        S_IDLE: if (cmd_valid) begin
          st <= S_CLR; dl <= cmd_dropless; cap <= cmd_cap; ne <= cmd_nexp; i <= '0; tot <= '0; ovf <= 1'b0;
        end
        S_CLR: begin
          i <= i + 1'b1;
          if (i == ne - 1'b1) begin st <= dl ? S_P0 : S_P1; i <= '0; end
        end
        S_P0: if (in_fire) begin
          if (!bad) tot <= tot + 1'b1;
          if (in_last) begin st <= S_PFX; i <= '0; sum <= '0; end
        end
        S_PFX: begin                          // off[e] = Σ n[0..e−1], n[e] 清為 0 (第二趟重新計數)
          sum <= sum + n[ci];
          i <= i + 1'b1;
          if (i == ne - 1'b1) begin st <= S_P1; i <= '0; end
        end
        S_P1: if (in_fire) begin
          out_valid <= 1'b1;
          out_exp   <= bad ? '0 : e;
          out_tok   <= in_tok;
          out_w     <= in_w;
          out_err   <= bad;
          out_drop  <= bad || full;
          out_pos   <= (bad || full) ? '0 : pos[PW-1:0];
          if (!dl && !bad) tot <= tot + 1'b1;
          if (in_last) begin st <= S_CNT; i <= '0; end
        end
        S_CNT: if (cnt_fire) begin
          i <= i + 1'b1;
          if (i == ne - 1'b1) begin st <= S_IDLE; done <= 1'b1; end
        end
        /*verilator coverage_off*/
        default: st <= S_IDLE;                // 不可達 (6 / 7 不是狀態編碼)
        /*verilator coverage_on*/
      endcase
      // 項目數上限: 計數 (tot) 到 2^PW 即溢位 (命令開始時清除)
      if (tot[PW] && st != S_IDLE) ovf <= 1'b1;
    end

  // 計數陣列 (無重置; CLR 清除)
  always_ff @(posedge gclk) begin
    if (st == S_CLR) n[ci] <= '0;
    if (st == S_PFX) begin off[ci] <= sum; n[ci] <= '0; end
    if (in_fire && !bad) n[e] <= r + 1'b1;   // 容量模式: 超出容量的也計數 (cnt_n 取 min)
  end

`ifndef SYNTHESIS
  /*verilator coverage_off*/
  initial assert (NEXP >= 2 && (NEXP & (NEXP - 1)) == 0 && CW >= 1)
    else $fatal(1, "npu_moe_disp: NEXP 須為 >= 2 的 2 的冪, PW > log2(NEXP)");
  /*verilator coverage_on*/
`endif
endmodule : npu_moe_disp
