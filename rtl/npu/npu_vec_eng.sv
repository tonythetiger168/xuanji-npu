// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU 向量引擎 (v6.3.10): 從 L2 取向量程式, 依序交給 npu_vec 執行。NPU Subsystem 的 VEC 命令由此執行。
`include "orca_pkg.sv"
`include "npu_vec_pkg.sv"

// npu_vec_eng: 命令 = {程式起始 line, 指令數 n (>= 1)}。
//   程式版面: 一個 64 B line 放 4 個指令, 第 i 個指令在 bit [128*i +: 77] (npu_vec_ins_t), 其餘位元不用。
//   取指: 兩個 line 的緩衝 (FIFO), 執行一個 line 的同時預取下一個; 與 npu_vec 共用一個 L2 讀取埠,
//         預取只在 npu_vec 沒有讀取請求的拍發出, 發出後保持到 grant。
//   完成: 第 n 個指令被 npu_vec 接受的下一拍 done 一拍脈衝; derr = 這次命令執行中遇到非法指令 (npu_vec.err,
//         命令開始時清除)。命令只在閒置時接受。
//   限制: 程式不可被自己的 ST 改寫 (預取可能已讀走舊內容)。
// 低功耗: 閒置且沒有命令時引擎的取指 / 控制停止時脈 (ICG); npu_vec 用未閘控的 clk (它只有 err 一個未閘控的 flop,
//   每個向量暫存器各有 ICG; 也讓 npu_vec 的 SVA 以未閘控時脈取樣, 不會被時脈關閉期間的 reset 誤報)。
module npu_vec_eng
  import npu_vec_pkg::*;
#(parameter int NV = 8)
(
  input  logic clk, rst_n,
  input  logic test_en,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [25:0] cmd_line,
  input  logic [15:0] cmd_n,               // 須 >= 1 (orca_npu_ss 把 n = 0 當 NOP)
  output logic        done,
  output logic        derr,                // 與 done 同拍有效
  output logic        busy,
  output logic         rd_req,
  output logic [25:0]  rd_line,
  input  logic         rd_gnt,
  input  logic [511:0] rd_data,
  output logic         wr_req,
  output logic [25:0]  wr_line,
  output logic [511:0] wr_data,
  input  logic         wr_gnt
);
  logic gclk;
  logic act;
  wire  en = act || cmd_valid || done;
  clock_gate u_icg (.clk(clk), .en(en), .test_en(test_en), .clk_out(gclk));

  // ---------------- 取指 ----------------
  logic [511:0] fb [2];                    // 取指緩衝
  logic [1:0]   fv;                        // 各緩衝有效
  logic         fw, fr;                    // 寫入 / 讀出的緩衝
  logic [1:0]   slot;                      // 讀出緩衝中下一個指令
  logic [25:0]  pc;                        // 下一個要取的 line
  logic [15:0]  nf, ni;                    // 尚未取的指令數 / 尚未發出的指令數
  logic         v_rd_req, v_rd_gnt, v_ins_ready, v_err;
  logic [25:0]  v_rd_line;
  // 讀取埠: 取指只在 npu_vec 沒有讀取請求時開始; 一旦發出, 保持到 grant (L2 介面規則: 請求與位址保持),
  // 這期間 npu_vec 的 LD 等待 (它自己保持請求)
  logic f_pend;                            // 取指已發出、尚未 grant
  wire  f_want = act && nf != '0 && !fv[fw];
  wire  f_sel  = f_pend || (f_want && !v_rd_req);
  assign rd_req   = f_sel || v_rd_req;
  assign rd_line  = f_sel ? pc : v_rd_line;
  assign v_rd_gnt = !f_sel && v_rd_req && rd_gnt;
  wire  f_gnt = f_sel && rd_gnt;

  // ---------------- 發出 ----------------
  wire          iv   = act && ni != '0 && fv[fr];
  npu_vec_ins_t ins;
  assign ins = npu_vec_ins_t'(fb[fr][128 * slot +: 77]);
  wire  fire = iv && v_ins_ready;
  wire  last = fire && ni == 16'd1;
  wire  ln_end = fire && (slot == 2'd3 || last);  // 這個緩衝用完
  wire  start = cmd_valid && cmd_ready;
  assign cmd_ready = !act && !done;
  assign busy = act || done;
  assign derr = v_err;                     // done 拍: 已含最後一個指令 (err 在命令開始時清除)

  always_ff @(posedge gclk or negedge rst_n)
    if (!rst_n) begin
      act <= 1'b0; done <= 1'b0; f_pend <= 1'b0; fv <= '0; fw <= 1'b0; fr <= 1'b0; slot <= '0;
      pc <= '0; nf <= '0; ni <= '0;
    end else begin
      done <= last;
      f_pend <= f_sel && !rd_gnt;
      if (start) begin
        act <= 1'b1; pc <= cmd_line; nf <= cmd_n; ni <= cmd_n; fv <= '0; fw <= 1'b0; fr <= 1'b0; slot <= '0;
      end else begin
        if (f_gnt) begin
          fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd1;
          nf <= (nf > 16'd4) ? nf - 16'd4 : '0;
        end
        if (fire) begin
          ni <= ni - 16'd1;
          slot <= ln_end ? 2'd0 : slot + 2'd1;
          if (ln_end) begin fv[fr] <= 1'b0; fr <= !fr; end
        end
        if (last) act <= 1'b0;
      end
    end
  always_ff @(posedge gclk)                // 資料暫存器: 只在取指 grant 拍寫入
    if (f_gnt) fb[fw] <= rd_data;

  npu_vec #(.NV(NV)) u_vec (
    .clk(clk), .rst_n, .test_en, .ins_valid(iv), .ins_ready(v_ins_ready), .ins, .err(v_err), .err_clr(start),
    .rd_req(v_rd_req), .rd_line(v_rd_line), .rd_gnt(v_rd_gnt), .rd_data,
    .wr_req, .wr_line, .wr_data, .wr_gnt);
`ifdef FORMAL_TOP_npu_vec_eng
  // ---------------- 形式驗證 (formal/npu_vec_eng; npu_vec 以契約替身取代; 模擬時不編入) ----------------
  // 檢查器從埠記錄: 命令的起始 line 與指令數、任選序號 f_i (anyconst) 的指令所在 line 被取回時的內容
  // (取指 = 讀取埠有 grant 但 npu_vec 的 rd_gnt 為 0), 並證明第 f_i 個交給 npu_vec 的指令等於程式第 f_i 個字;
  // 每個命令恰好發出 n 個指令後 done 一次; L2 讀取請求保持到 grant。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  always @(posedge clk) if (f_past && rst_n && $past(rst_n) && $past(cmd_valid && !cmd_ready)) begin
    assume (cmd_valid && cmd_line == $past(cmd_line) && cmd_n == $past(cmd_n));
  end
  always @(*) begin
    if (cmd_valid) begin assume (cmd_n != '0); end
    assume (!test_en);                                  // test_en 只影響 ICG (替身直通)
  end
  (* anyconst *) logic [15:0] f_i;
  logic        f_act, f_have;
  logic [25:0] f_l0;
  logic [15:0] f_n, f_iss;
  logic [76:0] f_word;
  wire  f_start = cmd_valid && cmd_ready;
  wire  f_issue = iv && v_ins_ready;                    // 替身 npu_vec 的指令埠
  wire  f_fetch = rd_req && rd_gnt && !v_rd_gnt;        // 埠有 grant 但沒轉給 npu_vec (替身的輸入埠) → 取指
  always @(posedge clk or negedge rst_n)
    if (!rst_n) begin f_act <= 1'b0; f_have <= 1'b0; f_iss <= '0; f_n <= '0; f_l0 <= '0; end
    else begin
      if (f_start) begin f_act <= 1'b1; f_l0 <= cmd_line; f_n <= cmd_n; f_iss <= '0; f_have <= 1'b0; end
      else begin
        if (f_issue) f_iss <= f_iss + 16'd1;
        if (done) f_act <= 1'b0;
        if (f_fetch && f_act && rd_line == f_l0 + 26'(f_i >> 2) && !f_have) begin
          f_have <= 1'b1; f_word <= rd_data[128 * f_i[1:0] +: 77];
        end
      end
    end
  always @(*) if (rst_n) begin
    assert (cmd_ready == !f_act);
    assert (f_iss <= f_n);
    if (f_issue && f_act && f_iss == f_i && f_i < f_n) begin assert (f_have && ins == f_word); end
    if (done) begin assert (f_act && f_iss == f_n); end
    if (f_act && f_iss != f_n) begin assert (!done); end
    // 歸納強化 (只在斷言中引用 RTL 狀態)
    assert (act == (f_act && f_iss != f_n) && done == (f_act && f_iss == f_n));
    if (act) begin assert (ni == f_n - f_iss); end
    if (f_pend) begin assert (f_want); end                     // 等待 grant 的取指: 條件不會消失
    if (act || done) begin assert (pc - f_l0 == 26'((17'(f_n) - 17'(nf) + 17'd3) >> 2) && nf <= ni); end
    // 取指 / 緩衝的帳: F = 已取的指令數, I = 已發出; 緩衝中的 line 數 = ceil(F/4) - I/4; 讀出位置 slot = I mod 4
    if (act) begin
      assert (slot == f_iss[1:0]);
      assert (32'(fv[0]) + 32'(fv[1]) == 32'(((17'(f_n) - 17'(nf) + 17'd3) >> 2) - 17'(f_iss >> 2)));
      assert (fw == ((fv == 2'b01 || fv == 2'b10) ? !fr : fr));
      if (fv != 2'b00) begin assert (fv[fr]); end
    end
    if (f_act) begin assert (f_have == (((17'(f_n) - 17'(nf) + 17'd3) >> 2) > 17'(f_i >> 2))); end   // f_i 所在的 line 已取回
    // 被追蹤的字還沒發出時, 在它的 line 對應的緩衝裡 (目前的 line → 讀出緩衝, 下一個 line → 另一個)
    if (act && f_have && f_iss <= f_i) begin
      assert ((f_i >> 2) <= (f_iss >> 2) + 16'd1);
      if ((f_i >> 2) == (f_iss >> 2)) begin assert (fb[fr][128 * f_i[1:0] +: 77] == f_word); end
      else begin assert (fb[!fr][128 * f_i[1:0] +: 77] == f_word); end
    end
  end
  always @(posedge clk) if (f_past && rst_n && $past(rst_n)) begin
    if ($past(rd_req && !rd_gnt)) begin assert (rd_req && rd_line == $past(rd_line)); end
  end
  always @(posedge clk) if (f_past && rst_n) begin
    cover (done && f_n == 16'd9);
    cover (f_issue && f_iss == f_i && f_i == 16'd6);
  end
`endif
endmodule : npu_vec_eng
