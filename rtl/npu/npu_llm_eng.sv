// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU LLM 引擎 (v6.3.13): NPU Subsystem 的 KV (op 5) 與 MOE (op 6) 命令。
//   KV:  paged KV cache 頁管理 (npu_kv_pager) 的控制命令, 以及把區塊表轉成 DMA 間接模式的索引表 (KV_MAP)。
//   MOE: 從 L2 讀 router 的 top-k 結果, 經 npu_moe_disp 算出位置, 寫出 gather 索引表、combine 表與各 expert 數量。
//   資料搬移由 DMA 間接模式 (索引 0xFFFFFFFF = 略過) 負責; 本引擎只讀寫 L2 中的小表格。
// 命令一次一個 (閒置時接受), 完成時 done 一拍。L2 介面與 npu_vec_eng 相同 (請求 / 位址 / 資料保持到 grant,
// 讀資料與 grant 同拍)。所有表格以 u32 little-endian 字組存放, 字組 i 在 line 的 bit [32i +: 32]。
//
// ---- KV 命令 (cmd_op = 0) ----
//   sub = b[2:0]: 0 ALLOC, 1 FREE, 2 FREE_SEQ, 3 SHARE (語意見 npu_kv_pager), 4 MAP; 5..7 → 狀態 BADARG
//   seq = c[7:0], slot = c[23:8]; SHARE 的來源序列 = ext[7:0]
//   ALLOC..SHARE: 結果 line 寫到 L2 line a: w0 狀態, w1 資料 (頁 / 解除數), w2 free_cnt
//   MAP: seq 的 slot .. slot + k − 1 → 索引表寫到 line a 起 (每 line 16 項): 有映射 = 頁 × ext (ext = 每頁的外部
//        line 數, 例如 KV 資料 = 每區塊的 line 數; 量化 scale 表 = 1), 沒有映射或 slot ≥ NSLOT = 0xFFFFFFFF;
//        之後一個結果 line (a + ⌈k/16⌉): w0 狀態 (全部有映射 OK, 否則 UNMAPPED), w1 有映射的數目, w2 free_cnt
//   seq ≥ NSEQ 或 (ALLOC..SHARE 的) slot ≥ NSLOT 或 SHARE 來源 ≥ NSEQ 或 MAP 的 k > NSLOT → 狀態 BADARG (5),
//   只寫結果 line 到 a, 不送 pager
//   狀態: 0 OK, 1 NOMEM, 2 MAPPED, 3 UNMAPPED, 4 REFMAX, 5 BADARG
// ---- MOE 命令 (cmd_op = 1) ----
//   T = k 個 token (1..2^TW − 1); token t 的 top-k 在 L2 line a + 2t (16 × u32 expert id) 與 a + 2t + 1 (16 × FP32 weight)
//   K = b[4:0] (1..16), nexp = b[12:5] (1..NEXP), dropless = b[13], cap = ext[15:0] (< 2^(PW − log2 NEXP)),
//   mult = ext[31:16] (token 的外部 line 偏移 = token × mult)。參數不合法 (含 T = 0 或 T ≥ 2^TW) → 只在 line c
//   寫狀態 line (w0 = 0xFFFFFFFF)。T·K > 2^PW − 1 → 狀態 line 的 ovf = 1, 位置以 PW bit 回繞 (無效)
//   輸出從 L2 line c 起依序:
//     GL (gather 索引表) ⌈P/16⌉ lines, P = dropless ? T·K : nexp·cap; 先全部填 0xFFFFFFFF,
//        每個沒被 drop 的項目: GL[pos] = token × mult (讀 − 改 − 寫)
//     CT (combine 表) ⌈T·K/8⌉ lines, 依輸入順序每項 2 字組 {pos (drop = 0xFFFFFFFF), weight (drop = 0)}
//     CN (各 expert) ⌈nexp/8⌉ lines, 每 expert 2 字組 {n, off}
//     狀態 line: w0 = {ovf, 錯誤 id 數 (31 bit)}, w1 = 有效項目數, w2 = P
//   expert id ≥ nexp (含 top-k 未用的 0xFFFFFFFF) → 錯誤項目 (drop)。
`include "npu_pkg.sv"

module npu_llm_eng
  import npu_kv_pkg::*;
#(
  parameter int NP = 64, NSEQ = 8, NSLOT = 16, BS = 16, RW = 3,    // KV pager
  parameter int NEXP = 16, PW = 14,                                // MoE dispatch
  parameter int TW = 12,                                           // MOE 命令的 token 數上限 2^TW − 1
  localparam int PGW = $clog2(NP), SW = $clog2(NSEQ), LW = $clog2(NSLOT), OW = $clog2(BS), EW = $clog2(NEXP),
  localparam int KDW = (PGW > LW ? PGW : LW) + 1,
  localparam int XW = (TW + 4 > PW ? TW + 4 : PW),                 // 項目數 / P 的寬度 (T·K ≤ (2^TW − 1)·16; ne·cap < 2^PW)
  localparam int CNW = (EW > 3 ? EW - 3 : 0) + 1                   // CN line 數 ⌈nexp/8⌉ ≤ 2^(EW−3) 的寬度
)(
  input  logic clk, rst_n,
  input  logic test_en,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic        cmd_op,              // 0 KV, 1 MOE
  input  logic [25:0] cmd_a, cmd_b, cmd_c,
  input  logic [31:0] cmd_ext,
  input  logic [15:0] cmd_k,
  output logic        done,
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
  localparam int POSW = LW + OW + 1;
  typedef enum logic [4:0] {
    S_IDLE, S_WR, S_DONE,
    S_KCMD, S_KRSP, S_MISS, S_MCAP, S_MRES,
    S_OCHK, S_OINIT, S_OCMD, S_OLD, S_OFEED, S_OOUT, S_ORMW_R, S_ORMW_W, S_OCT, S_ONEXT, S_OCNT, S_OSTAT
  } st_e;
  st_e st, st_ret;                         // st_ret: S_WR 寫完之後的狀態

  // ---------------- 命令欄位 (鎖存) ----------------
  logic        op;
  logic [25:0] a, c;
  logic [31:0] ext;
  logic [15:0] kk;
  logic [2:0]  sub;
  logic [7:0]  seq8, src8, ne8;
  logic [15:0] slot16;
  logic [4:0]  K;
  logic        dl;

  // ---------------- L2 存取 ----------------
  logic [511:0] wbuf, cbuf;                // 寫出的 line / 累積中的表格 line
  logic [25:0]  wline, rline;
  logic         rq;
  assign rd_req  = rq;
  assign rd_line = rline;
  assign wr_req  = (st == S_WR);
  assign wr_line = wline;
  assign wr_data = wbuf;
  wire   rd_ok   = rq && rd_gnt;
  wire   wr_ok   = wr_req && wr_gnt;

  function automatic logic [511:0] put32(logic [511:0] l, int w, logic [31:0] v);
    l[32 * w +: 32] = v; return l;
  endfunction
  function automatic logic [511:0] res3(logic [31:0] w0, logic [31:0] w1, logic [31:0] w2);
    return put32(put32(put32('0, 0, w0), 1, w1), 2, w2);
  endfunction

  // ---------------- KV pager ----------------
  logic p_cmd_ready, p_rsp_valid, p_tr_rvalid, p_tr_hit, p_tr_oor, p_busy;
  kv_st_e p_st;
  logic [KDW-1:0] p_data;
  logic [PGW+OW-1:0] p_line;
  logic [PGW:0] p_free;
  logic [LW:0]  mi, mhit;                  // MAP: 已處理數 / 有映射數 (k ≤ NSLOT)
  wire  [16:0]  m_slot = 17'(slot16) + 17'(mi);
  wire          m_in  = m_slot < 17'(NSLOT);
  npu_kv_pager #(.NP(NP), .NSEQ(NSEQ), .NSLOT(NSLOT), .BS(BS), .RW(RW), .POSW(POSW)) u_pager (
    .clk(clk), .rst_n(rst_n), .test_en(test_en),
    .cmd_valid(st == S_KCMD && !kv_bad && sub != 3'd4), .cmd_ready(p_cmd_ready), .cmd_op(kv_op_e'(sub[1:0])),
    .cmd_seq(seq8[SW-1:0]), .cmd_slot(slot16[LW-1:0]), .cmd_src_seq(src8[SW-1:0]),
    .rsp_valid(p_rsp_valid), .rsp_ready(1'b1), .rsp_st(p_st), .rsp_data(p_data),
    .tr_valid(st == S_MISS && m_in), .tr_seq(seq8[SW-1:0]), .tr_pos(POSW'(m_slot[LW-1:0]) << OW),
    .tr_rvalid(p_tr_rvalid), .tr_hit(p_tr_hit), .tr_oor(p_tr_oor), .tr_line(p_line),
    .free_cnt(p_free), .busy(p_busy));
  wire kv_bad = seq8 >= 8'(NSEQ) || ((sub == 3'd4) ? kk > 16'(NSLOT)
                                                      : (sub[2] || slot16 >= 16'(NSLOT) || (sub == 3'd3 && src8 >= 8'(NSEQ))));

  // ---------------- MoE dispatch ----------------
  logic d_cmd_ready, d_in_ready, d_out_valid, d_out_drop, d_out_err, d_cnt_valid, d_done, d_ovf, d_busy;
  logic [PW-1:0] d_out_pos;
  logic [EW-1:0] d_out_exp, d_cnt_exp;
  logic [TW-1:0] d_out_tok;
  logic [31:0]   d_out_w;
  logic [PW:0]   d_cnt_n, d_cnt_off;
  logic [511:0]  ib, vb;                   // 目前 token 的 id / weight line
  logic [TW-1:0] t;                        // 目前 token
  logic [3:0]    j;                        // token 內的項目 (0..K−1 ≤ 15)
  logic          pass;                     // 輸出趟 (容量模式恆為 1; 無丟棄模式第二趟)
  logic          rsel;                     // OLD: 0 讀 id line, 1 讀 weight line
  logic [XW-4:0] gl_n;                     // 各區 line 數: GL ⌈P/16⌉, CT ⌈T·K/8⌉, CN ⌈nexp/8⌉; P
  logic [XW-3:0] ct_n;
  logic [CNW-1:0] cn_n;
  logic [XW-1:0] p_tot;
  logic [XW-3:0] gi;                       // OINIT / combine / count 已寫的 line
  logic [3:0]    cw;                       // cbuf 已填的項目 (combine / count 每 line 8 項)
  logic [EW:0]   ce;                       // 已收的 count 數
  logic [XW-1:0] nerr, nval;
  logic          e_drop;
  logic [PW-1:0] e_pos;
  logic [TW+15:0] e_val;                   // token × mult
  wire  [31:0]   eid     = ib[32 * j +: 32];
  wire           e_last  = (16'(t) == kk - 1'b1) && (5'(j) == K - 1'b1);
  wire           moe_bad = K == 5'd0 || K > 5'd16 || ne8 == 8'd0 || ne8 > 8'(NEXP) || ext[15:0] >= 16'(1 << (PW - EW)) ||
                           kk == '0 || kk >= 16'(1 << TW);
  wire  [25:0]   ct_base = c + 26'(gl_n);
  wire  [25:0]   cn_base = ct_base + 26'(ct_n);
  npu_moe_disp #(.NEXP(NEXP), .TW(TW), .PW(PW), .WW(32)) u_disp (
    .clk(clk), .rst_n(rst_n), .test_en(test_en),
    .cmd_valid(st == S_OCMD), .cmd_ready(d_cmd_ready), .cmd_dropless(dl), .cmd_cap(ext[PW-EW-1:0]),
    .cmd_nexp((EW+1)'(ne8)),
    .in_valid(st == S_OFEED), .in_ready(d_in_ready), .in_exp((eid >= 32'(1 << (EW + 2))) ? '1 : (EW+2)'(eid)),
    .in_tok(t), .in_w(vb[32 * j +: 32]), .in_last(e_last),
    .out_valid(d_out_valid), .out_ready(st == S_OOUT), .out_pos(d_out_pos), .out_exp(d_out_exp), .out_tok(d_out_tok),
    .out_w(d_out_w), .out_drop(d_out_drop), .out_err(d_out_err),
    .cnt_valid(d_cnt_valid), .cnt_ready(st == S_OCNT), .cnt_exp(d_cnt_exp), .cnt_n(d_cnt_n), .cnt_off(d_cnt_off),
    .done(d_done), .ovf(d_ovf), .busy(d_busy));

  // ---------------- 時脈 ----------------
  logic gclk;
  wire  en = cmd_valid || st != S_IDLE || done;
  clock_gate u_icg (.clk(clk), .en(en || !rst_n), .test_en(test_en), .clk_out(gclk));
  assign cmd_ready = (st == S_IDLE);
  assign busy      = (st != S_IDLE) || done;

  always_ff @(posedge gclk or negedge rst_n)
    if (!rst_n) begin
      st <= S_IDLE; st_ret <= S_IDLE; done <= 1'b0; rq <= 1'b0; rline <= '0; wline <= '0; wbuf <= '0; cbuf <= '0;
      op <= 1'b0; a <= '0; c <= '0; ext <= '0; kk <= '0; sub <= '0; seq8 <= '0; src8 <= '0; ne8 <= '0; slot16 <= '0;
      K <= '0; dl <= 1'b0; mi <= '0; mhit <= '0;
      ib <= '0; vb <= '0; t <= '0; j <= '0; pass <= 1'b0; rsel <= 1'b0;
      gl_n <= '0; ct_n <= '0; cn_n <= '0; p_tot <= '0; gi <= '0; cw <= '0; ce <= '0; nerr <= '0; nval <= '0;
      e_drop <= 1'b0; e_pos <= '0; e_val <= '0;
    end else begin
      done <= 1'b0;
      case (st)
        S_IDLE: if (cmd_valid) begin
          op <= cmd_op; a <= cmd_a; c <= cmd_c; ext <= cmd_ext; kk <= cmd_k;
          sub <= cmd_b[2:0]; seq8 <= cmd_c[7:0]; slot16 <= cmd_c[23:8]; src8 <= cmd_ext[7:0];
          K <= cmd_b[4:0]; ne8 <= cmd_b[12:5]; dl <= cmd_b[13];
          mi <= '0; mhit <= '0; cbuf <= '0; t <= '0; j <= '0; rsel <= 1'b0; gi <= '0; cw <= '0; ce <= '0;
          nerr <= '0; nval <= '0;
          st <= cmd_op ? S_OCHK : S_KCMD;
        end
        // 寫一個 line (wline / wbuf), 完成後到 st_ret
        S_WR: if (wr_ok) st <= st_ret;
        S_DONE: begin done <= 1'b1; st <= S_IDLE; end

        // ---------------- KV ----------------
        S_KCMD:
          if (kv_bad) begin
            wbuf <= res3(32'd5, 32'd0, 32'(p_free)); wline <= a; st <= S_WR; st_ret <= S_DONE;
          end else if (sub == 3'd4) st <= (kk == '0) ? S_MRES : S_MISS;
          else st <= S_KRSP;                 // pager 此時必定閒置 (SVA a_pager_rdy)
        S_KRSP: if (p_rsp_valid) begin
          wbuf <= res3(32'(p_st), 32'(p_data), 32'(p_free)); wline <= a; st <= S_WR; st_ret <= S_DONE;
        end
        S_MISS: st <= S_MCAP;              // 轉換已送出 (slot 超出範圍時不送, 下一拍當未命中)
        S_MCAP: begin
          automatic logic hit = p_tr_hit;    // slot ≥ NSLOT 時沒有送轉換, tr_hit = 0
          automatic logic [31:0] e = hit ? 32'(p_line >> OW) * ext : 32'hFFFF_FFFF;
          automatic logic [511:0] nb = put32(cbuf, int'(4'(mi)), e);
          mhit <= mhit + (LW+1)'(hit);
          mi   <= mi + 1'b1;
          if (4'(mi) == 4'd15 || 16'(mi) == kk - 1'b1) begin
            wbuf <= nb; cbuf <= '0; wline <= a + 26'(mi >> 4); st <= S_WR;
            st_ret <= (16'(mi) == kk - 1'b1) ? S_MRES : S_MISS;
          end else begin cbuf <= nb; st <= S_MISS; end
        end
        S_MRES: begin                      // MAP 結果 line
          wbuf  <= res3((16'(mhit) == kk) ? 32'd0 : 32'd3, 32'(mhit), 32'(p_free));
          wline <= a + 26'((32'(kk) + 32'd15) >> 4); st <= S_WR; st_ret <= S_DONE;
        end

        // ---------------- MOE ----------------
        S_OCHK: begin
          // 寬度依合法命令的界限 (非法命令不用這些值)
          automatic logic [XW:0] tkk = (XW+1)'(kk[TW-1:0]) * (XW+1)'(K);
          automatic logic [XW:0] P = dl ? tkk : (XW+1)'(ne8[EW:0]) * (XW+1)'(ext[PW-EW-1:0]);
          p_tot <= XW'(P);
          gl_n  <= (XW-3)'((P + 15) >> 4);
          ct_n  <= (XW-2)'((tkk + 7) >> 3);
          cn_n  <= CNW'((32'(ne8[EW:0]) + 32'd7) >> 3);
          pass  <= !dl;
          if (moe_bad) begin               // 只寫狀態 line (w0 = 0xFFFFFFFF) 到 c
            wbuf <= res3(32'hFFFF_FFFF, 32'd0, 32'd0); wline <= c; st <= S_WR; st_ret <= S_DONE;
          end else st <= S_OINIT;
        end
        S_OINIT:                           // GL 全部填 0xFFFFFFFF
          if (gi < (XW-2)'(gl_n)) begin
            wbuf <= '1; wline <= c + 26'(gi); gi <= gi + 1'b1; st <= S_WR; st_ret <= S_OINIT;
          end else begin gi <= '0; st <= S_OCMD; end
        S_OCMD: begin st <= S_OLD; rq <= 1'b1; rline <= a; rsel <= 1'b0; end   // dispatch 必定閒置 (SVA a_disp_rdy)
        S_OLD: if (rd_ok) begin            // token t: id line a + 2t, weight line a + 2t + 1
          if (!rsel) begin ib <= rd_data; rsel <= 1'b1; rline <= rline + 1'b1; end
          else begin vb <= rd_data; rsel <= 1'b0; rq <= 1'b0; st <= S_OFEED; end
        end
        S_OFEED: if (d_in_ready) st <= pass ? S_OOUT : S_ONEXT;   // 第一個項目可能要等 dispatch 清除計數
        S_OOUT: begin                      // 輸入的下一拍必有輸出 (SVA a_out_v)
          cbuf <= put32(put32(cbuf, 2 * int'(cw[2:0]), d_out_drop ? 32'hFFFF_FFFF : 32'(d_out_pos)),
                        2 * int'(cw[2:0]) + 1, d_out_drop ? 32'd0 : d_out_w);
          cw   <= cw + 1'b1;
          nerr <= nerr + XW'(d_out_err);
          nval <= nval + XW'(!d_out_err);
          e_drop <= d_out_drop; e_pos <= d_out_pos;
          e_val  <= (TW+16)'(d_out_tok) * (TW+16)'(ext[31:16]);
          st <= d_out_drop ? S_OCT : S_ORMW_R;
          if (!d_out_drop) begin rq <= 1'b1; rline <= c + 26'(d_out_pos >> 4); end
        end
        S_ORMW_R: if (rd_ok) begin
          rq <= 1'b0;
          wbuf <= put32(rd_data, int'(e_pos[3:0]), 32'(e_val)); wline <= rline; st <= S_WR; st_ret <= S_OCT;
        end
        S_OCT:                             // combine line 滿或最後一項 → 寫出
          if (cw == 4'd8 || e_last) begin
            wbuf <= cbuf; cbuf <= '0; cw <= '0; wline <= ct_base + 26'(gi); gi <= gi + 1'b1; st <= S_WR; st_ret <= S_ONEXT;
          end else st <= S_ONEXT;
        S_ONEXT:                           // 下一項
          if (e_last) begin
            t <= '0; j <= '0;
            if (!pass) begin pass <= 1'b1; st <= S_OLD; rq <= 1'b1; rline <= a; end
            else begin gi <= '0; st <= S_OCNT; end
          end else if (5'(j) == K - 1'b1) begin
            j <= '0; t <= t + 1'b1; st <= S_OLD; rq <= 1'b1; rline <= a + {25'(t) + 25'd1, 1'b0};
          end else begin j <= j + 1'b1; st <= S_OFEED; end
        S_OCNT: begin                      // 輸入結束後 dispatch 已在輸出數量 (SVA a_cnt_v)
          automatic logic [511:0] nb = put32(put32(cbuf, 2 * int'(cw[2:0]), 32'(d_cnt_n)), 2 * int'(cw[2:0]) + 1, 32'(d_cnt_off));
          ce <= ce + 1'b1;
          if (cw == 4'd7 || 8'(ce) == ne8 - 1'b1) begin
            wbuf <= nb; cbuf <= '0; cw <= '0; wline <= cn_base + 26'(gi); gi <= gi + 1'b1; st <= S_WR;
            st_ret <= (8'(ce) == ne8 - 1'b1) ? S_OSTAT : S_OCNT;
          end else begin cbuf <= nb; cw <= cw + 1'b1; end
        end
        S_OSTAT: begin
          wbuf <= res3({d_ovf, 31'(nerr)}, 32'(nval), 32'(p_tot)); wline <= cn_base + 26'(cn_n); st <= S_WR; st_ret <= S_DONE;
        end
        /*verilator coverage_off*/
        default: st <= S_IDLE;
        /*verilator coverage_on*/
      endcase
    end
endmodule : npu_llm_eng
