// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU paged KV cache 管理 (v6.3.13): 頁配發 / 區塊表 / 共享參考計數 / token 位址轉換
// 取代 rtl/legacy/ai/cloud/npu_kv_cache.sv 的頁管理 (issue #2 P0; 舊 RTL 的錯誤見 dv/npu_kv/legacy_kv_repro_tb.sv)。
//
// 模型 (與 vLLM PagedAttention 相同): KV 以「區塊」為單位存放, 一個區塊 = BS 個 token, 放在一個實體頁。
//   每個序列有 NSLOT 個邏輯區塊 (slot), 區塊表 (seq, slot) → 實體頁。token 位置 pos 屬於 slot = pos / BS,
//   頁內 offset = pos % BS; KV line 位址 = page × BS + offset (KV 資料本身放在 L2 / 外部記憶體, 由 DMA / CU 存取)。
//
// 控制命令 (cmd_valid / cmd_ready; 回應 rsp_valid / rsp_ready, 一個回應暫存器; 命令依序執行):
//   KV_ALLOC    (seq, slot)          配一個空閒頁給 (seq, slot), 參考計數 = 1。      rsp_data = 頁
//   KV_FREE     (seq, slot)          解除 (seq, slot) 的映射, 參考計數 −1, 歸零時頁回到空閒表。 rsp_data = 頁
//   KV_FREE_SEQ (seq)                解除整個序列的映射 (逐 slot, NSLOT 拍)。         rsp_data = 解除的 slot 數
//   KV_SHARE    (seq, slot, src_seq) (seq, slot) 指向 (src_seq, slot) 的同一頁, 參考計數 +1 (prefix 共享 / fork)。
//                                                                                   rsp_data = 頁
//   狀態 rsp_st: OK / NOMEM (沒有空閒頁) / MAPPED (目標 slot 已有映射) / UNMAPPED (要釋放或共享來源的 slot 沒有映射) /
//                REFMAX (參考計數已達上限 2^RW − 1)。錯誤的命令不改變任何狀態。
//   by construction 不會 double free: 釋放只經由區塊表, 未映射的 slot 回 UNMAPPED。
//
// 位址轉換埠 (tr_*): 每拍一個, 延遲 1 拍, 不回壓。看到的是本拍控制命令執行「之前」的區塊表。
//   tr_hit = pos 在範圍內 (pos < NSLOT × BS) 且 slot 有映射; tr_oor = pos 超出範圍; 未命中時 tr_line = 0。
//
// 空閒表: 從未配發過的頁由水位 hw (0 → NP) 依序配發, 釋放的頁進 FIFO; 配發優先用 FIFO。重置後不需要初始化週期。
// 儲存: 區塊表有效位元 = flop (需重置); 區塊表頁號、參考計數、FIFO 可對應 SRAM (只在有效 / 已配發時讀)。
// 低功耗: 沒有命令 / 回應 / 轉換時停止時脈 (ICG)。
`include "npu_pkg.sv"

package npu_kv_pkg;
  typedef enum logic [1:0] { KV_ALLOC = 2'd0, KV_FREE = 2'd1, KV_FREE_SEQ = 2'd2, KV_SHARE = 2'd3 } kv_op_e;
  typedef enum logic [2:0] { KV_OK = 3'd0, KV_NOMEM = 3'd1, KV_MAPPED = 3'd2, KV_UNMAPPED = 3'd3, KV_REFMAX = 3'd4 } kv_st_e;
endpackage

module npu_kv_pager
  import npu_kv_pkg::*;
#(
  parameter int NP    = 64,     // 實體頁數 (2 的冪)
  parameter int NSEQ  = 8,      // 序列數 (2 的冪)
  parameter int NSLOT = 16,     // 每序列的區塊數 (2 的冪)
  parameter int BS    = 16,     // 每區塊 token 數 (2 的冪)
  parameter int RW    = 3,      // 參考計數寬度 (共享上限 2^RW − 1)
  parameter int POSW  = 20,     // token 位置寬度 (須 > log2(NSLOT × BS) 才能表達超出範圍)
  localparam int PW = $clog2(NP), SW = $clog2(NSEQ), LW = $clog2(NSLOT), OW = $clog2(BS),
  localparam int DW = (PW > LW ? PW : LW) + 1
)(
  input  logic clk, rst_n,
  input  logic test_en,
  // 控制
  input  logic          cmd_valid,
  output logic          cmd_ready,
  input  kv_op_e        cmd_op,
  input  logic [SW-1:0] cmd_seq,
  input  logic [LW-1:0] cmd_slot,
  input  logic [SW-1:0] cmd_src_seq,
  output logic          rsp_valid,
  input  logic          rsp_ready,
  output kv_st_e        rsp_st,
  output logic [DW-1:0] rsp_data,
  // 位址轉換
  input  logic            tr_valid,
  input  logic [SW-1:0]   tr_seq,
  input  logic [POSW-1:0] tr_pos,
  output logic            tr_rvalid,
  output logic            tr_hit,
  output logic            tr_oor,
  output logic [PW+OW-1:0] tr_line,
  // 狀態
  output logic [PW:0]   free_cnt,
  output logic          busy
);
  localparam int NE = NSEQ * NSLOT;
  localparam logic [RW-1:0] RMAX = '1;

  logic gclk;
  logic sweep;                                  // KV_FREE_SEQ 逐 slot 執行中
  wire  cmd_fire = cmd_valid && cmd_ready;
  wire  en = cmd_valid || sweep || rsp_valid || tr_valid || tr_rvalid;
  clock_gate u_icg (.clk(clk), .en(en || !rst_n), .test_en(test_en), .clk_out(gclk));

  logic [NE-1:0]       bt_v;                    // 區塊表有效
  logic [PW-1:0]       bt_p [NE];               // 區塊表頁號
  logic [RW-1:0]       rc   [NP];               // 參考計數
  logic [PW-1:0]       fq   [NP];               // 釋放的頁 FIFO
  logic [PW-1:0]       fq_rd, fq_wr;
  logic [PW:0]         fq_n, hw;                // FIFO 中的頁數 / 從未配發的頁的水位
  logic [SW-1:0]       sw_seq;
  logic [LW-1:0]       sw_slot;
  logic [LW:0]         sw_n;

  assign free_cnt  = (PW+1)'(NP) - hw + fq_n;
  assign cmd_ready = !sweep && (!rsp_valid || rsp_ready);
  assign busy      = sweep;

  // ---------------- 控制: 本拍操作的 entry ----------------
  wire [SW+LW-1:0] e   = sweep ? {sw_seq, sw_slot} : {cmd_seq, cmd_slot};
  wire [SW+LW-1:0] es  = {cmd_src_seq, cmd_slot};
  wire             ev  = bt_v[e];
  wire [PW-1:0]    ep  = bt_p[e];
  wire [PW-1:0]    sp  = bt_p[es];
  wire             fq_e = (fq_n == '0);
  wire [PW-1:0]    np  = fq_e ? hw[PW-1:0] : fq[fq_rd];       // 下一個配發的頁 (free_cnt > 0 時有效)

  // 解除 entry e 的映射 (FREE 與 FREE_SEQ 共用): 參考計數 −1, 歸零時推入 FIFO
  wire do_unmap = (sweep && ev) || (cmd_fire && cmd_op == KV_FREE && ev);
  wire push     = do_unmap && rc[ep] == RW'(1);

  kv_st_e st;
  always_comb begin
    st = KV_OK;
    case (cmd_op)
      KV_ALLOC:    if (ev) st = KV_MAPPED; else if (free_cnt == '0) st = KV_NOMEM;
      KV_FREE:     if (!ev) st = KV_UNMAPPED;
      KV_SHARE:    if (ev) st = KV_MAPPED; else if (!bt_v[es]) st = KV_UNMAPPED; else if (rc[sp] == RMAX) st = KV_REFMAX;
      default:     st = KV_OK;
    endcase
  end
  wire do_alloc = cmd_fire && cmd_op == KV_ALLOC && st == KV_OK;
  wire do_share = cmd_fire && cmd_op == KV_SHARE && st == KV_OK;
  wire sw_last  = sweep && sw_slot == LW'(NSLOT - 1);

  always_ff @(posedge gclk or negedge rst_n)
    if (!rst_n) begin
      bt_v <= '0; fq_rd <= '0; fq_wr <= '0; fq_n <= '0; hw <= '0;
      sweep <= 1'b0; sw_seq <= '0; sw_slot <= '0; sw_n <= '0;
      rsp_valid <= 1'b0; rsp_st <= KV_OK; rsp_data <= '0;
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 1'b0;
      // 回應
      if (cmd_fire && cmd_op != KV_FREE_SEQ) begin
        rsp_valid <= 1'b1;
        rsp_st    <= st;
        rsp_data  <= (st != KV_OK) ? '0 : DW'(cmd_op == KV_ALLOC ? np : cmd_op == KV_FREE ? ep : sp);
      end
      if (sw_last) begin
        rsp_valid <= 1'b1;
        rsp_st    <= KV_OK;
        rsp_data  <= DW'(sw_n + (LW+1)'(ev));
      end
      // FREE_SEQ 逐 slot
      if (cmd_fire && cmd_op == KV_FREE_SEQ) begin
        sweep <= 1'b1; sw_seq <= cmd_seq; sw_slot <= '0; sw_n <= '0;
      end else if (sweep) begin
        sw_slot <= sw_slot + 1'b1;
        sw_n    <= sw_n + (LW+1)'(ev);
        if (sw_last) sweep <= 1'b0;
      end
      // 區塊表 / 空閒表
      if (do_alloc) begin
        bt_v[e] <= 1'b1;
        if (fq_e) hw <= hw + 1'b1;
        else begin fq_rd <= fq_rd + 1'b1; fq_n <= fq_n - 1'b1; end
      end
      if (do_share) bt_v[e] <= 1'b1;
      if (do_unmap) bt_v[e] <= 1'b0;
      if (push) begin fq_wr <= fq_wr + 1'b1; fq_n <= fq_n + 1'b1; end
    end

  // 無重置的儲存 (SRAM 式): 只在 entry 有效 / 頁已配發時被讀取
  always_ff @(posedge gclk) begin
    if (do_alloc) begin bt_p[e] <= np; rc[np] <= RW'(1); end
    if (do_share) begin bt_p[e] <= sp; rc[sp] <= rc[sp] + 1'b1; end
    if (do_unmap) rc[ep] <= rc[ep] - 1'b1;
    if (push) fq[fq_wr] <= ep;
  end

  // ---------------- 位址轉換 (1 拍) ----------------
  wire [LW-1:0]    t_slot = tr_pos[OW +: LW];
  wire             t_oor  = (tr_pos >> (OW + LW)) != '0;
  wire [SW+LW-1:0] te     = {tr_seq, t_slot};
  wire             t_hit  = !t_oor && bt_v[te];
  always_ff @(posedge gclk or negedge rst_n)
    if (!rst_n) begin
      tr_rvalid <= 1'b0; tr_hit <= 1'b0; tr_oor <= 1'b0; tr_line <= '0;
    end else begin
      tr_rvalid <= tr_valid;
      tr_hit    <= tr_valid && t_hit;
      tr_oor    <= tr_valid && t_oor;
      tr_line   <= (tr_valid && t_hit) ? {bt_p[te], tr_pos[OW-1:0]} : '0;
    end

`ifndef SYNTHESIS
  /*verilator coverage_off*/   // 參數檢查 (elaboration 時固定, 不是設計邏輯)
  initial begin
    assert (NP >= 2 && (NP & (NP - 1)) == 0 && NSEQ >= 2 && (NSEQ & (NSEQ - 1)) == 0 &&
            NSLOT >= 2 && (NSLOT & (NSLOT - 1)) == 0 && BS >= 2 && (BS & (BS - 1)) == 0 && RW >= 2 && POSW > OW + LW)
      else $fatal(1, "npu_kv_pager: NP / NSEQ / NSLOT / BS 須為 >= 2 的 2 的冪, RW >= 2, POSW > log2(NSLOT*BS)");
  end
  /*verilator coverage_on*/
`endif
endmodule : npu_kv_pager
