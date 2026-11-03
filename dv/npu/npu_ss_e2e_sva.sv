// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss 端到端 SVA + functional coverage (只看子系統邊界, 任何環境都可 bind: 單元 CRV、APB 整合、SoC)
//   命令守恆: 每個被接受的命令 (依 id 計數) 恰好完成一次, 不會有未曾下過的 id 完成;
//   閒置 (busy = 0) 時沒有未完成的命令; 未完成命令數有上界 (佇列 QD + 分派後在途 OC_MAX);
//   不卡死: 有未完成命令時, 若既沒有任何進展 (命令 / 完成 / 外部記憶體 / host L2 握手、GEMM 步進)
//   也沒有在等環境 (完成回壓、外部 grant、DMA 讀回應), 連續 WD 拍即判定 hang。
// coverage (cg_e2e, 樣本於每個完成握手): 各 op 類別 × 端到端延遲 (含排隊)、各 op 類別的亂序完成 (越過更早的命令);
//   (cg_acc, 樣本於每個命令握手): 接受時的未完成數; 另有回壓情境的 cover property。
// hit[]: 與 coverage bins 一一對應的命中位元, 供 npu_ss_tb 的 +cdv (coverage-driven) 模式選擇下一個 batch 的激勵。
//=============================================================================
`include "orca_pkg.sv"
module npu_ss_e2e_sva
  import orca_pkg::*;
#(parameter int QD = 8, parameter int NCU = 4, parameter int WD = 4096)(
  input logic          clk, rst_n,
  input logic          cmd_valid, cmd_ready,
  input npu_tile_cmd_t cmd,
  input logic          cpl_valid, cpl_ready,
  input logic [7:0]    cpl_id,
  input logic          busy,
  input logic          m_rd_req, m_rd_gnt, m_rd_rvalid, m_wr_req, m_wr_gnt,
  input logic          h_rd_req, h_rd_gnt, h_wr_req, h_wr_gnt,
  input logic [47:0]   perf_steps,
  input logic          dma_rd_wait                   // DMA 有已發出、資料尚未寫入 L2 的讀
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  localparam int OC_MAX = 4 + 4 * NCU + 1 + 3;      // 與 npu_ss_sva 相同
  localparam int TMAX = QD + OC_MAX;
  localparam int NBIN = 28;
  // bins: [0:14] 延遲 (op 類別 * 3 + 桶), [15:19] 亂序 (op 類別), [20:24] 接受時未完成數, [25:27] 回壓情境
  wire acc = cmd_valid && cmd_ready;
  wire cpl = cpl_valid && cpl_ready;

  function automatic int unsigned cls(npu_tile_cmd_t c);   // 0 GEMM, 1 DMA_LD, 2 DMA_ST, 3 NOP / 保留, 4 VEC
    unique case (c.op)
      NPU_OP_GEMM: return 0;
      NPU_OP_DMA_LD: return 1;
      NPU_OP_DMA_ST: return 2;
      NPU_OP_VEC: return 4;
      default: return 3;
    endcase
  endfunction
  function automatic int unsigned lbin(longint unsigned l);
    return (l < 16) ? 0 : (l < 256) ? 1 : 2;
  endfunction

  logic [3:0]        pend  [256];
  logic [2:0]        pcls  [256];
  longint unsigned   pcyc  [256], pseq [256];
  int unsigned       tot, stall;
  longint unsigned   cyc, seq;
  logic [NBIN-1:0]   hit /* verilator public_flat_rd */;
  logic [47:0]       ps_q;

  covergroup cg_e2e with function sample(int unsigned c, int unsigned lb, bit ooo);
    cp_cls: coverpoint c { bins gemm = {0}; bins ld = {1}; bins st = {2}; bins nop = {3}; bins vec = {4}; }
    cp_lat: coverpoint lb { bins short = {0}; bins mid = {1}; bins long = {2}; }
    cp_ooo: coverpoint ooo { bins in_order = {0}; bins ooo = {1}; }
    x_lat:  cross cp_cls, cp_lat;
    x_ooo:  cross cp_cls, cp_ooo;
  endgroup
  covergroup cg_acc with function sample(int unsigned t);
    cp_tot: coverpoint t { bins idle = {0}; bins one = {1}; bins few = {[2:4]}; bins queue = {[5:QD]}; bins deep = {[QD+1:$]}; }
  endgroup
  cg_e2e u_cg_e2e = new;
  cg_acc u_cg_acc = new;

  // 進展 / 等待環境
  wire progress = acc || cpl || (m_rd_req && m_rd_gnt) || (m_wr_req && m_wr_gnt) || m_rd_rvalid ||
                  (h_rd_req && h_rd_gnt) || (h_wr_req && h_wr_gnt) || (perf_steps != ps_q);
  wire env_wait = (cpl_valid && !cpl_ready) || (m_rd_req && !m_rd_gnt) || (m_wr_req && !m_wr_gnt) || dma_rd_wait;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      foreach (pend[i]) pend[i] <= '0;
      tot <= 0; stall <= 0; seq <= 0;
    end else begin
      automatic int unsigned t = tot;
      ps_q <= perf_steps;
      if (cpl) begin
        automatic int unsigned c = pcls[cpl_id];
        automatic longint unsigned l = cyc - pcyc[cpl_id];
        automatic bit ooo = 1'b0;
        for (int i = 0; i < 256; i++) if (pend[i] != '0 && pseq[i] < pseq[cpl_id]) ooo = 1'b1;
        if (pend[cpl_id] == 4'd1) begin        // 同一 id 有多個在途時延遲 / 順序無意義, 不取樣
          u_cg_e2e.sample(c, lbin(l), ooo);
          hit[c * 3 + lbin(l)] <= 1'b1;
          if (ooo) hit[15 + c] <= 1'b1;
        end
        pend[cpl_id] <= pend[cpl_id] - 4'd1;
        t--;
      end
      if (acc) begin
        u_cg_acc.sample(tot);
        hit[20 + ((tot == 0) ? 0 : (tot == 1) ? 1 : (tot <= 4) ? 2 : (tot <= QD) ? 3 : 4)] <= 1'b1;
        pend[cmd.id] <= ((cpl && cpl_id == cmd.id) ? pend[cmd.id] - 4'd1 : pend[cmd.id]) + 4'd1;
        pcls[cmd.id] <= 3'(cls(cmd)); pcyc[cmd.id] <= cyc; pseq[cmd.id] <= seq;
        seq <= seq + 1;
        t++;
      end
      if (cpl_valid && !cpl_ready && tot >= 4) hit[25] <= 1'b1;
      if (cmd_valid && !cmd_ready && cpl_valid && !cpl_ready) hit[26] <= 1'b1;
      if (acc && cpl) hit[27] <= 1'b1;
      tot <= t;
      stall <= (tot == 0 || progress || env_wait) ? 0 : stall + 1;
    end
  end
  always @(posedge clk) cyc <= cyc + 1;
  initial begin hit = '0; cyc = 0; end

  a_cpl_known:  assert property (cpl |-> pend[cpl_id] != '0)
                else $error("npu_ss: completion id %0d was never issued (or already completed)", cpl_id);
  a_idle_empty: assert property (!busy |-> tot == 0)
                else $error("npu_ss: busy = 0 with %0d commands still outstanding", tot);
  a_tot_bound:  assert property (tot <= TMAX);
  a_no_hang:    assert property (stall < WD)
                else $error("npu_ss: %0d outstanding commands, no progress for %0d cycles", tot, WD);
  a_id_wrap:    assert property (acc |-> pend[cmd.id] != '1);

  c_cpl_bp_deep:  cover property (cpl_valid && !cpl_ready && tot >= 4);   // 完成回壓且多個命令在途
  c_both_bp:      cover property (cmd_valid && !cmd_ready && cpl_valid && !cpl_ready);
  c_acc_and_cpl:  cover property (acc && cpl);
endmodule

bind orca_npu_ss npu_ss_e2e_sva #(.QD(QD), .NCU(NCU)) u_e2e (
  .clk, .rst_n, .cmd_valid, .cmd_ready, .cmd, .cpl_valid, .cpl_ready, .cpl_id, .busy,
  .m_rd_req, .m_rd_gnt, .m_rd_rvalid, .m_wr_req, .m_wr_gnt,
  .h_rd_req, .h_rd_gnt, .h_wr_req, .h_wr_gnt, .perf_steps, .dma_rd_wait(u_dma.iss != u_dma.wr));
