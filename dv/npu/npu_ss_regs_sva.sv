// SPDX-License-Identifier: Apache-2.0
// npu_ss_regs 的 SVA: APB4 從端協定 (零等待、PSLVERR 只在存取階段)、命令輸出的 valid / ready 協定 (等待時保持)、
// cpl_ready 只在讀 CPL_POP 的存取拍、FIFO 數量守恆; 功能 coverage (每個暫存器的讀 / 寫、位元組寫入、溢位、W1C、中斷來源)。
// 以 bind 掛到 DUT, .* 連接同名訊號。
module npu_ss_regs_sva
  import orca_pkg::*;
#(parameter int CQ = 4) (
  input logic clk, rst_n,
  input logic psel, penable, pwrite, pready, pslverr, irq,
  input logic [11:0] paddr,
  input logic [3:0]  pstrb,
  input logic cmd_valid, cmd_ready, cpl_valid, cpl_ready,
  input npu_tile_cmd_t cmd,
  input logic push, push_req, pop, q_full, hit,
  input logic [$clog2(CQ):0] q_n,
  input logic clk_aon,                         // 常開 (未閘控) 時脈: retention 性質以此取樣
  input logic ret_save, ret_restore, wr,       // retention (v6.3.11)
  input logic [9:0] wa,
  input logic [3:0] r_ien, ien_ret
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  // 主端協定 (TB 遵守)
  // (penable 是共用訊號: 沒被選中的從端也會看到 penable = 1, 不假設 penable → psel)
  m_access: assume property (psel && !penable |=> psel && penable && $stable(paddr) && $stable(pwrite));
  m_rdstrb: assume property (psel && !pwrite |-> pstrb == '0);
  m_cmd:    assume property (1'b1);                                  // (命令接收端為 TB 模型, 無額外假設)

  a_ready:   assert property (pready);
  a_slverr:  assert property (pslverr |-> psel && penable && !hit);
  a_cmd_hold: assert property (cmd_valid && !cmd_ready |=> cmd_valid && $stable(cmd));
  a_cpl_rd:  assert property (cpl_ready |-> psel && penable && !pwrite);
  a_qn:      assert property (q_n <= ($clog2(CQ)+1)'(CQ));
  a_qcnt:    assert property (q_n == $past(q_n) + ($past(push) ? 1 : 0) - ($past(pop) ? 1 : 0));
  a_drop:    assert property (push_req && q_full |-> !push);
  // retention: IRQ_EN 只在 APB 寫入或 ret_restore 拍改變; restore 載入影子 (同拍寫入 IRQ_EN 時寫入優先)。
  // 以未閘控的 clk_aon 取樣: clk 被電源域閘控時, 域 reset 在時脈停止期間發生, 以 clk 取樣的 |=> 會誤報
  a_ien_hold: assert property (@(posedge clk_aon) !(wr && wa == 10'h007) && !ret_restore |=> $stable(r_ien));
  a_ien_rest: assert property (@(posedge clk_aon) ret_restore && !(wr && wa == 10'h007) |=> r_ien == $past(ien_ret));

  c_full_push: cover property (push_req && q_full);
  c_push_pop:  cover property (push && pop);
  c_full_pop:  cover property (q_full && pop ##1 push);
  c_cpl_pop:   cover property (cpl_ready && cpl_valid);
  c_cpl_empty: cover property (cpl_ready && !cpl_valid);
  c_irq:       cover property (irq);
  c_b2b:       cover property (psel && penable ##1 psel && !penable);  // 連續交易
  c_save:      cover property (ret_save && r_ien != '0);
  c_restore:   cover property (ret_restore && ien_ret != r_ien);        // 還原改變了 IRQ_EN
  c_rest_wr:   cover property (ret_restore && wr && wa == 10'h007);      // 還原與寫入同拍 (寫入優先)

  covergroup cg_apb @(posedge clk iff (rst_n && psel && penable));
    cp_reg: coverpoint paddr[11:2] {
      bins id = {'h000}; bins cfg0 = {'h001}; bins cfg1 = {'h002}; bins status = {'h005}; bins irq_st = {'h006};
      bins irq_en = {'h007}; bins ctl = {'h008}; bins ext = {'h009}; bins a = {'h00A}; bins b = {'h00B};
      bins c = {'h00C}; bins push = {'h00D}; bins pop = {'h010}; bins peek = {'h011}; bins perf = {'h014};
      bins pb_lo = {'h015}; bins pb_hi = {'h016}; bins ps_lo = {'h017}; bins ps_hi = {'h018};
      bins undef = default;
    }
    cp_dir: coverpoint pwrite;
    x_reg_dir: cross cp_reg, cp_dir;
    cp_strb: coverpoint pstrb iff (pwrite) { bins all = {4'hF}; bins none = {4'h0}; bins part = {[1:14]}; }
    cp_low: coverpoint paddr[1:0];
  endgroup
  cg_apb u_cg = new;
endmodule

bind npu_ss_regs npu_ss_regs_sva #(.CQ(CQ)) u_sva (.*);
