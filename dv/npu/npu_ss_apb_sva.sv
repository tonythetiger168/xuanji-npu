// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss_apb 邊界 SVA + functional coverage (APB 整合測試與 SoC 層都 bind):
//   APB4 協定 (主端行為以 assert 檢查, 錯誤代表 SoC 端的 APB 主端或互連有誤): setup → access 兩相、access 期間位址 /
//   方向 / 資料 / strobe 穩定、讀交易 pstrb = 0; 從端: pslverr 只在 access 且 pready 時有效;
//   busy (PMU 用) 為 0 時沒有外部記憶體請求、命令 FIFO 為空; 外部記憶體請求等 grant 期間保持。
// coverage: 每個 APB 交易的 讀寫 × busy × irq, 以及錯誤回應、背靠背交易。
//=============================================================================
module npu_ss_apb_sva (
  input logic        clk, rst_n,
  input logic        psel, penable, pwrite, pready, pslverr,
  input logic [11:0] paddr,
  input logic [31:0] pwdata,
  input logic [3:0]  pstrb,
  input logic        irq, busy, cmd_valid,
  input logic        m_rd_req, m_rd_gnt, m_wr_req, m_wr_gnt,
  input logic [31:0] m_rd_addr, m_wr_addr
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  wire setup  = psel && !penable;
  wire access = psel && penable;

  // penable 在多從端的 APB 上是共用的 (未選中時也可能為 1), 只檢查本從端被選中時的兩相
  a_apb_setup:   assert property (setup |=> access);
  a_apb_access:  assert property (access && !pready |=> access);
  a_apb_stable:  assert property (setup ##1 access |-> $stable(paddr) && $stable(pwrite) && $stable(pwdata) && $stable(pstrb));
  a_apb_rd_strb: assert property (access && !pwrite |-> pstrb == '0);
  a_apb_err:     assert property (pslverr |-> access && pready);
  a_idle:        assert property (!busy |-> !m_rd_req && !m_wr_req && !cmd_valid);
  a_mrd_hold:    assert property (m_rd_req && !m_rd_gnt |=> m_rd_req && $stable(m_rd_addr));
  a_mwr_hold:    assert property (m_wr_req && !m_wr_gnt |=> m_wr_req && $stable(m_wr_addr));

  covergroup cg_apb @(posedge clk iff rst_n && access && pready);
    cp_dir:  coverpoint pwrite;
    cp_busy: coverpoint busy;
    cp_irq:  coverpoint irq;
    cp_err:  coverpoint pslverr;
    x_ctx:   cross cp_dir, cp_busy, cp_irq;
  endgroup
  cg_apb u_cg_apb = new;
  c_b2b:      cover property (access ##1 setup);
  c_err_wr:   cover property (access && pwrite && pslverr);
  c_err_rd:   cover property (access && !pwrite && pslverr);
  c_irq_rise: cover property (!irq ##1 irq);
endmodule

bind orca_npu_ss_apb npu_ss_apb_sva u_apb_sva (
  .clk, .rst_n, .psel, .penable, .pwrite, .pready, .pslverr, .paddr, .pwdata, .pstrb, .irq, .busy, .cmd_valid,
  .m_rd_req, .m_rd_gnt, .m_wr_req, .m_wr_gnt, .m_rd_addr, .m_wr_addr);
