// SPDX-License-Identifier: Apache-2.0
// orca_npu_ss / npu_ss_dma 的 SVA: 佇列 / 分派 (等待旗標) / 完成合併 / DMA 協定 / 時脈閘控安全 + 功能 coverage。
// 以 bind 掛到 DUT。

module npu_ss_dma_sva #(parameter int OUT = 8) (
  input logic clk, rst_n,
  input logic cmd_valid, cmd_ready, done, act, st,
  input logic [15:0] cmd_n, n, iss, wr, cols, col,
  input logic [31:0] stride,
  input logic [$clog2(OUT):0] f_cnt,
  input logic h_rd_req, h_rd_gnt, h_wr_req, h_wr_gnt, m_rd_req, m_rd_gnt, m_rd_rvalid, m_wr_req, m_wr_gnt,
  input logic [25:0] h_rd_line, h_wr_line,
  input logic [31:0] m_rd_addr, m_wr_addr,
  input logic [511:0] h_wr_data, m_wr_data
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  // 帳: wr <= iss <= n, 在途 <= OUT, FIFO 內容 <= 在途
  a_cnt_order:  assert property (!act || (wr <= iss && iss <= n));
  a_outst:      assert property (!act || (iss - wr) <= 16'(OUT));   // RTL 只保留 outst 的低 log2(OUT)+1 bit
  a_fifo:       assert property (32'(f_cnt) <= 32'(iss - wr) || !act);
  a_fifo_idle:  assert property (!act |-> f_cnt == '0);
  // 有在途額度就發出讀取 (吞吐量: 在途上限 OUT 全部可用)
  a_issue:      assert property (act && iss != n && (iss - wr) < 16'(OUT) |-> (st ? h_rd_req : m_rd_req));
  // 一次只有一個方向動作
  a_dir:        assert property (!(st && (m_rd_req || h_wr_req)) && !(!st && (h_rd_req || m_wr_req)));
  // 請求保持到 grant, 位址 / 資料不變
  a_mrd_hold:   assert property (m_rd_req && !m_rd_gnt |=> m_rd_req && $stable(m_rd_addr));
  a_mwr_hold:   assert property (m_wr_req && !m_wr_gnt |=> m_wr_req && $stable(m_wr_addr) && $stable(m_wr_data));
  a_hrd_hold:   assert property (h_rd_req && !h_rd_gnt |=> h_rd_req && $stable(h_rd_line));
  a_hwr_hold:   assert property (h_wr_req && !h_wr_gnt |=> h_wr_req && $stable(h_wr_line) && $stable(h_wr_data));
  // 讀回應只會在 LD 進行中、且有已發出未回來的讀取時出現
  a_rvalid:     assert property (m_rd_rvalid |-> act && !st && 32'(iss - wr) > 32'(f_cnt));
  // 完成: 一拍脈衝; n = 0 的命令接受後下一拍完成
  a_done_once:  assert property (done |=> !done && !act);
  a_n0:         assert property (cmd_valid && cmd_ready && cmd_n == '0 |=> done);
  a_ready:      assert property (cmd_ready == !act);

  c_ld:         cover property (act && !st && done && n > 16'(OUT));
  // 2D (外部端每列 cols 個 line): LD / ST 跨過列邊界、最後一列不滿、列距 0 / 小於列寬 (重疊)
  c_2d_ld:      cover property (act && !st && done && cols != '0 && n > cols);
  c_2d_st:      cover property (act && st && done && cols != '0 && n > cols);
  c_2d_part:    cover property (act && done && cols > 16'd1 && (n % cols) != '0 && n > cols);
  c_2d_s0:      cover property (act && done && cols != '0 && stride == '0 && n > cols);
  c_2d_ovl:     cover property (act && done && cols > 16'd1 && stride != '0 && stride < 32'(cols) && n > cols);
  a_2d_col:     assert property (!act || cols == '0 || col < cols);
  c_st:         cover property (act && st && done && n > 16'(OUT));
  c_n0:         cover property (cmd_valid && cmd_ready && cmd_n == '0);
  c_full:       cover property (act && (iss - wr) == 16'(OUT) && iss != n);
  c_mrd_wait:   cover property (m_rd_req && !m_rd_gnt);
  c_mwr_wait:   cover property (m_wr_req && !m_wr_gnt);
  c_hrd_wait:   cover property (h_rd_req && !h_rd_gnt);
  c_hwr_wait:   cover property (h_wr_req && !h_wr_gnt);
  c_ld_both:    cover property (m_rd_req && m_rd_gnt && m_rd_rvalid && h_wr_req && h_wr_gnt);   // 讀、回應、寫同拍
  c_st_both:    cover property (h_rd_req && h_rd_gnt && m_wr_req && m_wr_gnt);
  c_mrd_wrap:   cover property (m_rd_req && m_rd_gnt && m_rd_addr == '1);
  c_hwr_wrap:   cover property (h_wr_req && h_wr_gnt && h_wr_line == '1);

  covergroup cg_dma @(posedge clk iff rst_n && cmd_valid && cmd_ready);
    cp_n:   coverpoint cmd_n { bins zero = {0}; bins one = {1}; bins le_out[] = {[2:OUT]}; bins mid = {[OUT+1:4*OUT]}; bins big = {[4*OUT+1:$]}; }
  endgroup
  cg_dma u_cg_dma = new;
  covergroup cg_outst @(posedge clk iff rst_n && act);
    cp_o: coverpoint (iss - wr) { bins n[] = {[0:OUT]}; }
    cp_f: coverpoint f_cnt { bins n[] = {[0:OUT]}; }
  endgroup
  cg_outst u_cg_outst = new;
endmodule

// 以 tile 的未閘控時脈取樣 (DMA 時脈關閉期間發生的非同步 reset 仍會中止進行中的 |=> 檢查)
bind orca_npu_ss npu_ss_dma_sva #(.OUT(OUT)) u_dma_sva (
  .clk, .rst_n, .cmd_valid(u_dma.cmd_valid), .cmd_ready(u_dma.cmd_ready), .done(u_dma.done), .act(u_dma.act), .st(u_dma.st),
  .cmd_n(u_dma.cmd_n), .n(u_dma.n), .iss(u_dma.iss), .wr(u_dma.wr), .cols(u_dma.cols), .col(u_dma.col), .stride(u_dma.stride),
  .f_cnt(u_dma.f_cnt),
  .h_rd_req(u_dma.h_rd_req), .h_rd_gnt(u_dma.h_rd_gnt), .h_wr_req(u_dma.h_wr_req), .h_wr_gnt(u_dma.h_wr_gnt),
  .m_rd_req(u_dma.m_rd_req), .m_rd_gnt(u_dma.m_rd_gnt), .m_rd_rvalid(u_dma.m_rd_rvalid),
  .m_wr_req(u_dma.m_wr_req), .m_wr_gnt(u_dma.m_wr_gnt),
  .h_rd_line(u_dma.h_rd_line), .h_wr_line(u_dma.h_wr_line), .m_rd_addr(u_dma.m_rd_addr), .m_wr_addr(u_dma.m_wr_addr),
  .h_wr_data(u_dma.h_wr_data), .m_wr_data(u_dma.m_wr_data));

module npu_ss_sva #(parameter int QD = 8, parameter int NCU = 4) (
  input logic clk, rst_n, test_en,
  input logic cmd_valid, cmd_ready, cpl_valid, cpl_ready, busy, irq,
  input logic [7:0] cpl_id,
  input logic [$clog2(4 + NCU + 4 + 1)-1:0] oc,
  input logic [$clog2(QD):0] q_cnt,
  input logic q_hv, go, q_pop, g_core, g_dma, g_vec, g_nop, d_take, v_take, h_nop, c_cmd_ready, d_cmd_ready, v_cmd_ready,
  input logic head_wt,
  input logic [2:0] head_op,
  input logic [15:0] head_k,
  input logic dma_cv, nop_cv, vec_cv, c_cpl_valid, c_cpl_ready, c_busy, d_busy, v_busy, d_done, v_done, d_en, load,
  input logic [3:0] src_v, sel,
  input logic d_rd_req, d_wr_req, v_rd_req, v_wr_req, h_rd_req, h_rd_gnt, h_wr_req, m_rd_req, m_wr_req, m_rd_rvalid,
  input logic vec_err
);
  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);
  // npu_core 內最多: 佇列 4 + 每個 CU 1 (執行中或待回報) + 輸出暫存 1; 另加 DMA 1 + NOP 1 + VEC 1
  localparam int OC_MAX = 4 + NCU + 1 + 3;

  a_q_bound:    assert property (q_cnt <= ($clog2(QD)+1)'(QD));
  a_cmd_ready:  assert property (cmd_ready == (q_cnt != ($clog2(QD)+1)'(QD)));
  a_oc_bound:   assert property (int'(oc) <= OC_MAX);
  // 分派: 一次最多一個去處; 等待旗標的命令只在 oc = 0 時分派; 有命令可分派且去處就緒時必分派
  a_disp_one:   assert property ($onehot0({g_core, g_dma, g_vec, g_nop}));
  a_wt:         assert property (q_pop && head_wt |-> oc == '0);
  a_no_stall:   assert property (q_hv && (!head_wt || oc == '0) &&
                                 ((head_op == 3'd0 && !h_nop && c_cmd_ready) || (head_op inside {3'd1, 3'd2} && !dma_cv && d_cmd_ready) ||
                                  (head_op == 3'd4 && !h_nop && !vec_cv && v_cmd_ready) || (h_nop && !nop_cv)) |-> q_pop);
  a_rsvd_nop:   assert property (q_hv && head_op > 3'd4 |-> h_nop);         // 保留的 op 當 NOP
  a_vec_k0:     assert property (g_vec |-> head_k != '0);                    // K = 0 的 VEC 不送引擎
  a_dma_slot:   assert property (d_done |-> !dma_cv);
  a_vec_slot:   assert property (v_done |-> !vec_cv);
  a_vec_err:    assert property (vec_err |-> v_done);
  // 完成合併: 選中一個有效來源; 輸出保持; irq = cpl_valid
  a_sel:        assert property (load |-> $onehot(sel) && (sel & ~src_v) == '0);
  a_cpl_hold:   assert property (cpl_valid && !cpl_ready |=> cpl_valid && $stable(cpl_id));
  a_cpl_src:    assert property (c_cpl_valid && !c_cpl_ready && cpl_valid && !cpl_ready |=> c_cpl_valid);
  a_irq:        assert property (irq == cpl_valid);
  // 公平: 待回報的來源最多被其他來源搶先 3 次 (round-robin, 4 個來源)
  int unsigned skip [4];
  always @(posedge clk or negedge rst_n)
    if (!rst_n) skip = '{0, 0, 0, 0};
    else if (load) for (int i = 0; i < 4; i++) skip[i] = sel[i] ? 0 : src_v[i] ? skip[i] + 1 : 0;
  a_fair:       assert property (skip[0] <= 3 && skip[1] <= 3 && skip[2] <= 3 && skip[3] <= 3);
  // 確切的 round-robin: 選中的是上一個選中者之後第一個有效的來源
  logic [1:0] rr_last;
  logic [3:0] rr_exp;
  always @(posedge clk or negedge rst_n)
    if (!rst_n) rr_last <= 2'd3;
    else if (load) rr_last <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : sel[2] ? 2'd2 : 2'd3;
  always_comb begin
    rr_exp = '0;
    for (int i = 3; i >= 0; i--) if (src_v[(int'(rr_last) + 1 + i) % 4]) rr_exp = 4'(1 << ((int'(rr_last) + 1 + i) % 4));
  end
  a_rr:         assert property (load |-> sel == rr_exp);
  // 有界: 輸出暫存空著且有來源時必載入
  a_load:       assert property (src_v != '0 && !cpl_valid |-> load);
  a_busy:       assert property (!busy |-> !q_hv && oc == '0 && !cpl_valid && !c_busy && !d_busy && !v_busy);
  // DMA / 向量引擎閒置時不使用 host 埠 (DMA 也不用外部埠); DMA 時脈關閉時必在 idle (test_en 時 ICG 強制開啟)
  a_dma_quiet:  assert property (!d_busy |-> !d_rd_req && !d_wr_req && !m_rd_req && !m_wr_req);
  a_vec_quiet:  assert property (!v_busy |-> !v_rd_req && !v_wr_req);
  a_host:       assert property ((h_rd_req == (d_rd_req || v_rd_req)) && (h_wr_req == (d_wr_req || v_wr_req)));
  a_gate_safe:  assert property (!test_en && !d_en |-> !d_busy && !g_dma && !m_rd_rvalid);

  c_q_full:     cover property (cmd_valid && !cmd_ready);
  c_cpl_bp:     cover property (cpl_valid && !cpl_ready);
  c_wt_stall:   cover property (q_hv && head_wt && oc != '0);
  c_core_full:  cover property (g_core && !c_cmd_ready);
  c_dma_busy:   cover property (g_dma && !d_cmd_ready);
  c_vec_busy:   cover property (g_vec && !v_cmd_ready);
  c_dma_slot:   cover property (go && head_op inside {3'd1, 3'd2} && dma_cv);
  c_vec_slot:   cover property (go && head_op == 3'd4 && !h_nop && vec_cv);
  c_nop_slot:   cover property (go && h_nop && nop_cv);
  c_k0:         cover property (g_nop && head_op == 3'd0);
  c_vec_k0:     cover property (g_nop && head_op == 3'd4);
  c_rsvd:       cover property (g_nop && head_op > 3'd4);
  c_overlap:    cover property (d_busy && c_busy && h_wr_req);
  c_host_share: cover property (d_rd_req && v_rd_req);                     // DMA 與向量引擎同時要 host 讀埠
  c_vec_core:   cover property (v_busy && c_busy);
  c_vec_err:    cover property (vec_err);
  c_all_src:    cover property (src_v == 4'b1111 && cpl_valid && !cpl_ready);
  c_gated:      cover property (!test_en && busy && !d_en);
  c_oc_max:     cover property (int'(oc) == OC_MAX);              // 上限可達 (界限是緊的)

  covergroup cg_disp @(posedge clk iff rst_n && q_pop);
    cp_op: coverpoint head_op { bins gemm = {0}; bins ld = {1}; bins st = {2}; bins nop = {3}; bins vec = {4}; bins rsvd = {[5:7]}; }
    cp_wt: coverpoint head_wt;
    cp_oc: coverpoint oc { bins zero = {0}; bins one = {1}; bins few = {[2:4]}; bins many = {[5:$]}; }
    x_op_wt: cross cp_op, cp_wt;
  endgroup
  cg_disp u_cg_disp = new;
  covergroup cg_cpl @(posedge clk iff rst_n && load);
    // 每個來源在「只有它」與「4 個都在等」兩種情況下都被選中過; 另含每對來源的競爭
    cp_sel: coverpoint {(sel[1] ? 2'd1 : sel[2] ? 2'd2 : sel[3] ? 2'd3 : 2'd0), src_v} {
      bins alone[] = {6'b00_0001, 6'b01_0010, 6'b10_0100, 6'b11_1000};
      bins all[]   = {6'b00_1111, 6'b01_1111, 6'b10_1111, 6'b11_1111};
      bins pair[]  = {6'b00_0011, 6'b01_0011, 6'b00_1001, 6'b11_1001, 6'b01_1010, 6'b11_1010, 6'b10_1100, 6'b11_1100};
    }
  endgroup
  cg_cpl u_cg_cpl = new;
  covergroup cg_occ @(posedge clk iff rst_n);
    cp_q:  coverpoint q_cnt { bins n[] = {[0:QD]}; }
    cp_ex: coverpoint {v_busy, d_busy, c_busy} { bins n[] = {[0:7]}; }
  endgroup
  cg_occ u_cg_occ = new;
endmodule

bind orca_npu_ss npu_ss_sva #(.QD(QD), .NCU(NCU)) u_sva (
  .clk, .rst_n, .test_en, .cmd_valid, .cmd_ready, .cpl_valid, .cpl_ready, .busy, .irq, .cpl_id, .oc, .q_cnt,
  .q_hv, .go, .q_pop, .g_core, .g_dma, .g_vec, .g_nop, .d_take, .v_take, .h_nop, .c_cmd_ready, .d_cmd_ready, .v_cmd_ready,
  .head_wt(head.wt), .head_op(head.op), .head_k(head.k),
  .dma_cv, .nop_cv, .vec_cv, .c_cpl_valid, .c_cpl_ready, .c_busy, .d_busy, .v_busy, .d_done, .v_done, .d_en, .load,
  .src_v, .sel, .d_rd_req, .d_wr_req, .v_rd_req, .v_wr_req, .h_rd_req, .h_rd_gnt, .h_wr_req, .m_rd_req, .m_wr_req,
  .m_rd_rvalid, .vec_err);
