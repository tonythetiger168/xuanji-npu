#!/bin/bash
# 形式驗證的 mutation 檢查: 每個 mutant 在 RTL 植入一個錯誤, 證明必須失敗 (FAIL 或 UNKNOWN = 歸納找到反例);
# 若 mutant 仍「證明通過」, 代表性質太弱或證明是空的。
# 需求同 run_formal.sh (z3 >= 4.12: 舊版 4.8 在此問題上會當掉)
set -uo pipefail
cd "$(dirname "$0")/../.."
export LANG=C.UTF-8 LC_ALL=C.UTF-8
M=build/formal_mut; rm -rf $M; mkdir -p $M
nk=0; nt=0; surv=0
mut() {   # block rtl name old new
  local b=$1 f=$2 n=$1.$3                 # 名稱加區塊前綴 (不同區塊可用同名 mutant, 工作目錄不互相覆蓋)
  [ -n "${ONLY:-}" ] && [ "$b" != "$ONLY" ] && return   # ONLY=<區塊>: 只跑該區塊的 mutant
  python3 - "$4" "$5" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1)); mkdir -p $M/$n
  local extra=""; [ -f formal/$b/stubs.sv ] && extra=formal/$b/stubs.sv   # 子模組契約替身
  [ "$b" = orca_npu_ss ] && extra="$extra rtl/npu/npu_l2_arb2.sv"           # 真的仲裁器 (不是替身)
  [[ "$b" =~ ^(npu_ss_regs|gs_raster_regs)$ ]] && extra="$extra rtl/common/ret_shadow.sv"   # retention 影子
  sv2v -DFORMAL -DFORMAL_TOP_$b -E Assert -Irtl/common rtl/common/orca_pkg.sv $extra $M/$n.sv > $M/$n/$b.v
  cp formal/$b/$b.sby $M/$n/
  (cd $M/$n && timeout 1800 sby -f $b.sby prove > sby.log 2>&1)
  # 只有證明真的失敗 (FAIL: 反例 / UNKNOWN: 歸納找到反例) 才算抓到; 逾時或工具錯誤不算
  if grep -qE "DONE \((FAIL|UNKNOWN)" $M/$n/sby.log && [[ "$b" =~ ^(orca_pmu|npu_l2_xbar|npu_ss_dma|npu_dma)$ ]]; then
    # orca_pmu / npu_l2_xbar / npu_ss_dma 的歸納需要引用內部狀態的輔助不變式: 另以 BMC (深度 30, 拿掉輔助不變式) 確認
    # mutant 違反的是埠層級的性質 (真的反例), 而不只是輔助不變式
    sv2v -DFORMAL -DFORMAL_TOP_$b -DFORMAL_NO_HELPER -E Assert -Irtl/common rtl/common/orca_pkg.sv $extra $M/$n.sv > $M/$n/$b.v
    sed 's/^prove: mode prove/prove: mode bmc/; s/^prove: depth [0-9]*/prove: depth 30/' formal/$b/$b.sby > $M/$n/$b.sby
    [[ "$b" =~ ^(npu_l2_xbar|npu_ss_dma|npu_dma)$ ]] && sed -i "s/^smtbmc z3/abc bmc3/" $M/$n/$b.sby   # 陣列 / 寬計數器: z3 BMC 30 拍太慢, 改用 ABC bmc3
    # npu_ss_dma (32 bit 位址、16 bit 計數器) 無輔助不變式時 BMC 每拍成本急增 (第 10 拍約 20 分鐘): 只做 8 拍 —
    # 每個 DMA mutant 在 cols = 1、n 小的命令下幾拍內就違反埠層級的位址 / 資料性質
    [[ "$b" =~ ^(npu_ss_dma|npu_dma)$ ]] && sed -i "s/^prove: depth 30/prove: depth 8/" $M/$n/$b.sby
    (cd $M/$n && timeout 1800 sby -f $b.sby prove > bmc.log 2>&1)
    # ABC 回報 FAIL (反例) 時 sby 可能無法產生波形而報 ERROR, 以 engine 的結果為準
    if grep -qE "DONE \(FAIL|returned FAIL" $M/$n/bmc.log; then nk=$((nk + 1)); echo "$n: killed (BMC counterexample, port-level property)"
    else echo "$n: SURVIVED (only the helper invariants fail)"; surv=1; fi
  elif grep -qE "DONE \((FAIL|UNKNOWN)" $M/$n/sby.log; then
    nk=$((nk + 1)); echo "$n: killed ($(grep -oE 'DONE \([A-Z]+' $M/$n/sby.log | tail -1))"
  elif grep -q "DONE (PASS" $M/$n/sby.log; then echo "$n: SURVIVED"; surv=1
  else
    # 歸納逾時: 以 ABC bmc3 (深度 30) 找反例 (未植入錯誤的設計在同一設定下必須通過, run_formal.sh 的 prove 已保證)
    sed 's/^prove: mode prove/prove: mode bmc/; s/^prove: depth [0-9]*/prove: depth 30/; s/^smtbmc z3/abc bmc3/' formal/$b/$b.sby > $M/$n/bmc.sby
    (cd $M/$n && timeout 1800 sby -f bmc.sby prove > bmc3.log 2>&1)
    if grep -q "was asserted in frame" $M/$n/bmc3.log; then nk=$((nk + 1)); echo "$n: killed (ABC bmc3 counterexample, $(grep -oE 'frame [0-9]+' $M/$n/bmc3.log | head -1))"
    else echo "$n: INCONCLUSIVE (timeout / tool error)"; surv=1; fi
  fi
}
D=rtl/npu/npu_ss_dma.sv
mut npu_ss_dma $D done_early  "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"
mut npu_ss_dma $D outst_over  "wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT));" "wire can_iss = act && (iss != n);"
mut npu_ss_dma $D no_rd_qual  "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);" "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : m_rd_gnt;"
mut npu_ss_dma $D waddr_iss   "wire e_step = st ? pop : iss_ok;" "wire e_step = iss_ok;"
mut npu_ss_dma $D wrap_off    "wire e_wrap = (cols != '0) && (col == cols - 1'b1);" "wire e_wrap = (cols != '0) && (col == cols);"
mut npu_ss_dma $D stride_cols "if (e_wrap) begin col <= '0; erow <= erow + 32'(stride); end" "if (e_wrap) begin col <= '0; erow <= erow + 32'(cols); end"
mut npu_ss_dma $D fifo_ptr    "if (pop)  f_rd <= f_rd + 1'b1;" "if (pop && !push)  f_rd <= f_rd + 1'b1;"
mut npu_ss_dma $D pushd_wrong "wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;" "wire [DW-1:0] push_d = st ? h_rd_data : h_rd_data;"
# npu_dma (v6.3.13): npu_ss_dma 的代表性 mutant + 載入時解碼
E=rtl/npu/npu_dma.sv
mut npu_dma $E done_early   "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"
mut npu_dma $E outst_over   "wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT));" "wire can_iss = act && (iss != n);"
mut npu_dma $E stride_cols  "if (e_wrap) begin col <= '0; erow <= erow + 32'(stride); end" "if (e_wrap) begin col <= '0; erow <= erow + 32'(cols); end"
mut npu_dma $E mx_tab       "8'd4, 8'd6, 8'd8, 8'd12}" "8'd4, 8'd5, 8'd8, 8'd12}"
mut npu_dma $E mx_top       "8'd4, 8'd6, 8'd8, 8'd12}" "8'd4, 8'd6, 8'd8, 8'd16}"
mut npu_dma $E i4_nosext    "return (f == 2'd1) ? {{4{x[3]}}, x} :" "return (f == 2'd1) ? {4'd0, x} :"
mut npu_dma $E u4_sext      "(f == 2'd2) ? {4'd0, x} :" "(f == 2'd2) ? {{4{x[3]}}, x} :"
mut npu_dma $E mx_nosign    "(x[3] ? 8'(-m) : m);" "(m);"
mut npu_dma $E nib_order    "wire [DW/2-1:0] f_sel = half ? f_head[DW-1:DW/2] : f_head[DW/2-1:0];" "wire [DW/2-1:0] f_sel = half ? f_head[DW/2-1:0] : f_head[DW-1:DW/2];"
mut npu_dma $E pop_first    "wire pop    = w_fire && (!dec || half);" "wire pop    = w_fire;"
mut npu_dma $E half_addr    "assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(wr));" "assign h_wr_line = dec ? (l2 + 26'(wr) + 26'(half)) : (l2 + 26'(wr));"
mut npu_dma $E dec_st       "dec <= !cmd_st && (cmd_fmt != 2'd0);" "dec <= (cmd_fmt != 2'd0);"
mut npu_dma $E half_noflip  "if (dec && w_fire) half <= ~half;" "if (dec && pop) half <= ~half;"
mut npu_ss_dma $D wr_no_hold  "assign m_wr_req  = act && f_nempty && st;" "assign m_wr_req  = act && f_nempty && st && !iss[0];"
T=rtl/npu/orca_npu_ss.sv
mut orca_npu_ss $T wt_ignored  "wire go     = q_hv && (!head.wt || oc == '0);" "wire go     = q_hv;"
mut orca_npu_ss $T oc_no_dec   "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop);"
mut orca_npu_ss $T oc_early    "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop) - OCW'(load || d_done);"
mut orca_npu_ss $T dma_slot    "head.op == NPU_OP_DMA_ST) && !dma_cv;" "head.op == NPU_OP_DMA_ST);"
mut orca_npu_ss $T k0_to_core  "((head.op == NPU_OP_GEMM || head.op == NPU_OP_VEC) && head.k == '0)" "1'b0"
mut orca_npu_ss $T cpl_no_hold "wire load = (src_v != '0) && (!cpl_valid || cpl_ready);" "wire load = (src_v != '0);"
mut orca_npu_ss $T cpl_fixed   "last      <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : sel[2] ? 2'd2 : 2'd3;" "last      <= 2'd3;"
mut orca_npu_ss $T dma_id      "if (d_take) dma_id <= head.id;" "if (g_dma) dma_id <= head.id;"
mut orca_npu_ss $T q_full      "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"
mut orca_npu_ss $T dma_gate    "assign d_en = d_busy || g_dma;" "assign d_en = d_busy;"
mut orca_npu_ss $T nop_id      "if (g_nop) begin nop_cv <= 1'b1; nop_id <= head.id; end" "if (g_nop) begin nop_cv <= 1'b1; end"
X=rtl/npu/npu_l2_xbar.sv
mut npu_l2_xbar $X rr_frozen   "if (rd_bv[b]) rd_rr[b] <= rd_bm[b];" "if (1'b0) rd_rr[b] <= rd_bm[b];"
mut npu_l2_xbar $X fixed_prio  "rot = {rq, rq} >> ((int'(rr) + 1) % NM);" "rot = {rq, rq} >> 0;"
mut npu_l2_xbar $X no_bank_chk "for (int i = 0; i < NM; i++) rq[i] = req[i] && bank_of(line[i]) == BW'(b);" "for (int i = 0; i < NM; i++) rq[i] = req[i];"
# (拿掉 wr_gnt 的 wr_req 條件是等價 mutant: wr_bm 只會是有請求者; 改測 grant 不比對得主)
mut npu_l2_xbar $X gnt_all     "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb] && rd_bm[rb] == MW'(m);" "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb];"
mut npu_l2_xbar $X wdata_m0    "if (wr_bv[b]) mem[b][row_of(wr_line[wr_bm[b]])] <= wr_data[wr_bm[b]];" "if (wr_bv[b]) mem[b][row_of(wr_line[wr_bm[b]])] <= wr_data[0];"
mut npu_l2_xbar $X rdata_bank0 "assign rd_data[m] = mem[rb][row_of(rd_line[m])];" "assign rd_data[m] = mem[0][row_of(rd_line[m])];"
mut npu_l2_xbar $X row_bits    "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)); endfunction" "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)) & ~DW'(1); endfunction"   # 列位元 0 卡在 0 (兩個 line 別名; 原本的 l[0 +: DW] 在 (bank, 低位元) 下仍一對一, 是等價 mutant)
mut npu_l2_xbar $X ls_nowait   "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= '0; end"
mut npu_l2_xbar $X ls_nowake   "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (1'b0) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end"
mut npu_l2_xbar $X ls_rdidle   "else if (rd_rq[b] || wr_rq[b]) idle[b] <= '0;" "else if (wr_rq[b]) idle[b] <= '0;"
mut npu_l2_xbar $X ls_rdy      "assign b_rdy[b]   = !ls[b] && wk[b] == '0;" "assign b_rdy[b]   = wk[b] == '0;"
mut npu_l2_xbar $X ls_early    "else if (idle[b] == IW'(LS_IDLE - 1)) begin" "else if (idle[b] == IW'(LS_IDLE - 2)) begin"
mut npu_l2_xbar $X rr_rot2     "m = MW'((int'(rr) + 1 + f) % NM);" "m = MW'((int'(rr) + f) % NM);"
C=rtl/npu/npu_core.sv
mut npu_core $C idle_first    "assign cu_cand    = (cu_idle != '0) ? cu_idle : cu_free;" "assign cu_cand    = cu_free;"
mut npu_core $C not_ready     "assign cu_free    = cu_ready;" "assign cu_free    = cu_ready | cu_done;"
mut npu_core $C multi_disp    "assign first_free = cu_cand & (~cu_cand + 1'b1);" "assign first_free = cu_cand;"
mut npu_core $C q_full        "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"
mut npu_core $C cpl_no_hold   "wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);" "wire cpl_load = (cpl_pend != '0);"
mut npu_core $C dack_always   "assign cu_dack    = ~cpl_pend;" "assign cu_dack    = '1;"
mut npu_core $C tag_from_head "else if (cu_done[i] && !cpl_pend[i]) cpl_tag[i] <= cu_dtag[i];" "else if (cu_done[i] && !cpl_pend[i]) cpl_tag[i] <= head.id;"
mut npu_core $C pend_noclear  "cpl_pend <= (cpl_pend & ~(cpl_load ? cpl_oh : '0)) | (cu_done & ~cpl_pend);" "cpl_pend <= cpl_pend | (cu_done & ~cpl_pend);"
mut npu_core $C rr_frozen     "else if (cpl_load) cpl_last <= cpl_oh;" "else if (1'b0) cpl_last <= cpl_oh;"
mut npu_core $C busy_no_cpl   "assign busy = (|cu_busy) || (|cpl_pend) || cpl_valid || q_head_v;" "assign busy = (|cu_busy) || (|cpl_pend) || q_head_v;"
mut npu_core $C busy_no_cu    "assign busy = (|cu_busy) || (|cpl_pend) || cpl_valid || q_head_v;" "assign busy = (|cpl_pend) || cpl_valid || q_head_v;"
V=rtl/npu/npu_vec.sv
mut npu_vec $V rdy_no_gnt   "assign ins_ready = is_ld ? rd_gnt" "assign ins_ready = is_ld ? 1'b1"
mut npu_vec $V st_writes    " && ins.op != VOP_ST;" ";"
mut npu_vec $V err_nostick  "    else if (err_clr)          err <= 1'b0;" "    else                       err <= 1'b0;"   # err 不黏著 (下一拍就清掉)
mut npu_vec $V ill_st       "(ins.op == VOP_ST && ins.fn > 3'd1)" "(ins.op == VOP_ST && ins.fn > 3'd2)"
mut npu_vec $V rd_line      "assign rd_line   = ins.line;" "assign rd_line   = ins.line ^ 26'd1;"
mut npu_vec $V reg_all      "if (we && ins.vd == 3'(r)) q <= wd;" "if (we) q <= wd;"
mut npu_vec $V st_illegal   "wire is_st = ins_valid && !illegal && ins.op == VOP_ST;" "wire is_st = ins_valid && ins.op == VOP_ST;"
R=rtl/npu/npu_ss_regs.sv
mut npu_ss_regs rtl/npu/npu_ss_regs.sv no_restore "      if (ret_restore) r_ien <= ien_ret;" ""
mut npu_ss_regs rtl/npu/npu_ss_regs.sv rest_lost_wr "      if (ret_restore) r_ien <= ien_ret;" "      if (ret_restore && !wr) r_ien <= ien_ret;"
mut npu_ss_regs rtl/npu/npu_ss_regs.sv save_always "ret_shadow #(.W(4)) u_ret_ien (.clk_aon, .por_n, .save(ret_save)," "ret_shadow #(.W(4)) u_ret_ien (.clk_aon, .por_n, .save(1'b1),"
mut npu_ss_regs $R push_full  "wire           push   = push_req && !q_full;" "wire           push   = push_req;"
mut npu_ss_regs $R head_wrong "assign cmd       = q[q_rd[CW-1:0]];" "assign cmd       = q[q_wr[CW-1:0]];"
mut npu_ss_regs $R ovf_no_w1c "ovf  <= (ovf  && !(w1c && pstrb[0] && pwdata[1]))" "ovf  <= (ovf)"
mut npu_ss_regs $R irq_nomask "assign irq = |(ist & r_ien);" "assign irq = |ist;"
mut npu_ss_regs $R slverr_none "assign pslverr = acc && !hit;" "assign pslverr = 1'b0;"
mut npu_ss_regs $R cpl_on_wr  "assign cpl_ready = rd && wa == A_CPL_POP;" "assign cpl_ready = acc && wa == A_CPL_POP;"
mut npu_ss_regs $R strobe     "return (old & ~m) | (nw & m);" "return nw;"
A2=rtl/npu/npu_l2_arb2.sv
mut npu_l2_arb2 $A2 nolock    "wire  r_sel  = r_lock ? r_own : r_pick;" "wire  r_sel  = r_pick;"
mut npu_l2_arb2 $A2 gnt_both  "assign rd_gnt    = {m_rd_gnt && r_sel, m_rd_gnt && !r_sel} & rd_req;" "assign rd_gnt    = {m_rd_gnt, m_rd_gnt} & rd_req;"
mut npu_l2_arb2 $A2 wdata     "assign m_wr_data = wr_data[w_sel];" "assign m_wr_data = wr_data[0];"
mut npu_l2_arb2 $A2 fixed     "wire  w_pick = (wr_req == 2'b11) ? !w_last : wr_req[1];" "wire  w_pick = wr_req[1];"
E=rtl/npu/npu_vec_eng.sv
mut npu_vec_eng $E hold       "wire  f_sel  = f_pend || (f_want && !v_rd_req);" "wire  f_sel  = f_want && !v_rd_req;"
mut npu_vec_eng $E slot_off   "  assign ins = npu_vec_ins_t'(fb[fr][128 * slot +: 77]);" "  assign ins = npu_vec_ins_t'(fb[!fr][128 * slot +: 77]);"
mut npu_vec_eng $E pc_skip    "fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd1;" "fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd2;"
mut npu_vec_eng $E done_early "  wire  last = fire && ni == 16'd1;" "  wire  last = fire && ni == 16'd2;"
mut npu_vec_eng $E rdy_done   "  assign cmd_ready = !act && !done;" "  assign cmd_ready = !act;"
mut orca_npu_ss $T vec_no_wait "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop && !vec_cv;" "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop;"
mut orca_npu_ss $T vec_rr      "automatic int s = (int'(last) + 1 + i) % 4;" "automatic int s = (int'(last) + 1 + i) % 3;"
mut orca_npu_ss $T vec_oc      "if (v_take) vec_id <= head.id;" "if (g_vec) vec_id <= head.id;"
X=rtl/soc/orca_apb_xbar.sv
mut orca_apb_xbar $X psel_ungated "assign s_psel    = psel ? hit_v : '0;" "assign s_psel    = hit_v;"
mut orca_apb_xbar $X unmap_ok     "pslverr = psel && penable && !hit;" "pslverr = 1'b0;"
mut orca_apb_xbar $X resp_err     "        pslverr = s_pslverr[i];" "        pslverr = 1'b0;"
S=rtl/soc/orca_sys_regs.sv
mut orca_sys_regs $S be_no_w1c   "berr <= (berr && !clr) || set;" "berr <= berr || set;"
mut orca_sys_regs $S ea_always   "if (set && !berr) begin" "if (set) begin"
mut orca_sys_regs $S irq_nomask  "assign irq = (own_irq && r_mhi) || |(irq_in & r_mlo);" "assign irq = berr || |irq_in;"
mut orca_sys_regs $S scr_strb    "if (wr && wa == A_SCRATCH) r_scr <= (r_scr & ~bm) | (pwdata & bm);" "if (wr && wa == A_SCRATCH) r_scr <= pwdata;"
Q=rtl/soc/orca_pmu.sv
mut orca_pmu $Q pdn_no_wait  "P_PDN:  if (!pd_pwr_ack[g]) st_n = P_OFF;" "P_PDN:  st_n = P_OFF;"
mut orca_pmu $Q pup_no_wait  "P_PUP:  if (pd_pwr_ack[g]) st_n = P_RST;" "P_PUP:  st_n = P_RST;"
mut orca_pmu $Q pdn_abort    "P_PDN:  if (!pd_pwr_ack[g]) st_n = P_OFF;" "P_PDN:  if (!pd_pwr_ack[g]) st_n = P_OFF; else if (wake_req) st_n = P_ON;"
mut orca_pmu $Q iso_late     "iso_q   <= !(st_n == P_ON || st_n == P_IDLE);" "iso_q   <= !(st_n == P_ON || st_n == P_IDLE || st_n == P_RET);"
mut orca_pmu $Q rst_early    "rstn_q  <= !(st_n == P_PDN || st_n == P_OFF || st_n == P_PUP);" "rstn_q  <= !(st_n == P_PDN || st_n == P_OFF);"
mut orca_pmu $Q no_save      "assign pd_ret_save[g]    = (st == P_IDLE) && (st_n == P_RET);" "assign pd_ret_save[g]    = 1'b0;"
mut orca_pmu $Q no_restore   "assign pd_ret_restore[g] = (st == P_RST);" "assign pd_ret_restore[g] = 1'b0;"
mut orca_pmu $Q rst_no_clk   "assign pd_clk_gate[g]    = !(st == P_ON || st == P_RST);" "assign pd_clk_gate[g]    = !(st == P_ON);"
mut orca_pmu $Q clk_in_pup   "assign pd_clk_gate[g]    = !(st == P_ON || st == P_RST);" "assign pd_clk_gate[g]    = !(st == P_ON || st == P_RST || st == P_PUP);"
mut orca_pmu $Q dv_settle   "wire dv_end = dv_cnt == (dv_chf ? TW'(F_LOCK - 1) : TW'(V_SETTLE - 1));" "wire dv_end = dv_cnt == (dv_chf ? TW'(F_LOCK - 1) : TW'(F_LOCK - 1));"   # 電壓未穩定就換頻率
mut orca_pmu $Q dv_order    "wire dv_up  = dv_tgt > dv_v;" "wire dv_up  = 1'b0;"   # 升檔先換頻率 (頻率超過電壓)
mut orca_pmu $Q dv_retgt    "else if (dv_new) begin dv_tgt <= dvfs_req; dv_cnt <= '0; end" "else if (dv_new) begin dv_tgt <= dvfs_req; dv_cnt <= dv_cnt; end"   # 改目標不重新計數
mut orca_pmu $Q dv_rej      "wire dv_new = dvfs_req <= 3'd4 && dvfs_req != dv_tgt;" "wire dv_new = dvfs_req != dv_tgt;"   # 不合法效能點仍接受
mut orca_pmu $Q dv_busy     "assign dvfs_busy     = dv_up || dv_chf || dv_dn;" "assign dvfs_busy     = dv_up || dv_dn;"   # 換頻率時不報 busy
P=rtl/soc/orca_pmu_regs.sv
mut orca_pmu_regs $P rej_writes  "if (wa == A_DVFS_REQ && pstrb[0] && !rej) r_dvfs <= pwdata[2:0];" "if (wa == A_DVFS_REQ && pstrb[0]) r_dvfs <= pwdata[2:0];"
mut orca_pmu_regs $P wake_level  "ev_wake[g] = (pd_state[g] == ST_ON) && !was_on[g];" "ev_wake[g] = (pd_state[g] == ST_ON);"
mut orca_pmu_regs $P off_no_w1c  "of   <= (of & ~NPD'(wclr[15:8])) | ev_off;" "of   <= of | ev_off;"
mut orca_pmu_regs $P wake_strb   "wire [7:0]  b0 = pstrb[0] ? pwdata[7:0] : 8'd0;" "wire [7:0]  b0 = pwdata[7:0];"
G=rtl/sis/gs_raster_regs.sv
mut gs_raster_regs rtl/sis/gs_raster_regs.sv no_restore "      if (ret_restore) {r_ien, r_bg} <= ret_q;" "      if (ret_restore) r_ien <= ret_q[27:24];"
mut gs_raster_regs rtl/sis/gs_raster_regs.sv save_always "ret_shadow #(.W(28)) u_ret (.clk_aon, .por_n, .save(ret_save)," "ret_shadow #(.W(28)) u_ret (.clk_aon, .por_n, .save(ret_save || ret_restore),"
mut gs_raster_regs $G start_busy  "assign start       = go && !r_busy;" "assign start       = go;"
mut gs_raster_regs $G push_full   "wire         push     = push_req && !q_full;" "wire         push     = push_req;"
mut gs_raster_regs $G head_wrong  "assign hd      = q[q_rd[CW-1:0]];" "assign hd      = q[q_wr[CW-1:0]];"
mut gs_raster_regs $G last_strb   "last: pstrb[0] && pwdata[0]" "last: pwdata[0]"
mut gs_raster_regs $G pop_novalid "assign out_ready = wr && wa == A_OUT_POP && out_valid;" "assign out_ready = wr && wa == A_OUT_POP;"
echo "=== formal mutation: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
