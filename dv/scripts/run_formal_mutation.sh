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
  python3 - "$4" "$5" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1)); mkdir -p $M/$n
  local extra=""; [ -f formal/$b/stubs.sv ] && extra=formal/$b/stubs.sv   # 子模組契約替身
  sv2v -DFORMAL -E Assert -Irtl/common rtl/common/orca_pkg.sv $extra $M/$n.sv > $M/$n/$b.v
  cp formal/$b/$b.sby $M/$n/
  (cd $M/$n && timeout 1800 sby -f $b.sby prove > sby.log 2>&1)
  # 只有證明真的失敗 (FAIL: 反例 / UNKNOWN: 歸納找到反例) 才算抓到; 逾時或工具錯誤不算
  if grep -qE "DONE \((FAIL|UNKNOWN)" $M/$n/sby.log; then
    nk=$((nk + 1)); echo "$n: killed ($(grep -oE 'DONE \([A-Z]+' $M/$n/sby.log | tail -1))"
  elif grep -q "DONE (PASS" $M/$n/sby.log; then echo "$n: SURVIVED"; surv=1
  else echo "$n: INCONCLUSIVE (timeout / tool error)"; surv=1; fi
}
D=rtl/npu/npu_ss_dma.sv
mut npu_ss_dma $D done_early  "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"
mut npu_ss_dma $D outst_over  "wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT));" "wire can_iss = act && (iss != n);"
mut npu_ss_dma $D no_rd_qual  "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);" "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : m_rd_gnt;"
mut npu_ss_dma $D waddr_iss   "assign m_wr_addr = ext + 32'(wr);" "assign m_wr_addr = ext + 32'(iss);"
mut npu_ss_dma $D fifo_ptr    "if (pop)  f_rd <= f_rd + 1'b1;" "if (pop && !push)  f_rd <= f_rd + 1'b1;"
mut npu_ss_dma $D pushd_wrong "wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;" "wire [DW-1:0] push_d = st ? h_rd_data : h_rd_data;"
mut npu_ss_dma $D wr_no_hold  "assign m_wr_req  = act && f_nempty && st;" "assign m_wr_req  = act && f_nempty && st && !iss[0];"
T=rtl/npu/orca_npu_ss.sv
mut orca_npu_ss $T wt_ignored  "wire go     = q_hv && (!head.wt || oc == '0);" "wire go     = q_hv;"
mut orca_npu_ss $T oc_no_dec   "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop);"
mut orca_npu_ss $T oc_early    "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop) - OCW'(load || d_done);"
mut orca_npu_ss $T dma_slot    "head.op == NPU_OP_DMA_ST) && !dma_cv;" "head.op == NPU_OP_DMA_ST);"
mut orca_npu_ss $T k0_to_core  "(head.op == NPU_OP_GEMM && head.k == '0)" "1'b0"
mut orca_npu_ss $T cpl_no_hold "wire load = (src_v != '0) && (!cpl_valid || cpl_ready);" "wire load = (src_v != '0);"
mut orca_npu_ss $T cpl_fixed   "last      <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : 2'd2;" "last      <= 2'd2;"
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
mut npu_l2_xbar $X row_bits    "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)); endfunction" "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return l[0 +: DW]; endfunction"
mut npu_l2_xbar $X rr_rot2     "m = MW'((int'(rr) + 1 + f) % NM);" "m = MW'((int'(rr) + f) % NM);"
C=rtl/npu/npu_core.sv
mut npu_core $C done_dispatch "assign cu_free    = cu_ready & ~cpl_pend & ~cu_done;" "assign cu_free    = cu_ready & ~cpl_pend;"
mut npu_core $C pend_dispatch "assign cu_free    = cu_ready & ~cpl_pend & ~cu_done;" "assign cu_free    = cu_ready & ~cu_done;"
mut npu_core $C multi_disp    "assign first_free = cu_free & (~cu_free + 1'b1);" "assign first_free = cu_free;"
mut npu_core $C q_full        "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"
mut npu_core $C cpl_no_hold   "wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);" "wire cpl_load = (cpl_pend != '0);"
mut npu_core $C tag_from_port "for (int i = 0; i < NCU; i++) if (cu_v[i]) cu_tag[i] <= head.id;" "for (int i = 0; i < NCU; i++) if (cu_v[i]) cu_tag[i] <= cmd_id;"
mut npu_core $C pend_noclear  "cpl_pend <= (cpl_pend & ~(cpl_load ? cpl_oh : '0)) | cu_done;" "cpl_pend <= cpl_pend | cu_done;"
mut npu_core $C rr_frozen     "else if (cpl_load) cpl_last <= cpl_oh;" "else if (1'b0) cpl_last <= cpl_oh;"
mut npu_core $C busy_no_cpl   "assign busy = !(&cu_ready) || (|cpl_pend) || cpl_valid || q_head_v;" "assign busy = !(&cu_ready) || (|cpl_pend) || q_head_v;"
echo "=== formal mutation: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
