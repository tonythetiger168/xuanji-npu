#!/bin/bash
# NPU Subsystem 檢查器的 mutation 測試 (每個產品組態): 每個 mutant 在 orca_npu_ss / npu_ss_dma 植入一個錯誤,
# npu_ss_tb (參考模型比對 + SVA) 必須抓到 (killed)。npu_core 子樹的 mutant 見 run_npu_mutation.sh。
# 用法: dv/scripts/run_npu_ss_mutation.sh <nano|x1|pro|all> [batches=12]
set -uo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NBT=${2:-20}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NBT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG" || exit 1
M=build/npu_ss_mut_$CFG; rm -rf "$M"; mkdir -p "$M"
echo "##### NPU subsystem mutation $CFG: N=$N NCU=$NCU NB=$NB"
TILE=rtl/npu/orca_npu_ss.sv DMA=rtl/npu/npu_ss_dma.sv ARB=rtl/npu/npu_l2_arb2.sv
FP="rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv
    rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv dv/vec/npu_vec_eng_sva.sv"
surv=0; nk=0; nt=0
mut() {   # name file old new
  local n=$1 f=$2
  python3 - "$3" "$4" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  local tile=$TILE dma=$DMA arb=$ARB
  case $f in $TILE) tile=$M/$n.sv ;; $DMA) dma=$M/$n.sv ;; $ARB) arb=$M/$n.sv ;; esac
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style \
    -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT --Mdir "$M/$n.obj" -Irtl/common -Irtl/fp \
    rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv $FP $arb dv/npu/npu_l2_arb2_sva.sv \
    rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
    rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv $dma $tile dv/npu/npu_ss_sva.sv dv/npu/npu_ss_tb.sv \
    --top-module npu_ss_tb > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "$M/$n.obj"; surv=1; return; }
  timeout 1800 "$M/$n.obj/Vnpu_ss_tb" +seed=7 +batches=$NBT +no_big > "$M/$n.log" 2>&1
  # 全 0 上電通過時, 再以隨機上電狀態 (未 reset 的儲存為隨機值) 執行
  grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log" \
    && timeout 1800 "$M/$n.obj/Vnpu_ss_tb" +seed=9 +batches=$NBT +no_big +verilator+rand+reset+2 +verilator+seed+2 > "$M/$n.log" 2>&1
  rm -rf "$M/$n.obj"
  if grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log"; then
    echo "$n: SURVIVED"; surv=1
  else
    nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH [A-Za-z0-9]*|u_[a-z_]*sva\.[A-Za-z_.0-9\[\]]*|ERROR: [a-z_ ]*' "$M/$n.log"))"
  fi
}
# ---- orca_npu_ss ----
mut dma_icg_rst  $TILE ".en(d_en || !rst_n)" ".en(d_en)"                                                    # reset 期間 DMA 時脈關閉 (隨機上電時不清除)
mut wt_ignored   $TILE "wire go     = q_hv && (!head.wt || oc == '0);" "wire go     = q_hv;"                        # 等待旗標無效
mut oc_no_dec    $TILE "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop);"                                 # 完成不扣帳 (wt 永遠等待)
mut oc_early     $TILE "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop) - OCW'(load || d_done);"          # DMA 完成時提早扣帳
mut dma_slot     $TILE "head.op == NPU_OP_DMA_ST) && !dma_cv;" "head.op == NPU_OP_DMA_ST);"                             # DMA 完成槽未空就分派
mut k0_to_core   $TILE "((head.op == NPU_OP_GEMM || head.op == NPU_OP_VEC) && head.k == '0)" "(head.op == NPU_OP_VEC && head.k == '0)"                                               # K = 0 送進 npu_core
mut cpl_no_hold  $TILE "wire load = (src_v != '0) && (!cpl_valid || cpl_ready);" "wire load = (src_v != '0);"         # 完成輸出不保持
mut cpl_fixed    $TILE "last      <= sel[0] ? 2'd0 : sel[1] ? 2'd1 : sel[2] ? 2'd2 : 2'd3;" "last      <= 2'd3;"                       # 完成合併固定優先權
mut dma_id       $TILE "if (d_take) dma_id <= head.id;" "if (g_dma) dma_id <= head.id;"                                # DMA 忙碌時 id 被覆蓋
mut q_full       $TILE "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"                         # 佇列滿仍收命令
mut dma_gate     $TILE "assign d_en = d_busy || g_dma;" "assign d_en = d_busy;"                                        # 閘控未涵蓋分派拍
mut st_dir       $TILE ".cmd_st(head.op == NPU_OP_DMA_ST)" ".cmd_st(head.op[0])"                                       # 方向解碼錯
mut nop_id       $TILE "if (g_nop) begin nop_cv <= 1'b1; nop_id <= head.id; end" "if (g_nop) begin nop_cv <= 1'b1; end"  # NOP 完成 id 錯
# ---- npu_ss_dma ----
mut dma_outst    $DMA "(outst != (OW+1)'(OUT))" "(outst != (OW+1)'(OUT - 1) && outst != (OW+1)'(OUT))"  # 在途上限少一 (功能仍正確, 由 a_issue 抓到吞吐量損失)
mut dma_overrun  $DMA "wire [OW:0] outst = iss[OW:0] - wr[OW:0];" "wire [OW:0] outst = {1'b0, iss[OW-1:0] - wr[OW-1:0]};"  # 在途計算溢位 → FIFO 覆蓋
mut dma_rd_qual  $DMA "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);" "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : m_rd_gnt;"  # 外部 gnt 不限定 req
mut dma_wr_qual  $DMA "wire pop    = st ? (m_wr_req && m_wr_gnt) : (h_wr_req && h_wr_gnt);" "wire pop    = st ? m_wr_gnt : (h_wr_req && h_wr_gnt);"
mut dma_waddr    $DMA "assign m_wr_addr = ext + 32'(wr);" "assign m_wr_addr = ext + 32'(iss);"                       # 寫位址用錯計數器
mut dma_l2line   $DMA "assign h_wr_line = l2 + 26'(wr);" "assign h_wr_line = l2 + 26'(iss);"
mut dma_done     $DMA "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"                # 讀完就完成 (寫尚未完成)
mut dma_fifo     $DMA "if (pop)  f_rd <= f_rd + 1'b1;" "if (pop && !push)  f_rd <= f_rd + 1'b1;"                  # FIFO 讀指標錯
mut dma_ext31    $DMA "assign m_rd_addr = ext + 32'(iss);" "assign m_rd_addr = {1'b0, ext[30:0]} + 32'(iss);"       # 外部位址高位遺失
mut dma_pushd    $DMA "wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;" "wire [DW-1:0] push_d = st ? h_rd_data : h_rd_data;"
mut dma_n0       $DMA "assign done      = act && (wr == n);" "assign done      = act && (wr == n) && n != '0;"       # n = 0 永不完成
# ---- VEC 分派 / 完成槽 / host 埠共用 (v6.3.10) ----
mut vec_no_wait  $TILE "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop && !vec_cv;" "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop;"   # 完成槽被佔用仍分派 (覆蓋 vec_id)
mut vec_k0       $TILE "((head.op == NPU_OP_GEMM || head.op == NPU_OP_VEC) && head.k == '0);" "(head.op == NPU_OP_GEMM && head.k == '0);"   # K = 0 的 VEC 送引擎
mut rsvd_op      $TILE "wire h_nop  = head.op == NPU_OP_NOP || head.op > NPU_OP_VEC ||" "wire h_nop  = head.op == NPU_OP_NOP ||"   # 保留 op 卡住
mut vec_id       $TILE "      if (v_take) vec_id <= head.id;" "      if (g_vec) vec_id <= head.id;"                                    # 引擎忙碌時也覆寫 id
mut vec_err_gate $TILE "assign vec_err = v_done && v_derr;" "assign vec_err = v_derr;"                                                  # vec_err 不只在完成拍
mut vec_sel      $TILE "cpl_id    <= sel[0] ? c_cpl_id : sel[1] ? dma_id : sel[2] ? nop_id : vec_id;" "cpl_id    <= sel[0] ? c_cpl_id : sel[1] ? dma_id : nop_id;"   # VEC 完成回報錯的 id
mut rr4          $TILE "automatic int s = (int'(last) + 1 + i) % 4;" "automatic int s = (int'(last) + 1 + i) % 3;"                          # round-robin 漏掉 VEC
# (busy 拿掉 v_busy 是等價 mutant: 形式證明的不變式 oc = core 內 + DMA + NOP + (v_busy || vec_cv), v_busy ⇒ oc != 0 ⇒ busy)
mut arb_nolock   $ARB "wire  r_sel  = r_lock ? r_own : r_pick;" "wire  r_sel  = r_pick;"                                               # 等待中換主端 (違反保持)
mut arb_gnt_both $ARB "assign rd_gnt    = {m_rd_gnt && r_sel, m_rd_gnt && !r_sel} & rd_req;" "assign rd_gnt    = {m_rd_gnt, m_rd_gnt} & rd_req;"   # grant 給兩個
mut arb_wdata    $ARB "assign m_wr_data = wr_data[w_sel];" "assign m_wr_data = wr_data[0];"                                          # 寫資料取錯主端
mut arb_fixed    $ARB "wire  w_pick = (wr_req == 2'b11) ? !w_last : wr_req[1];" "wire  w_pick = wr_req[1];"                            # 固定優先權
echo "=== $CFG: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
