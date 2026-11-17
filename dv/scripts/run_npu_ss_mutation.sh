#!/bin/bash
# NPU Subsystem 檢查器的 mutation 測試 (每個產品組態): 每個 mutant 在 orca_npu_ss / npu_dma 植入一個錯誤,
# npu_ss_tb (參考模型比對 + SVA) 必須抓到 (killed)。npu_core 子樹的 mutant 見 run_npu_mutation.sh。
# 用法: dv/scripts/run_npu_ss_mutation.sh <nano|x1|pro|all> [batches=20]   (ONLY=<regex>: 只跑名稱相符的 mutant)
#   SHARD=i/n: 只跑第 i 片 (0 <= i < n; 依出現順序第 k 個 mutant 屬於第 (k mod n) 片)。CI 以組態 × 分片的 matrix 執行,
#   每片的 mutant 都必須被抓到 (分片聯集 = 全部 mutant)。
set -uo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NBT=${2:-20}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NBT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG" || exit 1
SH_I=0; SH_N=1
if [ -n "${SHARD:-}" ]; then
  SH_I=${SHARD%/*}; SH_N=${SHARD#*/}
  [[ $SH_I =~ ^[0-9]+$ && $SH_N =~ ^[0-9]+$ ]] && [ "$SH_N" -ge 1 ] && [ "$SH_I" -lt "$SH_N" ] || { echo "bad SHARD=$SHARD"; exit 1; }
fi
M=build/npu_ss_mut_$CFG${SHARD:+_${SH_I}of${SH_N}}; rm -rf "$M"; mkdir -p "$M"
echo "##### NPU subsystem mutation $CFG: N=$N NCU=$NCU NB=$NB${SHARD:+ (shard $SH_I / $SH_N)}"
TILE=rtl/npu/orca_npu_ss.sv DMA=rtl/npu/npu_dma.sv ARB=rtl/npu/npu_l2_arb2.sv MC=rtl/npu/npu_dma_mc.sv RR=rtl/npu/npu_rr_lock.sv
FP="rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv
    rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv dv/vec/npu_vec_eng_sva.sv"
surv=0; nk=0; nt=0; mi=0
mut() {   # name file old new
  local n=$1 f=$2
  [ -n "${ONLY:-}" ] && ! [[ $n =~ ^(${ONLY})$ ]] && return   # ONLY=<regex>: 只跑名稱相符的 mutant
  mi=$((mi + 1)); [ $(( (mi - 1) % SH_N )) -eq "$SH_I" ] || return   # SHARD: 不屬於本片
  python3 - "$3" "$4" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  local tile=$TILE dma=$DMA arb=$ARB mc=$MC rr=$RR
  case $f in $TILE) tile=$M/$n.sv ;; $DMA) dma=$M/$n.sv ;; $ARB) arb=$M/$n.sv ;; $MC) mc=$M/$n.sv ;; $RR) rr=$M/$n.sv ;; esac
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style \
    -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT --Mdir "$M/$n.obj" -Irtl/common -Irtl/fp \
    rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv $FP $arb dv/npu/npu_l2_arb2_sva.sv dv/npu/npu_rr_lock_sva.sv \
    rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
    rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv $dma $rr $mc $tile rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv dv/npu/npu_ss_sva.sv dv/npu/npu_ss_e2e_sva.sv dv/npu/npu_ss_tb.sv \
    --top-module npu_ss_tb > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "$M/$n.obj"; surv=1; return; }
  timeout 1800 "$M/$n.obj/Vnpu_ss_tb" +seed=7 +batches=$NBT +no_big > "$M/$n.log" 2>&1
  # 全 0 上電通過時, 再以隨機上電狀態 (未 reset 的儲存為隨機值) 執行
  grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log" \
    && timeout 1800 "$M/$n.obj/Vnpu_ss_tb" +seed=9 +batches=$NBT +no_big +verilator+rand+reset+2 +verilator+seed+2 > "$M/$n.log" 2>&1
  # 再一個隨機上電樣本 (短): 不同的上電狀態讓「reset 期間閘控時脈」類 mutant 不依賴單一 seed 的運氣
  grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log" \
    && timeout 600 "$M/$n.obj/Vnpu_ss_tb" +seed=11 +batches=2 +no_big +verilator+rand+reset+2 +verilator+seed+3 > "$M/$n.log" 2>&1
  rm -rf "$M/$n.obj"
  if grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log"; then
    echo "$n: SURVIVED"; surv=1
  else
    nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH [A-Za-z0-9]*|u_[a-z_]*sva\.[A-Za-z_.0-9\[\]]*|ERROR: [a-z_ ]*' "$M/$n.log"))"
  fi
}
# ---- orca_npu_ss ----
mut dma_icg_rst  $MC ".en(c_busy[i] || c_take[i] || !rst_n)" ".en(c_busy[i] || c_take[i])"          # reset 期間 DMA 通道時脈關閉 (隨機上電時不清除)
mut wt_ignored   $TILE "wire go     = q_hv && (!head.wt || oc == '0);" "wire go     = q_hv;"                        # 等待旗標無效
mut oc_no_dec    $TILE "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop);"                                 # 完成不扣帳 (wt 永遠等待)
mut oc_early     $TILE "oc <= oc + OCW'(q_pop) - OCW'(load);" "oc <= oc + OCW'(q_pop) - OCW'(load || d_cpl_valid);"    # DMA 完成時提早扣帳
mut dma_slot     $MC "wire [NDC-1:0] c_free = c_ready & ~cv;" "wire [NDC-1:0] c_free = c_ready;"     # 通道完成槽未空就分派
mut k0_to_core   $TILE "((head.op == NPU_OP_GEMM || head.op == NPU_OP_VEC) && head.k == '0)" "(head.op == NPU_OP_VEC && head.k == '0)"                                               # K = 0 送進 npu_core
mut cpl_no_hold  $TILE "wire load = (src_v != '0) && (!cpl_valid || cpl_ready);" "wire load = (src_v != '0);"         # 完成輸出不保持
mut cpl_fixed    $TILE "last      <= sel[0] ? 3'd0 : sel[1] ? 3'd1 : sel[2] ? 3'd2 : sel[3] ? 3'd3 : 3'd4;" "last      <= 3'd4;"                       # 完成合併固定優先權
mut dma_id       $MC "if (c_take[i]) c_id[i] <= cmd_id;" "if (cmd_valid) c_id[i] <= cmd_id;"          # DMA 通道忙碌時 id 被覆蓋
mut q_full       $TILE "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"                         # 佇列滿仍收命令
mut dma_gate     $MC ".en(c_busy[i] || c_take[i] || !rst_n)" ".en(c_busy[i] || !rst_n)"              # 閘控未涵蓋分派拍
# ---- npu_dma_mc 多通道 (v6.3.13) ----
mut mc_take_two  $MC "if (c_free[i]) c_take = NDC'(1) << i;" "if (c_free[i]) c_take = c_take | (NDC'(1) << i);"   # 一個命令交給多個通道
mut mc_tag_ch    $MC "if (m_rd_rvalid) c_mrd_rv[t_head] = 1'b1;" "if (m_rd_rvalid) c_mrd_rv[0] = 1'b1;"                # 讀回應都送通道 0
mut mc_tag_push  $MC "wire t_push = m_rd_req && m_rd_gnt;" "wire t_push = m_rd_req;"                                      # tag 在 grant 前就記錄
mut mc_tag_in    $MC "for (int i = 0; i < NDC; i++) if (s_mrd[i]) t_in = CW'(i);" "for (int i = 0; i < NDC; i++) if (c_mrd_req[i]) t_in = CW'(i);"   # tag 記錄請求者而非得主
mut mc_hrd_line  $MC "if (s_hrd[i]) h_rd_line = c_hrd_line[i];" "if (c_hrd_req[i]) h_rd_line = c_hrd_line[i];"          # L2 讀位址不屬於得主
mut mc_hwr_data  $MC "if (s_hwr[i]) begin h_wr_line = c_hwr_line[i]; h_wr_data = c_hwr_data[i]; end" "if (s_hwr[i]) h_wr_line = c_hwr_line[i]; h_wr_data = c_hwr_data[0];"
mut mc_mwr_data  $MC "if (s_mwr[i]) begin m_wr_addr = c_mwr_addr[i]; m_wr_data = c_mwr_data[i]; end" "if (s_mwr[i]) m_wr_addr = c_mwr_addr[i]; m_wr_data = c_mwr_data[0];"
mut mc_mrd_addr  $MC "if (s_mrd[i]) m_rd_addr = c_mrd_addr[i];" "if (c_mrd_req[i]) m_rd_addr = c_mrd_addr[i];"
mut mc_cpl_id    $MC "for (int i = 0; i < NDC; i++) if (c_sel[i]) cpl_id = c_id[i];" "for (int i = 0; i < NDC; i++) if (cv[i]) cpl_id = c_id[i];"   # 完成 id 不屬於被回報的通道
mut mc_cv_clr    $MC "else if (c_pop && c_sel[i]) cv[i] <= 1'b0;" "else if (c_pop) cv[i] <= 1'b0;"                          # 回報一個時清掉所有完成槽
mut rr_nolock    $RR "assign sel     = lock ? own : pick;" "assign sel     = pick;"                                       # 等待 grant 時換得主
mut rr_gnt_all   $RR "assign gnt     = m_gnt ? sel : '0;" "assign gnt     = m_gnt ? req : '0;"                           # grant 給所有請求者
mut st_dir       $TILE ".cmd_st(head.op == NPU_OP_DMA_ST)" ".cmd_st(head.op[0])"                                       # 方向解碼錯
mut nop_id       $TILE "if (g_nop) begin nop_cv <= 1'b1; nop_id <= head.id; end" "if (g_nop) begin nop_cv <= 1'b1; end"  # NOP 完成 id 錯
# ---- npu_dma ----
mut dma_outst    $DMA "(outst != (OW+1)'(OUT))" "(outst != (OW+1)'(OUT - 1) && outst != (OW+1)'(OUT))"  # 在途上限少一 (功能仍正確, 由 a_issue 抓到吞吐量損失)
mut dma_overrun  $DMA "wire [OW:0] outst = iss[OW:0] - wr[OW:0];" "wire [OW:0] outst = {1'b0, iss[OW-1:0] - wr[OW-1:0]};"  # 在途計算溢位 → FIFO 覆蓋
# (dma_rd_qual / dma_wr_qual: 「外部 gnt 不限定 req」在 npu_dma_mc 內是等價 mutant — npu_rr_lock 只把 grant 給選中的請求者
#  (formal/npu_rr_lock 證明 sel ⊆ req), 通道看到的 gnt 必伴隨自己的 req。npu_rr_lock 的 rr_gnt_all 涵蓋 grant 分配錯誤)
mut dma_waddr    $DMA "wire e_step = st ? pop : iss_ok;" "wire e_step = iss_ok;"                       # 寫位址用錯計數器
mut dma_l2line   $DMA "assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(wr));" "assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(iss));"
mut dma_done     $DMA "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"                # 讀完就完成 (寫尚未完成)
mut dma_fifo     $DMA "if (pop)  f_rd <= f_rd + 1'b1;" "if (pop && !push)  f_rd <= f_rd + 1'b1;"                  # FIFO 讀指標錯
mut dma_ext31    $DMA "wire [31:0] e_addr = ind ? (ext + e_val + 32'(col)) : (erow + 32'(col));" "wire [31:0] e_addr = ind ? (ext + e_val + 32'(col)) : ({1'b0, erow[30:0]} + 32'(col));"       # 外部位址高位遺失
mut dma_pushd    $DMA "wire [DW-1:0] push_d = st ? h_rd_data : ld_skip ? '0 : m_rd_rdata;" "wire [DW-1:0] push_d = st ? h_rd_data : ld_skip ? '0 : h_rd_data;"
mut dma_n0       $DMA "assign done      = act && (wr == n);" "assign done      = act && (wr == n) && n != '0;"       # n = 0 永不完成
mut dma_2d_wrap  $DMA "((cols != '0) && (col == cols - 1'b1));" "((cols != '0) && (col == cols));"           # 2D: 列多一個 line
mut dma_2d_strd  $DMA "if (e_wrap) begin col <= '0; erow <= erow + 32'(stride); end" "if (e_wrap) begin col <= '0; erow <= erow + 32'(cols); end"   # 2D: 忽略列距
# ---- npu_dma 載入時解碼 (v6.3.13) ----
mut dec_fmt_bits $TILE ".cmd_fmt(head.b[17:16])" ".cmd_fmt(head.b[16:15])"                                     # 格式欄位位置錯
mut dec_mx_tab   $DMA "8'd4, 8'd6, 8'd8, 8'd12}" "8'd4, 8'd5, 8'd8, 8'd12}"                                                      # E2M1 1.5×2 錯
mut dec_mx_top   $DMA "8'd4, 8'd6, 8'd8, 8'd12}" "8'd4, 8'd6, 8'd8, 8'd14}"                                               # E2M1 最大值錯
mut dec_mx_sign  $DMA "(x[3] ? 8'(-m) : m);" "(m);"                               # MXFP4 符號遺失
mut dec_i4_sext  $DMA "return (f == 2'd1) ? {{4{x[3]}}, x} :" "return (f == 2'd1) ? {4'd0, x} :"                            # INT4 未符號延伸
mut dec_u4_sext  $DMA "(f == 2'd2) ? {4'd0, x} :" "(f == 2'd2) ? {{4{x[3]}}, x} :"                            # UINT4 被符號延伸
mut dec_nib      $DMA "wire [DW/2-1:0] f_sel = half ? f_head[DW-1:DW/2] : f_head[DW/2-1:0];" "wire [DW/2-1:0] f_sel = half ? f_head[DW/2-1:0] : f_head[DW-1:DW/2];"   # 前後半對調
mut dec_pop1     $DMA "wire pop    = w_fire && (!dec || half);" "wire pop    = w_fire;"                       # 前半寫完就釋放
mut dec_addr     $DMA "assign h_wr_line = dec ? (l2 + 26'({wr, half})) : (l2 + 26'(wr));" "assign h_wr_line = dec ? (l2 + 26'(wr) + 26'(half)) : (l2 + 26'(wr));"   # L2 位址未乘 2
mut dec_st       $DMA "dec <= !cmd_st && (cmd_fmt != 2'd0);" "dec <= (cmd_fmt != 2'd0);"                        # ST 也解碼
# ---- npu_dma 間接 gather / scatter (v6.3.13) ----
mut ind_bit      $TILE ".cmd_ind(head.b[18])" ".cmd_ind(head.b[19])"
mut ind_list     $TILE ".cmd_list(head.c)" ".cmd_list(head.a)"
mut ind_skip_val $DMA "wire        e_skip = ind && (e_val == 32'hFFFF_FFFF);" "wire        e_skip = ind && (e_val == 32'hFFFF_FFFE);"
mut ind_skip_nowait $DMA "wire ld_skip = can_iss && !st && e_skip && (pend_rd == '0);" "wire ld_skip = can_iss && !st && e_skip;"
mut ind_addr_col $DMA "wire [31:0] e_addr = ind ? (ext + e_val + 32'(col)) : (erow + 32'(col));" "wire [31:0] e_addr = ind ? (ext + e_val) : (erow + 32'(col));"
mut ind_list_line $DMA "assign h_rd_line = sel_l ? (list + 26'(lf)) : (l2 + 26'(iss));" "assign h_rd_line = sel_l ? (l2 + 26'(lf)) : (l2 + 26'(iss));"
mut ind_lfree    $DMA "wire l_free = ind && e_next && (ent == EW'(EPL - 1));" "wire l_free = ind && e_step && (ent == EW'(EPL - 1));"
mut ind_lslot    $DMA "lb[lbv[lh] ? lo : lh] <= h_rd_data;" "lb[lh] <= h_rd_data;"
mut ind_stskip   $DMA "assign m_wr_req  = st_wv && !e_skip;" "assign m_wr_req  = st_wv;"
mut ind_skip_data $DMA "ld_skip ? '0 : m_rd_rdata;" "ld_skip ? m_rd_rdata : m_rd_rdata;"
mut ind_noev     $DMA "&& (st || e_v);" "&& 1'b1;"
mut ind_cw0      $DMA "wire e_wrap = ind ? (col == ((cols == '0) ? 16'd0 : cols - 1'b1))" "wire e_wrap = ind ? (col == ((cols == '0) ? 16'd1 : cols - 1'b1))"
mut ind_nl       $DMA "LW'((32'(cmd_n) + EPL - 1) / EPL)" "LW'(32'(cmd_n) / EPL)"
mut ind_lbv_full $DMA "wire l_need  = act && ind && (lf != nl) && (lbv != 2'b11);" "wire l_need  = act && ind && (lf != nl);"
mut ind_n        $DMA "n <= cmd_ind ? cmd_n * ((cmd_cols == '0) ? 16'd1 : cmd_cols) : cmd_n;" "n <= cmd_n;"
# ---- VEC 分派 / 完成槽 / host 埠共用 (v6.3.10) ----
mut vec_no_wait  $TILE "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop && !vec_cv;" "wire g_vec  = go && head.op == NPU_OP_VEC && !h_nop;"   # 完成槽被佔用仍分派 (覆蓋 vec_id)
mut vec_k0       $TILE "((head.op == NPU_OP_GEMM || head.op == NPU_OP_VEC) && head.k == '0);" "(head.op == NPU_OP_GEMM && head.k == '0);"   # K = 0 的 VEC 送引擎
mut rsvd_op      $TILE "wire h_nop  = head.op == NPU_OP_NOP || head.op > NPU_OP_MOE ||" "wire h_nop  = head.op == NPU_OP_NOP ||"   # 保留 op 卡住
mut vec_id       $TILE "      if (v_take) vec_id <= head.id;" "      if (g_vec) vec_id <= head.id;"                                    # 引擎忙碌時也覆寫 id
mut vec_err_gate $TILE "assign vec_err = v_done && v_derr;" "assign vec_err = v_derr;"                                                  # vec_err 不只在完成拍
mut vec_sel      $TILE "sel[2] ? nop_id : sel[3] ? vec_id : llm_id;" "sel[2] ? nop_id : sel[3] ? nop_id : llm_id;"   # VEC 完成回報錯的 id
mut rr4          $TILE "automatic int s = (int'(last) + 1 + i) % NSRC;" "automatic int s = (int'(last) + 1 + i) % (NSRC - 1);"            # round-robin 漏掉 LLM
# ---- LLM 引擎 (KV / MOE, v6.3.13) 的接入 ----
mut llm_slot     $TILE "&& !llm_cv && !l_busy;" "&& !l_busy;"                                                   # 完成槽被佔用仍分派 (覆蓋 llm_id)
mut llm_id       $TILE "if (g_llm) llm_id <= head.id;" "if (l_done) llm_id <= head.id;"                         # id 在完成時才記 (已是別的命令)
mut llm_sel      $TILE "sel[3] ? vec_id : llm_id;" "sel[3] ? vec_id : vec_id;"                                  # LLM 完成回報錯的 id
mut llm_rsvd     $TILE "head.op > NPU_OP_MOE ||" "head.op > NPU_OP_VEC ||"                                      # KV / MOE 當 NOP
mut llm_op       $TILE ".cmd_op(head.op == NPU_OP_MOE)" ".cmd_op(head.op[0])"                                   # KV / MOE 解碼錯
mut llm_rd_line  $TILE ".rd_line({l_rd_line, p_rd_line})" ".rd_line({p_rd_line, p_rd_line})"                    # host 讀埠: LLM 的位址沒有送出
mut llm_wr_data  $TILE ".wr_data({l_wr_data, p_wr_data})" ".wr_data({p_wr_data, p_wr_data})"                    # host 寫埠: LLM 的資料錯
mut llm_clr      $TILE "else if (load && sel[4]) llm_cv <= 1'b0;" "else if (load) llm_cv <= 1'b0;"              # 別的來源被選中時 LLM 完成遺失
# (busy 拿掉 v_busy 是等價 mutant: 形式證明的不變式 oc = core 內 + DMA + NOP + (v_busy || vec_cv), v_busy ⇒ oc != 0 ⇒ busy)
mut arb_nolock   $ARB "wire  r_sel  = r_lock ? r_own : r_pick;" "wire  r_sel  = r_pick;"                                               # 等待中換主端 (違反保持)
mut arb_nolock_w $ARB "wire  w_sel  = w_lock ? w_own : w_pick;" "wire  w_sel  = w_pick;"                                               # 寫入等待中換主端
mut arb_gnt_both $ARB "assign rd_gnt    = {m_rd_gnt && r_sel, m_rd_gnt && !r_sel} & rd_req;" "assign rd_gnt    = {m_rd_gnt, m_rd_gnt} & rd_req;"   # grant 給兩個
mut arb_wdata    $ARB "assign m_wr_data = wr_data[w_sel];" "assign m_wr_data = wr_data[0];"                                          # 寫資料取錯主端
mut arb_fixed    $ARB "wire  w_pick = (wr_req == 2'b11) ? !w_last : wr_req[1];" "wire  w_pick = wr_req[1];"                            # 固定優先權
echo "=== $CFG: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
