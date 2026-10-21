#!/bin/bash
# NPU 檢查器的 mutation 測試 (每個產品組態): 每個 mutant 在 RTL 植入一個錯誤, 對應的 CRV + SVA 必須抓到 (killed)。
# 存活的 mutant 代表檢查器有漏洞 (或是等價 mutant, 須說明)。
#   npu_mma / npu_gemm_cu 的 mutant 以各自的 TB (該組態的 N) 執行; npu_l2_xbar / npu_core 的 mutant 以 npu_core_tb 執行。
# 用法: dv/scripts/run_npu_mutation.sh <nano|x1|pro|all> [count=120]
set -uo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NT=${2:-120}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG" || exit 1
M=build/npu_mut_$CFG; rm -rf "$M"; mkdir -p "$M"
echo "##### NPU mutation $CFG: N=$N NCU=$NCU NB=$NB"
MMA=rtl/npu/npu_mma.sv CU=rtl/npu/npu_gemm_cu.sv XB=rtl/npu/npu_l2_xbar.sv CORE=rtl/npu/npu_core.sv
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
  local mma=$MMA cu=$CU xb=$XB core=$CORE top args
  case $f in
    $MMA) mma=$M/$n.sv ;; $CU) cu=$M/$n.sv ;; $XB) xb=$M/$n.sv ;; $CORE) core=$M/$n.sv ;;
  esac
  local src="rtl/common/orca_pkg.sv rtl/common/clock_gate.sv $mma dv/npu/npu_mma_sva.sv"
  case $f in
    $MMA) top=npu_mma_tb; src="$src dv/npu/npu_mma_tb.sv"; args="+tiles=$((NT * 2))" ;;
    $CU)  top=npu_gemm_cu_tb; src="$src $cu dv/npu/npu_gemm_cu_sva.sv dv/npu/npu_gemm_cu_tb.sv"; args="+cmds=$NT" ;;
    *)    top=npu_core_tb; args="+cmds=$NT"
          src="$src $cu dv/npu/npu_gemm_cu_sva.sv $xb dv/npu/npu_l2_xbar_sva.sv $core dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv" ;;
  esac
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -GN=$N \
    $([ $top = npu_core_tb ] && echo "-GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH") --Mdir "$M/$n.obj" -Irtl/common \
    $src --top-module $top > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "$M/$n.obj"; surv=1; return; }
  "$M/$n.obj/V$top" +seed=7 $args > "$M/$n.log" 2>&1
  rm -rf "$M/$n.obj"                       # 每個 mutant 的建置目錄數百 MB: 用完即刪
  if grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log"; then
    echo "$n: SURVIVED"; surv=1
  else
    nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH|u_sva\.[A-Za-z_.0-9\[\]]*|ERROR: [a-z_ ]*' "$M/$n.log"))"
  fi
}
# ---- npu_mma ----
mut gate_no_own   $MMA "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = a_tok_edge[r] || (|tok_q[r][N-2:0]);"  # 閘控未涵蓋最右 PE 的殘留 token
mut gate_no_edge  $MMA "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = (|tok_q[r]);"     # 閘控未涵蓋左邊緣進入的 token
mut drain_short   $MMA "localparam int DRAIN = 2 * N - 1;" "localparam int DRAIN = 2 * N - 3;"             # 排空不足
mut no_sext_a     $MMA "wire signed [8:0]  ax = {sgn_a & a_in[7], a_in};" "wire signed [8:0]  ax = {1'b0, a_in};"
mut no_sext_b     $MMA "wire signed [8:0]  bx = {sgn_b & b_in[7], b_in};" "wire signed [8:0]  bx = {b_in[7], b_in};"
mut no_clr        $MMA "if (clr)       acc[r][c] <= '0;" "if (1'b0)       acc[r][c] <= '0;"                # 新 tile 不清累加器
mut row_no_ready  $MMA "if (st == S_OUT && out_ready) out_row" "if (st == S_OUT) out_row"                     # 忽略回壓
mut skew_b        $MMA "assign b_edge[i] = b_sr[i-1];" "assign b_edge[i] = (i > 1) ? b_sr[i-2] : b_vec[i];" # B skew 少一拍
mut sign_latch    $MMA "if (clr) begin sgn_a <= a_signed; sgn_b <= b_signed; end" "begin sgn_a <= a_signed; sgn_b <= b_signed; end"
mut data_no_tok   $MMA "if (t_in) begin a_q[r][c] <= a_in; b_q[r][c] <= b_in; end" "if (t_in) a_q[r][c] <= a_in;" # B 資料不前進
mut acc_trunc     $MMA "acc[r][c] + 32'(prod)" "acc[r][c] + 32'(prod[15:0])"                                  # 乘積截斷
mut start_busy    $MMA "S_IDLE:  if (start) st_n = S_RUN;" "S_IDLE:  if (start) st_n = S_RUN;
      S_OUT:   if (start) st_n = S_RUN;"                                                                   # 輸出中接受 start
mut last_early    $MMA "S_RUN:   if (fire_in && in_last) st_n = S_DRAIN;" "S_RUN:   if (in_last) st_n = S_DRAIN;"  # in_last 未握手即結束
# ---- npu_gemm_cu ----
mut cu_aptr_stuck $CU "a_ptr <= a_ptr + 1'b1; fa <= fa - 1'b1; end" "a_ptr <= a_ptr; fa <= fa - 1'b1; end"   # A line 指標不前進
mut cu_pop_early  $CU "wire pop  = feed && (step == SW'(SPL - 1) || mma_in_last);" "wire pop  = feed && (step == SW'(SPL - 2) || mma_in_last);"  # 提早換 line
mut cu_no_wgnt    $CU "if (wr_gnt) c_ptr <= c_ptr + 1'b1;" "c_ptr <= c_ptr + 1'b1;"                          # C 指標不等寫入 grant
mut cu_wr_early   $CU "if (wr_end) begin" "if (wr_end || wr_req) begin"                                     # 未拿到寫入 grant 就結束該列
mut cu_last       $CU "assign mma_in_last   = (k_left == 16'd1);" "assign mma_in_last   = (k_left == 16'd2);"  # 少一個 k 步
mut cu_fleft      $CU "fa <= FW'((32'(cmd_k) + SPL - 1) / SPL);" "fa <= FW'((32'(cmd_k) + SPL - 1) / SPL + 1);"   # A 多讀一個 line (帳目錯)
mut cu_bsel       $CU "assign b_vec         = buf_b[rptr][step * N * 8 +: N * 8];" "assign b_vec         = buf_b[wb][step * N * 8 +: N * 8];"  # 讀錯緩衝槽
mut cu_have       $CU "assign have          = (a_cnt != '0) && (b_cnt != '0);" "assign have          = (a_cnt != '0);"  # B 未到齊就餵入
if [ $N -eq 32 ]; then   # 雙讀取埠 (RP = 2) 與一列兩個 C line
  mut cu_no_rgnt  $CU "assign a_gnt = rd_gnt[0]; assign b_gnt = rd_gnt[1];" "assign a_gnt = rd_req[0]; assign b_gnt = rd_gnt[1];"  # A 埠不等讀取 grant
  mut cu_overflow $CU "assign a_req = run && (fa != '0) && (a_cnt != (PW+1)'(PF));" "assign a_req = run && (fa != '0);"  # A 緩衝滿仍預取
  mut cu_b_full   $CU "assign b_req = run && (fb != '0) && (b_cnt != (PW+1)'(PF));" "assign b_req = run && (fb != '0);"  # B 緩衝滿仍預取
  mut cu_hi_half  $CU "else if (out_fire) begin c_hi <= mma_out_vec[N-1:N/2];" "else if (out_fire) begin c_hi <= mma_out_vec[N/2-1:0];"  # 後半列放錯 (曾犯的錯)
  mut cu_one_half $CU "assign wr_end   = wr_gnt && c_half;" "assign wr_end   = wr_gnt;"                     # 每列只寫前半
else                     # 單一讀取埠, A / B 輪流
  mut cu_no_rgnt  $CU "assign a_gnt = rd_gnt[0] && !f_b; assign b_gnt = rd_gnt[0] && f_b;" "assign a_gnt = rd_req[0] && !f_b; assign b_gnt = rd_req[0] && f_b;"  # 不等讀取 grant
  mut cu_overflow $CU "assign a_req = run && !f_b && (fa != '0) && (a_cnt != (PW+1)'(PF));" "assign a_req = run && !f_b && (fa != '0);"  # 緩衝滿仍預取 (覆蓋未用的 line)
  mut cu_alt      $CU "else if (rd_gnt[0]) f_b <= ~f_b;" "else if (rd_req[0]) f_b <= ~f_b;"               # 未 grant 就換軌
fi
[ $N -lt 16 ] && mut cu_slot $CU "c_line_n[slot * N * 32 +: N * 32] = mma_out_vec;" "c_line_n[0 +: N * 32] = mma_out_vec;"  # C 列放錯位置 (N=16 每列一個 line, 不適用)
# ---- npu_l2_xbar ----
mut xb_two_gnt    $XB "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb] && rd_bm[rb] == MW'(m);" "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb];"  # 同 bank 多個讀 grant
mut xb_no_rr      $XB "if (rd_bv[b]) rd_rr[b] <= rd_bm[b];" "if (1'b0) rd_rr[b] <= rd_bm[b];"               # 固定優先權 (可能飢餓)
mut xb_no_wrr     $XB "if (wr_bv[b]) wr_rr[b] <= wr_bm[b];" "if (1'b0) wr_rr[b] <= wr_bm[b];"               # 寫入固定優先權
mut xb_row        $XB "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)); endfunction" "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l >> 1, BW, DW)); endfunction"  # bank 列位址錯
mut xb_wr_any     $XB "assign wr_gnt[m]  = wr_req[m] && wr_bv[wb] && wr_bm[wb] == WW'(m);" "assign wr_gnt[m]  = wr_req[m];"  # 寫入不仲裁
# ---- npu_core ----
mut core_done_disp $CORE "assign cu_free    = cu_ready & ~cpl_pend & ~cu_done;" "assign cu_free    = cu_ready & ~cpl_pend;"  # done 當拍分派 (已修正的 bug)
mut core_cpl_hold  $CORE "wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);" "wire cpl_load = (cpl_pend != '0);"  # 完成輸出不保持
mut core_gate      $CORE "assign cu_en[i] = !cu_ready[i] || cu_v[i] || cu_done[i];" "assign cu_en[i] = !cu_ready[i] || cu_v[i];"  # 閘控未涵蓋 done 脈衝
mut core_q_full    $CORE "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"                # 佇列滿仍收命令
mut core_tag       $CORE "if (cu_v[i]) cu_tag[i] <= head.id;" "if (cu_v[i]) cu_tag[i] <= head.id ^ 8'h01;"     # 完成 id 錯
mut core_perf_wr   $CORE "perf_busy <= perf_wdata_busy; perf_steps <= perf_wdata_steps;" "perf_busy <= perf_wdata_busy; perf_steps <= perf_steps;"  # 計數器寫入無效
[ $NCU -gt 1 ] && mut core_perf_multi $CORE "perf_steps <= perf_steps + 48'(\$countones(cu_feed));" "perf_steps <= perf_steps + 48'(|cu_feed);"  # 多 CU 同拍只計一次 (NCU=1 等價, 不適用)
echo "=== $CFG: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
