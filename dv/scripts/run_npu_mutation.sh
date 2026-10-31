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
  verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style -GN=$N \
    $([ $top = npu_core_tb ] && echo "-GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH") --Mdir "$M/$n.obj" -Irtl/common \
    $src --top-module $top > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "$M/$n.obj"; surv=1; return; }
  "$M/$n.obj/V$top" +seed=7 $args > "$M/$n.log" 2>&1
  # 全 0 上電通過時, 再以隨機上電狀態 (未 reset 的儲存為隨機值) 執行
  for r in 2 3; do
    grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log" \
      && "$M/$n.obj/V$top" +seed=$((7 + r)) $args +verilator+rand+reset+2 +verilator+seed+$r > "$M/$n.log" 2>&1
  done
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
mut no_sext_a     $MMA "wire signed [8:0]  ax = {f_in.sa & a_in[7], a_in};" "wire signed [8:0]  ax = {1'b0, a_in};"
mut no_sext_b     $MMA "wire signed [8:0]  bx = {f_in.sb & b_in[7], b_in};" "wire signed [8:0]  bx = {b_in[7], b_in};"
mut no_clr        $MMA "(f_in.f ? 32'd0 : acc[r][c])" "(acc[r][c])"                                     # 新 tile 不清累加器
mut row_no_ready  $MMA "if (ost == O_OUT && out_ready) out_row" "if (ost == O_OUT) out_row"                   # 忽略回壓
mut skew_b        $MMA "assign b_edge[i] = b_sr[i-1];" "assign b_edge[i] = (i > 1) ? b_sr[i-2] : b_vec[i];" # B skew 少一拍
mut sign_latch    $MMA "sa: sgn_a, sb: sgn_b" "sa: a_signed, sb: sgn_b"                                         # 有號性未在 start 鎖存
mut data_no_tok   $MMA "if (t_in) begin                        // 資料只在有 token 時更新
            a_q[r][c] <= a_in; b_q[r][c] <= b_in;" "if (t_in) begin
            a_q[r][c] <= a_in;"                                                                               # B 資料不前進
mut acc_trunc     $MMA "acc[r][c]) + 32'(prod)" "acc[r][c]) + 32'(prod[15:0])"                                  # 乘積截斷
mut start_busy    $MMA "if (go) begin ist <= I_RUN;" "if (start) begin ist <= I_RUN;"                           # 輸入端忙碌時接受 start
mut last_early    $MMA "O_IDLE:  if (fire_last) ost <= O_DRAIN;" "O_IDLE:  if (in_valid && in_last) ost <= O_DRAIN;"  # in_last 未握手即排空
mut res_no_last   $MMA "if (f_in.l) res[r][c] <= sum;" "res[r][c] <= sum;"                                      # 結果暫存被下一個 tile 覆蓋
mut no_last_stall $MMA "assign in_ready  = (ist == I_RUN) && !(in_last && ost != O_IDLE);" "assign in_ready  = (ist == I_RUN);"  # 前一個 tile 未輸出完就收最後一步
mut fl_no_skew    $MMA "fl_q[r][c] <= f_in;" "fl_q[r][c] <= fl_in;"                                              # 旗標未隨 token 傳遞
mut first_stuck   $MMA "        first <= 1'b0;" "        first <= first;"                                        # 每一步都清累加器
# (列 ICG 拿掉 !rst_n 是等價 mutant: reset 期間 row_act = 0 表示本列 token 已全 0, 資料暫存只在有 token 時使用, 累加器由
#  token 的 first 旗標清除; 仍保留 !rst_n 以符合共用規則。CU 層級的 ICG 同理, 見 npu_core 段落)
# ---- npu_gemm_cu ----
mut cu_aptr_stuck $CU "a_ptr <= a_ptr + 1'b1; fa <= fa - 1'b1; end" "a_ptr <= a_ptr; fa <= fa - 1'b1; end"   # A line 指標不前進
mut cu_pop_early  $CU "wire pop  = feed && (step == SW'(SPL - 1) || mma_in_last);" "wire pop  = feed && (step == SW'(SPL - 2) || mma_in_last);"  # 提早換 line
mut cu_no_wgnt    $CU "if (wr_gnt) c_ptr <= c_ptr + 1'b1;" "c_ptr <= c_ptr + 1'b1;"                          # C 指標不等寫入 grant
mut cu_wr_early   $CU "if (wr_end) wst <=" "if (wr_end || wr_req) wst <="                                      # 未拿到寫入 grant 就結束該列
mut cu_pend_tag   $CU "c_pend <= c_cmd; t_pend <= t_cmd;" "c_pend <= c_cmd; t_pend <= cmd_tag;"                  # tag 取自埠上 (下一個命令) 而非本命令
mut cu_take_early $CU "(wst == W_DONE && done_ready)" "(wst == W_DONE)"                                       # 完成未握手就換下一個 tile
mut cu_done_noack $CU "W_DONE: if (done_ready) wst <= W_IDLE;" "W_DONE: wst <= W_IDLE;"                           # 完成不等握手
mut cu_lr         $CU "lr <= (mma_out_row == RW'(N - 1));" "lr <= (mma_out_row == RW'(N - 2));"                   # 提早完成
mut cu_take_cmd   $CU "if (take) begin c_ptr <= c_pend;" "if (take) begin c_ptr <= c_cmd;"                        # 寫回器取錯 C 位址
mut cu_busy_pend  $CU "assign busy      = (fst != F_IDLE) || pend || (wst != W_IDLE);" "assign busy      = (fst != F_IDLE) || (wst != W_IDLE);"  # busy 漏掉等待輸出
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
mut xb_ls_nowait  $XB "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= '0; end"  # 喚醒不等 SRAM
mut xb_ls_nowake  $XB "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (1'b0) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end"  # 睡了不醒 (死結)
mut xb_ls_rdidle  $XB "else if (rd_rq[b] || wr_rq[b]) idle[b] <= '0;" "else if (wr_rq[b]) idle[b] <= '0;"  # 讀取中仍計閒置 (存取中睡著)
mut xb_ls_rdy     $XB "assign b_rdy[b]   = !ls[b] && wk[b] == '0;" "assign b_rdy[b]   = wk[b] == '0;"   # 睡眠中仍給 grant
mut xb_ls_early   $XB "else if (idle[b] == IW'(LS_IDLE - 1)) begin" "else if (idle[b] == IW'(LS_IDLE - 2)) begin"  # 提早睡
# ---- npu_core ----
# (CU 的 ICG 拿掉 !rst_n 在硬體上是等價 mutant: CU / npu_mma 的每個 flop 都是非同步 reset, reset 不需要時脈。
#  v6.3.12 之前會被抓到, 是因為 Verilator 中 rst_n 從 0 開始時沒有 negedge 事件, 非同步 reset 只在 reset 期間有時脈沿時
#  才生效 — 隨機上電狀態下結果看運氣 (Nano 存活、X1 / Pro 抓到)。仍保留 !rst_n 以符合共用規則, mutant 移除)
mut core_cpl_hold  $CORE "wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);" "wire cpl_load = (cpl_pend != '0);"  # 完成輸出不保持
mut core_gate      $CORE "assign cu_en[i] = cu_busy[i] || cu_v[i];" "assign cu_en[i] = cu_busy[i];"              # 閘控未涵蓋分派拍
mut core_dack      $CORE "assign cu_dack    = ~cpl_pend;" "assign cu_dack    = '1;"                              # 待回報槽未空就接受完成 (遺失)
[ $NCU -gt 1 ] && mut core_idle_1st  $CORE "assign cu_cand    = (cu_idle != '0) ? cu_idle : cu_free;" "assign cu_cand    = cu_free;"  # 不優先閒置 CU (NCU = 1 時等價)
mut core_q_full    $CORE "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"                # 佇列滿仍收命令
mut core_tag       $CORE "cpl_tag[i] <= cu_dtag[i];" "cpl_tag[i] <= cu_dtag[i] ^ 8'h01;"                       # 完成 id 錯
mut core_perf_wr   $CORE "perf_busy <= perf_wdata_busy; perf_steps <= perf_wdata_steps;" "perf_busy <= perf_wdata_busy; perf_steps <= perf_steps;"  # 計數器寫入無效
[ $NCU -gt 1 ] && mut core_perf_multi $CORE "perf_steps <= perf_steps + 48'(\$countones(cu_feed));" "perf_steps <= perf_steps + 48'(|cu_feed);"  # 多 CU 同拍只計一次 (NCU=1 等價, 不適用)
echo "=== $CFG: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
