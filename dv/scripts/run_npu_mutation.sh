#!/bin/bash
# npu_mma / npu_gemm_cu 檢查器的 mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, CRV + SVA 必須抓到 (killed)。
# 存活的 mutant 代表檢查器有漏洞 (或是等價 mutant, 須說明)。
# 用法: dv/scripts/run_npu_mutation.sh [tiles=150]
set -uo pipefail
cd "$(dirname "$0")/../.."
NT=${1:-150}
M=build/npu_mut; rm -rf "$M"; mkdir -p "$M"
surv=0
mut() {   # name old new [file]  (file 預設 rtl/ai/pe/npu_mma.sv)
  local n=$1 f=${4:-rtl/ai/pe/npu_mma.sv}
  python3 - "$2" "$3" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  local src="$M/$n.sv rtl/ai/cluster/npu_gemm_cu.sv" top=npu_mma_tb tbs="dv/npu/npu_mma_tb.sv" args="+tiles=$NT"
  if [[ $f == *npu_gemm_cu* ]]; then
    src="rtl/ai/pe/npu_mma.sv $M/$n.sv"; top=npu_gemm_cu_tb; tbs="dv/npu/npu_gemm_cu_sva.sv dv/npu/npu_gemm_cu_tb.sv"; args="+cmds=$((NT / 2))"
  else
    src="$M/$n.sv"
  fi
  verilator --binary --timing -j "${JOBS:-2}" --assert -Wno-fatal -Wno-lint -Wno-style --Mdir "$M/$n.obj" -Irtl/common \
    rtl/common/orca_pkg.sv rtl/common/clock_gate.sv $src dv/npu/npu_mma_sva.sv $tbs \
    --top-module $top > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; surv=1; return; }
  "$M/$n.obj/V$top" +seed=7 $args > "$M/$n.log" 2>&1
  if grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log"; then
    echo "$n: SURVIVED"; surv=1
  else
    echo "$n: killed ($(grep -m1 -oE 'MISMATCH|u_sva\.[A-Za-z_.0-9\[\]]*|ERROR: [a-z_ ]*' "$M/$n.log"))"
  fi
}
mut gate_no_own   "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = a_tok_edge[r] || (|tok_q[r][N-2:0]);"  # 閘控未涵蓋最右 PE 的殘留 token
mut gate_no_edge  "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = (|tok_q[r]);"     # 閘控未涵蓋左邊緣進入的 token
mut drain_short   "localparam int DRAIN = 2 * N - 1;" "localparam int DRAIN = 2 * N - 3;"             # 排空不足
mut no_sext_a     "wire signed [8:0]  ax = {sgn_a & a_in[7], a_in};" "wire signed [8:0]  ax = {1'b0, a_in};"
mut no_sext_b     "wire signed [8:0]  bx = {sgn_b & b_in[7], b_in};" "wire signed [8:0]  bx = {b_in[7], b_in};"
mut no_clr        "if (clr)       acc[r][c] <= '0;" "if (1'b0)       acc[r][c] <= '0;"                # 新 tile 不清累加器
mut row_no_ready  "if (st == S_OUT && out_ready) out_row" "if (st == S_OUT) out_row"                     # 忽略回壓
mut skew_b        "assign b_edge[i] = b_sr[i-1];" "assign b_edge[i] = (i > 1) ? b_sr[i-2] : b_vec[i];" # B skew 少一拍
mut sign_latch    "if (clr) begin sgn_a <= a_signed; sgn_b <= b_signed; end" "begin sgn_a <= a_signed; sgn_b <= b_signed; end"
mut data_no_tok   "if (t_in) begin a_q[r][c] <= a_in; b_q[r][c] <= b_in; end" "if (t_in) a_q[r][c] <= a_in;" # B 資料不前進
mut acc_trunc     "acc[r][c] + 32'(prod)" "acc[r][c] + 32'(prod[15:0])"                                  # 乘積截斷
mut start_busy    "S_IDLE:  if (start) st_n = S_RUN;" "S_IDLE:  if (start) st_n = S_RUN;
      S_OUT:   if (start) st_n = S_RUN;"                                                                   # 輸出中接受 start
mut last_early    "S_RUN:   if (fire_in && in_last) st_n = S_DRAIN;" "S_RUN:   if (in_last) st_n = S_DRAIN;"  # in_last 未握手即結束
CU=rtl/ai/cluster/npu_gemm_cu.sv
mut cu_aptr_stuck "a_ptr <= a_ptr + 1'b1;" "a_ptr <= a_ptr;" $CU                                        # A line 指標不前進
mut cu_step_wrap  "else if (step == SW'(SPL - 1))   st <= G_RD_A;" "else if (step == SW'(SPL - 2))   st <= G_RD_A;" $CU  # 提早換 line
mut cu_slot       "c_line[c_slot * N * 32 +: N * 32] <= mma_out_vec;" "c_line[0 +: N * 32] <= mma_out_vec;" $CU    # C 列放錯位置
mut cu_no_gnt     "G_WR: if (wr_gnt) begin" "G_WR: begin" $CU                                              # 不等寫入 grant
mut cu_rd_no_gnt  "G_RD_B: if (rd_gnt) begin" "G_RD_B: begin" $CU                                          # 不等讀取 grant
mut cu_done_early "if (c_slot == \$clog2(RPL)'(RPL - 1)) st <= G_WR;" "st <= G_WR;" $CU                   # 每列都寫一次 line
mut cu_last       "assign mma_in_last  = (k_left == 16'd1);" "assign mma_in_last  = (k_left == 16'd2);" $CU   # 少一個 k 步
[ "$surv" -eq 0 ] && echo "=== all mutants killed" || { echo "=== some mutants survived"; exit 1; }
