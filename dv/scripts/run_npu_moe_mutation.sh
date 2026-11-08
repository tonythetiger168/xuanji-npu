#!/bin/bash
# npu_moe_disp mutation 測試: 每個 mutant 植入一個錯誤, 以 seed 1 / 2 (各 150 個命令) 執行 npu_moe_disp_tb, 必須失敗。
# 用法: dv/scripts/run_npu_moe_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/npu_moe_mut; rm -rf $M; mkdir -p $M
R=rtl/npu/npu_moe_disp.sv
nk=0; nt=0; surv=0
mut() {   # name old new
  local n=$1
  python3 - "$2" "$3" "$M/$n.sv" "$R" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if s.count(o) != 1: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common --Mdir $M/$n.obj \
    rtl/common/orca_pkg.sv rtl/common/clock_gate.sv $M/$n.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_moe/npu_moe_disp_tb.sv \
    --top-module npu_moe_disp_tb > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  local why=""
  for s in 1 2; do
    (cd $M && ./$n.obj/Vnpu_moe_disp_tb +seed=$s +cycles=20000 > $n.$s.log 2>&1)
    if ! grep -q "TB PASS" $M/$n.$s.log || grep -q "Assertion failed" $M/$n.$s.log; then
      why="seed $s: $(grep -m1 -oE 'MISMATCH [a-z_]+|ERROR[^,]*|u_sva\.[A-Za-z_]*' $M/$n.$s.log)"; break
    fi
  done
  rm -rf $M/$n.obj
  if [ -z "$why" ]; then echo "$n: SURVIVED"; surv=1; else nk=$((nk + 1)); echo "$n: killed ($why)"; fi
}
mut bad_ge      "wire          bad  = in_exp >= (EW+2)'(ne);" "wire          bad  = in_exp > (EW+2)'(ne);"            # id = nexp 當有效
mut bad_nexp    "wire          bad  = in_exp >= (EW+2)'(ne);" "wire          bad  = in_exp >= (EW+2)'(NEXP);"         # 不依 cmd_nexp
mut full_ge     "wire          full = !dl && r >= (PW+1)'(cap);" "wire          full = !dl && r > (PW+1)'(cap);"   # 容量多一個
mut full_dl     "wire          full = !dl && r >= (PW+1)'(cap);" "wire          full = r >= (PW+1)'(cap);"          # 無丟棄模式也丟
mut pos_dl      "dl ? off[e] + r :" "dl ? off[e] :"                                                                   # 無丟棄模式少 rank
mut pos_cap     "(PW+1)'(e) * (PW+1)'(cap) + r;" "(PW+1)'(e) * (PW+1)'(cap);"                                          # 容量模式少 rank
mut pfx_sum     "sum <= sum + n[ci];" "sum <= n[ci];"                                                                 # 前綴和錯
mut pfx_clr     "begin off[ci] <= sum; n[ci] <= '0; end" "begin off[ci] <= sum; end"                                  # 第二趟不清計數
mut clr_skip    "if (st == S_CLR) n[ci] <= '0;" "if (1'b0) n[ci] <= '0;"                                               # 命令開始不清計數
mut cnt_min     "assign cnt_n     = (!dl && n[ci] > (PW+1)'(cap)) ? (PW+1)'(cap) : n[ci];" "assign cnt_n     = n[ci];" # 容量模式數量不取 min
mut cnt_off     "assign cnt_off   = dl ? off[ci] :" "assign cnt_off   = 1'b1 ? off[ci] :"                          # 容量模式起點錯
mut cnt_last    "if (i == ne - 1'b1) begin st <= S_IDLE; done <= 1'b1; end" "if (i == ne - 2'd2) begin st <= S_IDLE; done <= 1'b1; end" # 少一筆數量
mut drop_pos    "out_pos   <= (bad || full) ? '0 : pos[PW-1:0];" "out_pos   <= pos[PW-1:0];"                         # drop 時輸出垃圾位置
mut err_flag    "out_err   <= bad;" "out_err   <= 1'b0;"                                                               # 不報錯
mut o_free      "wire   o_free    = !out_valid || out_ready;" "wire   o_free    = 1'b1;"                                 # 輸出暫存覆蓋
mut ovf_never   "if (tot[PW] && st != S_IDLE) ovf <= 1'b1;" "if (1'b0) ovf <= 1'b1;"                                   # 不報溢位
mut ovf_bad     "if (!bad) tot <= tot + 1'b1;" "tot <= tot + 1'b1;"                                                    # 錯誤項目也計入上限
mut icg         "wire  en = cmd_valid || st != S_IDLE || out_valid || done;" "wire  en = cmd_valid || st != S_IDLE || done;" # 閘控漏掉輸出暫存
mut cap_latch   "dl <= cmd_dropless; cap <= cmd_cap;" "dl <= cmd_dropless;"                                           # cap 未鎖存
echo "npu_moe_disp mutation: killed $nk / $nt"
[ "$surv" -eq 0 ]
