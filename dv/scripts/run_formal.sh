#!/bin/bash
# 形式驗證 (SymbiYosys): RTL 先以 sv2v 轉成 Verilog (Yosys 0.33 的 frontend 不支援本專案的 SystemVerilog), 再跑各 .sby
# 需求: yosys + yosys-abc, sby, sv2v, z3
# 用法: dv/scripts/run_formal.sh [block ...]   (預設: 全部)
set -uo pipefail
cd "$(dirname "$0")/../.."
export LANG=C.UTF-8 LC_ALL=C.UTF-8      # sv2v 需要 UTF-8 locale (原始碼含中文註解)
declare -A SRC=(
  [npu_ss_dma]="rtl/npu/npu_ss_dma.sv"
)
blocks=${*:-${!SRC[@]}}
rc=0
for b in $blocks; do
  d=formal/$b
  sv2v -Irtl/common rtl/common/orca_pkg.sv ${SRC[$b]} > $d/$b.v || { echo "$b: sv2v FAIL"; rc=1; continue; }
  (cd $d && sby -f $b.sby > sby_run.log 2>&1) || rc=1
  # 每個 task 都必須回報 DONE (PASS); 沒有回報 (例如 sby 當掉) 也算失敗
  for t in $(sed -n '/^\[tasks\]/,/^\[/{/^[a-z]/p}' $d/$b.sby); do
    if grep -q "\[${b}_$t\] DONE (PASS" $d/sby_run.log; then echo "$b/$t: PASS"
    else echo "$b/$t: FAIL ($(grep -oE "\[${b}_$t\] DONE \([A-Z]+" $d/sby_run.log || echo 'no result'))"; rc=1; fi
  done
done
exit $rc
