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
  (cd $d && sby -f $b.sby > sby_run.log 2>&1)
  for t in $(grep -oE "^\[npu[^]]*\]|DONE \([A-Z]+" $d/sby_run.log | true); do :; done
  grep -E "\] DONE" $d/sby_run.log | sed "s/^/$b: /"
  grep -q "DONE (FAIL\|DONE (ERROR\|DONE (UNKNOWN" $d/sby_run.log && rc=1
done
exit $rc
