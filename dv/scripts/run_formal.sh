#!/bin/bash
# 形式驗證 (SymbiYosys): RTL 先以 sv2v 轉成 Verilog (Yosys 0.33 的 frontend 不支援本專案的 SystemVerilog), 再跑各 .sby
# 需求: yosys + yosys-abc, sby, sv2v, z3
# 用法: dv/scripts/run_formal.sh [block ...]   (預設: 全部)
# FORMAL_TOP_<block>: 只在該區塊是證明頂層時才編入的檢查 (子模組的環境假設不可約束上層)
set -uo pipefail
cd "$(dirname "$0")/../.."
export LANG=C.UTF-8 LC_ALL=C.UTF-8      # sv2v 需要 UTF-8 locale (原始碼含中文註解)
declare -A SRC=(
  [npu_ss_dma]="rtl/npu/npu_ss_dma.sv"
  [npu_dma]="rtl/npu/npu_dma.sv"
  [orca_npu_ss]="formal/orca_npu_ss/stubs.sv rtl/npu/npu_l2_arb2.sv rtl/npu/orca_npu_ss.sv"
  [npu_l2_xbar]="rtl/npu/npu_l2_xbar.sv"
  [npu_core]="formal/npu_core/stubs.sv rtl/npu/npu_core.sv"
  [npu_vec]="formal/npu_vec/stubs.sv rtl/npu/npu_vec.sv"
  [npu_ss_regs]="formal/npu_ss_regs/stubs.sv rtl/common/ret_shadow.sv rtl/npu/npu_ss_regs.sv"
  [npu_l2_arb2]="rtl/npu/npu_l2_arb2.sv"
  [npu_vec_eng]="formal/npu_vec_eng/stubs.sv rtl/npu/npu_vec_eng.sv"
  [orca_apb_xbar]="rtl/soc/orca_apb_xbar.sv"
  [orca_sys_regs]="formal/orca_sys_regs/stubs.sv rtl/soc/orca_sys_regs.sv"
  [orca_pmu_regs]="formal/orca_pmu_regs/stubs.sv rtl/soc/orca_pmu_regs.sv"
  [orca_pmu]="rtl/soc/orca_pmu.sv"
  [gs_raster_regs]="formal/gs_raster_regs/stubs.sv rtl/common/ret_shadow.sv rtl/sis/gs_raster_regs.sv"
)
blocks=${*:-${!SRC[@]}}
rc=0
for b in $blocks; do
  d=formal/$b
  sv2v -DFORMAL -DFORMAL_TOP_$b -E Assert -Irtl/common rtl/common/orca_pkg.sv ${SRC[$b]} > $d/$b.v || { echo "$b: sv2v FAIL"; rc=1; continue; }
  # sv2v 預設會刪除 assertion (-E Assert 保留); 轉出的設計沒有 assert 時整個證明是空的
  grep -q "assert" $d/$b.v || { echo "$b: no assertions after sv2v (vacuous)"; rc=1; continue; }
  (cd $d && sby -f $b.sby > sby_run.log 2>&1) || rc=1
  # 每個 task 都必須回報 DONE (PASS); 沒有回報 (例如 sby 當掉) 也算失敗
  for t in $(sed -n '/^\[tasks\]/,/^\[/{/^[a-z]/p}' $d/$b.sby); do
    if grep -q "\[${b}_$t\] DONE (PASS" $d/sby_run.log; then echo "$b/$t: PASS"
    else echo "$b/$t: FAIL ($(grep -oE "\[${b}_$t\] DONE \([A-Z]+" $d/sby_run.log || echo 'no result'))"; rc=1; fi
  done
done
exit $rc
