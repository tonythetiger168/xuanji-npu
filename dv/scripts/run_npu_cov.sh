#!/bin/bash
# NPU coverage closure, 每個產品組態各自收斂:
#   npu_mma_tb (N) + npu_gemm_cu_tb (N) + npu_core_tb (N, NCU, NB, DEPTH) 的 CRV 多個 seed, 加上效能量測,
#   以「模組層級」合併 (同一模組在不同 TB / 不同 instance 的同一點合併; generate 範圍仍分開),
#   line / branch / expr / toggle / FSM / SVA cover / covergroup 套用 dv/npu/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_cov.sh <nano|x1|pro|all> [seeds=4] [count=200]
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NS=${2:-4}; NT=${3:-200}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NS" "$NT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG"
B=build/npu_cov_$CFG; rm -rf "$B"; mkdir -p "$B"
echo "##### NPU config $CFG: N=$N NCU=$NCU NB=$NB DEPTH=$DEPTH (peak $((N * N * NCU)) MAC/cycle)"
SRC="-Irtl/common rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv"
CU="rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv"
build() {   # top -G... -- sources...
  local top=$1; shift
  verilator --binary --timing -j "${JOBS:-4}" --assert --coverage --coverage-max-width 32768 \
    -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/$top" $SRC "$@" --top-module $top > "$B/$top.build.log" 2>&1 \
    || { tail -20 "$B/$top.build.log"; echo "BUILD FAIL $top"; exit 1; }
}
build npu_mma_tb -GN=$N dv/npu/npu_mma_tb.sv
build npu_gemm_cu_tb -GN=$N $CU dv/npu/npu_gemm_cu_tb.sv
build npu_core_tb -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH $CU rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
  rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv
python3 dv/scripts/wide_signals.py "$B/npu_core_tb" 32768 npu_core_tb__DOT__dut__DOT__ | tee "$B/toggle_excluded.txt"
fail=0
run() {   # top name tag plusargs...
  local top=$1 n=$2 tag=$3; shift 3
  if (cd "$B" && ./$top/V$top "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log"; then
    echo "PASS $n  $(grep -m1 "^$tag:" "$B/$n.log")"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|Assertion|ERROR|Error" "$B/$n.log" || true; fail=1
  fi
}
for ((s = 1; s <= NS; s++)); do
  run npu_mma_tb mma_s$s MMA +seed=$s +tiles=$((NT * 2))
  run npu_gemm_cu_tb cu_s$s GEMMCU +seed=$s +cmds=$NT
  run npu_core_tb core_s$s NPUCORE +seed=$s +cmds=$NT
done
run npu_core_tb perf PERF +seed=1 +perf +min_util=$MIN_UTIL
for pk in 64 256 1024; do run npu_core_tb perf_k$pk PERF +seed=1 +perf +perf_k=$pk; done   # 較短 K 的利用率 (只報告)
run npu_core_tb long NPUCORE +seed=1 +long
# 豁免: 共用 (dv/npu/waivers.txt) + 組態專屬 (dv/npu/waivers_<cfg>.txt: 參數造成的結構常數)
W=$B/waivers.txt; cat dv/npu/waivers.txt dv/npu/waivers_$CFG.txt > "$W" 2>/dev/null || true
echo "=== RTL structural coverage (module-level; line / branch / expr / toggle / fsm)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/(ai/(pe/npu_mma|cluster/npu_gemm_cu|memory/npu_l2_xbar|npu_core)|common/clock_gate)' \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/npu/' --types covergroup,user \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### $CFG PASS" || { echo "##### $CFG FAIL"; exit 1; }
