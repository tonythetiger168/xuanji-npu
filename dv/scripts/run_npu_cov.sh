#!/bin/bash
# NPU 運算核心 (npu_mma) 與 GEMM 計算單元 (npu_gemm_cu) coverage closure: 各自的 CRV 多個 seed, 合併 line / branch / expr / toggle / FSM /
# SVA cover / covergroup, 套用 dv/npu/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_cov.sh [seeds=4] [tiles=400]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NT=${2:-400}
B=build/npu_cov; rm -rf "$B"; mkdir -p "$B"
SRC="-Irtl/common rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/ai/pe/npu_mma.sv dv/npu/npu_mma_sva.sv"
build() {   # top extra-sources...
  local top=$1; shift
  verilator --binary --timing -j "${JOBS:-4}" --assert --coverage --coverage-max-width 32768 \
    -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/$top" $SRC "$@" --top-module $top > "$B/$top.build.log" 2>&1 \
    || { tail -20 "$B/$top.build.log"; echo "BUILD FAIL $top"; exit 1; }
}
build npu_mma_tb dv/npu/npu_mma_tb.sv
build npu_gemm_cu_tb rtl/ai/cluster/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv dv/npu/npu_gemm_cu_tb.sv
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
  run npu_mma_tb mma_s$s MMA +seed=$s +tiles=$NT
  run npu_gemm_cu_tb cu_s$s GEMMCU +seed=$s +cmds=$((NT / 2))
done
W=dv/npu/waivers.txt; [ -f "$W" ] || W=/dev/null
echo "=== RTL structural coverage (line / branch / expr / toggle / fsm) ==="
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^rtl/ai/(pe/npu_mma|cluster/npu_gemm_cu)' --waivers "$W" --require 100 || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user) ==="
# npu_mma 的 cover / covergroup 在 npu_mma_tb 量測 (CU 內部不會出現 busy 時 start 等情境; 其斷言在 CU 內仍檢查)
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^dv/npu/' --types covergroup,user \
  --hier-exclude '^npu_gemm_cu_tb\.dut\.u_mma' --waivers "$W" --require 100 || fail=1
[ "$fail" -eq 0 ]
