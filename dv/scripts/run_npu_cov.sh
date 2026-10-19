#!/bin/bash
# NPU 運算核心 (npu_mma) coverage closure: CRV 多個 seed, 合併 line / branch / expr / toggle / FSM /
# SVA cover / covergroup, 套用 dv/npu/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_cov.sh [seeds=4] [tiles=400]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NT=${2:-400}
B=build/npu_cov; rm -rf "$B"; mkdir -p "$B"
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage --coverage-max-width 32768 \
  -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/obj" \
  -Irtl/common rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/ai/pe/npu_mma.sv dv/npu/npu_mma_sva.sv dv/npu/npu_mma_tb.sv \
  --top-module npu_mma_tb > "$B/build.log" 2>&1 || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
fail=0
for ((s = 1; s <= NS; s++)); do
  if (cd "$B" && ./obj/Vnpu_mma_tb +seed=$s +tiles=$NT +verilator+coverage+file+cov_s$s.dat > s$s.log 2>&1) && grep -q "TB PASS" "$B/s$s.log"; then
    echo "PASS s$s  $(grep -m1 '^MMA:' "$B/s$s.log")"
  else
    echo "FAIL s$s"; grep -m5 -E "MISMATCH|Assertion|ERROR|Error" "$B/s$s.log" || true; fail=1
  fi
done
W=dv/npu/waivers.txt; [ -f "$W" ] || W=/dev/null
echo "=== RTL structural coverage (line / branch / expr / toggle / fsm) ==="
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^rtl/ai/pe/npu_mma' --waivers "$W" --require 100 || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user) ==="
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^dv/npu/' --types covergroup,user --waivers "$W" --require 100 || fail=1
[ "$fail" -eq 0 ]
