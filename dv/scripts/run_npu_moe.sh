#!/bin/bash
# NPU MoE dispatch (rtl/npu/npu_moe_disp.sv) 驗證: CRV + scoreboard (多個 seed + 隨機上電), 兩組參數;
# coverage: line / branch / expr / toggle (小組態) + covergroup + SVA cover, 套用 dv/npu_moe/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_moe.sh [seeds=4] [cmds=300]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NC=${2:-300}
B=build/npu_moe; rm -rf "$B"; mkdir -p "$B"
VFLAGS="--binary --timing -j ${JOBS:-4} --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common"
SRC="rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_moe_disp.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_moe/npu_moe_disp_tb.sv"
verilator $VFLAGS --coverage --Mdir "$B/small" $SRC --top-module npu_moe_disp_tb > "$B/build_small.log" 2>&1 \
  || { tail -20 "$B/build_small.log"; echo "BUILD FAIL small"; exit 1; }
verilator $VFLAGS --Mdir "$B/big" -GNEXP=64 -GTW=12 -GPW=14 $SRC --top-module npu_moe_disp_tb > "$B/build_big.log" 2>&1 \
  || { tail -20 "$B/build_big.log"; echo "BUILD FAIL big"; exit 1; }
fail=0
run() {   # 名稱 執行檔 plusargs...
  local n=$1 x=$2; shift 2
  if (cd "$B" && ./$x "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log" \
     && ! grep -q "Assertion failed" "$B/$n.log"; then
    echo "PASS $n  $(grep 'MOEDISP: commands' "$B/$n.log" | sed 's/MOEDISP: //')"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|ERROR|Assertion" "$B/$n.log" || true; fail=1
  fi
}
for ((s = 1; s <= NS; s++)); do run small_s$s small/Vnpu_moe_disp_tb +seed=$s +cmds=$NC; done
for r in 1 2; do run small_rand$r small/Vnpu_moe_disp_tb +seed=$((r + 50)) +cmds=$((NC / 2)) +verilator+rand+reset+2 +verilator+seed+$r; done
run big_s1 big/Vnpu_moe_disp_tb +seed=101 +cmds=$((NC / 3))
echo "=== RTL structural coverage (line / branch / expr / toggle) ==="
python3 dv/scripts/cov_report.py "$B"/cov_small_*.dat --rtl '^rtl/npu/npu_moe_disp' --expect rtl/npu/npu_moe_disp.sv --waivers dv/npu_moe/waivers.txt --require 100 || fail=1
echo "=== functional coverage (covergroup) + SVA cover ==="
python3 dv/scripts/cov_report.py "$B"/cov_small_*.dat --rtl '^dv/npu_moe/' --types covergroup,user --waivers dv/npu_moe/waivers.txt --require 100 || fail=1
[ "$fail" -eq 0 ]
