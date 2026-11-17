#!/bin/bash
# NPU LLM 引擎 (rtl/npu/npu_llm_eng.sv: KV / MOE 命令) 驗證: CRV + L2 模型 + scoreboard (每個命令後比對整個 L2),
# 多個 seed + 隨機上電; coverage: line / branch / expr / toggle (npu_llm_eng) + covergroup + SVA cover 100%。
#   子模組 npu_kv_pager / npu_moe_disp 的 coverage 在各自的腳本收斂 (run_npu_kv.sh / run_npu_moe.sh)。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_llm.sh [seeds=4] [cmds=400]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NC=${2:-400}
B=build/npu_llm; rm -rf "$B"; mkdir -p "$B"
SRC="rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv
     dv/npu_kv/npu_kv_pager_sva.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_llm/npu_llm_eng_sva.sv dv/npu_llm/npu_llm_eng_tb.sv"
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style -Irtl/common --Mdir "$B/obj" \
  $SRC --top-module npu_llm_eng_tb > "$B/build.log" 2>&1 || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
fail=0
run() {
  local n=$1; shift
  if (cd "$B" && ./obj/Vnpu_llm_eng_tb "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log" \
     && ! grep -q "Assertion failed" "$B/$n.log"; then
    echo "PASS $n  $(grep 'LLMENG: KV' "$B/$n.log" | sed 's/LLMENG: //')"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|ERROR|Assertion" "$B/$n.log" || true; fail=1
  fi
}
for ((s = 1; s <= NS; s++)); do run s$s +seed=$s +cmds=$NC; done
for r in 1 2; do run rand$r +seed=$((r + 50)) +cmds=$((NC / 2)) +verilator+rand+reset+2 +verilator+seed+$r; done
echo "=== RTL structural coverage (npu_llm_eng) ==="
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^rtl/npu/npu_llm_eng' --expect rtl/npu/npu_llm_eng.sv --waivers dv/npu_llm/waivers.txt --require 100 || fail=1
echo "=== functional coverage (covergroup) + SVA cover ==="
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --rtl '^dv/npu_llm/' --types covergroup,user --waivers dv/npu_llm/waivers.txt --require 100 || fail=1
[ "$fail" -eq 0 ]
