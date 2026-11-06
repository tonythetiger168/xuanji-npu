#!/bin/bash
# NPU paged KV cache 管理 (rtl/npu/npu_kv_pager.sv) 驗證 (issue #2 P0):
#   1. 舊 RTL (rtl/legacy/ai/cloud/npu_kv_cache.sv) 的失敗重現: 預期 FAIL (舊 RTL 未修), 印出各項 BUG 供參考。
#   2. 新 RTL 兩組參數: 小 (NP 16 / NSEQ 4 / NSLOT 8 / BS 4 / RW 2, 碰撞與耗盡頻繁) 與預設 (NP 64 / NSEQ 8 / NSLOT 16 / BS 16 / RW 3);
#      CRV + scoreboard + 後門不變式, 多個 seed + 隨機上電狀態。
#   3. coverage: line / branch / expr / toggle (小組態) + covergroup + SVA cover, 套用 dv/npu_kv/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_kv.sh [seeds=4] [cycles=40000]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NC=${2:-40000}
B=build/npu_kv; rm -rf "$B"; mkdir -p "$B"
VFLAGS="--binary --timing -j ${JOBS:-4} --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common"
SRC="rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_kv_pager.sv dv/npu_kv/npu_kv_pager_sva.sv dv/npu_kv/npu_kv_pager_tb.sv"
fail=0

echo "=== 舊 RTL 失敗重現 (預期: TB FAIL) ==="
verilator $VFLAGS --Mdir "$B/legacy" rtl/common/orca_pkg.sv rtl/legacy/ai/cloud/npu_kv_cache.sv dv/npu_kv/legacy_kv_repro_tb.sv \
  --top-module legacy_kv_repro_tb > "$B/legacy_build.log" 2>&1 || { tail -20 "$B/legacy_build.log"; echo "BUILD FAIL legacy"; exit 1; }
"$B/legacy/Vlegacy_kv_repro_tb" | grep -E '^(BUG|TB)' || true

verilator $VFLAGS --coverage --Mdir "$B/small" $SRC --top-module npu_kv_pager_tb > "$B/build_small.log" 2>&1 \
  || { tail -20 "$B/build_small.log"; echo "BUILD FAIL small"; exit 1; }
verilator $VFLAGS --Mdir "$B/dflt" -GNP=64 -GNSEQ=8 -GNSLOT=16 -GBS=16 -GRW=3 -GPOSW=20 $SRC --top-module npu_kv_pager_tb \
  > "$B/build_dflt.log" 2>&1 || { tail -20 "$B/build_dflt.log"; echo "BUILD FAIL default"; exit 1; }
run() {   # 名稱 執行檔 plusargs...
  local n=$1 x=$2; shift 2
  if (cd "$B" && ./$x "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log" \
     && ! grep -q "Assertion failed" "$B/$n.log"; then
    echo "PASS $n  $(grep 'KVPAGER: rsp' "$B/$n.log" | sed 's/KVPAGER: //')"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|INVARIANT|Assertion|ERROR" "$B/$n.log" || true; fail=1
  fi
}
echo "=== npu_kv_pager CRV ==="
for ((s = 1; s <= NS; s++)); do run small_s$s small/Vnpu_kv_pager_tb +seed=$s +cycles=$NC; done
for r in 1 2; do run small_rand$r small/Vnpu_kv_pager_tb +seed=$((r + 50)) +cycles=$((NC / 2)) +verilator+rand+reset+2 +verilator+seed+$r; done
for ((s = 1; s <= 2; s++)); do run dflt_s$s dflt/Vnpu_kv_pager_tb +seed=$((s + 100)) +cycles=$NC; done
grep -h "^KVPAGER: ALLOC" "$B"/small_s1.log || true
echo "=== RTL structural coverage (line / branch / expr / toggle) ==="
python3 dv/scripts/cov_report.py "$B"/cov_small_*.dat --rtl '^rtl/npu/npu_kv_pager' --expect rtl/npu/npu_kv_pager.sv --waivers dv/npu_kv/waivers.txt --require 100 || fail=1
echo "=== functional coverage (covergroup) + SVA cover ==="
python3 dv/scripts/cov_report.py "$B"/cov_small_*.dat --rtl '^dv/npu_kv/' --types covergroup,user --waivers dv/npu_kv/waivers.txt --require 100 || fail=1
[ "$fail" -eq 0 ]
