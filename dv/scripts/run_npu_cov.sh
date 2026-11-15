#!/bin/bash
# NPU coverage closure, 每個產品組態各自收斂:
#   npu_mma_tb (N) + npu_gemm_cu_tb (N) + npu_core_tb (N, NCU, NB, DEPTH) 的 CRV 多個 seed, 加上效能量測,
#   以「模組層級」合併 (同一模組在不同 TB / 不同 instance 的同一點合併; generate 範圍仍分開),
#   line / branch / expr / toggle / FSM / SVA cover / covergroup 套用 dv/npu/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_cov.sh <nano|x1|pro|all> [seeds=4] [count=200]
# 選項 (環境變數, 供長時間的組態 (Max) 續跑或分割到多個 CI job):
#   RESUME=1   保留建置目錄: 原始碼雜湊與上次相同的 TB 不重建, 已 PASS 且有 coverage 檔的執行不重跑
#   ONLY=<re>  只執行名稱符合 regex 的執行 (mma_s1 / cu_rand2 / core_s3 / perf_k64 / long ...)
#   NOREPORT=1 不合併 coverage (分割執行時, 由最後一個 job 合併所有 cov_*.dat)
#   REPORT_ONLY=1 不建置、不執行, 只合併建置目錄中已有的 cov_*.dat
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NS=${2:-4}; NT=${3:-200}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NS" "$NT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG"
B=build/npu_cov_$CFG; [ "${RESUME:-0}${REPORT_ONLY:-0}" = 00 ] && rm -rf "$B"; mkdir -p "$B"
echo "##### NPU config $CFG: N=$N NCU=$NCU NB=$NB DEPTH=$DEPTH (peak $((N * N * NCU)) MAC/cycle)"
SRC_FILES="rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv"
SRC="-Irtl/common -Idv/npu $SRC_FILES"
CU="rtl/fp/fp_from_int.sv rtl/fp/fp_add.sv rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv"
build() {   # top -G... -- sources...
  local top=$1; shift
  # 建置印記: 參數 + 原始碼內容的雜湊; RESUME 時相同就不重建
  local stamp; stamp=$( { echo "$*"; cat $SRC_FILES $(printf '%s\n' "$@" | grep -v '^-'); } | sha1sum | cut -c1-40)
  if [ "${RESUME:-0}" = 1 ] && [ -x "$B/$top/V$top" ] && [ "$(cat "$B/$top.stamp" 2>/dev/null)" = "$stamp" ]; then
    echo "(reuse build $top)"; return
  fi
  rm -f "$B/$top.stamp"
  verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert --coverage --coverage-max-width 32768 \
    -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/$top" $SRC "$@" --top-module $top > "$B/$top.build.log" 2>&1 \
    || { tail -20 "$B/$top.build.log"; echo "BUILD FAIL $top"; exit 1; }
  echo "$stamp" > "$B/$top.stamp"
}
fail=0
if [ "${REPORT_ONLY:-0}" != 1 ]; then
# FSM coverage (state / arc): pyslang 依本組態的參數抽出各 DUT 的狀態機, 產生 SVA cover (dv/scripts/gen_fsm_cov.py)
NPU_RTL="rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/fp/fp_from_int.sv rtl/fp/fp_add.sv rtl/npu/npu_mma.sv rtl/npu/npu_gemm_cu.sv rtl/npu/npu_l2_xbar.sv rtl/npu/npu_core.sv"
python3 dv/scripts/gen_fsm_cov.py -I rtl/common $NPU_RTL --top npu_core -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH \
  --split -o "$B" --json "$B/fsm.json" > "$B/fsm.txt"
F=$B/fsm_npu_mma.sv; FC="$F $B/fsm_npu_gemm_cu.sv"
build npu_mma_tb -GN=$N dv/npu/npu_mma_tb.sv $F
build npu_gemm_cu_tb -GN=$N $CU dv/npu/npu_gemm_cu_tb.sv $FC
build npu_core_tb -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH $CU rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
  rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv $FC
python3 dv/scripts/wide_signals.py "$B/npu_core_tb" 32768 npu_core_tb__DOT__dut__DOT__ | tee "$B/toggle_excluded.txt"
run() {   # top name tag plusargs...
  local top=$1 n=$2 tag=$3; shift 3
  [ -n "${ONLY:-}" ] && ! [[ $n =~ ^(${ONLY})$ ]] && return
  if [ "${RESUME:-0}" = 1 ] && [ -f "$B/cov_$n.dat" ] && grep -q "TB PASS" "$B/$n.log" 2>/dev/null; then
    echo "PASS $n  (cached) $(grep -m1 "^$tag:" "$B/$n.log")"; return
  fi
  rm -f "$B/cov_$n.dat"
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
# 上電時未 reset 的儲存為隨機值 (Verilator 隨機初值; 被閘控的 flop 在 reset 期間也要收到時脈才會被 reset)
for r in 1 2; do
  run npu_mma_tb mma_rand$r MMA +seed=$((r + 50)) +tiles=$NT +verilator+rand+reset+2 +verilator+seed+$r
  run npu_gemm_cu_tb cu_rand$r GEMMCU +seed=$((r + 50)) +cmds=$((NT / 2)) +verilator+rand+reset+2 +verilator+seed+$r
  run npu_core_tb core_rand$r NPUCORE +seed=$((r + 50)) +cmds=$((NT / 2)) +verilator+rand+reset+2 +verilator+seed+$r
done
run npu_core_tb perf PERF +seed=1 +perf +min_util=$MIN_UTIL
for pk in 64 256 1024; do run npu_core_tb perf_k$pk PERF +seed=1 +perf +perf_k=$pk; done   # 較短 K 的利用率 (只報告)
run npu_core_tb long NPUCORE +seed=1 +long
fi
if [ "${NOREPORT:-0}" = 1 ]; then [ "$fail" -eq 0 ] && echo "##### $CFG runs PASS (report skipped)" && exit 0; echo "##### $CFG FAIL"; exit 1; fi
# 豁免: 共用 (dv/npu/waivers.txt) + 組態專屬 (dv/npu/waivers_<cfg>.txt: 參數造成的結構常數)
W=$B/waivers.txt; cat dv/npu/waivers.txt dv/npu/waivers_$CFG.txt > "$W" 2>/dev/null || true
echo "=== RTL structural coverage (module-level; line / branch / expr / toggle / fsm)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl "^(rtl/(npu/(npu_mma|npu_gemm_cu|npu_l2_xbar|npu_core)|common/clock_gate)|$B/fsm_)" --expect rtl/npu/npu_mma.sv --expect rtl/npu/npu_gemm_cu.sv --expect rtl/npu/npu_l2_xbar.sv --expect rtl/npu/npu_core.sv \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/npu/npu_(mma|gemm_cu|l2_xbar|core)_sva' --types covergroup,user \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### $CFG PASS" || { echo "##### $CFG FAIL"; exit 1; }
