#!/bin/bash
# NPU Subsystem (orca_npu_ss) coverage closure, 每個產品組態各自收斂:
#   npu_ss_tb (orca_npu_ss + npu_dma + npu_core) 的 CRV 多個 seed,
#   量測 subsystem 自身的 RTL (orca_npu_ss / npu_dma / clock_gate) 與 dv/npu/npu_ss_sva 的 covergroup / SVA cover;
#   npu_core 子樹在 dv/scripts/run_npu_cov.sh 收斂 (此處仍帶著其 SVA 一起執行)。
#   line / branch / expr / toggle / covergroup / SVA cover 套用 dv/npu/waivers*.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_ss_cov.sh <nano|x1|pro|all> [seeds=6] [batches=30]
# 選項 (環境變數; 同 run_npu_cov.sh): RESUME=1 (原始碼相同不重建、已 PASS 的執行不重跑)、ONLY=<regex> (只執行
#   名稱符合者: s1 / rand2 / perf13 ...)、NOREPORT=1 (不合併 coverage)、REPORT_ONLY=1 (只合併已有的 cov_*.dat)
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NS=${2:-6}; NBT=${3:-30}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NS" "$NBT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG"
B=build/npu_ss_cov_$CFG; [ "${RESUME:-0}${REPORT_ONLY:-0}" = 00 ] && rm -rf "$B"; mkdir -p "$B"
echo "##### NPU subsystem config $CFG: N=$N NCU=$NCU NB=$NB DEPTH=$DEPTH DMA OUT=$TILE_OUT"
SRC="-Irtl/common -Irtl/fp rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv
  rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv
  rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv dv/vec/npu_vec_eng_sva.sv
  rtl/npu/npu_l2_arb2.sv dv/npu/npu_l2_arb2_sva.sv dv/npu/npu_rr_lock_sva.sv
  rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv
  rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv rtl/npu/npu_dma.sv rtl/npu/npu_rr_lock.sv rtl/npu/npu_dma_mc.sv rtl/npu/orca_npu_ss.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv
  dv/npu_kv/npu_kv_pager_sva.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_llm/npu_llm_eng_sva.sv
  dv/npu/npu_ss_sva.sv dv/npu/npu_ss_e2e_sva.sv dv/npu/npu_ss_tb.sv"
if [ "${REPORT_ONLY:-0}" != 1 ]; then
# 建置印記: 參數 + 原始碼內容的雜湊; RESUME 時相同就不重建
stamp=$( { echo "$N $NCU $NB $DEPTH $TILE_OUT"; cat $(printf '%s\n' $SRC | grep -v '^-'); } | sha1sum | cut -c1-40)
if [ "${RESUME:-0}" = 1 ] && [ -x "$B/tb/Vnpu_ss_tb" ] && [ "$(cat "$B/tb.stamp" 2>/dev/null)" = "$stamp" ]; then
  echo "(reuse build)"
else
  rm -f "$B/tb.stamp"
verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert --coverage --coverage-max-width 32768 \
  -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/tb" -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT $SRC \
  --top-module npu_ss_tb > "$B/build.log" 2>&1 || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
  echo "$stamp" > "$B/tb.stamp"
fi
python3 dv/scripts/wide_signals.py "$B/tb" 32768 npu_ss_tb__DOT__dut__DOT__ | tee "$B/toggle_excluded.txt"
fi
fail=0
run() {   # name tag plusargs...
  local n=$1 tag=$2; shift 2
  [ -n "${ONLY:-}" ] && ! [[ $n =~ ^(${ONLY})$ ]] && return
  if [ "${RESUME:-0}" = 1 ] && [ -f "$B/cov_$n.dat" ] && grep -q "TB PASS" "$B/$n.log" 2>/dev/null; then
    echo "PASS $n  (cached) $(grep -m1 "^$tag:" "$B/$n.log")"; return
  fi
  rm -f "$B/cov_$n.dat"
  if (cd "$B" && ./tb/Vnpu_ss_tb "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log"; then
    echo "PASS $n  $(grep -m1 "^$tag:" "$B/$n.log")"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|Assertion|ERROR|Error" "$B/$n.log" || true; fail=1
  fi
}
if [ "${REPORT_ONLY:-0}" != 1 ]; then
for ((s = 1; s <= NS; s++)); do run s$s NPUSS +seed=$s +batches=$NBT; done
# coverage-driven: 32-bit 隨機 seed (每次執行不同; 以 CDV_SEEDS="a b" 重現), 依端到端 coverage 的未命中 bin 調整激勵直到收斂
for cs in ${CDV_SEEDS:-$(od -An -tu4 -N8 /dev/urandom)}; do
  run cdv$cs CDV +seed=$cs +batches=$((NBT * 4)) +cdv +no_big
  [ -f "$B/cdv$cs.log" ] && ! grep -q "TB PASS" "$B/cdv$cs.log" && echo "  (rerun: CDV_SEEDS=$cs)"
done
for r in 1 2; do   # 上電時未 reset 的儲存為隨機值 (Verilator 隨機初值)
  run rand$r NPUSS +seed=$((r + 50)) +batches=$((NBT / 2)) +verilator+rand+reset+2 +verilator+seed+$r
done
for lat in 0 $((TILE_OUT - 3)) $((TILE_OUT * 2)) 100; do   # DMA 吞吐量 (只報告): min(1, OUT / (L + 3)) line/拍
  run perf$lat PERFDMA +seed=1 +perf_dma=$lat
done
fi
if [ "${NOREPORT:-0}" = 1 ]; then [ "$fail" -eq 0 ] && echo "##### $CFG runs PASS (report skipped)" && exit 0; echo "##### $CFG FAIL"; exit 1; fi
W=$B/waivers.txt; cat dv/npu/waivers_ss.txt dv/npu/waivers_ss_$CFG.txt > "$W" 2>/dev/null || true
echo "=== RTL structural coverage (module-level; line / branch / expr / toggle)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/(npu/(orca_npu_ss|npu_dma|npu_dma_mc|npu_rr_lock|npu_l2_arb2)|common/clock_gate)' --expect rtl/npu/orca_npu_ss.sv --expect rtl/npu/npu_dma.sv \
  --expect rtl/npu/npu_l2_arb2.sv --expect rtl/npu/npu_dma_mc.sv --expect rtl/npu/npu_rr_lock.sv \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/npu/npu_(ss|ss_e2e|l2_arb2|rr_lock)_sva' --types covergroup,user \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### $CFG PASS" || { echo "##### $CFG FAIL"; exit 1; }
