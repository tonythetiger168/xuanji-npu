#!/bin/bash
# NPU Subsystem (orca_npu_ss) coverage closure, 每個產品組態各自收斂:
#   npu_ss_tb (orca_npu_ss + npu_ss_dma + npu_core) 的 CRV 多個 seed,
#   量測 subsystem 自身的 RTL (orca_npu_ss / npu_ss_dma / clock_gate) 與 dv/npu/npu_ss_sva 的 covergroup / SVA cover;
#   npu_core 子樹在 dv/scripts/run_npu_cov.sh 收斂 (此處仍帶著其 SVA 一起執行)。
#   line / branch / expr / toggle / covergroup / SVA cover 套用 dv/npu/waivers*.txt 後要求 100%。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_ss_cov.sh <nano|x1|pro|all> [seeds=6] [batches=30]
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NS=${2:-6}; NBT=${3:-30}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NS" "$NBT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG"
B=build/npu_ss_cov_$CFG; rm -rf "$B"; mkdir -p "$B"
echo "##### NPU subsystem config $CFG: N=$N NCU=$NCU NB=$NB DEPTH=$DEPTH DMA OUT=$TILE_OUT"
SRC="-Irtl/common rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv
  rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv
  rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv rtl/npu/npu_ss_dma.sv rtl/npu/orca_npu_ss.sv
  dv/npu/npu_ss_sva.sv dv/npu/npu_ss_tb.sv"
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage --coverage-max-width 32768 \
  -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/tb" -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT $SRC \
  --top-module npu_ss_tb > "$B/build.log" 2>&1 || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
python3 dv/scripts/wide_signals.py "$B/tb" 32768 npu_ss_tb__DOT__dut__DOT__ | tee "$B/toggle_excluded.txt"
fail=0
for ((s = 1; s <= NS; s++)); do
  if (cd "$B" && ./tb/Vnpu_ss_tb +seed=$s +batches=$NBT +verilator+coverage+file+cov_s$s.dat > s$s.log 2>&1) && grep -q "TB PASS" "$B/s$s.log"; then
    echo "PASS s$s  $(grep -m1 '^NPUSS:' "$B/s$s.log")"
  else
    echo "FAIL s$s"; grep -m5 -E "MISMATCH|Assertion|ERROR|Error" "$B/s$s.log" || true; fail=1
  fi
done
for lat in 0 $((TILE_OUT - 3)) $((TILE_OUT * 2)) 100; do   # DMA 吞吐量 (只報告): min(1, OUT / (L + 3)) line/拍
  if (cd "$B" && ./tb/Vnpu_ss_tb +seed=1 +perf_dma=$lat +verilator+coverage+file+cov_perf$lat.dat > perf$lat.log 2>&1) && grep -q "TB PASS" "$B/perf$lat.log"; then
    echo "PASS perf_dma lat=$lat  $(grep -m1 '^PERFDMA:' "$B/perf$lat.log")"
  else echo "FAIL perf_dma lat=$lat"; fail=1; fi
done
W=$B/waivers.txt; cat dv/npu/waivers_ss.txt dv/npu/waivers_ss_$CFG.txt > "$W" 2>/dev/null || true
echo "=== RTL structural coverage (module-level; line / branch / expr / toggle)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/(npu/(orca_npu_ss|npu_ss_dma)|common/clock_gate)' --expect rtl/npu/orca_npu_ss.sv --expect rtl/npu/npu_ss_dma.sv \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/npu/npu_ss_sva' --types covergroup,user \
  --waivers "$W" --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### $CFG PASS" || { echo "##### $CFG FAIL"; exit 1; }
