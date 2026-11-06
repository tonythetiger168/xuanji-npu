#!/bin/bash
# NPU CRV seed sweep: 一次編譯 (帶 --assert, 不帶 coverage), 平行跑大量 32-bit 隨機 seed, 列出失敗的 seed。
#   seed 預設由 /dev/urandom 產生 (每次執行不同), 全部記錄在 $B/seeds.txt; 以 SEEDS="s1 s2 ..." 指定則重跑那些 seed。
#   WAVES=1: 失敗的 seed (最多 WAVES_MAX 個, 預設 3) 以 --trace-fst 另外編譯的版本重跑, 波形存成 $B/wave_<seed>.fst。
#   CDV=1: 每個 seed 都以 +cdv (coverage-driven) 執行 (只對 ss 有意義)。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_seeds.sh <core|ss> <nano|x1|pro|max> [seeds=1000] [batches=10]  (core: 命令數 = batches × 10)
#   JOBS=平行執行數 (預設 nproc)
set -uo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
KIND=${1:?core|ss}; CFG=${2:?cfg}; NSEED=${3:-1000}; NBT=${4:-10}
npu_cfg "$CFG" || exit 1
P=${JOBS:-$(nproc)}
B=build/npu_seeds_${KIND}_$CFG; mkdir -p "$B"
CU="rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv"
if [ "$KIND" = core ]; then
  SRC="-Irtl/common rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv $CU
       rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv"
  TOP=npu_core_tb; TAG=NPUCORE; ARGS="+cmds=$((NBT * 10))"   # core TB: 命令數 = batches × 10
else
  SRC="-Irtl/common -Irtl/fp rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv
       rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv
       rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv dv/vec/npu_vec_eng_sva.sv
       rtl/npu/npu_l2_arb2.sv dv/npu/npu_l2_arb2_sva.sv $CU rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv
       rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv rtl/npu/npu_dma.sv rtl/npu/orca_npu_ss.sv dv/npu/npu_ss_sva.sv
       dv/npu/npu_ss_e2e_sva.sv dv/npu/npu_ss_tb.sv -GOUT=$TILE_OUT"
  TOP=npu_ss_tb; TAG=NPUSS; ARGS="+batches=$NBT +no_big"
  [ "${CDV:-0}" = 1 ] && ARGS="$ARGS +cdv"
fi
VF="--binary --timing -j ${BJOBS:-4} --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH"
build() {   # dir extra-flags
  # shellcheck disable=SC2086
  verilator $VF $2 --Mdir "$1" $SRC --top-module $TOP > "$1.log" 2>&1 || { tail -20 "$1.log"; echo "BUILD FAIL"; exit 1; }
}
echo "##### NPU $KIND seed sweep, config $CFG: N=$N NCU=$NCU NB=$NB, $NSEED seeds x $NBT batches, $P parallel"
t0=$(date +%s)
build "$B/tb" ""
echo "build: $(( $(date +%s) - t0 )) s"
if [ -n "${SEEDS:-}" ]; then printf '%s\n' $SEEDS > "$B/seeds.txt"
else od -An -tu4 -N$((4 * NSEED)) /dev/urandom | tr -s ' ' '\n' | sed '/^$/d' > "$B/seeds.txt"; fi
rm -rf "$B/logs"; mkdir -p "$B/logs"
export B TOP ARGS
t1=$(date +%s)
# shellcheck disable=SC2016
xargs -P "$P" -I{} sh -c '"$B/tb/V$TOP" +seed={} $ARGS > "$B/logs/{}.log" 2>&1; \
  if grep -q "TB PASS" "$B/logs/{}.log" && ! grep -q "Assertion failed" "$B/logs/{}.log"; then rm -f "$B/logs/{}.log"; fi' < "$B/seeds.txt"
ls "$B/logs" | sed 's/\.log$//' > "$B/fails.txt"
nf=$(wc -l < "$B/fails.txt"); n=$(wc -l < "$B/seeds.txt")
echo "ran $n seeds in $(( $(date +%s) - t1 )) s: $((n - nf)) pass, $nf fail (seeds: $B/seeds.txt)"
if [ "$nf" -ne 0 ]; then
  echo "failing seeds (rerun: SEEDS=\"...\" $0 $KIND $CFG):"; head -20 "$B/fails.txt"
  for s in $(head -5 "$B/fails.txt"); do echo "--- seed $s"; grep -m3 -E "MISMATCH|Assertion|ERROR|Error|FAIL" "$B/logs/$s.log" || tail -3 "$B/logs/$s.log"; done
  if [ "${WAVES:-0}" = 1 ]; then
    build "$B/tb_waves" "--trace-fst --trace-structs +define+WAVES"
    for s in $(head -"${WAVES_MAX:-3}" "$B/fails.txt"); do
      # shellcheck disable=SC2086
      "$B/tb_waves/V$TOP" +seed="$s" $ARGS +dump="$B/wave_$s.fst" > "$B/logs/$s.waves.log" 2>&1
      echo "waveform: $B/wave_$s.fst"
    done
  fi
  exit 1
fi
