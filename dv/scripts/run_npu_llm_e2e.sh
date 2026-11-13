#!/bin/bash
# NPU Subsystem 端到端 (paged attention 的 KV cache、MoE 層; 期望值由 numpy 計算): dv/npu_llm/llm_e2e.py
# 需求: Verilator >= 5.052, python3 + numpy
# 用法: dv/scripts/run_npu_llm_e2e.sh [nano|x1|pro] [seeds=3]
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NS=${2:-3}
npu_cfg "$CFG"
B=build/npu_llm_e2e_$CFG; rm -rf "$B"; mkdir -p "$B"
SRC="-Irtl/common -Irtl/fp rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_mma.sv
  rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv
  rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv rtl/npu/npu_vec_eng.sv rtl/npu/npu_l2_arb2.sv
  rtl/npu/npu_gemm_cu.sv rtl/npu/npu_l2_xbar.sv rtl/npu/npu_core.sv rtl/npu/npu_dma.sv rtl/npu/npu_rr_lock.sv rtl/npu/npu_dma_mc.sv
  rtl/npu/orca_npu_ss.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv
  dv/npu/npu_ss_sva.sv dv/npu/npu_ss_e2e_sva.sv dv/npu_kv/npu_kv_pager_sva.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_llm/npu_llm_eng_sva.sv
  dv/npu_llm/npu_llm_e2e_tb.sv"
# shellcheck disable=SC2086
verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/tb" \
  -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT $SRC --top-module npu_llm_e2e_tb > "$B/build.log" 2>&1 \
  || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
rc=0
for ((s = 1; s <= NS; s++)); do
  echo "##### $CFG seed $s"
  python3 dv/npu_llm/llm_e2e.py "$B/tb/Vnpu_llm_e2e_tb" "$B/s$s" "$N" "$s" || rc=1
done
[ "$rc" -eq 0 ] && echo "##### npu_llm_e2e $CFG PASS" || { echo "##### npu_llm_e2e $CFG FAIL"; exit 1; }
