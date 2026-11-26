#!/bin/bash
# NPU 大組態 (max) 的 CI smoke: 不帶 coverage 編譯 (帶 --assert), 跑一個短的 CRV seed。
# max (smoke 只建 MAX_SMOKE_NCU 個 CU, 見下) 的 coverage 建置需 ~9 GB (Verilator 前端) 且完整 CRV 遠超 CI 時限; 100% coverage / mutation 由
# nano / x1 / pro 三個組態收斂 (同一份 RTL, 只有參數不同), max 只確認能編譯並正確執行。
# 需求: Verilator >= 5.052
# 用法: dv/scripts/run_npu_smoke.sh <core|ss> <cfg> [cmds=4]
set -euo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
KIND=${1:?core|ss}; CFG=${2:?cfg}; NC=${3:-4}
npu_cfg "$CFG"
# v6.3.16 起每個 PE 有自己的 FP32 加法器 (npu_pe_mac): max 每個 CU (32×32 PE) 的 Verilator 前端約 1.2 GB、
# C++ 編譯數分鐘, 24 個 CU 超過 15 GB (CI runner 與本機都被 OOM 終止)。CU 都是同一模組 (參數相同), smoke 只建
# MAX_SMOKE_NCU (預設 2) 個 CU: 仍驗 N=32 的資料路徑與多 CU 分派 / L2 仲裁。
if [ "$CFG" = max ]; then NCU=${MAX_SMOKE_NCU:-2}; fi
B=build/npu_smoke_${KIND}_$CFG; rm -rf "$B"; mkdir -p "$B"
VF="--binary --timing -j ${JOBS:-2} --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style --Mdir $B/tb"
CU0="rtl/npu/npu_gemm_cu.sv dv/npu/npu_gemm_cu_sva.sv"
CU="rtl/fp/fp_from_int.sv rtl/fp/fp_add.sv $CU0"   # npu_gemm_cu 的 MX 寫回用到 fp_from_int / fp_add (npu_ss 的檔案清單已含)
echo "##### NPU $KIND smoke, config $CFG: N=$N NCU=$NCU NB=$NB DEPTH=$DEPTH"
t0=$(date +%s)
if [ "$KIND" = core ]; then
  # shellcheck disable=SC2086
  verilator $VF -Irtl/common -Idv/npu rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_pe_mac.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv \
    -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH $CU rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
    rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv --top-module npu_core_tb > "$B/build.log" 2>&1 \
    || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
  RUN=("$B/tb/Vnpu_core_tb" +seed=1 +smoke +cmds="$NC"); TAG=NPUCORE
else
  # shellcheck disable=SC2086
  verilator $VF -Irtl/common -Idv/npu -Irtl/fp rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_pe_mac.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv \
    rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv \
    rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv dv/vec/npu_vec_eng_sva.sv \
    rtl/npu/npu_l2_arb2.sv dv/npu/npu_l2_arb2_sva.sv $CU0 rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv \
    rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv rtl/npu/npu_dma.sv rtl/npu/npu_rr_lock.sv rtl/npu/npu_dma_mc.sv rtl/npu/orca_npu_ss.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv dv/npu/npu_ss_sva.sv dv/npu/npu_ss_e2e_sva.sv dv/npu/npu_ss_tb.sv \
    -GN=$N -GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH -GOUT=$TILE_OUT --top-module npu_ss_tb > "$B/build.log" 2>&1 \
    || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
  RUN=("$B/tb/Vnpu_ss_tb" +seed=1 +batches=1 +cmds="$NC" +no_big); TAG=NPUSS
fi
t1=$(date +%s); echo "build ${KIND} $CFG: $((t1 - t0)) s"
if "${RUN[@]}" > "$B/run.log" 2>&1 && grep -q "TB PASS" "$B/run.log"; then
  echo "PASS $KIND $CFG smoke  $(grep -m1 "^$TAG:" "$B/run.log" || true)  ($(( $(date +%s) - t1 )) s)"
else
  echo "FAIL $KIND $CFG smoke"; grep -m5 -E "MISMATCH|Assertion|ERROR|Error" "$B/run.log" || tail -5 "$B/run.log"; exit 1
fi
