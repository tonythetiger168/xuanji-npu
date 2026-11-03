#!/bin/bash
# NPU 向量核 (rtl/npu/npu_vec.sv) 驗證: 多個 seed 的受約束隨機指令串 (seed 1 另含 FP8 / FP4 碼 × 全部比例、
# 16 bit 碼的窮舉解碼) + kernel 程式 (softmax / LayerNorm / RMSNorm / GELU / SiLU, vec_kernels.py), 期望值來自
# dv/vec/vec_ref.py; coverage (line / branch / expr / toggle 套用 dv/vec/waivers.txt 後 100%, covergroup + SVA cover 100%)。
#   rtl/fp 的 fp_add / fp_mul / sfu / fp_narrow / fp_widen 在 run_fp.sh 收斂, 這裡只統計新的元件。
# 需求: Verilator >= 5.052, Python 3 + numpy
# 用法: dv/scripts/run_vec.sh [seeds=4] [n_ins=6000]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NI=${2:-6000}
B=build/vec_cov; rm -rf "$B"; mkdir -p "$B"
FP="rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv"
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style -Irtl/common -Irtl/fp --Mdir "$B/obj" \
  rtl/common/orca_pkg.sv rtl/common/clock_gate.sv $FP rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv dv/vec/npu_vec_tb.sv \
  --top-module npu_vec_tb > "$B/build.log" 2>&1 || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style -Irtl/common -Irtl/fp --Mdir "$B/eobj" \
  rtl/common/orca_pkg.sv rtl/common/clock_gate.sv $FP rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv \
  dv/vec/npu_vec_eng_sva.sv dv/vec/npu_vec_eng_tb.sv --top-module npu_vec_eng_tb > "$B/build_eng.log" 2>&1 \
  || { tail -20 "$B/build_eng.log"; echo "BUILD FAIL eng"; exit 1; }
fail=0
run() {   # 名稱 base plusargs...  (EXE 預設 obj/Vnpu_vec_tb)
  local n=$1 v=$2; shift 2
  if (cd "$B" && ./${EXE:-obj/Vnpu_vec_tb} +vec=$v "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log" \
     && ! grep -q "Assertion failed" "$B/$n.log"; then
    echo "PASS $n  $(grep -m1 -E '^(VEC|VECENG):' "$B/$n.log")"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|Assertion|ERROR" "$B/$n.log" || true; fail=1
  fi
}
for ((s = 1; s <= NS; s++)); do
  python3 dv/vec/gen_vec_vectors.py "$B/v$s" $s "$NI" > /dev/null
  run s$s v$s +seed=$s
done
python3 dv/vec/vec_kernels.py "$B" | tee "$B/kernels.txt" || fail=1
for k in $(awk '/^KERNEL/ {print $2}' "$B/kernels.txt"); do
  run k_$k k_$k +seed=3
  run perf_$k k_$k +perf
done
# 向量引擎 (取指 / 預取 / 與 LD 共用讀取埠 / 命令完成): 多個命令的程式, 隨機 grant
for ((s = 1; s <= NS; s++)); do
  python3 dv/vec/gen_eng_vectors.py "$B/e$s" $s 60 > /dev/null
  EXE=eobj/Vnpu_vec_eng_tb run eng_s$s e$s +seed=$s +gnt_p=$((30 + 20 * s))
done
EXE=eobj/Vnpu_vec_eng_tb run eng_perf e1 +perf
# 上電時未 reset 的儲存為隨機值
for r in 1 2; do EXE=eobj/Vnpu_vec_eng_tb run eng_rand$r e$r +seed=$((r + 50)) +gnt_p=50 +verilator+rand+reset+2 +verilator+seed+$r; done
echo "=== RTL structural coverage (line / branch / expr / toggle)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/(npu/npu_vec(_eng)?|fp/(fp_max|fp_small_dec|fp_mx_scale|fp_from_int))\.sv' \
  --expect rtl/npu/npu_vec.sv --expect rtl/npu/npu_vec_eng.sv --expect rtl/fp/fp_max.sv --expect rtl/fp/fp_small_dec.sv --expect rtl/fp/fp_mx_scale.sv --expect rtl/fp/fp_from_int.sv \
  --waivers dv/vec/waivers.txt --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/vec/npu_vec(_eng)?_sva' --types covergroup,user \
  --expect dv/vec/npu_vec_sva.sv --expect dv/vec/npu_vec_eng_sva.sv --waivers dv/vec/waivers.txt --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### vec PASS" || { echo "##### vec FAIL"; exit 1; }
