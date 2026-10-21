#!/bin/bash
# 共用浮點函式庫 (rtl/fp) 驗證: 多個 seed 的向量 (位元精確參考 dv/fp/fp_ref.py) + coverage 收斂。
#   line / branch / expr / toggle 套用 dv/fp/waivers.txt 後要求 100%。
# 需求: Verilator >= 5.052, Python 3
# 用法: dv/scripts/run_fp.sh [seeds=4] [n_random=20000]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NR=${2:-20000}
B=build/fp_cov; rm -rf "$B"; mkdir -p "$B"
python3 dv/fp/fp_ref.py 20000 > "$B/ref_selfcheck.log" || { cat "$B/ref_selfcheck.log"; echo "REF SELF-CHECK FAIL"; exit 1; }
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style --Mdir "$B/obj" \
  rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv dv/fp/fp_tb.sv --top-module fp_tb > "$B/build.log" 2>&1 \
  || { tail -20 "$B/build.log"; echo "BUILD FAIL"; exit 1; }
fail=0
for ((s = 1; s <= NS; s++)); do
  python3 dv/fp/gen_vectors.py "$B/vec_s$s" $s "$NR"
  if (cd "$B" && ./obj/Vfp_tb +vec=vec_s$s +verilator+coverage+file+cov_s$s.dat > s$s.log 2>&1) && grep -q "TB PASS" "$B/s$s.log"; then
    echo "PASS s$s  $(grep -m1 '^FP:' "$B/s$s.log")"
  else
    echo "FAIL s$s"; grep -m5 -E "MISMATCH|ERROR" "$B/s$s.log" || true; fail=1
  fi
done
echo "=== RTL structural coverage (line / branch / expr / toggle)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/fp/' \
  --expect rtl/fp/fp_add.sv --expect rtl/fp/fp_mul.sv --expect rtl/fp/fp_narrow.sv --expect rtl/fp/fp_widen.sv \
  --waivers dv/fp/waivers.txt --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### fp PASS" || { echo "##### fp FAIL"; exit 1; }
