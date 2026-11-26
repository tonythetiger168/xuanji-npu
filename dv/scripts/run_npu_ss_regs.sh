#!/bin/bash
# NPU Subsystem APB 暫存器 (rtl/npu/npu_ss_regs.sv) 驗證: 受約束隨機 APB 對參考模型 (多個 seed) + orca_npu_ss_apb 整合測試
# (只經 APB 操作真的 orca_npu_ss); coverage (line / branch / expr / toggle 套用 dv/npu/waivers_regs.txt 後 100%,
# covergroup + SVA cover 100%)。暫存器表見 docs/apb.md。
# 用法: dv/scripts/run_npu_ss_regs.sh [seeds=4] [cycles=200000]
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${1:-4}; NC=${2:-200000}
B=build/npu_ss_regs; rm -rf "$B"; mkdir -p "$B"
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style -Irtl/common --Mdir "$B/u" \
  rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/common/ret_shadow.sv rtl/npu/npu_ss_regs.sv dv/npu/npu_ss_regs_sva.sv dv/npu/npu_ss_regs_tb.sv --top-module npu_ss_regs_tb \
  > "$B/build_u.log" 2>&1 || { tail -20 "$B/build_u.log"; echo "BUILD FAIL"; exit 1; }
verilator --binary --timing -j "${JOBS:-4}" --assert --coverage -Wno-fatal -Wno-lint -Wno-style -Irtl/common -Irtl/fp --Mdir "$B/i" \
  rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_pe_mac.sv rtl/npu/npu_mma.sv dv/npu/npu_mma_sva.sv rtl/npu/npu_gemm_cu.sv \
  rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv \
  rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv dv/vec/npu_vec_sva.sv rtl/npu/npu_vec_eng.sv \
  dv/vec/npu_vec_eng_sva.sv rtl/npu/npu_l2_arb2.sv \
  dv/npu/npu_gemm_cu_sva.sv rtl/npu/npu_l2_xbar.sv dv/npu/npu_l2_xbar_sva.sv rtl/npu/npu_core.sv dv/npu/npu_core_sva.sv \
  rtl/npu/npu_dma.sv rtl/npu/npu_rr_lock.sv rtl/npu/npu_dma_mc.sv rtl/npu/orca_npu_ss.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv rtl/npu/npu_llm_eng.sv dv/npu/npu_ss_sva.sv dv/npu/npu_ss_e2e_sva.sv rtl/common/ret_shadow.sv rtl/npu/npu_ss_regs.sv dv/npu/npu_ss_regs_sva.sv \
  rtl/npu/orca_npu_ss_apb.sv dv/npu/npu_ss_apb_sva.sv dv/npu/orca_npu_ss_apb_tb.sv --top-module orca_npu_ss_apb_tb > "$B/build_i.log" 2>&1 || { tail -20 "$B/build_i.log"; echo "BUILD FAIL"; exit 1; }
fail=0
run() {   # 名稱 執行檔 plusargs...
  local n=$1 x=$2; shift 2
  if (cd "$B" && ./$x "$@" +verilator+coverage+file+cov_$n.dat > $n.log 2>&1) && grep -q "TB PASS" "$B/$n.log" \
     && ! grep -q "Assertion failed" "$B/$n.log"; then
    echo "PASS $n  $(grep -m1 -E '^(REGS|NPUSSAPB):' "$B/$n.log")"
  else
    echo "FAIL $n"; grep -m5 -E "MISMATCH|Assertion|ERROR" "$B/$n.log" || true; fail=1
  fi
}
for ((s = 1; s <= NS; s++)); do run s$s u/Vnpu_ss_regs_tb +seed=$s +cycles=$NC; done
run int i/Vorca_npu_ss_apb_tb
echo "=== RTL structural coverage (line / branch / expr / toggle)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^rtl/npu/npu_ss_regs\.sv' --expect rtl/npu/npu_ss_regs.sv \
  --waivers dv/npu/waivers_regs.txt --require 100 --list "${LIST:-40}" || fail=1
echo "=== functional coverage (covergroup) + SVA cover (user)"
python3 dv/scripts/cov_report.py "$B"/cov_*.dat --by-module --rtl '^dv/npu/npu_ss_regs_sva' --types covergroup,user \
  --expect dv/npu/npu_ss_regs_sva.sv --waivers dv/npu/waivers_regs.txt --require 100 --list "${LIST:-40}" || fail=1
[ "$fail" -eq 0 ] && echo "##### npu_ss_regs PASS" || { echo "##### npu_ss_regs FAIL"; exit 1; }
