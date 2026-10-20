#!/bin/bash
# 形式驗證的 mutation 檢查: 每個 mutant 在 RTL 植入一個錯誤, 證明必須失敗 (FAIL 或 UNKNOWN = 歸納找到反例);
# 若 mutant 仍「證明通過」, 代表性質太弱或證明是空的。
# 需求同 run_formal.sh (z3 >= 4.12: 舊版 4.8 在此問題上會當掉)
set -uo pipefail
cd "$(dirname "$0")/../.."
export LANG=C.UTF-8 LC_ALL=C.UTF-8
M=build/formal_mut; rm -rf $M; mkdir -p $M
nk=0; nt=0; surv=0
mut() {   # block rtl name old new
  local b=$1 f=$2 n=$3
  python3 - "$4" "$5" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1)); mkdir -p $M/$n
  sv2v -DFORMAL -E Assert -Irtl/common rtl/common/orca_pkg.sv $M/$n.sv > $M/$n/$b.v
  cp formal/$b/$b.sby $M/$n/
  (cd $M/$n && timeout 1200 sby -f $b.sby prove > sby.log 2>&1)
  if grep -q "DONE (PASS" $M/$n/sby.log; then echo "$n: SURVIVED"; surv=1
  else nk=$((nk + 1)); echo "$n: killed ($(grep -oE 'DONE \([A-Z]+' $M/$n/sby.log))"; fi
}
D=rtl/npu/npu_ss_dma.sv
mut npu_ss_dma $D done_early  "assign done      = act && (wr == n);" "assign done      = act && (iss == n);"
mut npu_ss_dma $D outst_over  "wire can_iss = act && (iss != n) && (outst != (OW+1)'(OUT));" "wire can_iss = act && (iss != n);"
mut npu_ss_dma $D no_rd_qual  "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);" "wire iss_ok = st ? (h_rd_req && h_rd_gnt) : m_rd_gnt;"
mut npu_ss_dma $D waddr_iss   "assign m_wr_addr = ext + 32'(wr);" "assign m_wr_addr = ext + 32'(iss);"
mut npu_ss_dma $D fifo_ptr    "if (pop)  f_rd <= f_rd + 1'b1;" "if (pop && !push)  f_rd <= f_rd + 1'b1;"
mut npu_ss_dma $D pushd_wrong "wire [DW-1:0] push_d = st ? h_rd_data : m_rd_rdata;" "wire [DW-1:0] push_d = st ? h_rd_data : h_rd_data;"
mut npu_ss_dma $D wr_no_hold  "assign m_wr_req  = act && f_nempty && st;" "assign m_wr_req  = act && f_nempty && st && !iss[0];"
echo "=== formal mutation: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
