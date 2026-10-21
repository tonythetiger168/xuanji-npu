#!/bin/bash
# 浮點函式庫 mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, 以 seed 1 的向量執行 fp_tb, 必須失敗。
# 用法: dv/scripts/run_fp_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/fp_mut; rm -rf $M; mkdir -p $M
python3 dv/fp/gen_vectors.py $M/vec 1 20000
FILES="rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv"
nk=0; nt=0; surv=0
mut() {   # name file old new
  local n=$1 f=$2
  python3 - "$3" "$4" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if s.count(o) != 1: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1))
  local src=""; for x in $FILES; do [ "$x" = "$f" ] && src="$src $M/$n.sv" || src="$src $x"; done
  verilator --binary --timing -j "${JOBS:-4}" -Wno-fatal -Wno-lint -Wno-style --Mdir $M/$n.obj $src dv/fp/fp_tb.sv \
    --top-module fp_tb > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  $M/$n.obj/Vfp_tb +vec=$M/vec > $M/$n.log 2>&1; rm -rf $M/$n.obj
  if grep -q "TB PASS" $M/$n.log; then echo "$n: SURVIVED"; surv=1
  else nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH [a-z0-9_]+' $M/$n.log))"; fi
}
A=rtl/fp/fp_add.sv; X=rtl/fp/fp_mul.sv; N=rtl/fp/fp_narrow.sv; W=rtl/fp/fp_widen.sv
mut add_tie      $A "qs = {1'b0, q} + (M+2)'(rb && (st || q[0]));" "qs = {1'b0, q} + (M+2)'(rb);"            # 平手不偶數化
# (遠距對齊丟掉 sticky 是等價 mutant: 移位 >= M+3 時結果必為大數, RTL 已改為不計 sticky)
mut add_negzero  $A "y = {sa & sb, E'(0), M'(0)};" "y = {sa | sb, E'(0), M'(0)};"                              # (-0) + (+0) 符號錯
mut add_cancel   $A "y = '0;                                 // 精確相消 → +0" "y = {s_big, E'(0), M'(0)};"     # 相消變 -0
mut add_ftz      $A "else if (kr < EW'(1 - BIAS))   y_fin = {s_big, E'(0), M'(0)};" "else if (kr < EW'(-BIAS))   y_fin = {s_big, E'(0), M'(0)};"  # FTZ 邊界錯一個
mut add_ovf      $A "if (kr > EW'(BIAS))            y_fin = {s_big, EMAX, M'(0)};" "if (kr > EW'(BIAS + 1))            y_fin = {s_big, EMAX, M'(0)};"  # 溢位邊界錯一個
mut add_infnan   $A "(a_inf && b_inf && sub)" "(1'b0)"                                                         # inf - inf 不是 NaN
mut add_lzc      $A "for (int i = M + 1; i < L; i++)" "for (int i = M + 3; i < L; i++)"                       # 大量相消時正規化錯
mut add_swap     $A "wire swap = {eb, fb} > {ea, fa};" "wire swap = eb > ea;"                                 # 同指數時大小判斷錯
mut add_daz      $A "wire [M:0]    ma = a_zero ? '0 : {1'b1, fa}" "wire [M:0]    ma = {!a_zero, fa}"           # 次正規數尾數帶入 (曾犯的錯)
mut add_carry    $A "kr = qs[M+1] ? k + EW'(1) : k;" "kr = k;"                                                # 捨入進位不加指數
mut add_align    $A "if (d < E'(M + 3)) ys = {m_sml, (M + 2)'(0)} >> d;" "if (d < E'(M + 2)) ys = {m_sml, (M + 2)'(0)} >> d;"  # 精確對齊範圍少一
mut add_far      $A "else               ys = '0;" "else               ys = (L - 1)'(m_sml);"                   # 遠距時仍把小數加進來 (未對齊)
mut mul_tie      $X "qs = {1'b0, q} + (M+1)'(rb && (st || q[0]));" "qs = {1'b0, q} + (M+1)'(rb && st);"     # 平手一律捨去
mut mul_sticky   $X "rb = p[M];   st = |p[M-1:0];" "rb = p[M];   st = |p[M-1:1];"                             # sticky 少一位
mut mul_carry    $X "kr = qs[M] ? k + EW'(1) : k;" "kr = k;"                                                  # 捨入進位不加指數
mut mul_zinf     $X "(a_inf && b_zero) || (a_zero && b_inf)" "1'b0"                                          # 0 × inf 不是 NaN
mut mul_sign     $X "wire s = sa ^ sb;" "wire s = sa;"                                                         # 符號錯
mut mul_norm     $X "if (p[2*M+1]) begin" "if (p[2*M+1] && p[2*M]) begin"                                     # 乘積 >= 2 的正規化錯
mut mul_ftz      $X "else if (kr < EW'(1 - BIAS))   y_fin = {s, E'(0), M'(0)};" "else if (kr < EW'(-BIAS))   y_fin = {s, E'(0), M'(0)};"
mut nar_tie      $N "wire [E+MO-1:0] r = {e, keep} + (E+MO)'(rb && (st || keep[0]));" "wire [E+MO-1:0] r = {e, keep} + (E+MO)'(rb);"  # 平手不偶數化
mut nar_nan      $N "if (e == EMAX && f != '0) y = {1'b0, EMAX, 1'b1, (MO - 1)'(0)};" "if (e == EMAX && f != '0) y = {s, EMAX, keep};"  # NaN 截斷後可能變 inf
mut nar_daz      $N "else if (e == '0)         y = {s, E'(0), MO'(0)};" "else if (1'b0)         y = {s, E'(0), MO'(0)};"  # 次正規數不清零
mut wid_daz      $W "else if (e == '0)         y = {s, E'(0), MO'(0)};" "else if (1'b0)         y = {s, E'(0), MO'(0)};"
mut wid_nan      $W "if (e == EMAX && f != '0) y = {1'b0, EMAX, 1'b1, (MO - 1)'(0)};" "if (e == EMAX && f != '0) y = {s, e, f, (MO - MI)'(0)};"  # NaN 不標準化
echo "=== fp mutation: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
