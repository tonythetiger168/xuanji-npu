#!/bin/bash
# 浮點函式庫 mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, 以 seed 1 的向量執行 fp_tb, 必須失敗。
# 用法: dv/scripts/run_fp_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/fp_mut; rm -rf $M; mkdir -p $M
python3 dv/fp/gen_vectors.py $M/vec 1 20000
python3 dv/fp/gen_sfu_vectors.py $M/sfu_vec.hex 1 20000
FILES="rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv"
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
  local top=fp_tb tbsrc=dv/fp/fp_tb.sv args="+vec=$M/vec"
  if [ "$f" = rtl/fp/sfu.sv ]; then   # SFU: 位元一致向量 + 精度掃描 (exp 掃描縮小以節省時間)
    top=sfu_tb; tbsrc=dv/fp/sfu_tb.sv; args="+vec=$M/sfu_vec.hex +exp_n=200000"; src="$M/$n.sv"
  else src=$(echo $src | sed 's| [^ ]*sfu.sv||; s|^[^ ]*sfu.sv ||'); fi
  verilator --binary --timing -j "${JOBS:-4}" -Wno-fatal -Wno-lint -Wno-style -Irtl/fp --Mdir $M/$n.obj $src $tbsrc \
    --top-module $top > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  $M/$n.obj/V$top $args > $M/$n.log 2>&1; rm -rf $M/$n.obj
  if grep -q "TB PASS" $M/$n.log; then echo "$n: SURVIVED"; surv=1
  else nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH [a-z0-9_=]+|ERROR: op [0-9]' $M/$n.log))"; fi
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
U=rtl/fp/sfu.sv
mut sfu_tie      $U "wire          inc  = rb && (st || q[0]);" "wire          inc  = rb;"                   # 平手不偶數化
mut sfu_norm     $U "wire          hi   = yu[30];" "wire          hi   = yu[29];"                          # 正規化選錯
mut sfu_rsq_par  $U "c = eu[0] ? SFU_TAB_RSQRT_O[idx] : SFU_TAB_RSQRT_E[idx];" "c = eu[0] ? SFU_TAB_RSQRT_E[idx] : SFU_TAB_RSQRT_O[idx];"  # 指數奇偶選錯表
mut sfu_rsq_k    $U "2'd1:    k = -10'(eu >>> 1);" "2'd1:    k = -10'(eu >>> 1) + 10'sd1;"               # rsqrt 指數錯
mut sfu_p2       $U "p2 = 16'((41'(c2) * 41'(t2)) >> 25);" "p2 = 16'((41'(c2) * 41'(t2)) >> 26);"          # exp2 二次項比例錯
# (p1 / floor 的 >>> 改 >> 是等價 mutant: 位移後立即截斷到低位, 移入的位元不影響結果)
mut sfu_p1scale  $U "p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 16);" "p1 = 26'((52'(c1) * signed'({27'b0, t})) >>> 15);"  # recip / rsqrt 一次項比例錯
mut sfu_log2e    $U "if (op == 2'd3) w = 42'((91'(v) * signed'({1'b0, SFU_LOG2E})) >>> SFU_LOG2E_SH);" "if (op == 2'd3) w = 42'((91'(v) * signed'({1'b0, SFU_LOG2E})) >>> (SFU_LOG2E_SH + 1));"  # exp 換底錯
mut sfu_floor    $U "default: k = 10'(w >>> 32);" "default: k = 10'(w >>> 31);"                           # exp2 的整數部分錯
mut sfu_ftz      $U "else if (e < -11'sd126)  y_fin = '0;" "else if (e < -11'sd127)  y_fin = '0;"           # FTZ 邊界錯一個
mut sfu_ovf      $U "if (e > 11'sd127)        y_fin = PINF[30:0];" "if (e > 11'sd128)        y_fin = PINF[30:0];"  # 溢位邊界錯一個
mut sfu_rcp_zero $U "else if (x_zero) y = {s, PINF[30:0]};
        else             y = {s, y_fin};" "else if (x_zero) y = {s, 31'b0};
        else             y = {s, y_fin};"                                                                # recip(0) 不是 inf
mut sfu_rsq_neg  $U "if (x_nan || (s && !x_zero)) y = QNAN;" "if (x_nan) y = QNAN;"                           # rsqrt(負數) 不是 NaN
# (|x| 範圍界改成 135 是等價 mutant: ex = 134 時一般路徑本來就溢位成 inf / FTZ 成 0; 界只在定點值超出 42 bit 時才必要)
mut sfu_exp_big  $U "else if (x_inf || ex >= 8'd134) y = s ? '0 : PINF;" "else if (x_inf || ex >= 8'd200) y = s ? '0 : PINF;"  # 大 |x| 時定點溢位
mut sfu_carry    $U "e = 11'(k) - 11'(F) + (hi ? 11'sd7 : 11'sd6) + 11'sd23 + (qs[23] ? 11'sd1 : 11'sd0);" "e = 11'(k) - 11'(F) + (hi ? 11'sd7 : 11'sd6) + 11'sd23;"  # 捨入進位不加指數
# (exp 的 ±0 / 次正規輸入: m >> 118 自然為 0, RTL 已移除多餘的分支)
echo "=== fp mutation: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
