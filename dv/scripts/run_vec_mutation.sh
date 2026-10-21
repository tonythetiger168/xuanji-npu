#!/bin/bash
# NPU 向量核 mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, 以 seed 1 (含窮舉解碼) 與 seed 2 的指令串執行 npu_vec_tb,
# 必須失敗 (資料不符或 SVA 失敗)。
# 用法: dv/scripts/run_vec_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/vec_mut; rm -rf $M; mkdir -p $M
python3 dv/vec/gen_vec_vectors.py $M/v1 1 6000 > /dev/null
python3 dv/vec/gen_vec_vectors.py $M/v2 2 6000 > /dev/null
FILES="rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/fp/fp_add.sv rtl/fp/fp_mul.sv rtl/fp/fp_narrow.sv rtl/fp/fp_widen.sv rtl/fp/sfu.sv
       rtl/fp/fp_max.sv rtl/fp/fp_small_dec.sv rtl/fp/fp_mx_scale.sv rtl/fp/fp_from_int.sv rtl/npu/npu_vec.sv"
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
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common -Irtl/fp --Mdir $M/$n.obj \
    $src dv/vec/npu_vec_sva.sv dv/vec/npu_vec_tb.sv --top-module npu_vec_tb > $M/$n.build 2>&1 \
    || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  local why=""
  for v in v1 v2; do
    (cd $M && ./$n.obj/Vnpu_vec_tb +vec=$v +seed=5 > $n.$v.log 2>&1)
    if ! grep -q "TB PASS" $M/$n.$v.log || grep -q "Assertion failed" $M/$n.$v.log; then
      why="$v: $(grep -m1 -oE 'MISMATCH [a-z]+|u_sva\.[A-Za-z_.0-9\[\]]*' $M/$n.$v.log)"; break
    fi
  done
  rm -rf $M/$n.obj
  if [ -z "$why" ]; then echo "$n: SURVIVED"; surv=1; else nk=$((nk + 1)); echo "$n: killed ($why)"; fi
}
V=rtl/npu/npu_vec.sv D=rtl/fp/fp_small_dec.sv S=rtl/fp/fp_mx_scale.sv I=rtl/fp/fp_from_int.sv X=rtl/fp/fp_max.sv
# ---- npu_vec: 解碼 / 交握 / 寫回 / 格式選擇 ----
mut vec_ill_op   $V "wire illegal = (ins.op > VOP_RMAX) ||" "wire illegal = 1'b0 ||"                          # 未定義 op 不報錯
mut vec_ill_ld   $V "(ins.op == VOP_LD && ins.fn == 3'd7) ||" ""                                                # LD fn 7 當合法
mut vec_ill_st   $V "(ins.op == VOP_ST && ins.fn > 3'd1)" "(ins.op == VOP_ST && ins.fn > 3'd2)"                 # ST fn 2 當合法
mut vec_ill_sfu  $V "(ins.op == VOP_SFU && ins.fn > 3'd5)" "(ins.op == VOP_SFU && ins.fn > 3'd6)"               # SFU fn 6 當合法
mut vec_rdy_ld   $V "assign ins_ready = is_ld ? rd_gnt" "assign ins_ready = is_ld ? 1'b1"                       # 不等讀取 grant
mut vec_rdy_st   $V "is_st ? wr_gnt : 1'b1;" "is_st ? 1'b1 : 1'b1;"                                             # 不等寫入 grant
mut vec_we_st    $V " && ins.op != VOP_ST;" ";"                                                                  # ST 也寫暫存器
mut vec_err      $V "else if (fire && illegal)  err <= 1'b1;" "else err <= fire && illegal;"                     # err 不黏著
mut vec_part16   $V "rd_data[256 * ins.part[0] +: 256]" "rd_data[256 * ins.part[1] +: 256]"                      # 16 bit 格式 part 選錯
mut vec_part4    $V "rd_data[64 * ins.part +: 64]" "rd_data[64 * ins.part[1:0] +: 64]"                           # FP4 part 少一位
mut vec_sub      $V "b[i][31] ^ (ins.op == VOP_SUB)" "b[i][31]"                                                  # SUB 不取負
mut vec_mx_sel   $V "3'd5:    d_mx = {d_e5, 1'b0};" "3'd5:    d_mx = d_e4;"                                      # E5M2 用 E4M3 解碼
mut vec_fp16     $V ".BIAS(15), .SP(0)) u_h" ".BIAS(15), .SP(1)) u_h"                                             # FP16 特殊值規則錯
mut vec_int      $V "3'd3:    ld_v[i] = d_i;" "3'd3:    ld_v[i] = rd_data[32*i +: 32];"                         # INT32 不轉換
mut vec_tree_s   $V ".a(ts[2*k]), .b(ts[2*k+1])" ".a(ts[2*k]), .b(ts[2*k])"                                      # 歸約和取錯節點
mut vec_tree_m   $V ".a(tm[2*k]), .b(tm[2*k+1])" ".a(tm[2*k+1]), .b(tm[2*k+1])"                                  # 歸約 max 取錯節點
mut vec_gate     $V ".en(we && ins.vd == 3'(r))" ".en(we && ins.vd == 3'(r) && ins.op != VOP_SPLAT)"              # clock gate 漏開 (SPLAT 寫不進)
mut vec_st_bf    $V ".a(b[i]), .y(st_bf[VL+i])" ".a(a[i]), .y(st_bf[VL+i])"                                       # BF16 儲存的後半取錯
mut vec_st_fmt   $V "assign wr_data = (ins.fn == 3'd0)" "assign wr_data = (ins.fn == 3'd1)"                       # 儲存格式顛倒
mut vec_rsum     $V "VOP_RSUM: wd = {VL{ts[2*VL-2]}};" "VOP_RSUM: wd = {VL{ts[2*VL-3]}};"                         # 歸約少最後一層
# ---- fp_small_dec ----
mut dec_sub_exp  $D "8'(128 - BIAS) - 8'(sh)" "8'(127 - BIAS) - 8'(sh)"                                          # 次正規指數錯
mut dec_sub_frac $D "M'(fr << sh)" "M'(fr << (sh - 1))"                                                          # 次正規尾數少移一位
mut dec_norm_exp $D "8'(ex) + 8'(127 - BIAS)" "8'(ex) + 8'(128 - BIAS)"                                          # 正規指數偏一
mut dec_e4_nan   $D "(ex == '1 && fr == '1) ? QN" "(ex == '1) ? QN"                                               # E4M3 把 448 等當 NaN
mut dec_inf      $D "((fr != '0) ? QN : {s, 8'hFF, M'(0)})" "QN"                                                  # IEEE inf 變 NaN
mut dec_zero     $D "if (ex == '0 && fr == '0) yv = {s, 8'd0, M'(0)};" "if (ex == '0 && fr == '0) yv = '0;"      # -0 變 +0
# ---- fp_mx_scale ----
mut sc_nan       $S "if (scale == 8'hFF)            y = QN;" "if (1'b0)            y = QN;"                      # 比例 NaN 未處理
mut sc_ovf       $S "else if (be > 10'sd254)" "else if (be > 10'sd255)"                                           # 溢位邊界錯一個
mut sc_ftz       $S "else if (be < 10'sd1)" "else if (be < 10'sd0)"                                               # FTZ 邊界錯一個
mut sc_bias      $S "- 10'sd127;" "- 10'sd126;"                                                                   # 比例偏一
mut sc_special   $S "else if (e == 8'hFF || e == '0) y = x;" "else if (e == 8'hFF) y = x;"                       # 0 被縮放 (FTZ 判斷錯)
# ---- fp_from_int ----
mut int_tie      $I "((|n[6:1]) || n[8])" "(|n[6:1])"                                                            # 平手不偶數化
mut int_carry    $I "8'(158 - lz) + 8'(q[23])" "8'(158 - lz)"                                                    # 捨入進位不加指數
mut int_abs      $I "wire [31:0] m = s ? -a : a;" "wire [31:0] m = s ? ~a : a;"                                  # 負數絕對值少 1
mut int_sticky   $I "(|n[6:1])" "(|n[6:2])"                                                                      # sticky 少一位
# ---- fp_max ----
mut max_zero     $X "(ad[31] != bd[31]) ? !ad[31]" "(ad[31] != bd[31]) ? ad[31]"                                  # 異號時取負者
mut max_neg      $X "(ad[31] ? (ad[30:0] < bd[30:0])" "(ad[31] ? (ad[30:0] > bd[30:0])"                          # 負數比較方向錯
mut max_daz      $X "wire [31:0] ad = (a[30:23] == '0) ? {a[31], 31'd0} : a;" "wire [31:0] ad = a;"              # 次正規數未 DAZ
mut max_nan      $X "(nan_a || nan_b)" "(nan_a)"                                                                 # b 為 NaN 未處理
# ---- npu_vec_eng (以引擎 TB 執行: 多命令程式, 隨機 grant) ----
python3 dv/vec/gen_eng_vectors.py $M/e1 1 60 > /dev/null
python3 dv/vec/gen_eng_vectors.py $M/e2 2 60 > /dev/null
mut_e() {   # name old new
  local n=$1
  python3 - "$2" "$3" "$M/$n.sv" rtl/npu/npu_vec_eng.sv <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if s.count(o) != 1: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common -Irtl/fp --Mdir "$M/$n.obj" \
    $FILES dv/vec/npu_vec_sva.sv "$M/$n.sv" dv/vec/npu_vec_eng_sva.sv dv/vec/npu_vec_eng_tb.sv --top-module npu_vec_eng_tb \
    > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "${M:?}/${n:?}.obj"; surv=1; return; }
  local why=""
  for v in e1 e2; do
    (cd "$M" && "./$n.obj/Vnpu_vec_eng_tb" +vec=$v +seed=3 +gnt_p=60 > "$n.$v.log" 2>&1)
    if ! grep -q "TB PASS" "$M/$n.$v.log" || grep -q "Assertion failed" "$M/$n.$v.log"; then
      why="$v: $(grep -m1 -oE 'MISMATCH [a-z]+|ERROR: [a-z ]+|u_sva\.[A-Za-z_.0-9\[\]]*' "$M/$n.$v.log")"; break
    fi
  done
  rm -rf "${M:?}/${n:?}.obj"
  if [ -z "$why" ]; then echo "$n: SURVIVED"; surv=1; else nk=$((nk + 1)); echo "$n: killed ($why)"; fi
}
mut_e eng_hold    "wire  f_sel  = f_pend || (f_want && !v_rd_req);" "wire  f_sel  = f_want && !v_rd_req;"            # 預取請求不保持 (曾犯的錯)
mut_e eng_vprio   "wire  f_sel  = f_pend || (f_want && !v_rd_req);" "wire  f_sel  = f_pend || f_want;"                # 取指搶走 npu_vec 等待中的讀取
mut_e eng_slot    "  assign ins = npu_vec_ins_t'(fb[fr][128 * slot +: 77]);" "  assign ins = npu_vec_ins_t'(fb[fr][128 * slot +: 76]);"  # 指令少一位
mut_e eng_lnend   "wire  ln_end = fire && (slot == 2'd3 || last);" "wire  ln_end = fire && slot == 2'd3;"         # 最後一個 line 不釋放
mut_e eng_nf      "nf <= (nf > 16'd4) ? nf - 16'd4 : '0;" "nf <= (nf > 16'd3) ? nf - 16'd4 : '0;"                 # 剩餘取指數錯
mut_e eng_pc      "fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd1;" "fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd2;"   # 取指位址跳過
mut_e eng_fw      "fv[fw] <= 1'b1; fw <= !fw; pc <= pc + 26'd1;" "fv[fw] <= 1'b1; pc <= pc + 26'd1;"              # 寫入緩衝不輪換
mut_e eng_done    "  wire  last = fire && ni == 16'd1;" "  wire  last = fire && ni == 16'd2;"                       # 少執行一個指令
mut_e eng_errclr  ".err(v_err), .err_clr(start)," ".err(v_err), .err_clr(1'b0),"                                  # 命令開始不清 err
mut_e eng_ready   "  assign cmd_ready = !act && !done;" "  assign cmd_ready = !act;"                                # done 拍又接受命令
mut_e eng_gate    "  wire  en = act || cmd_valid || done;" "  wire  en = act || cmd_valid;"                       # done 拍時脈關閉
mut_e eng_vgnt    "  assign v_rd_gnt = !f_sel && v_rd_req && rd_gnt;" "  assign v_rd_gnt = v_rd_req && rd_gnt;"     # 取指的 grant 也給 npu_vec
echo "=== vec mutation: $nk / $nt mutants killed"
[ $surv -eq 0 ] && [ $nk -eq $nt ]
