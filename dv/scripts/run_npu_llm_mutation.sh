#!/bin/bash
# npu_llm_eng mutation 測試: 每個 mutant 植入一個錯誤, 以 seed 1 / 2 (各 200 個命令) 執行 npu_llm_eng_tb, 必須失敗。
# 用法: dv/scripts/run_npu_llm_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/npu_llm_mut; rm -rf $M; mkdir -p $M
R=rtl/npu/npu_llm_eng.sv
nk=0; nt=0; surv=0
mut() {   # name old new
  local n=$1
  python3 - "$2" "$3" "$M/$n.sv" "$R" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if s.count(o) != 1: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common --Mdir $M/$n.obj \
    rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/npu/npu_kv_pager.sv rtl/npu/npu_moe_disp.sv $M/$n.sv \
    dv/npu_kv/npu_kv_pager_sva.sv dv/npu_moe/npu_moe_disp_sva.sv dv/npu_llm/npu_llm_eng_sva.sv dv/npu_llm/npu_llm_eng_tb.sv \
    --top-module npu_llm_eng_tb > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  local why=""
  for s in 1 2; do
    (cd $M && ./$n.obj/Vnpu_llm_eng_tb +seed=$s +cmds=200 > $n.$s.log 2>&1)
    if ! grep -q "TB PASS" $M/$n.$s.log || grep -q "Assertion failed" $M/$n.$s.log; then
      why="seed $s: $(grep -m1 -oE 'MISMATCH [A-Z]+|ERROR[^,]*|u_sva\.[A-Za-z_]*' $M/$n.$s.log)"; break
    fi
  done
  rm -rf $M/$n.obj
  if [ -z "$why" ]; then echo "$n: SURVIVED"; surv=1; else nk=$((nk + 1)); echo "$n: killed ($why)"; fi
}
mut kv_bad_seq   "wire kv_bad = seq8 >= 8'(NSEQ) ||" "wire kv_bad = 1'b0 ||"                                           # seq 超出範圍不報錯
mut kv_bad_k     "? kk > 16'(NSLOT)" "? 1'b0"                                                                             # MAP k 超過 NSLOT 不報錯
mut kv_bad_src   "(sub == 3'd3 && src8 >= 8'(NSEQ))" "1'b0"                                                              # SHARE 來源超出範圍
mut kv_res_st    "wbuf <= res3(32'(p_st), 32'(p_data), 32'(p_free));" "wbuf <= res3(32'd0, 32'(p_data), 32'(p_free));" # 狀態遺失
mut kv_res_line  "wbuf <= res3(32'(p_st), 32'(p_data), 32'(p_free)); wline <= a;" "wbuf <= res3(32'(p_st), 32'(p_data), 32'(p_free)); wline <= c;" # 結果寫錯位置
mut map_mult     "32'(p_line >> OW) * ext" "32'(p_line >> OW)"                                                            # MAP 沒乘 mult
mut map_miss     ": 32'hFFFF_FFFF;" ": 32'h0;"                                                                            # 未映射寫 0 (DMA 會搬頁 0)
mut map_range    ".tr_valid(st == S_MISS && m_in)" ".tr_valid(st == S_MISS)"                                             # slot ≥ NSLOT 時回繞查到別的 slot
mut map_line     "wline <= a + 26'(mi >> 4);" "wline <= a;"                                                               # 多 line 時都寫第一個
mut map_res      "(16'(mhit) == kk) ? 32'd0 : 32'd3" "32'd0"                                                              # 有未映射仍回 OK
mut moe_bad_tw   "kk == '0 || kk >= 16'(1 << TW);" "kk == '0;"                                                            # T 超過上限不報錯
mut moe_init     "wbuf <= '1; wline <= c + 26'(gi);" "wbuf <= '0; wline <= c + 26'(gi);"                                  # GL 初值 0 (未填的槽不會被略過)
mut moe_init_n   "if (gi < (XW-2)'(gl_n)) begin" "if (gi + 1'b1 < (XW-2)'(gl_n)) begin"                                    # GL 少初始化一個 line
mut moe_rd_w     "else begin vb <= rd_data;" "else begin vb <= ib;"                                                       # weight 用錯 line
mut moe_rmw      "wbuf <= put32(rd_data, int'(e_pos[3:0]), 32'(e_val));" "wbuf <= put32('1, int'(e_pos[3:0]), 32'(e_val));" # 寫 GL 時覆蓋同 line 其他項
mut moe_mult     "(TW+16)'(d_out_tok) * (TW+16)'(ext[31:16]);" "(TW+16)'(d_out_tok);"                                     # gather 項沒乘 mult
mut moe_drop_w   "d_out_drop ? 32'd0 : d_out_w);" "d_out_w);"                                                              # drop 的權重沒清 0
mut moe_ct_flush "if (cw == 4'd8 || e_last) begin" "if (cw == 4'd8) begin"                                                # 最後一個不滿的 combine line 沒寫
mut moe_next_tok "rline <= a + {25'(t) + 25'd1, 1'b0};" "rline <= a + {25'(t), 1'b0};"                                    # 下一個 token 讀錯
mut moe_pass2    "if (!pass) begin pass <= 1'b1; st <= S_OLD; rq <= 1'b1; rline <= a; end" "if (!pass) begin pass <= 1'b1; st <= S_OLD; rq <= 1'b1; rline <= a + 26'd2; end" # 第二趟從 token 1 開始
mut moe_cnt_last "st_ret <= (8'(ce) == ne8 - 1'b1) ? S_OSTAT : S_OCNT;" "st_ret <= S_OSTAT;"                              # 只寫第一個 CN line
mut moe_stat     "wbuf <= res3({d_ovf, 31'(nerr)}, 32'(nval), 32'(p_tot));" "wbuf <= res3({1'b0, 31'(nerr)}, 32'(nval), 32'(p_tot));" # 溢位旗標遺失
mut icg          "wire  en = cmd_valid || st != S_IDLE || done;" "wire  en = cmd_valid || st != S_IDLE;"                    # 閘控漏掉 done 拍
echo "npu_llm_eng mutation: killed $nk / $nt"
[ "$surv" -eq 0 ]
