#!/bin/bash
# npu_kv_pager mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, 以小組態 seed 1 / 2 (各 20000 拍) 執行 npu_kv_pager_tb,
# 必須失敗 (scoreboard / 不變式不符或 SVA 失敗)。
# 用法: dv/scripts/run_npu_kv_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/npu_kv_mut; rm -rf $M; mkdir -p $M
R=rtl/npu/npu_kv_pager.sv
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
    rtl/common/npu_pkg.sv rtl/common/clock_gate.sv $M/$n.sv dv/npu_kv/npu_kv_pager_sva.sv dv/npu_kv/npu_kv_pager_tb.sv \
    --top-module npu_kv_pager_tb > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  local why=""
  for s in 1 2; do
    (cd $M && ./$n.obj/Vnpu_kv_pager_tb +seed=$s +cycles=20000 > $n.$s.log 2>&1)
    if ! grep -q "TB PASS" $M/$n.$s.log || grep -q "Assertion failed" $M/$n.$s.log; then
      why="seed $s: $(grep -m1 -oE 'MISMATCH [a-z_]+|INVARIANT|u_sva\.[A-Za-z_]*' $M/$n.$s.log)"; break
    fi
  done
  rm -rf $M/$n.obj
  if [ -z "$why" ]; then echo "$n: SURVIVED"; surv=1; else nk=$((nk + 1)); echo "$n: killed ($why)"; fi
}
mut alloc_mapped  "KV_ALLOC:    if (ev) st = KV_MAPPED; else" "KV_ALLOC:    if (1'b0) st = KV_MAPPED; else"      # 已映射的 slot 再配發 (覆蓋, 頁遺失)
mut alloc_nomem   "else if (free_cnt == '0) st = KV_NOMEM;" "else if (1'b0) st = KV_NOMEM;"                                                  # 耗盡時仍配發
mut free_unmapped "KV_FREE:     if (!ev) st = KV_UNMAPPED;" "KV_FREE:     ;"                                    # 未映射也釋放 (double free)
mut share_mapped  "KV_SHARE:    if (ev) st = KV_MAPPED; else if" "KV_SHARE:    if" # 共享到已映射的 slot
mut share_src     "else if (!bt_v[es]) st = KV_UNMAPPED;" "else if (1'b0) st = KV_UNMAPPED;"                                                    # 來源未映射也共享
mut share_refmax  "else if (rc[sp] == RMAX) st = KV_REFMAX;" "else if (1'b0) st = KV_REFMAX;"                                                # 參考計數溢位
mut push_rc       "rc[ep] == RW'(1);" "rc[ep] != RW'(0);"                                                        # 共享頁第一次解除就回收
mut push_never    "wire push     = do_unmap &&" "wire push     = 1'b0 && do_unmap &&"                           # 不回收 (頁遺失)
mut pop_fifo_last "wire [PW-1:0]    np  = fq_e ? hw[PW-1:0] : fq[fq_rd];" "wire [PW-1:0]    np  = hw[PW-1:0];"      # 不用 FIFO (水位越界)
mut fq_rd_inc     "else begin fq_rd <= fq_rd + 1'b1; fq_n <= fq_n - 1'b1; end" "else begin fq_n <= fq_n - 1'b1; end" # FIFO 讀指標不動
mut hw_inc        "if (fq_e) hw <= hw + 1'b1;" "if (fq_e) hw <= hw;"                                                               # 水位不動 (同頁重複配發)
mut free_cnt_eq   "assign free_cnt  = (PW+1)'(NP) - hw + fq_n;" "assign free_cnt  = (PW+1)'(NP) - hw;"         # 空閒數漏 FIFO
mut share_rc      "rc[sp] <= rc[sp] + 1'b1;" "rc[sp] <= rc[sp];"                                                 # 共享不加參考計數
mut alloc_rc      "rc[np] <= RW'(1);" "rc[np] <= RW'(2);"                                                        # 配發的參考計數錯
mut sweep_last    "wire sw_last  = sweep && sw_slot == LW'(NSLOT - 1);" "wire sw_last  = sweep && sw_slot == LW'(NSLOT - 2);" # FREE_SEQ 少最後一個 slot
mut sweep_n       "rsp_data  <= DW'(sw_n + (LW+1)'(ev));" "rsp_data  <= DW'(sw_n);"                              # 解除數少算最後一個
mut sweep_seq     "sweep ? {sw_seq, sw_slot} : {cmd_seq, cmd_slot};" "sweep ? {cmd_seq, sw_slot} : {cmd_seq, cmd_slot};" # FREE_SEQ 用到後來的 seq 輸入
mut ready_sweep   "assign cmd_ready = !sweep && (" "assign cmd_ready = ("                                      # FREE_SEQ 中受理命令
mut ready_rsp     "(!rsp_valid || rsp_ready);" "1'b1;"                                                          # 回應未取走就受理 (覆蓋回應)
mut rsp_err_data  "rsp_data  <= (st != KV_OK) ? '0 :" "rsp_data  <= 1'b0 ? '0 :"                                  # 錯誤回應帶資料
mut tr_oor        "wire             t_oor  = (tr_pos >> (OW + LW)) != '0;" "wire             t_oor  = 1'b0;"     # 不檢查範圍 (回繞)
mut tr_slot       "wire [LW-1:0]    t_slot = tr_pos[OW +: LW];" "wire [LW-1:0]    t_slot = tr_pos[LW-1:0];"     # slot 取錯位元 (舊 RTL 的錯誤類型)
mut tr_off        "{bt_p[te], tr_pos[OW-1:0]}" "{bt_p[te], {OW{1'b0}}}"                                         # 頁內 offset 遺失
mut tr_miss_line  "(tr_valid && t_hit) ? {bt_p[te], tr_pos[OW-1:0]} : '0;" "{bt_p[te], tr_pos[OW-1:0]};"   # 未命中時輸出垃圾位址
mut icg_tr        "wire  en = cmd_valid || sweep || rsp_valid || tr_valid || tr_rvalid;" "wire  en = cmd_valid || sweep || rsp_valid || tr_valid;" # 時脈閘控漏掉轉換結果清除
mut icg_sweep     "wire  en = cmd_valid || sweep ||" "wire  en = cmd_valid ||"                                   # 時脈閘控漏掉 FREE_SEQ
echo "npu_kv_pager mutation: killed $nk / $nt"
[ "$surv" -eq 0 ]
