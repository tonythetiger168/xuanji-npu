#!/bin/bash
# npu_ss_regs mutation 測試: 每個 mutant 在 RTL 植入一個錯誤, 受約束隨機 APB TB (seed 1, 參考模型 + SVA) 必須失敗。
# 用法: dv/scripts/run_npu_ss_regs_mutation.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
M=build/npu_ss_regs_mut; rm -rf $M; mkdir -p $M
F=rtl/npu/npu_ss_regs.sv
nk=0; nt=0; surv=0
mut() {   # name old new
  local n=$1
  python3 - "$2" "$3" "$M/$n.sv" "$F" <<'PY' || { echo "$n: pattern not found"; nt=$((nt + 1)); surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if s.count(o) != 1: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --assert -Wno-fatal -Wno-lint -Wno-style -Irtl/common --Mdir $M/$n.obj \
    rtl/common/orca_pkg.sv rtl/common/clock_gate.sv rtl/common/ret_shadow.sv $M/$n.sv dv/npu/npu_ss_regs_sva.sv dv/npu/npu_ss_regs_tb.sv --top-module npu_ss_regs_tb \
    > $M/$n.build 2>&1 || { echo "$n: BUILD FAIL"; rm -rf $M/$n.obj; surv=1; return; }
  $M/$n.obj/Vnpu_ss_regs_tb +seed=1 +cycles=100000 > $M/$n.log 2>&1; rm -rf $M/$n.obj
  if grep -q "TB PASS" $M/$n.log && ! grep -q "Assertion failed" $M/$n.log; then echo "$n: SURVIVED"; surv=1
  else nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH [0-9]+ [a-z_]+|u_sva\.[A-Za-z_.0-9]*' $M/$n.log))"; fi
}
mut slverr_none "assign pslverr = acc && !hit;" "assign pslverr = 1'b0;"                                     # 未定義位址不報錯
mut hit_hole    "A_PB_HI, A_PS_LO, A_PS_HI};" "A_PB_HI, A_PS_LO};"                     # 漏一個已定義位址
mut strobe      "return (old & ~m) | (nw & m);" "return nw;"                                                 # 忽略 PSTRB
mut ctl_rsvd    "r_ctl <= merge(r_ctl, pwdata, bm) & 32'hFFFF_FF3F;" "r_ctl <= merge(r_ctl, pwdata, bm);"                                                        # 保留位元可寫
mut push_full   "wire           push   = push_req && !q_full;" "wire           push   = push_req;"           # 滿時仍寫入
mut full_off    "wire           q_full = (q_n == (CW+1)'(CQ));" "wire           q_full = (q_n == (CW+1)'(CQ - 1));"  # 少一項就算滿
mut head_wrong  "assign cmd       = q[q_rd[CW-1:0]];" "assign cmd       = q[q_wr[CW-1:0]];"                # 送出錯的 FIFO 項
mut cpl_on_wr   "assign cpl_ready = rd && wa == A_CPL_POP;" "assign cpl_ready = acc && wa == A_CPL_POP;"    # 寫 CPL_POP 也取走
mut peek_pops   "assign cpl_ready = rd && wa == A_CPL_POP;" "assign cpl_ready = rd && (wa == A_CPL_POP || wa == A_CPL_PEEK);"  # PEEK 也取走
mut ovf_no_w1c  "ovf  <= (ovf  && !(w1c && pwdata[1]))" "ovf  <= (ovf)"                                     # CMD_OVF 清不掉
mut berr_bit    "berr <= (berr && !(w1c && pwdata[2]))" "berr <= (berr && !(w1c && pwdata[1]))"              # 清錯位元
mut w1c_strb    "wire  w1c = wr && wa == A_IRQ_ST && pstrb[0];" "wire  w1c = wr && wa == A_IRQ_ST;"          # W1C 忽略 PSTRB
mut irq_nomask  "assign irq = |(ist & r_ien);" "assign irq = |ist;"                                         # 中斷不看致能
mut perf_strb   "wr && wa == A_PERF_CTRL && pstrb[0] && pwdata[0]" "wr && wa == A_PERF_CTRL && pwdata[0]"   # 清除忽略 PSTRB
mut shadow_hi   "if (rd && wa == A_PB_LO) sh_b <= perf_busy[47:32];" "if (rd && wa == A_PB_HI) sh_b <= perf_busy[47:32];"  # 讀 HI 才存高位 (不一致)
mut shadow_wr   "if (rd && wa == A_PS_LO) sh_s <= perf_steps[47:32];" "if (acc && wa == A_PS_LO) sh_s <= perf_steps[47:32];"  # 寫 LO 也存
mut free_cnt    "4'(CQ - int'(q_n))" "4'(q_n)"                                                              # FREE 欄位錯
mut cpl_data    "cpl_valid ? cpl_id : 8'd0" "cpl_id"                                                        # 沒有完成時仍回 id
mut cfg_depth   "8'(\$clog2(DEPTH))" "8'(DEPTH)"                                                            # CFG0 欄位錯
mut busy_q      "ss_busy || cmd_valid" "ss_busy"                                                            # FIFO 有命令時 BUSY 為 0
mut rst_ctl     "r_ctl <= '0; r_ext <= '0;" "r_ext <= '0;"                                                  # CMD_CTL 不 reset
mut op_hi       "stg.op  = npu_op_e'({r_ctl[5], r_ctl[1:0]});" "stg.op  = npu_op_e'({1'b0, r_ctl[1:0]});"   # OP[2] 遺失 (VEC 變 GEMM)
mut verr_no_w1c "verr <= (verr && !(w1c && pwdata[3])) || vec_err;" "verr <= verr || vec_err;"            # VEC_ERR 清不掉
mut verr_none   "verr <= (verr && !(w1c && pwdata[3])) || vec_err;" "verr <= (verr && !(w1c && pwdata[3]));"  # 沒記錄 VEC 錯誤
mut icg_w       "u_icg_w (.clk(clk), .en(wr || ret_restore || !rst_n)" "u_icg_w (.clk(clk), .en((wr && wa != A_CMD_EXT) || ret_restore || !rst_n)"   # 時脈閘控漏掉一個暫存器
mut icg_q       ".en(push || !rst_n)" ".en((push && !cmd_ready) || !rst_n)"                              # FIFO 同拍 push / pop 時沒有時脈
mut icg_rest    "u_icg_w (.clk(clk), .en(wr || ret_restore || !rst_n)" "u_icg_w (.clk(clk), .en(wr || !rst_n)"   # 還原拍沒有時脈
mut no_restore  "      if (ret_restore) r_ien <= ien_ret;" ""                                          # 不還原
mut rest_lost   "      if (ret_restore) r_ien <= ien_ret;" "      if (ret_restore && !wr) r_ien <= ien_ret;"  # 同拍有其他寫入時沒還原
mut save_always ".save(ret_save), .d(r_ien)" ".save(1'b1), .d(r_ien)"                                  # 影子一直跟著主暫存器
mut save_none   ".save(ret_save), .d(r_ien)" ".save(1'b0), .d(r_ien)"                                  # 從不存檔
echo "=== npu_ss_regs mutation: $nk / $nt mutants killed"
[ $surv -eq 0 ] && [ $nk -eq $nt ]
