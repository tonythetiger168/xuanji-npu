#!/bin/bash
# NPU 檢查器的 mutation 測試 (每個產品組態): 每個 mutant 在 RTL 植入一個錯誤, 對應的 CRV + SVA 必須抓到 (killed)。
# 存活的 mutant 代表檢查器有漏洞 (或是等價 mutant, 須說明)。
#   npu_mma / npu_gemm_cu 的 mutant 以各自的 TB (該組態的 N) 執行; npu_l2_xbar / npu_core 的 mutant 以 npu_core_tb 執行。
# 用法: dv/scripts/run_npu_mutation.sh <nano|x1|pro|all> [count=120]
set -uo pipefail
cd "$(dirname "$0")/../.."
. dv/scripts/npu_cfg.sh
CFG=${1:-x1}; NT=${2:-120}
if [ "$CFG" = all ]; then
  rc=0; for c in nano x1 pro; do bash "$0" $c "$NT" || rc=1; done; exit $rc
fi
npu_cfg "$CFG" || exit 1
M=build/npu_mut_$CFG; rm -rf "$M"; mkdir -p "$M"
echo "##### NPU mutation $CFG: N=$N NCU=$NCU NB=$NB"
MMA=rtl/npu/npu_mma.sv CU=rtl/npu/npu_gemm_cu.sv XB=rtl/npu/npu_l2_xbar.sv CORE=rtl/npu/npu_core.sv
surv=0; nk=0; nt=0
mut() {   # name file old new
  local n=$1 f=$2
  python3 - "$3" "$4" "$M/$n.sv" "$f" <<'PY' || { echo "$n: pattern not found"; surv=1; return; }
import sys
o, nw, out, f = sys.argv[1:5]
s = open(f).read()
if o not in s: sys.exit(1)
open(out, 'w').write(s.replace(o, nw, 1))
PY
  local mma=$MMA cu=$CU xb=$XB core=$CORE top args
  case $f in
    $MMA) mma=$M/$n.sv ;; $CU) cu=$M/$n.sv ;; $XB) xb=$M/$n.sv ;; $CORE) core=$M/$n.sv ;;
  esac
  local src="rtl/common/npu_pkg.sv rtl/common/clock_gate.sv rtl/fp/fp_from_int.sv rtl/fp/fp_add.sv $mma dv/npu/npu_mma_sva.sv"
  case $f in
    $MMA) top=npu_mma_tb; src="$src dv/npu/npu_mma_tb.sv"; args="+tiles=$((NT * 2))" ;;
    $CU)  top=npu_gemm_cu_tb; src="$src $cu dv/npu/npu_gemm_cu_sva.sv dv/npu/npu_gemm_cu_tb.sv"; args="+cmds=$NT" ;;
    *)    top=npu_core_tb; args="+cmds=$NT"
          src="$src $cu dv/npu/npu_gemm_cu_sva.sv $xb dv/npu/npu_l2_xbar_sva.sv $core dv/npu/npu_core_sva.sv dv/npu/npu_core_tb.sv" ;;
  esac
  nt=$((nt + 1))
  verilator --binary --timing -j "${JOBS:-4}" --output-groups 64 --assert -Wno-fatal -Wno-lint -Wno-style -GN=$N \
    $([ $top = npu_core_tb ] && echo "-GNCU=$NCU -GNB=$NB -GDEPTH=$DEPTH") --Mdir "$M/$n.obj" -Irtl/common -Idv/npu \
    $src --top-module $top > "$M/$n.build" 2>&1 || { echo "$n: BUILD FAIL"; rm -rf "$M/$n.obj"; surv=1; return; }
  "$M/$n.obj/V$top" +seed=7 $args > "$M/$n.log" 2>&1
  # 全 0 上電通過時, 再以隨機上電狀態 (未 reset 的儲存為隨機值) 執行
  for r in 2 3; do
    grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log" \
      && "$M/$n.obj/V$top" +seed=$((7 + r)) $args +verilator+rand+reset+2 +verilator+seed+$r > "$M/$n.log" 2>&1
  done
  rm -rf "$M/$n.obj"                       # 每個 mutant 的建置目錄數百 MB: 用完即刪
  if grep -q "TB PASS" "$M/$n.log" && ! grep -q "Assertion failed" "$M/$n.log"; then
    echo "$n: SURVIVED"; surv=1
  else
    nk=$((nk + 1)); echo "$n: killed ($(grep -m1 -oE 'MISMATCH|u_sva\.[A-Za-z_.0-9\[\]]*|ERROR: [a-z_ ]*' "$M/$n.log"))"
  fi
}
# ---- npu_mma ----
mut gate_no_own   $MMA "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = a_tok_edge[r] || (|tok_q[r][N-2:0]);"  # 閘控未涵蓋最右 PE 的殘留 token
mut gate_no_edge  $MMA "assign row_act[r] = a_tok_edge[r] || (|tok_q[r]);" "assign row_act[r] = (|tok_q[r]);"     # 閘控未涵蓋左邊緣進入的 token
mut drain_short   $MMA "localparam int DRAIN = 2 * N - 1;" "localparam int DRAIN = 2 * N - 3;"             # 排空不足
mut no_sext_a     $MMA "wire signed [8:0]  ax = {f_in.sa & a_in[7], a_in};" "wire signed [8:0]  ax = {1'b0, a_in};"
mut no_sext_b     $MMA "wire signed [8:0]  bx = {f_in.sb & b_in[7], b_in};" "wire signed [8:0]  bx = {b_in[7], b_in};"
mut no_clr        $MMA "(f_in.f ? 32'd0 : acc[r][c])" "(acc[r][c])"                                     # 新 tile 不清累加器
mut row_no_ready  $MMA "if (ost == O_OUT && out_ready) out_row" "if (ost == O_OUT) out_row"                   # 忽略回壓
mut skew_b        $MMA "assign b_edge[i] = b_sr[i-1];" "assign b_edge[i] = (i > 1) ? b_sr[i-2] : b_vec[i];" # B skew 少一拍
mut sign_latch    $MMA "sa: sgn_a, sb: sgn_b" "sa: a_signed, sb: sgn_b"                                         # 有號性未在 start 鎖存
mut data_no_tok   $MMA "if (t_in) begin                        // 資料只在有 token 時更新
            a_q[r][c] <= a_in; b_q[r][c] <= b_in;" "if (t_in) begin
            a_q[r][c] <= a_in;"                                                                               # B 資料不前進
mut acc_trunc     $MMA "acc[r][c]) + 32'(prod)" "acc[r][c]) + 32'(prod[15:0])"                                  # 乘積截斷
mut start_busy    $MMA "if (go) begin ist <= I_RUN;" "if (start) begin ist <= I_RUN;"                           # 輸入端忙碌時接受 start
mut last_early    $MMA "O_IDLE:  if (fire_last) ost <= O_DRAIN;" "O_IDLE:  if (in_valid && in_last) ost <= O_DRAIN;"  # in_last 未握手即排空
mut res_no_last   $MMA "if (f_in.l) res[r][c] <= sum;" "res[r][c] <= sum;"                                      # 結果暫存被下一個 tile 覆蓋
mut no_last_stall $MMA "assign in_ready  = (ist == I_RUN) && !(in_last && ost != O_IDLE);" "assign in_ready  = (ist == I_RUN);"  # 前一個 tile 未輸出完就收最後一步
mut fl_no_skew    $MMA "fl_q[r][c] <= f_in;" "fl_q[r][c] <= fl_in;"                                              # 旗標未隨 token 傳遞
mut first_stuck   $MMA "        first <= 1'b0;" "        first <= first;"                                        # 每一步都清累加器
# (列 ICG 拿掉 !rst_n 是等價 mutant: reset 期間 row_act = 0 表示本列 token 已全 0, 資料暫存只在有 token 時使用, 累加器由
#  token 的 first 旗標清除; 仍保留 !rst_n 以符合共用規則。CU 層級的 ICG 同理, 見 npu_core 段落)
# ---- npu_gemm_cu ----
mut cu_aptr_stuck $CU "a_ptr <= a_ptr + 1'b1; fa <= fa - 1'b1; end" "a_ptr <= a_ptr; fa <= fa - 1'b1; end"   # A line 指標不前進
mut cu_pop_early  $CU "wire pop  = feed && (step == SW'(SPL - 1) || mma_in_last);" "wire pop  = feed && (step == SW'(SPL - 2) || mma_in_last);"  # 提早換 line
mut cu_no_wgnt    $CU "W_WR: if (wr_gnt) begin c_ptr <= c_ptr + 1'b1; w_ln <= w_ln + 1'b1; wst <= w_nxt; end" "W_WR: begin c_ptr <= c_ptr + 1'b1; w_ln <= w_ln + 1'b1; if (wr_gnt) wst <= w_nxt; end"                          # C 指標不等寫入 grant
mut cu_wr_early   $CU "W_WR: if (wr_gnt) begin" "W_WR: if (wr_gnt || wr_req) begin"                         # 未拿到寫入 grant 就結束該 line
# ---- 累加 (v6.3.14) ----
mut cu_acc_nosum  $CU "assign wr_data   = m_w ? wr_mx : x_w ? wr_sum : wr_d;" "assign wr_data   = m_w ? wr_mx : wr_d;"                        # 累加命令不加舊值
mut cu_acc_pend   $CU "x_pend <= x_cmd;" "x_pend <= cmd_acc;"                                                     # 累加旗標取自埠上 (下一個命令)
mut cu_acc_lock   $CU "assign w_sel = w_rq && !lk_f;" "assign w_sel = w_rq;"                                       # 搶走取線器未完成的請求
mut cu_acc_skiprd $CU "          wst <= o_nxt;" "          wst <= (o_nxt == W_RD) ? W_WR : o_nxt;"             # 累加不讀舊值 (o_line 為上一次的)
mut cu_acc_rdgnt  $CU "W_RD: if (w_gnt) begin" "W_RD: if (rd_gnt[0]) begin"                                         # 把取線器的 grant 當成自己的
mut cu_acc_fb     $CU "else if (g0f) f_b <= ~f_b;" "else if (rd_gnt[0]) f_b <= ~f_b;"                               # 寫回器的 grant 也切換 A / B
mut cu_acc_line   $CU "assign rd_line = w_sel ? w_line : f_ln0;" "assign rd_line = f_ln0;"                           # 讀舊值的位址錯
# MX 區塊比例 (v6.3.15)
mut cu_mx_off     $CU "assign wr_data   = m_w ? wr_mx : x_w ? wr_sum : wr_d;" "assign wr_data   = x_w ? wr_sum : wr_d;"   # MX 命令不轉 FP32
mut cu_mx_pend    $CU "m_pend <= m_cmd;" "m_pend <= cmd_mx;"                                                   # MX 旗標取自埠上 (下一個命令)
mut cu_mx_spend   $CU "s_pend <= s_cmd;" "s_pend <= cmd_s_line;"                                               # 比例 line 取自埠上
mut cu_mx_nosc    $CU "W_IDLE: if (take) wst <= m_pend ? W_SC : W_OUT;" "W_IDLE: if (take) wst <= W_OUT;"       # 閒置後的 MX 命令不讀比例
mut cu_mx_nosc2   $CU "W_DONE: if (take) wst <= m_pend ? W_SC : W_OUT;" "W_DONE: if (take) wst <= W_OUT;"       # 連續命令時不讀比例 (用上一個的)
mut cu_mx_scgnt   $CU "W_SC: if (w_gnt) begin" "W_SC: if (rd_gnt[0]) begin"                                     # 把取線器的 grant 當成比例讀取
mut cu_mx_line    $CU "wire [25:0]  w_line = (wst == W_SC) ? s_w : c_ptr;" "wire [25:0]  w_line = c_ptr;"       # 比例 line 位址錯
mut cu_mx_sb      $CU "sc_b[i] <= rd_data[0][256 + 8 * i +: 8];" "sc_b[i] <= rd_data[0][8 * i +: 8];"           # sb 取錯 byte
mut cu_mx_ij      $CU "sc_a[RW'({w_ln, 4'(e)} >> RW)], eb = sc_b[RW'({w_ln, 4'(e)})];" "sc_a[RW'({w_ln, 4'(e)})], eb = sc_b[RW'({w_ln, 4'(e)} >> RW)];"  # 列 / 欄比例對調
# (take 時 w_ln 不歸零是等價 mutant: 每個命令恰好寫 N*N/16 個 line, w_ln 為 log2 寬, 寫完自然回到 0; 中途 reset 也清除)
mut cu_mx_bias    $CU "- 11'sd254;" "- 11'sd253;"                                                              # 指數偏差錯
mut cu_mx_nan     $CU "if (ea == 8'hFF || eb == 8'hFF)" "if (ea == 8'hFF)"                                     # sb = NaN 未處理
mut cu_mx_ovf     $CU "else if (be > 11'sd254)" "else if (be > 11'sd255)"                                      # 溢位邊界錯 (指數 255 → NaN 樣式)
mut cu_mx_unf     $CU "else if (be < 11'sd1)" "else if (be < 11'sd0)"                                          # 下溢邊界錯 (指數 0 → 非正規)
mut cu_mx_zero    $CU "else if (f[30:23] == '0)        g = f;" "else if (1'b0)        g = f;"                   # 部分和 0 被縮放成非 0
mut cu_mx_acc     $CU "assign wr_mx[e] = x_w ? y : g;" "assign wr_mx[e] = g;"                                   # MX 累加不加舊值
mut cu_pend_tag   $CU "c_pend <= c_cmd; t_pend <= t_cmd;" "c_pend <= c_cmd; t_pend <= cmd_tag;"                  # tag 取自埠上 (下一個命令) 而非本命令
mut cu_take_early $CU "(wst == W_DONE && done_ready)" "(wst == W_DONE)"                                       # 完成未握手就換下一個 tile
mut cu_done_noack $CU "else if (done_ready) wst <= W_IDLE;" "else wst <= W_IDLE;"                           # 完成不等握手
mut cu_lr         $CU "lr <= (mma_out_row == RW'(N - 1));" "lr <= (mma_out_row == RW'(N - 2));"                   # 提早完成
mut cu_take_cmd   $CU "if (take) begin c_ptr <= c_pend;" "if (take) begin c_ptr <= c_cmd;"                        # 寫回器取錯 C 位址
mut cu_busy_pend  $CU "assign busy      = (fst != F_IDLE) || pend || (wst != W_IDLE);" "assign busy      = (fst != F_IDLE) || (wst != W_IDLE);"  # busy 漏掉等待輸出
mut cu_last       $CU "assign mma_in_last   = (k_left == 16'd1);" "assign mma_in_last   = (k_left == 16'd2);"  # 少一個 k 步
mut cu_fleft      $CU "fa <= FW'((32'(cmd_k) + SPL - 1) / SPL);" "fa <= FW'((32'(cmd_k) + SPL - 1) / SPL + 1);"   # A 多讀一個 line (帳目錯)
mut cu_bsel       $CU "assign b_vec         = buf_b[rptr][step * N * 8 +: N * 8];" "assign b_vec         = buf_b[wb][step * N * 8 +: N * 8];"  # 讀錯緩衝槽
mut cu_have       $CU "assign have          = (a_cnt != '0) && (b_cnt != '0);" "assign have          = (a_cnt != '0);"  # B 未到齊就餵入
if [ $N -eq 32 ]; then   # 雙讀取埠 (RP = 2) 與一列兩個 C line
  mut cu_no_rgnt  $CU "assign a_gnt = g0f; assign b_gnt = rd_gnt[1];" "assign a_gnt = rd_req[0] && !w_sel; assign b_gnt = rd_gnt[1];"  # A 埠不等讀取 grant
  mut cu_overflow $CU "assign a_req = run && (fa != '0) && (a_cnt != (PW+1)'(PF));" "assign a_req = run && (fa != '0);"  # A 緩衝滿仍預取
  mut cu_b_full   $CU "assign b_req = run && (fb != '0) && (b_cnt != (PW+1)'(PF));" "assign b_req = run && (fb != '0);"  # B 緩衝滿仍預取
  mut cu_hi_half  $CU "else if (out_fire) begin c_hi <= mma_out_vec[N-1:N/2];" "else if (out_fire) begin c_hi <= mma_out_vec[N/2-1:0];"  # 後半列放錯 (曾犯的錯)
  mut cu_one_half $CU "assign w_nxt    = c_half ? (lr ? W_DONE : W_OUT) : (x_w ? W_RD : W_WR);" "assign w_nxt    = lr ? W_DONE : W_OUT;"                     # 每列只寫前半
else                     # 單一讀取埠, A / B 輪流
  mut cu_no_rgnt  $CU "assign a_gnt = g0f && !f_b; assign b_gnt = g0f && f_b;" "assign a_gnt = rd_req[0] && !w_sel && !f_b; assign b_gnt = rd_req[0] && !w_sel && f_b;"  # 不等讀取 grant
  mut cu_overflow $CU "assign a_req = run && !f_b && (fa != '0) && (a_cnt != (PW+1)'(PF));" "assign a_req = run && !f_b && (fa != '0);"  # 緩衝滿仍預取 (覆蓋未用的 line)
  mut cu_alt      $CU "else if (g0f) f_b <= ~f_b;" "else if (rd_req[0] && !w_sel) f_b <= ~f_b;"               # 未 grant 就換軌
fi
[ $N -lt 16 ] && mut cu_slot $CU "c_line_n[slot * N * 32 +: N * 32] = mma_out_vec;" "c_line_n[0 +: N * 32] = mma_out_vec;"  # C 列放錯位置 (N=16 每列一個 line, 不適用)
# ---- npu_l2_xbar ----
mut xb_two_gnt    $XB "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb] && rd_bm[rb] == MW'(m);" "assign rd_gnt[m]  = rd_req[m] && rd_bv[rb];"  # 同 bank 多個讀 grant
mut xb_no_rr      $XB "if (rd_bv[b]) rd_rr[b] <= rd_bm[b];" "if (1'b0) rd_rr[b] <= rd_bm[b];"               # 固定優先權 (可能飢餓)
mut xb_no_wrr     $XB "if (wr_bv[b]) wr_rr[b] <= wr_bm[b];" "if (1'b0) wr_rr[b] <= wr_bm[b];"               # 寫入固定優先權
mut xb_row        $XB "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l, BW, DW)); endfunction" "function automatic logic [DW-1:0] row_of(logic [25:0] l);  return DW'(npu_l2_row(l >> 1, BW, DW)); endfunction"  # bank 列位址錯
mut xb_wr_any     $XB "assign wr_gnt[m]  = wr_req[m] && wr_bv[wb] && wr_bm[wb] == WW'(m);" "assign wr_gnt[m]  = wr_req[m];"  # 寫入不仲裁
mut xb_ls_nowait  $XB "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= '0; end"  # 喚醒不等 SRAM
mut xb_ls_nowake  $XB "if (rd_rq[b] || wr_rq[b]) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end" "if (1'b0) begin ls[b] <= 1'b0; wk[b] <= KW'(LS_WAKE); end"  # 睡了不醒 (死結)
mut xb_ls_rdidle  $XB "else if (rd_rq[b] || wr_rq[b]) idle[b] <= '0;" "else if (wr_rq[b]) idle[b] <= '0;"  # 讀取中仍計閒置 (存取中睡著)
mut xb_ls_rdy     $XB "assign b_rdy[b]   = !ls[b] && wk[b] == '0;" "assign b_rdy[b]   = wk[b] == '0;"   # 睡眠中仍給 grant
mut xb_ls_early   $XB "else if (idle[b] == IW'(LS_IDLE - 1)) begin" "else if (idle[b] == IW'(LS_IDLE - 2)) begin"  # 提早睡
# ---- npu_core ----
# (CU 的 ICG 拿掉 !rst_n 在硬體上是等價 mutant: CU / npu_mma 的每個 flop 都是非同步 reset, reset 不需要時脈。
#  v6.3.12 之前會被抓到, 是因為 Verilator 中 rst_n 從 0 開始時沒有 negedge 事件, 非同步 reset 只在 reset 期間有時脈沿時
#  才生效 — 隨機上電狀態下結果看運氣 (Nano 存活、X1 / Pro 抓到)。仍保留 !rst_n 以符合共用規則, mutant 移除)
mut core_cpl_hold  $CORE "wire cpl_load = (cpl_pend != '0) && (!cpl_valid || cpl_ready);" "wire cpl_load = (cpl_pend != '0);"  # 完成輸出不保持
mut core_gate      $CORE "assign cu_en[i] = cu_busy[i] || cu_v[i];" "assign cu_en[i] = cu_busy[i];"              # 閘控未涵蓋分派拍
mut core_dack      $CORE "assign cu_dack    = ~cpl_pend;" "assign cu_dack    = '1;"                              # 待回報槽未空就接受完成 (遺失)
[ $NCU -gt 1 ] && mut core_idle_1st  $CORE "assign cu_cand    = (cu_idle != '0) ? cu_idle : cu_free;" "assign cu_cand    = cu_free;"  # 不優先閒置 CU (NCU = 1 時等價)
mut core_mx        $CORE ".cmd_mx(head.mx)," ".cmd_mx(1'b0),"                                       # MX 旗標未傳到 CU
mut core_sl        $CORE ".cmd_s_line(head.sl)," ".cmd_s_line(head.c),"                               # 比例 line 傳錯
mut core_q_full    $CORE "assign cmd_ready = (q_cnt != (QW+1)'(QD));" "assign cmd_ready = 1'b1;"                # 佇列滿仍收命令
mut core_tag       $CORE "cpl_tag[i] <= cu_dtag[i];" "cpl_tag[i] <= cu_dtag[i] ^ 8'h01;"                       # 完成 id 錯
mut core_perf_wr   $CORE "perf_busy <= perf_wdata_busy; perf_steps <= perf_wdata_steps;" "perf_busy <= perf_wdata_busy; perf_steps <= perf_steps;"  # 計數器寫入無效
[ $NCU -gt 1 ] && mut core_perf_multi $CORE "perf_steps <= perf_steps + 48'(\$countones(cu_feed));" "perf_steps <= perf_steps + 48'(|cu_feed);"  # 多 CU 同拍只計一次 (NCU=1 等價, 不適用)
echo "=== $CFG: $nk / $nt mutants killed"
[ "$surv" -eq 0 ] || exit 1
