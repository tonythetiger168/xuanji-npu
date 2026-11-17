// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_kv_pager — constrained-random TB + scoreboard (issue #2 P0)
//   參考模型 (本檔): 區塊表 mv / mp、參考計數 mrc、空閒頁 = 水位 mhw + FIFO mfq (配發順序由規格決定, 逐位元比對)。
//   每拍: 轉換結果與模型比對 (看本拍命令之前的表); 回應依序比對; cmd_ready / rsp_valid 與模型一致;
//   不變式 (後門): 每個頁恰好在「空閒 (水位以上或 FIFO 中)」與「被映射」其中之一, 參考計數 = 映射數 →
//   無重疊配發、無遺失、free_cnt 不超界。
//   情境 (+mode, 每個 phase 隨機): 0 均衡、1 配發壓力 (耗盡)、2 共享鏈 (達上限)、3 FREE_SEQ 密集、4 大量釋放
//   (FIFO 回收 / 滿)。回應回壓 (always / 50% / 稀少), 轉換 pos 偏向邊界 (範圍末端、剛好超出、最大值)。
//   隨機 test_en; 偶爾在 FREE_SEQ 或回應等待中 reset。
// plusargs: +seed=N  +cycles=N (預設 40000)
//=============================================================================
`include "npu_pkg.sv"
module npu_kv_pager_tb
  import npu_kv_pkg::*;
#(parameter int NP = 16, NSEQ = 4, NSLOT = 8, BS = 4, RW = 2, POSW = 7);
  localparam int PW = $clog2(NP), SW = $clog2(NSEQ), LW = $clog2(NSLOT), OW = $clog2(BS);
  localparam int DW = (PW > LW ? PW : LW) + 1, NE = NSEQ * NSLOT, RMAX = (1 << RW) - 1, NPOS = NSLOT * BS;

  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, rsp_ready = 1'b0, tr_valid = 1'b0;
  kv_op_e cmd_op = KV_ALLOC;
  logic [SW-1:0] cmd_seq = '0, cmd_src_seq = '0, tr_seq = '0;
  logic [LW-1:0] cmd_slot = '0;
  logic [POSW-1:0] tr_pos = '0;
  logic cmd_ready, rsp_valid, tr_rvalid, tr_hit, tr_oor, busy;
  kv_st_e rsp_st;
  logic [DW-1:0] rsp_data;
  logic [PW+OW-1:0] tr_line;
  logic [PW:0] free_cnt;
  always #5 clk = ~clk;

  npu_kv_pager #(.NP(NP), .NSEQ(NSEQ), .NSLOT(NSLOT), .BS(BS), .RW(RW), .POSW(POSW)) dut (.*);

  // ---------------- 參考模型 ----------------
  bit mv [NE]; int mp [NE]; int mrc [NP]; int mfq [$]; int mhw;
  bit msw; int msw_seq, msw_slot, msw_n;
  typedef struct { kv_st_e st; int data; kv_op_e op; } rsp_t;
  rsp_t rq [$];
  typedef struct { bit v, hit, oor; int line; } tr_t;
  tr_t tq;                                   // 上一拍送出的轉換的期望
  int unsigned seed, ncyc, n_err = 0, n_rsp = 0, n_tr = 0, n_hit = 0, n_rst = 0, n_inv = 0;
  int unsigned n_st [5][4];                  // [st][op]
  int mode, bp;
  bit fired;

  function automatic int ent(int s, int l); return s * NSLOT + l; endfunction
  function automatic int mfree(); return NP - mhw + mfq.size(); endfunction
  function automatic void mreset();
    foreach (mv[i]) mv[i] = 0; foreach (mrc[i]) mrc[i] = 0;
    mfq.delete(); mhw = 0; msw = 0; rq.delete(); tq = '{default: 0};
  endfunction
  function automatic void unmap(int e);
    int p = mp[e];
    mv[e] = 0; mrc[p]--;
    if (mrc[p] == 0) mfq.push_back(p);
  endfunction
  function automatic rsp_t mcmd(kv_op_e op, int s, int l, int ss);
    rsp_t r; int e = ent(s, l), es = ent(ss, l), p;
    r.op = op; r.st = KV_OK; r.data = 0;
    case (op)
      KV_ALLOC: if (mv[e]) r.st = KV_MAPPED; else if (mfree() == 0) r.st = KV_NOMEM;
                else begin
                  if (mfq.size() == 0) begin p = mhw; mhw = mhw + 1; end else p = mfq.pop_front();
                  mv[e] = 1; mp[e] = p; mrc[p] = 1; r.data = p;
                end
      KV_FREE:  if (!mv[e]) r.st = KV_UNMAPPED; else begin r.data = mp[e]; unmap(e); end
      KV_SHARE: if (mv[e]) r.st = KV_MAPPED; else if (!mv[es]) r.st = KV_UNMAPPED; else if (mrc[mp[es]] == RMAX) r.st = KV_REFMAX;
                else begin mv[e] = 1; mp[e] = mp[es]; mrc[mp[es]]++; r.data = mp[es]; end
      default: ;
    endcase
    return r;
  endfunction
  function automatic tr_t mtr(int s, longint pos);
    tr_t t; int l = int'((pos >> OW) % NSLOT), e = ent(s, l);
    t.v = 1; t.oor = (pos >= NPOS); t.hit = !t.oor && mv[e];
    t.line = t.hit ? mp[e] * BS + int'(pos % BS) : 0;
    return t;
  endfunction

  // ---------------- 功能 coverage ----------------
  covergroup cg_cmd with function sample(int op, int st);
    // 合法組合: ALLOC {OK, NOMEM, MAPPED}, FREE {OK, UNMAPPED}, FREE_SEQ {OK}, SHARE {OK, MAPPED, UNMAPPED, REFMAX}
    cp_os: coverpoint op * 8 + st { bins alloc_ok = {0}; bins alloc_nomem = {1}; bins alloc_mapped = {2};
                                    bins free_ok = {8}; bins free_unmapped = {11}; bins free_seq = {16};
                                    bins share_ok = {24}; bins share_mapped = {26}; bins share_unmapped = {27}; bins share_refmax = {28}; }
  endgroup
  covergroup cg_misc with function sample(int what, int v);
    cp_seq_n: coverpoint v iff (what == 0) { bins none = {0}; bins some = {[1:NSLOT-1]}; bins all = {NSLOT}; }
    cp_tr:    coverpoint v iff (what == 1) { bins miss = {0}; bins hit = {1}; bins oor = {2}; bins hit_last_pos = {3}; bins oor_first = {4}; }
    cp_free:  coverpoint v iff (what == 2) { bins empty = {0}; bins full = {NP}; }
    cp_rc:    coverpoint v iff (what == 3) { bins one = {1}; bins shared = {[2:RMAX-1]}; bins max = {RMAX}; }
  endgroup
  cg_cmd  u_cg_cmd  = new;
  cg_misc u_cg_misc = new;

  // ---------------- 不變式 (後門) ----------------
  task automatic check_inv();
    int cnt [NP]; int refs [NP];
    foreach (cnt[i]) begin cnt[i] = 0; refs[i] = 0; end
    for (int p = int'(dut.hw); p < NP; p++) cnt[p]++;
    for (int i = 0; i < int'(dut.fq_n); i++) cnt[dut.fq[(int'(dut.fq_rd) + i) % NP]]++;
    for (int e = 0; e < NE; e++) if (dut.bt_v[e]) refs[dut.bt_p[e]]++;
    for (int p = 0; p < NP; p++) begin
      bit used = refs[p] != 0;
      if (cnt[p] + int'(used) != 1 || (used && int'(dut.rc[p]) != refs[p])) begin
        $display("INVARIANT ERROR page %0d: free-list count %0d, mappings %0d, rc %0d", p, cnt[p], refs[p], dut.rc[p]);
        n_err++;
      end
      if (used) u_cg_misc.sample(3, refs[p]);
    end
    if (int'(free_cnt) != mfree()) begin $display("MISMATCH free_cnt %0d model %0d", free_cnt, mfree()); n_err++; end
    n_inv++;
  endtask

  // ---------------- 刺激 ----------------
  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  // 挑一個已映射 (want=1) 或未映射 (want=0) 的 entry; 找不到時隨機
  function automatic int pick_ent(bit want);
    int c [$];
    for (int e = 0; e < NE; e++) if (mv[e] == want) c.push_back(e);
    return (c.size() == 0) ? ur(0, NE - 1) : c[ur(0, c.size() - 1)];
  endfunction
  task automatic new_cmd();
    int r = ur(0, 99), e;
    kv_op_e op;
    case (mode)
      1: op = (r < 75) ? KV_ALLOC : (r < 90) ? KV_FREE : (r < 95) ? KV_SHARE : KV_FREE_SEQ;
      2: op = (r < 25) ? KV_ALLOC : (r < 85) ? KV_SHARE : (r < 97) ? KV_FREE : KV_FREE_SEQ;
      3: op = (r < 50) ? KV_ALLOC : (r < 60) ? KV_SHARE : (r < 70) ? KV_FREE : KV_FREE_SEQ;
      4: op = (r < 20) ? KV_ALLOC : (r < 30) ? KV_SHARE : (r < 90) ? KV_FREE : KV_FREE_SEQ;
      default: op = (r < 40) ? KV_ALLOC : (r < 65) ? KV_FREE : (r < 90) ? KV_SHARE : KV_FREE_SEQ;
    endcase
    case (op)
      KV_FREE: e = pick_ent(ur(0, 9) != 0);
      KV_SHARE: begin
        int src = pick_ent(1);
        // 目標: 同一 slot、另一個序列 (大多未映射); 共享鏈模式偏向同一個來源頁
        cmd_src_seq = SW'(src / NSLOT);
        e = ent(ur(0, NSEQ - 1), src % NSLOT);
        if (ur(0, 9) == 0) cmd_src_seq = SW'(ur(0, NSEQ - 1));
      end
      KV_ALLOC: e = pick_ent(ur(0, 9) == 0);
      default:  e = ent(ur(0, NSEQ - 1), 0);
    endcase
    cmd_op = op; cmd_seq = SW'(e / NSLOT); cmd_slot = LW'(e % NSLOT);
    cmd_valid = 1'b1;
  endtask
  task automatic drive();
    // 命令: 未被接受時保持 (fired = 上一個 posedge 已被接受)
    if (fired || !cmd_valid) begin
      cmd_valid = 1'b0;
      if (ur(0, 99) < 70) new_cmd();
    end
    case (bp) 0: rsp_ready = 1'b1; 1: rsp_ready = 1'($urandom()); default: rsp_ready = (ur(0, 7) == 0); endcase
    tr_valid = (ur(0, 99) < 80);
    tr_seq = SW'(ur(0, NSEQ - 1));
    case (ur(0, 9))
      0: tr_pos = POSW'(NPOS - 1);
      1: tr_pos = POSW'(NPOS);
      2: tr_pos = '1;
      3: tr_pos = POSW'(ur(NPOS, (1 << POSW) - 1));
      default: tr_pos = POSW'(ur(0, NPOS - 1));
    endcase
  endtask

  task automatic do_reset();
    rst_n = 1'b0; cmd_valid = 1'b0; tr_valid = 1'b0; fired = 1'b0;
    repeat (ur(1, 3)) @(negedge clk);
    mreset(); rst_n = 1'b1;
  endtask

  // ---------------- 主迴圈: 在 negedge 驅動、比對 (下一個 posedge 生效) ----------------
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cycles=%d", ncyc)) ncyc = 40000;
    process::self().srandom(seed);
    $display("KVPAGER: seed %0d cycles %0d NP %0d NSEQ %0d NSLOT %0d BS %0d RW %0d", seed, ncyc, NP, NSEQ, NSLOT, BS, RW);
    mreset();
    @(negedge clk); do_reset();
    for (int unsigned c = 0; c < ncyc; c++) begin
      if (c % 2000 == 0) begin
        mode = ur(0, 4); bp = ur(0, 2); test_en = 1'($urandom());
      end
      drive();
      #1;
      // 1. 上一拍的轉換結果
      if (tr_rvalid !== tq.v || tr_hit !== tq.hit || tr_oor !== tq.oor || int'(tr_line) !== tq.line) begin
        $display("MISMATCH tr @%0d: dut v%0d hit%0d oor%0d line %0d, model v%0d hit%0d oor%0d line %0d",
                 c, tr_rvalid, tr_hit, tr_oor, tr_line, tq.v, tq.hit, tq.oor, tq.line); n_err++;
      end
      // 2. 回應與 ready
      if (rsp_valid !== (rq.size() != 0)) begin $display("MISMATCH rsp_valid @%0d: dut %0d model %0d", c, rsp_valid, rq.size()); n_err++; end
      if (cmd_ready !== (!msw && (rq.size() == 0 || rsp_ready))) begin $display("MISMATCH cmd_ready @%0d", c); n_err++; end
      if (busy !== msw) begin $display("MISMATCH busy @%0d", c); n_err++; end
      if (rsp_valid && rq.size() != 0 && (rsp_st !== rq[0].st || int'(rsp_data) !== rq[0].data)) begin
        $display("MISMATCH rsp @%0d op %s: dut st %s data %0d, model st %s data %0d", c, rq[0].op.name(), rsp_st.name(), rsp_data,
                 rq[0].st.name(), rq[0].data); n_err++;
      end
      check_inv();
      // 3. 本拍 (下一個 posedge) 的轉換期望: 看命令之前的表
      tq = tr_valid ? mtr(int'(tr_seq), longint'(tr_pos)) : '{default: 0};
      if (tr_valid) begin
        n_tr++; n_hit += tq.hit;
        u_cg_misc.sample(1, tq.oor ? (tr_pos == POSW'(NPOS) ? 4 : 2) : tq.hit ? (tr_pos == POSW'(NPOS - 1) ? 3 : 1) : 0);
      end
      // 4. 回應被取走 → 命令 → FREE_SEQ 一步
      if (rsp_valid && rsp_ready && rq.size() != 0) begin void'(rq.pop_front()); n_rsp++; end
      fired = cmd_valid && cmd_ready;
      if (fired) begin
        if (cmd_op == KV_FREE_SEQ) begin msw = 1; msw_seq = int'(cmd_seq); msw_slot = 0; msw_n = 0; u_cg_cmd.sample(int'(KV_FREE_SEQ), 0); n_st[0][2]++; end
        else begin
          rsp_t r;
          r = mcmd(cmd_op, int'(cmd_seq), int'(cmd_slot), int'(cmd_src_seq));   // 宣告與賦值分開 (initial 內的區域變數是 static)
          rq.push_back(r); u_cg_cmd.sample(int'(cmd_op), int'(r.st)); n_st[int'(r.st)][int'(cmd_op)]++;
        end
        u_cg_misc.sample(2, mfree());
      end else if (msw) begin
        int e;
        e = ent(msw_seq, msw_slot);
        if (mv[e]) begin unmap(e); msw_n++; end
        if (msw_slot == NSLOT - 1) begin
          rsp_t r; r.op = KV_FREE_SEQ; r.st = KV_OK; r.data = msw_n; rq.push_back(r); msw = 0;
          u_cg_misc.sample(0, msw_n);
        end
        msw_slot++;
      end
      u_cg_misc.sample(2, mfree());
      if (n_err > 20) break;
      @(negedge clk);
      // 偶爾 reset (FREE_SEQ 中 / 回應等待中)
      if (ur(0, 2999) == 0 || (msw && ur(0, 399) == 0) || (rq.size() != 0 && !rsp_ready && ur(0, 399) == 0)) begin
        do_reset(); n_rst++;
      end
    end
    $display("KVPAGER: rsp %0d tr %0d (hit %0d) resets %0d inv-checks %0d", n_rsp, n_tr, n_hit, n_rst, n_inv);
    $display("KVPAGER: ALLOC ok/nomem/mapped %0d/%0d/%0d  FREE ok/unmapped %0d/%0d  FREE_SEQ %0d  SHARE ok/mapped/unmapped/refmax %0d/%0d/%0d/%0d",
             n_st[0][0], n_st[1][0], n_st[2][0], n_st[0][1], n_st[3][1], n_st[0][2], n_st[0][3], n_st[2][3], n_st[3][3], n_st[4][3]);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL: %0d error(s)", n_err);
    $finish;
  end
endmodule
