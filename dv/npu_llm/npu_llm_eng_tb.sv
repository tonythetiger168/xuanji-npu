// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_llm_eng — constrained-random TB + scoreboard
//   L2 模型: 稀疏的 line 陣列 (未寫過的 line 讀到依位址決定的偽隨機內容); 讀 / 寫 grant 隨機延遲。
//   參考模型 (本檔): KV 頁管理 (區塊表、參考計數、水位 + FIFO 空閒表) 與 MoE dispatch (兩種模式) 依規格
//   算出每個命令應寫的 line; 每個命令完成後, 比對 L2 的完整內容 (DUT 不可寫到期望以外的 line)。
//   命令組合: KV ALLOC / FREE / FREE_SEQ / SHARE / MAP (mult、跨越 NSLOT、k = 0) 與非法參數 (BADARG),
//   MOE (容量 / 無丟棄、K = 1..16、偏斜分布、top-k 未用槽 0xFFFFFFFF、超出範圍 id、非法參數)。
//   隨機 test_en; 偶爾在命令中途 reset (模型同步清空 KV 狀態)。
// plusargs: +seed=N  +cmds=N (預設 400)
//=============================================================================
`include "orca_pkg.sv"
module npu_llm_eng_tb
  import npu_kv_pkg::*;
#(parameter int NP = 16, NSEQ = 4, NSLOT = 32, BS = 4, RW = 2, NEXP = 16, PW = 9, TW = 6);
  localparam int EW = $clog2(NEXP), CAPMAX = (1 << (PW - EW)) - 1, MAXE = (1 << PW) - 1, RMAX = (1 << RW) - 1;
  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_op = 1'b0;
  logic [25:0] cmd_a = '0, cmd_b = '0, cmd_c = '0;
  logic [31:0] cmd_ext = '0;
  logic [15:0] cmd_k = '0;
  logic cmd_ready, done, busy, rd_req, rd_gnt, wr_req, wr_gnt;
  logic [25:0] rd_line, wr_line;
  logic [511:0] rd_data, wr_data;
  always #5 clk = ~clk;

  npu_llm_eng #(.NP(NP), .NSEQ(NSEQ), .NSLOT(NSLOT), .BS(BS), .RW(RW), .NEXP(NEXP), .PW(PW), .TW(TW)) dut (.*);

  // ---------------- L2 ----------------
  logic [511:0] mem [int], exp_m [int];
  int gnt_p;
  function automatic logic [511:0] dflt(int l);
    logic [511:0] v;
    for (int w = 0; w < 16; w++) v[32 * w +: 32] = 32'(l * 2654435761 + w * 40503 + 7);
    return v;
  endfunction
  function automatic int L(longint l); return int'(l & 64'h3FF_FFFF); endfunction   // L2 line 位址 26 bit 回繞
  function automatic logic [511:0] rd(int l); return mem.exists(l) ? mem[l] : dflt(l); endfunction
  function automatic logic [511:0] erd(int l); l = L(l); return exp_m.exists(l) ? exp_m[l] : dflt(l); endfunction
  function automatic void ewr(int l, int w, logic [31:0] v);
    logic [511:0] x;
    l = L(l); x = erd(l); x[32 * w +: 32] = v; exp_m[l] = x;
  endfunction
  function automatic void eset(int l, logic [511:0] v); exp_m[L(l)] = v; endfunction
  assign rd_data = rd(int'(rd_line));
  always @(negedge clk) begin
    rd_gnt <= rd_req && ($urandom_range(99, 0) < gnt_p);
    wr_gnt <= wr_req && ($urandom_range(99, 0) < gnt_p);
  end
  always @(posedge clk) if (rst_n && wr_req && wr_gnt) mem[int'(wr_line)] = wr_data;

  // ---------------- KV 參考模型 ----------------
  bit mv [NSEQ][NSLOT]; int mp [NSEQ][NSLOT]; int mrc [NP]; int mfq [$]; int mhw;
  function automatic int mfree(); return NP - mhw + mfq.size(); endfunction
  function automatic void kv_reset();
    foreach (mv[s, l]) mv[s][l] = 0; foreach (mrc[p]) mrc[p] = 0; mfq.delete(); mhw = 0;
  endfunction
  function automatic void unmap(int s, int l);
    int p = mp[s][l]; mv[s][l] = 0; mrc[p]--; if (mrc[p] == 0) mfq.push_back(p);
  endfunction
  function automatic void kv_res(int line, int st, int data, int fc);
    logic [511:0] x = '0; x[31:0] = 32'(st); x[63:32] = 32'(data); x[95:64] = 32'(fc); eset(line, x);
  endfunction
  function automatic void kv_model(int sub, int seq, int slot, int src, int a, int k, int mult);
    int st = 0, data = 0, p;
    bit bad = seq >= NSEQ || sub > 4 || (sub != 4 && slot >= NSLOT) || (sub == 3 && src >= NSEQ) || (sub == 4 && k > NSLOT);
    if (bad) begin kv_res(a, 5, 0, mfree()); return; end
    case (sub)
      0: if (mv[seq][slot]) st = 2; else if (mfree() == 0) st = 1;
         else begin
           if (mfq.size() == 0) begin p = mhw; mhw++; end else p = mfq.pop_front();
           mv[seq][slot] = 1; mp[seq][slot] = p; mrc[p] = 1; data = p;
         end
      1: if (!mv[seq][slot]) st = 3; else begin data = mp[seq][slot]; unmap(seq, slot); end
      2: for (int l = 0; l < NSLOT; l++) if (mv[seq][l]) begin unmap(seq, l); data++; end
      3: if (mv[seq][slot]) st = 2; else if (!mv[src][slot]) st = 3; else if (mrc[mp[src][slot]] == RMAX) st = 4;
         else begin p = mp[src][slot]; mv[seq][slot] = 1; mp[seq][slot] = p; mrc[p]++; data = p; end
      4: begin
        int hit = 0;
        for (int i = 0; i < k; i++) begin
          int l = slot + i;
          bit h = l < NSLOT && mv[seq][l];
          ewr(a + i / 16, i % 16, h ? 32'(mp[seq][l] * mult) : 32'hFFFF_FFFF);
          hit += h;
        end
        // 索引 line 中未用的字組: 0
        for (int i = k; i % 16 != 0; i++) ewr(a + i / 16, i % 16, 32'd0);
        kv_res(a + (k + 15) / 16, (hit == k) ? 0 : 3, hit, mfree());
        return;
      end
      default: ;
    endcase
    kv_res(a, st, data, mfree());
  endfunction

  // ---------------- MoE 參考模型 ----------------
  function automatic void moe_model(int a, int c, int T, int K, int ne, bit dl, int cap, int mult);
    int n [NEXP], off [NEXP], P, nval = 0, nerr = 0, gl, ct, cn, sl, i = 0;
    bit bad = K == 0 || K > 16 || ne == 0 || ne > NEXP || cap > CAPMAX || T == 0 || T >= (1 << TW);
    if (bad) begin
      logic [511:0] x = '0; x[31:0] = 32'hFFFF_FFFF; eset(c, x); return;
    end
    P = dl ? T * K : ne * cap;
    gl = (P + 15) / 16; ct = (T * K + 7) / 8; cn = (ne + 7) / 8;
    for (int l = 0; l < gl; l++) eset(c + l, '1);
    foreach (n[e]) begin n[e] = 0; off[e] = 0; end
    if (dl) begin
      for (int t = 0; t < T; t++) for (int jj = 0; jj < K; jj++) begin
        logic [31:0] id = erd(a + 2 * t)[32 * jj +: 32];
        if (id < ne) n[id]++;
      end
      for (int e = 1; e < NEXP; e++) off[e] = off[e - 1] + n[e - 1];
      foreach (n[e]) n[e] = 0;
    end else foreach (off[e]) off[e] = e * cap;
    for (int t = 0; t < T; t++) for (int jj = 0; jj < K; jj++) begin
      logic [31:0] id = erd(a + 2 * t)[32 * jj +: 32];
      logic [31:0] w  = erd(a + 2 * t + 1)[32 * jj +: 32];
      bit drop = 1; int pos = 0;
      if (id < ne) begin
        int r = n[id]; n[id]++; nval++;
        if (dl || r < cap) begin drop = 0; pos = (off[id] + r) % (1 << PW); end   // 溢位時位置以 PW bit 回繞
      end else nerr++;
      if (!drop) ewr(c + pos / 16, pos % 16, 32'(longint'(t) * mult));
      ewr(c + gl + i / 8, 2 * (i % 8), drop ? 32'hFFFF_FFFF : 32'(pos));
      ewr(c + gl + i / 8, 2 * (i % 8) + 1, drop ? 32'd0 : w);
      i++;
    end
    for (; i % 8 != 0; i++) begin ewr(c + gl + i / 8, 2 * (i % 8), 32'd0); ewr(c + gl + i / 8, 2 * (i % 8) + 1, 32'd0); end
    for (int e = 0; e < ne; e++) begin
      ewr(c + gl + ct + e / 8, 2 * (e % 8), 32'(dl ? n[e] : (n[e] > cap ? cap : n[e])));
      ewr(c + gl + ct + e / 8, 2 * (e % 8) + 1, 32'(off[e]));
    end
    for (int e = ne; e % 8 != 0; e++) begin ewr(c + gl + ct + e / 8, 2 * (e % 8), 32'd0); ewr(c + gl + ct + e / 8, 2 * (e % 8) + 1, 32'd0); end
    sl = c + gl + ct + cn;
    begin
      logic [511:0] x = '0;
      x[31:0] = {1'(nval > MAXE), 31'(nerr)}; x[63:32] = 32'(nval); x[95:64] = 32'(P);
      eset(sl, x);
    end
  endfunction

  // ---------------- coverage ----------------
  covergroup cg with function sample(int kind, int v);
    cp_kv:  coverpoint v iff (kind == 0) { bins alloc = {0}; bins free = {1}; bins free_seq = {2}; bins share = {3}; bins map = {4}; bins bad = {5}; }
    cp_kst: coverpoint v iff (kind == 1) { bins ok = {0}; bins nomem = {1}; bins mapped = {2}; bins unmapped = {3}; bins refmax = {4}; bins badarg = {5}; }
    cp_map: coverpoint v iff (kind == 2) { bins k0 = {0}; bins k1 = {1}; bins line = {16}; bins multi = {[17:NSLOT]}; bins mid = {[2:15]}; bins over = {[NSLOT+1:$]}; }
    cp_moe: coverpoint v iff (kind == 3) { bins cap = {0}; bins dropless = {1}; bins bad = {2}; bins k16 = {3}; bins k1 = {4}; bins ovf = {5}; }
  endgroup
  cg u_cg = new;

  // ---------------- 驅動 ----------------
  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  int unsigned seed, ncmd, n_err = 0, n_kv = 0, n_moe = 0, n_rst = 0, n_lines = 0;

  task automatic issue(bit op, int a, int b, int c, int ext, int k);
    cmd_valid = 1'b1; cmd_op = op; cmd_a = 26'(a); cmd_b = 26'(b); cmd_c = 26'(c); cmd_ext = 32'(ext); cmd_k = 16'(k);
    do @(posedge clk); while (!cmd_ready);
    #1 cmd_valid = 1'b0;
  endtask
  task automatic compare(string what);
    int keys [$];
    foreach (mem[l]) keys.push_back(l);
    foreach (exp_m[l]) if (!mem.exists(l)) keys.push_back(l);
    foreach (keys[i]) if (rd(keys[i]) !== erd(keys[i])) begin
      $display("MISMATCH %s line %0d", what, keys[i]);
      for (int w = 0; w < 16; w++) if (rd(keys[i])[32 * w +: 32] !== erd(keys[i])[32 * w +: 32])
        $display("    w%0d dut %h model %h", w, rd(keys[i])[32 * w +: 32], erd(keys[i])[32 * w +: 32]);
      n_err++;
    end
    n_lines += keys.size();
  endtask

  initial begin
    int wd;
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cmds=%d", ncmd)) ncmd = 400;
    process::self().srandom(seed);
    $display("LLMENG: seed %0d cmds %0d", seed, ncmd);
    kv_reset(); gnt_p = 70;
    repeat (2) @(negedge clk); rst_n = 1'b1; @(negedge clk);
    for (int unsigned ci = 0; ci < ncmd; ci++) begin
      bit op, rst_mid; int a, b, c, ext, k;
      mem.delete(); exp_m.delete();                   // 每個命令從乾淨的 L2 比對 (KV 狀態保留在 DUT 與模型)
      test_en = 1'($urandom()); gnt_p = (ur(0, 2) == 0) ? 100 : ur(20, 90);
      op = (ci >= 16 + NSEQ) && (ur(0, 2) == 0);
      // 位址: 整個 26 bit 範圍 (含接近頂端時回繞); 輸入 a 與輸出 c 相距 2^25, 不重疊
      a = (ur(0, 3) == 0) ? 26'h3FF_FFFF - ur(0, 40) : int'($urandom() & 32'h3FF_FFFF);
      c = L(longint'(a) + (1 << 25));
      if (!op) begin
        int sub, seq, slot, src;
        int r;
        r = ur(0, 99);                                // initial 內的區域變數是 static: 不可在宣告時給值
        // ALLOC 40 / SHARE 25 / FREE 13 / MAP 15 / FREE_SEQ 3 / 非法 sub 4 — 讓空閒頁耗盡與參考計數到上限都會發生
        sub = (r < 40) ? 0 : (r < 65) ? 3 : (r < 78) ? 1 : (r < 93) ? 4 : (r < 96) ? 2 : ur(5, 7);
        seq = ur(0, NSEQ - 1); slot = ur(0, NSLOT - 1); src = ur(0, NSEQ - 1);
        if (sub == 3 || sub == 1) begin               // 偏向有映射的 slot (SHARE 的來源 / FREE 的目標)
          int c2 [$];
          int e2;
          c2.delete();
          foreach (mv[s2, l2]) if (mv[s2][l2]) c2.push_back(s2 * NSLOT + l2);
          if (c2.size() != 0 && ur(0, 4) != 0) begin
            e2 = c2[ur(0, c2.size() - 1)];
            slot = e2 % NSLOT;
            if (sub == 3) src = e2 / NSLOT; else seq = e2 / NSLOT;
          end
        end
        if ((sub == 0 || sub == 3) && ur(0, 1)) seq = 0;   // 集中在序列 0, 讓 MAP 有大量有映射的 slot
        if (sub == 4 && ur(0, 1)) begin seq = 0; slot = 0; end
        if (ur(0, 19) == 0) seq = ur(NSEQ, 255);
        if (ur(0, 19) == 0) slot = ur(NSLOT, 65535);
        if (ur(0, 19) == 0) src = ur(NSEQ, 255);
        // 開頭的 directed 命令 (每次 reset 後的第一段): 序列 0 配滿 16 個 slot、slot 0 共享到其他序列直到參考計數上限、
        // MAP 整個序列 (有映射數 16 以上)
        if (ci < 16 + NSEQ) begin
          op = 0;
          if (ci < 16) begin sub = 0; seq = 0; slot = ci; end
          else if (ci < 15 + NSEQ) begin sub = 3; seq = ci - 15; slot = 0; src = 0; end
          else begin sub = 4; seq = 0; slot = 0; end
        end
        k = (ci == 15 + NSEQ) ? NSLOT : (sub == 4) ? ((ur(0, 9) == 0) ? 0 : (ur(0, 9) == 0) ? ur(NSLOT + 1, 65535) : ur(1, NSLOT)) : int'($urandom() & 16'hFFFF);
        if (sub == 4 && ur(0, 3) == 0 && ci >= 16 + NSEQ) slot = (ur(0, 1)) ? ur(0, NSLOT + 4) : 65535 - ur(0, 3);
        ext = (sub == 4) ? ((ur(0, 1)) ? BS : int'($urandom())) : (src | int'($urandom() & 32'hFFFF_FF00));
        b = sub | int'($urandom() & 32'h3FF_FFF8);
        c = seq | (slot << 8) | int'($urandom() & 32'h300_0000);
        kv_model(sub, seq, slot, src, a, k, ext);
        u_cg.sample(0, (seq >= NSEQ || sub > 4 || (sub != 4 && slot >= NSLOT) || (sub == 3 && src >= NSEQ)) ? 5 : sub);
        if (sub == 4) u_cg.sample(2, k);
        u_cg.sample(1, int'(erd((sub == 4 && !(seq >= NSEQ) && k <= NSLOT) ? a + (k + 15) / 16 : a)[31:0]));
        n_kv++;
      end else begin
        int T, K, ne, cap, mult, dmode, hot; bit dl, ovr;
        T = (ur(0, 9) == 0) ? 1 : (ur(0, 9) == 0) ? (1 << TW) - 1 : ur(1, 24); K = (ur(0, 4) == 0) ? 16 : ur(1, 8);
        ne = (ur(0, 3) == 0) ? NEXP : ur(1, NEXP); dl = 1'($urandom());
        cap = (ur(0, 4) == 0) ? ur(0, 1) : ur(2, 16);
        mult = ur(0, 3) == 0 ? 1 : ur(2, 300);
        ovr = (ur(0, 19) == 0);
        if (ovr) begin dl = 1; K = 16; T = ur(MAXE / 16 + 1, (1 << TW) - 1); end   // 項目數超過 2^PW − 1
        dmode = ur(0, 2); hot = ur(0, ne - 1);
        if (ur(0, 14) == 0) begin                        // 非法參數
          case (ur(0, 5)) 0: K = 0; 1: K = ur(17, 31); 2: ne = 0; 3: ne = ur(NEXP + 1, 255); 4: T = ur(1 << TW, 65535); default: T = 0; endcase
        end
        if (ur(0, 14) == 0) cap = ur(CAPMAX + 1, 65535);
        for (int t = 0; t < T && t < (1 << TW); t++) begin
          logic [511:0] il = '0, vl = '0;
          for (int jj = 0; jj < 16; jj++) begin
            int id;
            case (dmode) 1: id = (ur(0, 3) != 0) ? hot : ur(0, ne - 1); 2: id = hot; default: id = ur(0, ne - 1); endcase
            if (ur(0, 19) == 0) id = ur(ne, 1000);
            if (ur(0, 29) == 0) id = 32'hFFFF_FFFF;
            if (ne == 0) id = ur(0, 3);
            il[32 * jj +: 32] = 32'(id); vl[32 * jj +: 32] = $urandom();
          end
          if (t < (1 << TW)) begin      // 非法 T 時只放前面幾個 (不會被讀)
            mem[L(a + 2 * t)] = il; mem[L(a + 2 * t + 1)] = vl; eset(a + 2 * t, il); eset(a + 2 * t + 1, vl);
          end
        end
        b = K | (ne << 5) | (int'(dl) << 13);
        if (ur(0, 4) == 0) mult = ur(60000, 65535);
        b = b | int'($urandom() & 32'h3FF_C000);         // 未用的位元
        ext = (cap & 32'hFFFF) | (mult << 16);
        k = T;
        moe_model(a, c, T, K, ne, dl, cap, mult);
        u_cg.sample(3, (K == 0 || K > 16 || ne == 0 || ne > NEXP || cap > CAPMAX || T == 0 || T >= (1 << TW)) ? 2 : ovr ? 5 : (K == 16) ? 3 : (K == 1) ? 4 : int'(dl));
        n_moe++;
      end
      rst_mid = (ur(0, 39) == 0) && ci >= 16 + NSEQ;
      repeat (ur(0, 3)) @(negedge clk);                // 命令之間隨機閒置
      issue(op, a, b, c, ext, k);
      wd = 0;
      fork
        begin
          while (!done) begin @(posedge clk); #1; wd++; if (wd > 150000) begin $display("ERROR: command timeout"); $display("TB FAIL: timeout"); $finish; end end
        end
        begin if (rst_mid) repeat (ur(1, 40)) @(posedge clk); else wait (1'b0); end
      join_any
      disable fork;
      if (rst_mid && !done && busy) begin
        #1 rst_n = 1'b0; repeat (2) @(negedge clk); rst_n = 1'b1; @(negedge clk);   // reset 跨過一個取樣沿 (SVA 的 disable iff 在取樣時評估)
        kv_reset(); n_rst++;
        continue;
      end
      @(negedge clk);
      compare(op ? "MOE" : "KV");
      if (n_err > 10) break;
    end
    $display("LLMENG: KV %0d MOE %0d resets %0d lines-compared %0d", n_kv, n_moe, n_rst, n_lines);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL: %0d error(s)", n_err);
    $finish;
  end
endmodule
