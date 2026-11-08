// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_moe_disp — constrained-random TB + scoreboard
//   每個命令: 隨機模式 (容量 / 無丟棄)、cap (含 0 與 1)、cmd_nexp (1..NEXP)、項目數 (1..上限, 偶爾超過 2^PW − 1)、
//   expert 分布 (均勻 / 偏斜到少數熱門 expert / 全部同一個 / 偶爾超出範圍的 id)、token 與 weight 隨機。
//   參考模型 (本檔): 依規格算出每一項的 pos / drop / err 與各 expert 的 {n, off}; 輸出依序逐項比對。
//   另外檢查: 非 drop 項目的 pos 兩兩相異且落在該 expert 的區域內 (等於 dispatch 是一個排列 / 單射)。
//   輸入隨機空拍、輸出與數量回壓隨機; 隨機 test_en; 偶爾在命令中途 reset。
// plusargs: +seed=N  +cmds=N (預設 300)
//=============================================================================
`include "orca_pkg.sv"
module npu_moe_disp_tb #(parameter int NEXP = 8, TW = 8, PW = 9, WW = 16);
  localparam int EW = $clog2(NEXP), CW = PW - EW, MAXE = (1 << PW) - 1;
  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_dropless = 1'b0, in_valid = 1'b0, in_last = 1'b0, out_ready = 1'b0, cnt_ready = 1'b0;
  logic [CW-1:0] cmd_cap = '0;
  logic [EW:0] cmd_nexp = '0;
  logic [EW+1:0] in_exp = '0;
  logic [TW-1:0] in_tok = '0;
  logic [WW-1:0] in_w = '0;
  logic cmd_ready, in_ready, out_valid, out_drop, out_err, cnt_valid, done, ovf, busy;
  logic [PW-1:0] out_pos;
  logic [EW-1:0] out_exp, cnt_exp;
  logic [TW-1:0] out_tok;
  logic [WW-1:0] out_w;
  logic [PW:0] cnt_n, cnt_off;
  always #5 clk = ~clk;

  npu_moe_disp #(.NEXP(NEXP), .TW(TW), .PW(PW), .WW(WW)) dut (.*);

  typedef struct { int exp, tok, w; } ent_t;
  typedef struct { int pos, exp, tok, w; bit drop, err; } out_t;
  ent_t L [$];
  out_t X [$];
  int xn [NEXP], xoff [NEXP];
  bit cdl; int ccap, cne, bpo, bpc, gap;
  int unsigned seed, ncmd, n_err = 0, n_ent = 0, n_drop = 0, n_bad = 0, n_rst = 0, n_ovf = 0, n_done = 0;
  bit aborted;
  int unsigned wd_cnt, wd_lim;
  // watchdog: 每個命令的拍數上限 (回壓最稀少時每項約 6 拍); mutant 讓命令永遠不結束時以失敗結束
  always @(negedge clk) begin
    wd_cnt++;
    if (wd_lim != 0 && wd_cnt > wd_lim) begin $display("ERROR: command timeout"); $display("TB FAIL: timeout"); $finish; end
  end

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction

  covergroup cg_cmd with function sample(bit dl, int cap, int ne, int nent, int ndrop, int nbad, bit ov);
    cp_dl:   coverpoint dl;
    cp_cap:  coverpoint cap iff (!dl) { bins zero = {0}; bins one = {1}; bins mid = {[2:(1 << CW) - 2]}; bins max = {(1 << CW) - 1}; }
    cp_ne:   coverpoint ne { bins one = {1}; bins mid = {[2:NEXP-1]}; bins all = {NEXP}; }
    cp_nent: coverpoint nent { bins one = {1}; bins few = {[2:16]}; bins many = {[17:MAXE-1]}; bins limit = {MAXE}; bins over = {[MAXE+1:$]}; }
    cp_drop: coverpoint ndrop iff (!dl) { bins none = {0}; bins some = {[1:$]}; }
    cp_bad:  coverpoint nbad { bins none = {0}; bins some = {[1:$]}; }
    cp_ovf:  coverpoint ov;
    x_dl_bad: cross cp_dl, cp_bad;
  endgroup
  covergroup cg_cnt with function sample(bit dl, int n, int cap);
    cp_n: coverpoint n { bins zero = {0}; bins some = {[1:$]}; }
    cp_full: coverpoint (!dl && n == cap && cap != 0);
  endgroup
  cg_cmd u_cg_cmd = new;
  cg_cnt u_cg_cnt = new;

  // ---------------- 參考模型 ----------------
  function automatic void model();
    int n [NEXP]; int tot = 0;
    foreach (n[e]) n[e] = 0;
    X.delete();
    if (cdl) begin
      foreach (L[j]) if (L[j].exp < cne) n[L[j].exp]++;
      xoff[0] = 0;
      for (int e = 1; e < NEXP; e++) xoff[e] = xoff[e - 1] + ((e - 1 < cne) ? n[e - 1] : 0);
      foreach (n[e]) begin xn[e] = n[e]; n[e] = 0; end
    end else foreach (xoff[e]) xoff[e] = e * ccap;
    foreach (L[j]) begin
      out_t o; o.tok = L[j].tok; o.w = L[j].w; o.err = L[j].exp >= cne; o.exp = o.err ? 0 : L[j].exp;
      o.drop = o.err; o.pos = 0;
      if (!o.err) begin
        int r = n[o.exp];
        n[o.exp]++;
        if (!cdl && r >= ccap) o.drop = 1;
        else o.pos = (xoff[o.exp] + r) % (1 << PW);
      end
      X.push_back(o);
    end
    if (!cdl) foreach (n[e]) xn[e] = (n[e] > ccap) ? ccap : n[e];
  endfunction

  // ---------------- 驅動 / 監看 ----------------
  task automatic send_pass();
    foreach (L[j]) begin
      if (aborted) return;
      while (ur(0, 99) < gap) begin in_valid = 1'b0; @(negedge clk); if (aborted) return; end
      in_valid = 1'b1; in_exp = (EW+2)'(L[j].exp); in_tok = TW'(L[j].tok); in_w = WW'(L[j].w); in_last = (j == L.size() - 1);
      do @(posedge clk); while (!in_ready && !aborted);
      @(negedge clk);
    end
    in_valid = 1'b0; in_last = 1'b0;
  endtask
  task automatic drive();
    cmd_valid = 1'b1; cmd_dropless = cdl; cmd_cap = CW'(ccap); cmd_nexp = (EW+1)'(cne);
    do @(posedge clk); while (!cmd_ready && !aborted);
    @(negedge clk); cmd_valid = 1'b0;
    if (cdl) send_pass();
    send_pass();
  endtask
  task automatic mon_out(input bit chk);
    int j = 0;
    bit seen [int];
    while (j < X.size() && !aborted) begin
      out_ready = (bpo == 0) ? 1'b1 : (bpo == 1) ? 1'($urandom()) : (ur(0, 5) == 0);
      @(posedge clk);
      if (aborted) break;
      if (out_valid && out_ready) begin
        out_t x = X[j];
        if (chk && (out_pos !== PW'(x.pos) || out_drop !== x.drop || out_err !== x.err || out_exp !== EW'(x.exp) ||
            out_tok !== TW'(x.tok) || out_w !== WW'(x.w))) begin
          $display("MISMATCH out #%0d: dut pos %0d e %0d tok %0d drop %0d err %0d / model pos %0d e %0d tok %0d drop %0d err %0d",
                   j, out_pos, out_exp, out_tok, out_drop, out_err, x.pos, x.exp, x.tok, x.drop, x.err); n_err++;
        end
        if (chk && !out_drop) begin
          if (seen.exists(int'(out_pos))) begin $display("ERROR: pos %0d dispatched twice", out_pos); n_err++; end
          seen[int'(out_pos)] = 1;
          if (int'(out_pos) < xoff[out_exp] || int'(out_pos) >= xoff[out_exp] + xn[out_exp] + (cdl ? 0 : 0)) begin
            $display("ERROR: pos %0d outside expert %0d region [%0d, +%0d)", out_pos, out_exp, xoff[out_exp], xn[out_exp]); n_err++;
          end
        end
        j++;
      end
      @(negedge clk);
    end
    out_ready = 1'b0;
  endtask
  task automatic mon_cnt(input bit chk);
    int e = 0;
    while (e < cne && !aborted) begin
      cnt_ready = (bpc == 0) ? 1'b1 : 1'($urandom());
      @(posedge clk);
      if (aborted) break;
      if (cnt_valid && cnt_ready) begin
        if (chk && (cnt_exp !== EW'(e) || int'(cnt_n) !== xn[e] || int'(cnt_off) !== xoff[e])) begin
          $display("MISMATCH cnt e %0d: dut e %0d n %0d off %0d / model n %0d off %0d", e, cnt_exp, cnt_n, cnt_off, xn[e], xoff[e]); n_err++;
        end
        u_cg_cnt.sample(cdl, xn[e], ccap);
        e++;
        #1 if (e == cne && done !== 1'b1) begin $display("ERROR: no done after last count"); n_err++; end
      end
      @(negedge clk);
    end
    cnt_ready = 1'b0;
  endtask

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cmds=%d", ncmd)) ncmd = 300;
    process::self().srandom(seed);
    $display("MOEDISP: seed %0d cmds %0d NEXP %0d PW %0d", seed, ncmd, NEXP, PW);
    repeat (2) @(negedge clk); rst_n = 1'b1;
    for (int unsigned c = 0; c < ncmd; c++) begin
      int nent, dmode, hot, ndrop, nbad; bit ov, rst_mid;
      ndrop = 0; nbad = 0;                           // initial 內的區域變數是 static: 每個命令明確清除
      test_en = 1'($urandom());
      cdl = 1'($urandom()); cne = (ur(0, 3) == 0) ? NEXP : ur(1, NEXP);
      case (ur(0, 4)) 0: ccap = 0; 1: ccap = 1; 2: ccap = (1 << CW) - 1; default: ccap = ur(2, 12); endcase
      // 4: 總數超過上限、但扣掉超出範圍的 id 後剛好在上限內 (錯誤項目不計入上限)
      case (ur(0, 10)) 0: nent = 1; 1: nent = MAXE; 2: nent = MAXE + ur(40, 80); 3: nent = ur(17, MAXE - 1); 4: nent = MAXE + 2; default: nent = ur(2, 40); endcase
      dmode = ur(0, 3); hot = ur(0, cne - 1);
      L.delete();
      for (int j = 0; j < nent; j++) begin
        ent_t x;
        case (dmode)
          1: x.exp = (ur(0, 3) != 0) ? hot : ur(0, cne - 1);
          2: x.exp = hot;
          default: x.exp = ur(0, cne - 1);
        endcase
        if (ur(0, 19) == 0 || (nent == MAXE + 2 && j < 8)) x.exp = ur(cne, (1 << (EW + 2)) - 1);     // 超出範圍的 id
        x.tok = ur(0, (1 << TW) - 1); x.w = ur(0, (1 << WW) - 1);
        L.push_back(x);
      end
      model();
      foreach (X[j]) begin ndrop += X[j].drop && !X[j].err; nbad += X[j].err; end
      ov = (nent - nbad) > MAXE;                     // 只計有效的項目
      bpo = ur(0, 2); bpc = ur(0, 1); gap = (ur(0, 2) == 0) ? 0 : ur(5, 40);
      aborted = 1'b0;
      wd_cnt = 0; wd_lim = 40 * (2 * nent + NEXP) + 1000;
      rst_mid = (ur(0, 24) == 0);
      fork
        drive();
        mon_out(!ov);
        mon_cnt(!ov);
        if (rst_mid) begin
          repeat (ur(1, 3 * nent + NEXP)) @(negedge clk);
          aborted = 1'b1;
        end
      join_any
      if (rst_mid) begin
        wait fork;
        if (aborted) begin
          rst_n = 1'b0; cmd_valid = 1'b0; in_valid = 1'b0; in_last = 1'b0; out_ready = 1'b0; cnt_ready = 1'b0;
          @(negedge clk); rst_n = 1'b1; n_rst++;
          if (ovf !== 1'b0 || busy !== 1'b0 || out_valid !== 1'b0) begin $display("ERROR: state after reset"); n_err++; end
          continue;
        end
      end else wait fork;
      @(negedge clk);
      if (ovf !== ov) begin $display("MISMATCH ovf: dut %0d model %0d (entries %0d)", ovf, ov, nent); n_err++; end
      if (busy) begin $display("ERROR: busy after command"); n_err++; end
      u_cg_cmd.sample(cdl, ccap, cne, nent, ndrop, nbad, ov);
      n_ent += nent; n_drop += ndrop; n_bad += nbad; n_ovf += ov; n_done++;
      if (n_err > 20) break;
    end
    $display("MOEDISP: commands %0d entries %0d dropped %0d bad-id %0d overflow %0d resets %0d", n_done, n_ent, n_drop, n_bad, n_ovf, n_rst);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL: %0d error(s)", n_err);
    $finish;
  end
endmodule
