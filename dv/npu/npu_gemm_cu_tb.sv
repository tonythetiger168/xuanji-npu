// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_gemm_cu — constrained-random TB
//   L2 行為模型 (64 B line, 組合式讀取) + 隨機 rd_gnt / wr_gnt 延遲 (模擬與其他 L2 使用者的仲裁);
//   每個命令隨機 K (含 >= 32768 的長命令, 累加器模 2^32 環繞)、有號性、A / B / C line 位址 (互不重疊;
//   高位元隨機 — 模型只解碼低 15 bit), 以整數 GEMM 參考模型比對寫回的 C, 並檢查 C 區以外沒有被寫。
//   命令之間隨機間隔, 偶爾在運算中途 reset。v6.3.14: 約 1/3 的命令為累加 (C += A × B, 舊的 C = 隨機背景值)。
//   v6.3.15: 約 1/3 的命令為 MX (FP32 C, E8M0 比例 line; 參考函式 dv/npu/npu_mx_ref.svh): 比例多在 2^±27 內,
//   偶爾極端值 (溢位 / FTZ) 與 0xFF (NaN); 累加時舊的 C 多為與結果量級相近的 FP32, 偶爾任意位元 (NaN / inf / 次正規)。
//   v6.3.16: 約 1/3 的命令為浮點 (FP8 E4M3 / E5M2 混合, 或 BF16 byte-planar; 參考 dv/npu/npu_fp_ref.svh), 可與 MX / 累加並用。
//   v6.3.12 命令重疊: 命令一受理就準備下一個 (CU 最多同時持有 3 個), 每個命令放在獨立的 slot 區域;
//   done_ready 隨機回壓; 完成依受理順序並以 tag 比對。長命令 (K >= 32768) 用整個記憶體, 串行執行。
// plusargs: +seed=N  +cmds=N (預設 200)
//=============================================================================
`include "npu_fp_ref.svh"
module npu_gemm_cu_tb #(parameter int N = 8);
  localparam int RP = (N == 32) ? 2 : 1;         // CU 讀取埠數 (與 npu_gemm_cu 相同公式)
  localparam int KMAX = 65535;
  localparam int AMAX = (KMAX * N + 63) / 64;    // A / B 各最多佔的 line 數 (N=16: 16384, N=32: 32768)
  localparam int LW = (N == 32) ? 17 : 16, LINES = 1 << LW;   // 模型只解碼低 LW bit
  localparam int CL = (4 * N * N + 63) / 64;     // C 佔的 line 數
  localparam int RS = (N == 32) ? 43600 : 21800; // 每塊區域的 line 數 (>= AMAX + 2, 三塊 <= LINES)
  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready, done, feeding, busy;
  logic done_ready = 1'b1;
  logic [7:0] cmd_tag = '0, done_tag;
  logic [25:0] cmd_a_line = '0, cmd_b_line = '0, cmd_c_line = '0;
  logic [15:0] cmd_k = 16'd1;
  logic cmd_a_signed = 1'b0, cmd_b_signed = 1'b0, cmd_acc = 1'b0, cmd_mx = 1'b0, cmd_fp = 1'b0, cmd_bf = 1'b0;
  logic [25:0] cmd_s_line = '0;
  logic [RP-1:0] rd_req, rd_gnt;
  logic wr_req, wr_gnt;
  logic [RP-1:0][25:0] rd_line;
  logic [25:0] wr_line;
  logic [RP-1:0][511:0] rd_data;
  logic [511:0] wr_data;
  always #5 clk = ~clk;

  npu_gemm_cu #(.N(N)) dut (.*);

  // ---------------- L2 模型 ----------------
  logic [511:0] mem [LINES];
  int gnt_pct = 100;                           // 每拍給 grant 的機率 (%)
  logic [RP-1:0] rg;                           // 每個讀取埠各自隨機 grant
  logic wg;
  always @(negedge clk) begin
    for (int p = 0; p < RP; p++) rg[p] = ($urandom_range(99, 0) < gnt_pct);
    wg = ($urandom_range(99, 0) < gnt_pct);
  end
  assign rd_gnt  = rd_req & rg;
  assign wr_gnt  = wr_req && wg;
  for (genvar p = 0; p < RP; p++) begin : g_rd
    assign rd_data[p] = mem[rd_line[p][LW-1:0]];
  end
  always @(posedge clk) if (rst_n && wr_gnt) mem[wr_line[LW-1:0]] <= wr_data;

  // ---------------- 參考模型與檢查 ----------------
  localparam int SL = 1024;                    // 重疊命令的 slot 大小 (line): A / B 各 <= KO*N/64 + 1, C = CL
  localparam int KO = 600;                     // 重疊命令的 K 上限
  localparam int NSLOT = 8;                    // slot 輪用 (在途命令最多 3 + 1 個 < NSLOT)
  typedef struct {
    logic [25:0] a, b, c, s;                   // s: MX 比例 line
    int          k;
    bit          sa, sb, acc, mx, fp, bf, serial, whole;   // whole: 用整個記憶體 (長命令); fp / bf: 浮點格式
    logic [7:0]  tag;
    logic [31:0] C [N][N];
  } rec_t;
  rec_t q [$];                                 // 已受理、尚未完成握手 (依受理順序)
  rec_t cur;                                   // 正在送出的命令
  int unsigned n_fp = 0, n_bf = 0;
  int unsigned seed, ncmds, n_ok = 0, n_rst = 0, n_err = 0, n_ovl = 0, n_three = 0, n_dwait = 0;
  logic [7:0] A[N][KMAX], B[KMAX][N];

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic logic [7:0] pick(int mode);
    case (mode)
      1: case (ur(0, 4)) 0: return 8'h00; 1: return 8'h7F; 2: return 8'h80; 3: return 8'hFF; default: return 8'($urandom()); endcase
      2: return (ur(0, 5) == 0) ? 8'($urandom()) : 8'h00;
      3: return (ur(0, 3) != 0) ? 8'h00 : (ur(0, 1) == 0) ? 8'h01 : 8'hFF;   // 稀疏 0 / ±1: 部分和常為 ±1 (MX 轉換的指數 127)
      4: return (ur(0, 7) != 0) ? 8'hFF : 8'($urandom());   // 接近最大的無號值: 中長 K 時部分和 > 2^23 (FP32 尾數最低位有效)
      default: return 8'($urandom());
    endcase
  endfunction
  function automatic void put8(int unsigned line, int unsigned off, logic [7:0] v);   // off: line 起算的 byte 偏移
    mem[(line + off / 64) % LINES][(off % 64) * 8 +: 8] = v;
  endfunction
  function automatic logic [31:0] get32(int unsigned line, int unsigned off);
    return mem[(line + off / 64) % LINES][(off % 64) * 8 +: 32];
  endfunction
  function automatic logic [7:0] pick_sc();                                 // E8M0 比例
    case (ur(0, 15))
      0: return 8'(ur(0, 30));
      1: return 8'(ur(225, 254));
      2: return (ur(0, 3) == 0) ? 8'hFF : 8'(ur(0, 254));
      default: return 8'(ur(100, 154));
    endcase
  endfunction
  function automatic logic [31:0] pick_old();                               // 累加的舊 C (FP32)
    return (ur(0, 3) == 0) ? $urandom() : {1'($urandom()), 8'(ur(110, 175)), 23'($urandom())};
  endfunction

  // 準備一個命令: 寫入 A / B、背景垃圾, 算出參考 C。serial: 長命令, 用整個記憶體的三塊區域;
  // 否則放在 slot 區域 (A / B / C 在 slot 內的順序隨機)
  function automatic void make_cmd(int k, bit serial, int slot, int unsigned t);
    int mode = ur(0, 4), fmode = ur(0, 2);
    int al = (k * N + 63) / 64;                // A / B 各佔的 line 數
    int perm = ur(0, 5);
    int ra = (perm < 2) ? 0 : (perm < 4) ? 1 : 2;
    int rb = (ra + 1 + perm % 2) % 3, rc = 3 - ra - rb;
    int base[3];
    longint s;
    logic [7:0] av[], bv[];
    if (serial) begin
      for (int r = 0; r < 3; r++) base[r] = RS * r;
      cur.a = {(26-LW)'($urandom()), LW'(ur(base[ra], base[ra] + RS - 2 - al))};
      cur.b = {(26-LW)'($urandom()), LW'(ur(base[rb], base[rb] + RS - 2 - al))};
      cur.c = {(26-LW)'($urandom()), LW'(ur(base[rc], base[rc] + RS - 2 - CL))};
      cur.s = {(26-LW)'($urandom()), LW'(base[rc] + RS - 1)};
    end else begin
      // slot 內: A / B 各 (SL - CL) / 2 line (先後隨機), C 在最後 CL line
      bit sw = 1'($urandom());
      cur.a = {(26-LW)'($urandom()), LW'(slot * SL + (sw ? (SL - CL) / 2 : 0))};
      cur.b = {(26-LW)'($urandom()), LW'(slot * SL + (sw ? 0 : (SL - CL) / 2))};
      cur.c = {(26-LW)'($urandom()), LW'(slot * SL + SL - CL)};
      cur.s = {(26-LW)'($urandom()), LW'(slot * SL + SL - CL - 1)};
    end
    cur.k = k; cur.sa = 1'($urandom()); cur.sb = 1'($urandom()); cur.acc = (ur(0, 2) == 0); cur.mx = (ur(0, 2) == 0); cur.fp = (ur(0, 2) == 0); cur.bf = (ur(0, 1) == 0);   // bf 在整數命令也隨機 (應被忽略)
    if (cur.fp && k > KO) fmode = 0;                                  // 長命令: 有限值 (少 NaN / inf)
    cur.serial = serial; cur.whole = serial; cur.tag = 8'(t);
    for (int l = 0; l < al + 1; l++) begin mem[(cur.a[LW-1:0] + l) % LINES] = {16{$urandom()}}; mem[(cur.b[LW-1:0] + l) % LINES] = {16{$urandom()}}; end
    for (int l = 0; l < CL; l++) mem[(cur.c[LW-1:0] + l) % LINES] = {16{$urandom()}};
    mem[cur.s[LW-1:0]] = {16{$urandom()}};
    if (cur.mx) begin
      for (int i = 0; i < N; i++) begin put8(cur.s[LW-1:0], i, pick_sc()); put8(cur.s[LW-1:0], 32 + i, pick_sc()); end
    end
    if ((cur.mx || cur.fp) && cur.acc)
      for (int e = 0; e < N * N; e++) mem[(cur.c[LW-1:0] + e / 16) % LINES][(e % 16) * 32 +: 32] = pick_old();
    for (int kk = 0; kk < k; kk++) for (int i = 0; i < N; i++) begin
      if (!cur.fp) begin A[i][kk] = pick(mode); B[kk][i] = pick(mode); end
      else if (!cur.bf) begin A[i][kk] = fpr_pick8(cur.sa, fmode); B[kk][i] = fpr_pick8(cur.sb, fmode); end
      else if (kk % 2 == 0) begin                                       // BF16: 第 kk / kk+1 步 = 元素的低 / 高位元組
        {A[i][(kk + 1) % KMAX], A[i][kk]} = fpr_pick16(fmode); {B[(kk + 1) % KMAX][i], B[kk][i]} = fpr_pick16(fmode);
      end
    end
    for (int kk = 0; kk < k; kk++) for (int i = 0; i < N; i++) begin
      put8(cur.a[LW-1:0], kk * N + i, A[i][kk]); put8(cur.b[LW-1:0], kk * N + i, B[kk][i]);
    end
    av = new[k]; bv = new[k];
    if (cur.mx && ur(0, 3) == 0) for (int i = 0; i < N; i++) begin   // 邊界: 對角元素 C[i][i] 縮放後的指數落在溢位 / 下溢邊界
      logic [7:0] ea, eb;
      logic [31:0] p;
      for (int kk = 0; kk < k; kk++) begin av[kk] = A[i][kk]; bv[kk] = B[kk][i]; end
      p = fpr_dot(k, av, bv, cur.sa, cur.sb, cur.fp, cur.bf);
      if (cur.fp ? mx_edge_sc_f(p, (ur(0, 1) == 0) ? ur(-1, 2) : ur(253, 256), ea, eb)
                 : mx_edge_sc(p, (ur(0, 1) == 0) ? ur(-1, 2) : ur(253, 256), ea, eb)) begin
        put8(cur.s[LW-1:0], i, ea); put8(cur.s[LW-1:0], 32 + i, eb);
      end
    end
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
      for (int kk = 0; kk < k; kk++) begin av[kk] = A[i][kk]; bv[kk] = B[kk][j]; end
      cur.C[i][j] = fpr_elem(cur.fp, fpr_dot(k, av, bv, cur.sa, cur.sb, cur.fp, cur.bf), cur.mx, 8'(get32(cur.s[LW-1:0], i)),
                             8'(get32(cur.s[LW-1:0], 32 + j)), cur.acc, get32(cur.c[LW-1:0], 4 * (i * N + j)));
    end
  endfunction

  function automatic void check_done();
    if (q.size() == 0) begin $display("ERROR: done without outstanding command"); n_err++; return; end
    if (done_tag !== q[0].tag) begin $display("ERROR: done_tag %h expected %h", done_tag, q[0].tag); n_err++; end
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
      if (get32(q[0].c[LW-1:0], 4 * (i * N + j)) !== q[0].C[i][j]) begin
        if (n_err < 20) $display("MISMATCH k=%0d sa=%0d sb=%0d acc=%0d mx=%0d fp=%0d bf=%0d C[%0d][%0d] dut=%h ref=%h", q[0].k, q[0].sa, q[0].sb, q[0].acc, q[0].mx, q[0].fp, q[0].bf, i, j,
                                 get32(q[0].c[LW-1:0], 4 * (i * N + j)), q[0].C[i][j]);
        n_err++;
      end
    n_fp += q[0].fp; n_bf += q[0].fp && q[0].bf;
    void'(q.pop_front()); n_ok++;
  endfunction

  // 命令的 K: 依序 = 長命令兩個 (隨機 >= 32768、KMAX), 之後短 / 中 / 單步混合
  function automatic int pick_k(int unsigned t);
    if (t == 0) return ur(32768, KMAX);
    if (t == 1) return KMAX;
    return (ur(0, 5) == 0) ? 1 : (ur(0, 3) == 0) ? ur(60, 300) : (ur(0, 15) == 0) ? ur(301, KO) : ur(1, 70);
  endfunction

  initial begin
    int unsigned t = 0, cyc = 0, slot = 0, gap = 0, dr_hold = 0, last_cyc = 0;
    bit has = 0;
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("cmds=%d", ncmds)) ncmds = 200;
    void'($urandom(seed));
    repeat (3) @(posedge clk); #1 rst_n = 1'b1;
    while ((t < ncmds + 2 || has || q.size() != 0) && n_err <= 30) begin
      // ---- 驅動 (posedge 後 #1) ----
      // 長命令執行中不準備下一個 (它用整個記憶體)
      if (!has && t < ncmds + 2 && gap == 0 && !(q.size() != 0 && q[$].whole)) begin
        int k;                                         // initial 內帶初值的區域變數是 static: 分開賦值
        bit serial;
        k = pick_k(t);
        serial = (k > KO) || (ur(0, 9) == 0);          // 長命令或偶爾串行 (等前面全部完成)
        if (!serial || q.size() == 0) begin
          make_cmd(k, k > KO, slot, t); cur.serial = serial;
          slot = (slot + 1) % NSLOT; t++; has = 1;
          test_en = (ur(0, 3) == 0);
          gnt_pct = (ur(0, 2) == 0) ? 100 : ur(20, 90);
        end
      end else if (gap != 0) gap--;
      cmd_valid = has && !(cur.serial && q.size() != 0);
      cmd_a_line = cmd_valid ? cur.a : 26'($urandom()); cmd_b_line = cmd_valid ? cur.b : 26'($urandom());
      cmd_c_line = cmd_valid ? cur.c : 26'($urandom()); cmd_k = cmd_valid ? 16'(cur.k) : 16'($urandom());
      cmd_a_signed = cmd_valid ? cur.sa : 1'($urandom()); cmd_b_signed = cmd_valid ? cur.sb : 1'($urandom());
      cmd_tag = cmd_valid ? cur.tag : 8'($urandom()); cmd_acc = cmd_valid ? cur.acc : 1'($urandom());
      cmd_mx = cmd_valid ? cur.mx : 1'($urandom()); cmd_s_line = cmd_valid ? cur.s : 26'($urandom());
      cmd_fp = cmd_valid ? cur.fp : 1'($urandom()); cmd_bf = cmd_valid ? cur.bf : 1'($urandom());
      if (dr_hold != 0) begin done_ready = 1'b0; dr_hold--; end
      else begin done_ready = (ur(0, 9) != 0); if (ur(0, 49) == 0) dr_hold = ur(1, 40); end
      // ---- 邊沿前取樣握手 (#2: 避開 negedge 的 grant 亂數) ----
      #2;
      if (wr_req && wr_gnt) begin
        if (q.size() == 0 || wr_line < q[0].c || wr_line >= q[0].c + 26'(CL)) begin
          $display("ERROR: write outside C region line=%h", wr_line); n_err++;
        end
      end
      if (done && !done_ready) n_dwait++;
      if (done && done_ready) begin check_done(); last_cyc = cyc; end
      if (cmd_valid && cmd_ready) begin
        if (q.size() != 0) n_ovl++;
        q.push_back(cur); has = 0; gap = (ur(0, 3) == 0) ? ur(1, 4) : 0;
        if (q.size() == 3) n_three++;
      end
      @(posedge clk); #1; cyc++;
      // 偶爾在運算中途 reset: 丟棄在途命令
      if (q.size() != 0 && !q[0].whole && ur(0, 1999) == 0) begin
        rst_n = 1'b0; cmd_valid = 1'b0; repeat (ur(1, 3)) @(posedge clk); #1 rst_n = 1'b1; n_rst++;
        if (!cmd_ready || busy || done) begin $display("ERROR: not idle after reset"); n_err++; end
        q.delete(); has = 0;
      end
      if (cyc - last_cyc > 40 * (KMAX + 3 * N) + 400) begin $display("ERROR: timeout"); n_err++; break; end
    end
    repeat (2) @(posedge clk);
    if (busy) begin $display("ERROR: busy at end"); n_err++; end
    $display("GEMMCU: seed=%0d cmds_ok=%0d (fp %0d bf16 %0d) resets=%0d overlapped=%0d three=%0d done_wait=%0d errors=%0d",
             seed, n_ok, n_fp, n_bf, n_rst, n_ovl, n_three, n_dwait, n_err);
    if (n_ovl == 0 || n_three == 0) begin $display("ERROR: no overlap exercised"); n_err++; end
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
