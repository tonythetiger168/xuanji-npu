// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss — CRV TB (每個產品組態各跑一次, 參數由 -G 指定)
//   每個 batch 產生一段隨機命令程式 (GEMM / DMA_LD / DMA_ST / NOP / VEC, 含 K = 0、line 數 0、位址環繞),
//   VEC: 先以 DMA_LD 從外部記憶體載入向量程式, 緊接著 VEC 執行 (程式為 line 複製 / 填值 / 偶爾非法指令,
//   參考模型逐指令執行; 完整 ISA 在 npu_vec / npu_vec_eng 的單元驗證);
//   以「依序執行」的參考模型 (L2 + 外部記憶體) 求得預期結果。產生器追蹤自上一個等待旗標 (wt) 以來
//   所有命令的讀 / 寫範圍: 新命令與其衝突時必設 wt (另有隨機 wt) — 因此 DUT 的並行執行必須等同依序執行。
//   外部記憶體模型: 讀 / 寫 grant 隨機 (與 req 無關)、讀回應依序且延遲隨機 (0..L 拍後、可間斷)。
//   batch 結束 (所有完成都已收到且 busy = 0) 時比對: 整個 L2 (後門) 與外部記憶體所有被寫過的 line、
//   每個 id 恰好完成一次、wt 命令的完成必在所有先前命令之後、perf_steps = 起始值 + 所有 GEMM 的 K。
//   每個 batch 隨機: 各埠 grant / 回壓機率、test_en、效能計數器起始值; 偶爾在 batch 中途 reset
//   (之後以 DUT 狀態重新同步參考模型)。
// plusargs: +seed=N  +batches=N (預設 30)  +cmds=N (每個 batch 的命令數上限, 預設 60)  +no_big (不跑長 DMA)
//           +perf_dma=L: DMA 吞吐量量測 (外部埠全速, 讀取延遲固定 L 拍)
//=============================================================================
module npu_ss_tb #(parameter int N = 8, parameter int NCU = 4, parameter int NB = 4, parameter int DEPTH = 1024,
                    parameter int OUT = 32);
  import orca_pkg::*;
  localparam int QD = 8;                                  // 與 orca_npu_ss 預設一致
  localparam int LINES = NB * DEPTH, SPL = 64 / N, CL = (4 * N * N + 63) / 64;
  localparam int LB = $clog2(LINES);

  logic clk = 1'b0, rst_n = 1'b0, test_en = 1'b0;
  logic cmd_valid = 1'b0, cmd_ready;
  npu_tile_cmd_t cmd = '0;
  logic cpl_valid, cpl_ready = 1'b0, irq, busy, vec_err;
  logic [7:0] cpl_id;
  logic m_rd_req, m_rd_gnt = 1'b0, m_rd_rvalid = 1'b0, m_wr_req, m_wr_gnt = 1'b0;
  logic [31:0] m_rd_addr, m_wr_addr;
  logic [511:0] m_rd_rdata = '0, m_wr_data;
  logic perf_wr = 1'b0;
  logic [47:0] perf_wdata_busy = '0, perf_wdata_steps = '0, perf_busy, perf_steps;
  always #5 clk = ~clk;

  orca_npu_ss #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(QD), .OUT(OUT)) dut (.*);

  function automatic int ur(int lo, int hi); return lo + int'($urandom_range(hi - lo, 0)); endfunction
  function automatic bit rnd(int pct); return ur(0, 99) < pct; endfunction

  // ---------------- 外部記憶體 (DUT 側) 與參考模型 ----------------
  int unsigned salt;
  logic [511:0] emem [int unsigned];
  logic [511:0] rext [int unsigned];
  logic [511:0] rl2 [LINES];
  function automatic logic [511:0] einit(int unsigned a);
    logic [511:0] d;
    for (int w = 0; w < 16; w++) d[w * 32 +: 32] = (a * 32'd2654435761) ^ (32'(w) * 32'd40503 + salt) ^ ((a >> 7) * 32'd97);
    return d;
  endfunction
  function automatic logic [511:0] erd(int unsigned a); return emem.exists(a) ? emem[a] : einit(a); endfunction
  function automatic logic [511:0] rerd(int unsigned a); return rext.exists(a) ? rext[a] : einit(a); endfunction
  function automatic int l2i(logic [25:0] l); return int'(l) % LINES; endfunction
  task automatic l2_bd_wr(int i, logic [511:0] d);
    dut.u_core.u_l2.mem[npu_l2_bank(26'(i), $clog2(NB), $clog2(DEPTH))][npu_l2_row(26'(i), $clog2(NB), $clog2(DEPTH))] = d;
  endtask
  function automatic logic [511:0] l2_bd(int i);
    return dut.u_core.u_l2.mem[npu_l2_bank(26'(i), $clog2(NB), $clog2(DEPTH))][npu_l2_row(26'(i), $clog2(NB), $clog2(DEPTH))];
  endfunction

  // 向量程式 (VEC): 一個 line 4 個指令, 第 j 個在 bit [128j +: 77]; 參考模型只解讀 TB 產生的子集
  int vec_err_exp = 0, vec_err_got = 0;
  function automatic npu_vec_pkg::npu_vec_ins_t vins(logic [25:0] pl, int i);
    return npu_vec_pkg::npu_vec_ins_t'(rl2[l2i(pl + 26'(i / 4))][128 * (i % 4) +: 77]);
  endfunction
  logic [511:0] vreg [8];
  function automatic void vec_exec(npu_tile_cmd_t c);
    bit e = 0;
    for (int i = 0; i < int'(c.k); i++) begin
      npu_vec_pkg::npu_vec_ins_t x;
      x = vins(c.a, i);
      case (x.op)
        npu_vec_pkg::VOP_LD:    vreg[x.vd] = rl2[l2i(x.line)];                   // fn 0 (FP32): 原樣
        npu_vec_pkg::VOP_ST:    rl2[l2i(x.line)] = vreg[x.va];
        npu_vec_pkg::VOP_SPLAT: vreg[x.vd] = {16{x.imm}};
        default:                e = 1;                                           // 非法 op (TB 只產生 11..15)
      endcase
    end
    if (e) vec_err_exp++;
  endfunction

  // 參考模型: 依序執行一個命令
  function automatic void ref_exec(npu_tile_cmd_t c);
    case (c.op)
      NPU_OP_DMA_LD: for (int i = 0; i < int'(c.k); i++) rl2[l2i(c.a + 26'(i))] = rerd(ext_of(c, i));
      NPU_OP_DMA_ST: for (int i = 0; i < int'(c.k); i++) rext[ext_of(c, i)] = rl2[l2i(c.a + 26'(i))];
      NPU_OP_GEMM: if (c.k != 0) begin
        logic [31:0] C [];
        C = new[N * N];
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
          longint acc = 0;
          for (int kk = 0; kk < int'(c.k); kk++) begin
            logic [7:0] av = rl2[l2i(c.a + 26'(kk / SPL))][((kk % SPL) * N + i) * 8 +: 8];
            logic [7:0] bv = rl2[l2i(c.b + 26'(kk / SPL))][((kk % SPL) * N + j) * 8 +: 8];
            acc += (c.sa ? longint'($signed(av)) : longint'(av)) * (c.sb ? longint'($signed(bv)) : longint'(bv));
          end
          C[i * N + j] = 32'(acc);
        end
        for (int l = 0; l < CL; l++) for (int e = 0; e < 16; e++) rl2[l2i(c.c + 26'(l))][e * 32 +: 32] = C[l * 16 + e];
      end
      NPU_OP_VEC: vec_exec(c);
      default: ;
    endcase
  endfunction

  // ---------------- 命令產生器 (衝突時設 wt) ----------------
  bit acc_l2r [LINES], acc_l2w [LINES];
  bit acc_er [int unsigned], acc_ew [int unsigned];
  logic [25:0] hot_l2 [8];
  int unsigned hot_ext [6];
  function automatic void acc_clear();
    foreach (acc_l2r[i]) begin acc_l2r[i] = 0; acc_l2w[i] = 0; end
    acc_er.delete(); acc_ew.delete();
  endfunction
  // 命令的讀 / 寫範圍
  typedef struct { int l2r [$]; int l2w [$]; int unsigned er [$]; int unsigned ew [$]; } fp_t;
  function automatic fp_t footprint(npu_tile_cmd_t c);
    fp_t f;
    case (c.op)
      NPU_OP_DMA_LD: for (int i = 0; i < int'(c.k); i++) begin f.l2w.push_back(l2i(c.a + 26'(i))); f.er.push_back(ext_of(c, i)); end
      NPU_OP_DMA_ST: for (int i = 0; i < int'(c.k); i++) begin f.l2r.push_back(l2i(c.a + 26'(i))); f.ew.push_back(ext_of(c, i)); end
      NPU_OP_GEMM: if (c.k != 0) begin
        for (int l = 0; l < (int'(c.k) + SPL - 1) / SPL; l++) begin f.l2r.push_back(l2i(c.a + 26'(l))); f.l2r.push_back(l2i(c.b + 26'(l))); end
        for (int l = 0; l < CL; l++) f.l2w.push_back(l2i(c.c + 26'(l)));
      end
      NPU_OP_VEC: for (int i = 0; i < int'(c.k); i++) begin                  // 程式 line + 指令的 LD / ST
        npu_vec_pkg::npu_vec_ins_t x;
        x = vins(c.a, i);
        if (i % 4 == 0) f.l2r.push_back(l2i(c.a + 26'(i / 4)));
        if (x.op == npu_vec_pkg::VOP_LD) f.l2r.push_back(l2i(x.line));
        if (x.op == npu_vec_pkg::VOP_ST) f.l2w.push_back(l2i(x.line));
      end
      default: ;
    endcase
    return f;
  endfunction
  function automatic bit conflicts(fp_t f);
    foreach (f.l2w[i]) if (acc_l2r[f.l2w[i]] || acc_l2w[f.l2w[i]]) return 1;
    foreach (f.l2r[i]) if (acc_l2w[f.l2r[i]]) return 1;
    foreach (f.ew[i])  if (acc_er.exists(f.ew[i]) || acc_ew.exists(f.ew[i])) return 1;
    foreach (f.er[i])  if (acc_ew.exists(f.er[i])) return 1;
    return 0;
  endfunction
  function automatic void acc_add(fp_t f);
    foreach (f.l2w[i]) acc_l2w[f.l2w[i]] = 1;
    foreach (f.l2r[i]) acc_l2r[f.l2r[i]] = 1;
    foreach (f.ew[i])  acc_ew[f.ew[i]] = 1;
    foreach (f.er[i])  acc_er[f.er[i]] = 1;
  endfunction
  int p_hot = 50;
  function automatic logic [25:0] pick_l2(int span);
    logic [25:0] l = rnd(p_hot) ? hot_l2[ur(0, 7)] + 26'(ur(0, 3)) : 26'(ur(0, LINES - 1));
    if (rnd(8)) l = 26'(LINES - ur(1, span + 1));        // 跨越 L2 容量折返
    l[25:LB] = (26 - LB)'($urandom());                   // 高位元 (xbar 只解碼低 LB bit)
    if (rnd(3)) l = 26'h3FF_FFFF - 26'(ur(0, span));     // 跨越 26 bit line 位址環繞
    return l;
  endfunction
  function automatic int unsigned pick_ext();
    int unsigned a;
    case (ur(0, 9))
      0:       a = 32'hFFFF_FFFF - 32'(ur(0, 20));   // 外部位址環繞
      1, 2, 3: a = $urandom();
      default: a = hot_ext[ur(0, 5)] + 32'(ur(0, 4));
    endcase
    if (a[31:28] == 4'hE) a[31:28] = 4'hD;               // 0xE000_0000.. 留給向量程式
    return a;
  endfunction
  function automatic logic [15:0] pick_n();
    case (ur(0, 9))
      0:       return 16'd0;
      1:       return 16'd1;
      2, 3, 4: return 16'(ur(2, OUT));
      5, 6, 7: return 16'(ur(OUT + 1, 4 * OUT));
      default: return 16'(ur(4 * OUT + 1, 6 * OUT));
    endcase
  endfunction
  function automatic logic [15:0] pick_k();
    case (ur(0, 11))
      0:       return 16'd0;
      1:       return 16'd1;
      2:       return 16'(SPL);
      3, 4, 5: return 16'(ur(2, 3 * SPL));
      6, 7, 8: return 16'(ur(1, 64));
      default: return 16'(ur(65, 260));
    endcase
  endfunction
  function automatic bit disjoint(int a0, int an, int b0, int bn);   // L2 範圍 (模 LINES) 不重疊
    for (int i = 0; i < an; i++) for (int j = 0; j < bn; j++) if ((a0 + i) % LINES == (b0 + j) % LINES) return 0;
    return 1;
  endfunction
  // mode: 0 = 混合; 1 = GEMM / NOP 突發 (填滿 npu_core 佇列, 各來源的完成同時待回報)
  // DMA 2D: b[15:0] = 每列 line 數 (0 = 1D), c = 外部列距; 第 i 個 line 的外部位址
  function automatic logic [31:0] ext_of(npu_tile_cmd_t c, int i);
    int unsigned cols = int'(c.b[15:0]);
    if (cols == 0) return c.ext + 32'(i);
    return c.ext + 32'(i / cols) * 32'(c.c) + 32'(i % cols);
  endfunction
  function automatic void dma_2d(ref npu_tile_cmd_t c);
    if (rnd(40)) begin            // 2D: 列距大於 / 等於 / 小於 (重疊) / 0 (同一列重複) 每列 line 數, 或較大
      c.b = 26'(rnd(10) ? ur(1, 2) : rnd(8) ? ur(32768, 65535) : ur(1, 12));   // 很寬的列: n < cols, 只有一列
      case (ur(0, 4))
        0: c.c = 26'(int'(c.b[15:0]));
        1: c.c = 26'(int'(c.b[15:0]) + ur(1, 40));
        2: c.c = 26'(ur(0, int'(c.b[15:0])));
        3: c.c = 26'(ur(0, 3));
        default: c.c = 26'(ur(0, 4096));
      endcase
      if (int'(c.k) > 512) c.k = 16'(ur(1, 512));    // 範圍有限 (< 0x20_0000 line): 不碰到 VEC 程式區 0xE000_0000
      if (c.ext >= 32'hDFE0_0000 && c.ext < 32'hE000_0000) c.ext = c.ext - 32'h0020_0000;
    end else begin
      c.b = rnd(50) ? '0 : 26'({10'($urandom()), 16'h0});   // 1D (b 的高位元不影響)
    end
  endfunction
  function automatic npu_tile_cmd_t gen_cmd(logic [7:0] id, int p_wt, int mode);
    npu_tile_cmd_t c;
    fp_t f;
    c = '0; c.id = id; c.ext = $urandom(); c.a = 26'($urandom()); c.b = 26'($urandom()); c.c = 26'($urandom());
    c.k = 16'($urandom()); c.sa = 1'($urandom()); c.sb = 1'($urandom());
    case (mode == 1 ? (rnd(65) ? 0 : rnd(70) ? 19 : 7) : ur(0, 19))
      0, 1, 2, 3, 4, 5, 6: begin
        int al;
        c.op = NPU_OP_GEMM; c.k = pick_k(); al = (int'(c.k) + SPL - 1) / SPL;
        c.a = pick_l2(al); c.b = rnd(10) ? c.a : pick_l2(al);
        do c.c = pick_l2(CL);
        while (c.k != 0 && !(disjoint(l2i(c.c), CL, l2i(c.a), al) && disjoint(l2i(c.c), CL, l2i(c.b), al)));
      end
      7, 8, 9, 10, 11, 12: begin c.op = NPU_OP_DMA_LD; c.k = (mode == 1) ? 16'(ur(0, 3)) : pick_n(); c.a = pick_l2(int'(c.k)); c.ext = pick_ext(); dma_2d(c); end
      13, 14, 15, 16, 17:  begin c.op = NPU_OP_DMA_ST; c.k = pick_n(); c.a = pick_l2(int'(c.k)); c.ext = pick_ext(); dma_2d(c); end
      default: c.op = rnd(80) ? NPU_OP_NOP : npu_op_e'(ur(5, 7));     // 5..7 保留: 當 NOP
    endcase
    f = footprint(c);
    c.wt = rnd(p_wt);
    if (conflicts(f)) c.wt = 1'b1;
    if (c.wt) acc_clear();
    acc_add(f);
    return c;
  endfunction

  // 長 DMA (line 數接近 2^16, L2 範圍折返多次): 涵蓋 DMA 計數器的所有位元
  function automatic npu_tile_cmd_t big_dma(logic [7:0] id, bit st);
    npu_tile_cmd_t c = '0;
    c.id = id; c.op = st ? NPU_OP_DMA_ST : NPU_OP_DMA_LD; c.wt = 1'b1;
    c.k = st ? 16'(ur(40000, 65534)) : 16'hFFFF; c.a = 26'($urandom()); c.ext = $urandom();
    acc_clear(); acc_add(footprint(c));
    return c;
  endfunction

  // VEC: 回傳 {DMA_LD 程式, VEC}; 程式放外部記憶體 0xE000_0000 起 (pick_ext 不會產生這個區域)
  int unsigned vec_ext = 32'hE000_0000;
  function automatic npu_tile_cmd_t mk_cmd(npu_tile_cmd_t c, int p_wt);   // wt 與衝突追蹤 (同 gen_cmd)
    fp_t f;
    f = footprint(c);
    c.wt = rnd(p_wt);
    if (conflicts(f)) c.wt = 1'b1;
    if (c.wt) acc_clear();
    acc_add(f);
    return c;
  endfunction
  function automatic void gen_vec(logic [7:0] id0, logic [7:0] id1, int p_wt, output npu_tile_cmd_t c0, output npu_tile_cmd_t c1);
    npu_vec_pkg::npu_vec_ins_t p [$];
    int m, np;
    logic [25:0] src, dst, pl;
    bit fill, burst, rburst;
    m = rnd(30) ? ur(9, 24) : ur(1, 8); src = pick_l2(m); fill = rnd(30);
    burst  = rnd(25);                                     // 連續 ST (同一暫存器寫 m 個 line): 與 DMA 搶 L2 寫入埠
    rburst = !burst && rnd(30);                           // 連續 LD (m 個 line), 最後一個 ST: 與 DMA ST 搶 L2 讀取埠
    if (rburst) fill = 0;
    do dst = pick_l2(m); while (!disjoint(l2i(dst), m, l2i(src), m) && rnd(70));   // 多半不重疊 (重疊也合法)
    for (int i = 0; i < m; i++) begin
      npu_vec_pkg::npu_vec_ins_t x;
      x = '0;
      if (!burst || i == 0) begin
        if (fill) begin x.op = npu_vec_pkg::VOP_SPLAT; x.vd = 3'(i); x.imm = $urandom(); end
        else begin x.op = npu_vec_pkg::VOP_LD; x.vd = 3'(i); x.line = src + 26'(i); end
        p.push_back(x);
      end
      if (rnd(4)) begin x = '0; x.op = npu_vec_pkg::vop_e'(ur(11, 15)); p.push_back(x); end   // 非法指令
      if (rburst && i != m - 1) continue;
      x = '0; x.op = npu_vec_pkg::VOP_ST; x.va = burst ? 3'd0 : 3'(i); x.line = dst + 26'(i);
      p.push_back(x);
    end
    np = (p.size() + 3) / 4;
    do pl = pick_l2(np); while (!disjoint(l2i(pl), np, l2i(dst), m));   // 程式不可被自己改寫
    for (int l = 0; l < np; l++) begin
      logic [511:0] d;
      d = '0;
      for (int j = 0; j < 4 && 4 * l + j < p.size(); j++) d[128 * j +: 77] = p[4 * l + j];
      rext[vec_ext + 32'(l)] = d; emem[vec_ext + 32'(l)] = d;
    end
    c0 = '0; c0.id = id0; c0.op = NPU_OP_DMA_LD; c0.k = 16'(np); c0.a = pl; c0.ext = vec_ext;
    vec_ext += 32'(np);
    c0 = mk_cmd(c0, p_wt);
    ref_exec(c0);                                         // 程式進入參考 L2, VEC 的 footprint 才能解讀
    c1 = '0; c1.id = id1; c1.op = NPU_OP_VEC; c1.a = pl; c1.k = rnd(5) ? 16'd0 : 16'(p.size());
    c1.ext = $urandom(); c1.b = 26'($urandom()); c1.c = 26'($urandom());
    c1 = mk_cmd(c1, p_wt);
  endfunction

  // ---------------- 驅動 / 監看 (posedge 取樣, 非阻塞驅動) ----------------
  npu_tile_cmd_t prog [$];
  int pidx = 0, ncpl = 0, n_err = 0, n_rst = 0, n_batch = 0, n_cmds = 0;
  int cpl_pos [int];                     // id -> 完成順序
  int prog_pos [int];                    // id -> 程式中的位置
  int p_cmd = 80, p_cpl = 80, p_rgnt = 80, p_wgnt = 80, p_rv = 80, rlat = 4;
  typedef struct { logic [511:0] d; longint t; } rsp_t;
  rsp_t rq [$];
  longint cyc = 0, last_t = 0;
  int fix_lat = -1;                      // +perf_dma: 固定讀取延遲
  longint cpl_cyc [int];                 // id -> 完成拍
  bit in_batch = 0;

  always @(posedge clk) begin
    cyc++;
    if (!rst_n) begin
      cmd_valid <= 1'b0; cpl_ready <= 1'b0; m_rd_gnt <= 1'b0; m_wr_gnt <= 1'b0; m_rd_rvalid <= 1'b0;
      rq.delete();
    end else begin
      // 外部讀: grant → 依序回應 (延遲 0..rlat, 回應之間可間斷)
      if (m_rd_req && m_rd_gnt) begin
        automatic longint t = cyc + longint'(fix_lat >= 0 ? fix_lat : ur(0, rlat));
        if (t <= last_t) t = last_t + 1;
        last_t = t;
        rq.push_back('{erd(m_rd_addr), t});
      end
      if (rq.size() != 0 && rq[0].t <= cyc && rnd(p_rv)) begin m_rd_rvalid <= 1'b1; m_rd_rdata <= rq.pop_front().d; end
      else begin m_rd_rvalid <= 1'b0; m_rd_rdata <= {16{$urandom()}}; end
      m_rd_gnt <= rnd(p_rgnt);
      // 外部寫: grant 拍完成
      if (m_wr_req && m_wr_gnt) emem[m_wr_addr] = m_wr_data;
      m_wr_gnt <= rnd(p_wgnt);
      // 命令
      if (cmd_valid && cmd_ready) pidx++;
      if (cmd_valid && !cmd_ready) ;                       // 保持
      else if (in_batch && pidx < prog.size() && rnd(p_cmd)) begin cmd_valid <= 1'b1; cmd <= prog[pidx]; end
      else begin
        automatic npu_tile_cmd_t j = '0;
        j.id = 8'($urandom()); j.op = npu_op_e'($urandom()); j.k = 16'($urandom()); j.a = 26'($urandom());
        cmd_valid <= 1'b0; cmd <= j;
      end
      // 完成
      if (cpl_valid && cpl_ready) begin
        if (!prog_pos.exists(int'(cpl_id)) || cpl_pos.exists(int'(cpl_id)) || prog_pos[int'(cpl_id)] >= pidx) begin
          if (n_err < 20) $display("ERROR: completion for unknown / duplicate / unissued id %0d", cpl_id);
          n_err++;
        end else begin cpl_pos[int'(cpl_id)] = ncpl; cpl_cyc[int'(cpl_id)] = cyc; end
        ncpl++;
      end
      cpl_ready <= rnd(p_cpl);
      if (irq !== cpl_valid) begin $display("ERROR: irq != cpl_valid"); n_err++; end
      if (vec_err) vec_err_got++;
    end
  end

  // watchdog
  longint last_prog = 0;
  int ncpl_q = 0, pidx_q = 0;
  always @(posedge clk) begin
    if (ncpl != ncpl_q || pidx != pidx_q || !in_batch || (m_rd_req && m_rd_gnt) || (m_wr_req && m_wr_gnt)) begin
      ncpl_q = ncpl; pidx_q = pidx; last_prog = cyc;
    end
    else if (cyc - last_prog > 200_000) begin
      $display("ERROR: watchdog - no progress for 200k cycles (issued %0d / %0d, completed %0d)", pidx, prog.size(), ncpl);
      $display("TB FAIL"); $finish;
    end
  end

  // ---------------- batch ----------------
  task automatic resync();                // reset 後: 參考模型改用 DUT 的實際狀態
    for (int i = 0; i < LINES; i++) rl2[i] = l2_bd(i);
    rext = emem;
  endtask

  task automatic check_state();
    int bad = 0;
    for (int i = 0; i < LINES; i++)
      if (l2_bd(i) !== rl2[i]) begin
        if (bad < 5) $display("MISMATCH L2 line %0d (batch %0d)\n  dut=%h\n  ref=%h", i, n_batch, l2_bd(i), rl2[i]);
        bad++;
      end
    foreach (rext[a]) if (erd(a) !== rext[a]) begin
      if (bad < 10) $display("MISMATCH ext line %h (batch %0d)", a, n_batch);
      bad++;
    end
    foreach (emem[a]) if (!rext.exists(a) && emem[a] !== einit(a)) begin
      if (bad < 10) $display("MISMATCH ext line %h written by DUT only (batch %0d)", a, n_batch);
      bad++;
    end
    n_err += bad;
  endtask

  task automatic batch(int ncmd, bit big = 0);
    logic [7:0] ids [256];
    longint ksum = 0;
    longint unsigned ps0;
    int rst_at, p_wt, mode;
    n_batch++;
    // 各埠壓力
    p_cmd  = (ur(0, 3) == 0) ? 100 : ur(20, 95);
    p_cpl  = (ur(0, 3) == 0) ? ur(3, 15) : ur(50, 100);
    p_rgnt = (ur(0, 3) == 0) ? ur(5, 30) : ur(60, 100);
    p_wgnt = (ur(0, 3) == 0) ? ur(5, 30) : ur(60, 100);
    p_rv   = ur(30, 100);
    rlat   = (ur(0, 2) == 0) ? 0 : (ur(0, 1) == 0) ? ur(1, 12) : ur(OUT, 3 * OUT);   // 長延遲: 在途額度用盡
    p_wt   = ur(0, 3) * 10;
    mode   = (ur(0, 3) == 0) ? 1 : 0;
    p_hot  = (mode == 1) ? 0 : 50;
    if (mode == 1) p_wt = 0;
    test_en = rnd(15);
    if (big) begin p_rgnt = 100; p_wgnt = 100; p_rv = 100; end   // 長 DMA: 外部埠全速
    for (int i = 0; i < 8; i++) hot_l2[i] = 26'(ur(0, LINES - 1));
    for (int i = 0; i < 6; i++) hot_ext[i] = $urandom();
    // 效能計數器起始值 (含接近 2^48 的值)
    @(posedge clk);
    perf_wr <= 1'b1;
    perf_wdata_busy <= {16'($urandom()), 32'($urandom())};
    perf_wdata_steps <= rnd(30) ? 48'hFFFF_FFFF_FFFF - 48'(ur(0, 3000)) : {16'($urandom()), 32'($urandom())};
    @(posedge clk);
    perf_wr <= 1'b0;
    @(posedge clk);
    ps0 = perf_steps;
    // 程式 (id 在 batch 內唯一)
    foreach (ids[i]) ids[i] = 8'(i);
    ids.shuffle();
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    acc_clear();
    vec_err_exp = 0; vec_err_got = 0;
    for (int i = 0; i < ncmd; i++) begin
      if (!big && mode == 0 && i + 1 < ncmd && rnd(20)) begin             // DMA_LD 程式 + VEC
        npu_tile_cmd_t c0, c1;
        gen_vec(ids[i], ids[i + 1], p_wt, c0, c1);
        prog.push_back(c0); prog_pos[int'(c0.id)] = i;
        prog.push_back(c1); prog_pos[int'(c1.id)] = i + 1;
        ref_exec(c1);
        i++;
        if (i + 1 < ncmd && rnd(50)) begin          // 緊接一個 DMA_ST (wt = 0): 與 VEC 同時讀 host L2 埠 (仲裁器的爭用)
          npu_tile_cmd_t c2;
          c2 = '0; c2.id = ids[i + 1]; c2.op = NPU_OP_DMA_ST; c2.k = 16'(ur(8, 48)); c2.a = pick_l2(int'(c2.k)); c2.ext = pick_ext();
          c2 = mk_cmd(c2, 0);
          prog.push_back(c2); prog_pos[int'(c2.id)] = i + 1;
          ref_exec(c2);
          i++;
        end
        continue;
      end
      begin
        automatic npu_tile_cmd_t c = (big && i == 1) ? big_dma(ids[i], 0) : (big && i == 2) ? big_dma(ids[i], 1) :
                                    gen_cmd(ids[i], p_wt, mode);
        prog.push_back(c); prog_pos[int'(c.id)] = i;
        ref_exec(c);
        if (c.op == NPU_OP_GEMM) ksum += longint'(c.k);
      end
    end
    rst_at = (ur(0, 9) == 0 && !big) ? ur(10, 3000) : -1;
    in_batch = 1;
    for (int w = 0; ncpl != ncmd && !(rst_at >= 0 && w >= rst_at); w++) @(posedge clk);
    in_batch = 0;
    if (ncpl != ncmd) begin                // 中途 reset: 丟棄在途命令, 參考模型改用 DUT 狀態
      @(posedge clk); rst_n <= 1'b0;
      repeat (ur(1, 4)) @(posedge clk);
      rst_n <= 1'b1;
      @(posedge clk);
      resync();
      n_rst++;
      return;
    end
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);        // 所有完成都已收到: busy 須在數拍內落下
    repeat (3) @(posedge clk);
    if (busy) begin $display("ERROR: busy after all completions"); $display("TB FAIL"); $finish; end
    check_state();
    // wt 命令的完成必在所有先前命令之後
    foreach (prog[i]) if (prog[i].wt)
      for (int j = 0; j < i; j++)
        if (cpl_pos[int'(prog[j].id)] > cpl_pos[int'(prog[i].id)]) begin
          if (n_err < 20) $display("ERROR: wt command #%0d (id %0d) completed before earlier command #%0d", i, prog[i].id, j);
          n_err++;
        end
    if (vec_err_got != vec_err_exp) begin
      $display("ERROR: vec_err pulses %0d != expected %0d (batch %0d)", vec_err_got, vec_err_exp, n_batch); n_err++;
    end
    if (longint'(perf_steps) != longint'(48'(ps0 + longint'(ksum)))) begin
      $display("ERROR: perf_steps %0d != %0d + %0d", perf_steps, ps0, ksum); n_err++;
    end
    n_cmds += ncmd;
  endtask

  // +perf_dma: 外部埠全速、固定讀取延遲 L 時的 DMA 吞吐量 (line/拍); LD 2048 line 後 (wt) ST 2048 line
  task automatic perf_dma();
    npu_tile_cmd_t c;
    longint t0, t_ld, t_st;
    p_cmd = 100; p_cpl = 100; p_rgnt = 100; p_wgnt = 100; p_rv = 100; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); cpl_cyc.delete(); pidx = 0; ncpl = 0;
    c = '0; c.op = NPU_OP_DMA_LD; c.id = 8'd1; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h1000_0000;
    prog.push_back(c); prog_pos[1] = 0; ref_exec(c);
    c = '0; c.op = NPU_OP_DMA_ST; c.id = 8'd2; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h2000_0000; c.wt = 1'b1;
    prog.push_back(c); prog_pos[2] = 1; ref_exec(c);
    @(posedge clk); t0 = cyc; in_batch = 1;
    while (ncpl != 2) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    t_ld = cpl_cyc[1] - t0; t_st = cpl_cyc[2] - cpl_cyc[1];
    $display("PERFDMA: OUT=%0d read_latency=%0d LD %0d lines %0d cycles (%.3f line/cycle)  ST %0d lines %0d cycles (%.3f line/cycle)",
             OUT, fix_lat, 2048, t_ld, 2048.0 / real'(t_ld), 2048, t_st, 2048.0 / real'(t_st));
  endtask

  // directed: 完成輸出回壓期間填滿所有槽 (npu_core 佇列 + 每個 CU + 其輸出暫存 + tile 輸出暫存, 再加 DMA 與 NOP),
  //           oc 達到上限 CQD + NCU + 4 (界限是緊的); 再加兩個 VEC (程式以後門預先放進 L2): 第二個等引擎忙碌、
  //           再等 VEC 完成槽; 之後放開回壓 (4 個來源同時待回報) 並照常比對
  task automatic fill_all(int prime = 3);   // prime: round-robin 起點 (0 GEMM / 1 DMA / 2 NOP / 3 VEC / 4 VEC, 只有 GEMM 競爭 /
                                            //        5 VEC, 只有 DMA 與 VEC 競爭 / 6 NOP, 只有 NOP 與 VEC 競爭)
    localparam int K = 200, AL = (K + SPL - 1) / SPL, STRIDE = 2 * AL + CL;
    localparam int NG = 4 + NCU + 2;     // npu_core 最多 4 + NCU + 1 個, 另一個已在 tile 輸出暫存
    npu_tile_cmd_t c;
    int ng, oc_exp;
    longint unsigned ps0;
    ps0 = perf_steps;
    n_batch++;
    p_cmd = 100; p_cpl = 0; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    // 兩個 VEC 程式 (各填 2 個 line), 放在 GEMM / DMA 用到的範圍之後
    for (int v = 0; v < 2; v++) begin
      logic [511:0] d;
      npu_vec_pkg::npu_vec_ins_t x;
      d = '0;
      for (int j = 0; j < 4; j++) begin
        x = '0;
        if (j % 2 == 0) begin x.op = npu_vec_pkg::VOP_SPLAT; x.vd = 3'(j); x.imm = $urandom(); end
        else begin x.op = npu_vec_pkg::VOP_ST; x.va = 3'(j - 1); x.line = 26'((NG + 1) * STRIDE + 8 + 4 * v + j); end
        d[128 * j +: 77] = x;
      end
      l2_bd_wr(l2i(26'((NG + 1) * STRIDE + v)), d); rl2[l2i(26'((NG + 1) * STRIDE + v))] = d;
    end
    // 起點: prime != 0 時先送一個該來源的命令, 等它進入輸出暫存 (回壓中) 後才送其餘命令, 使 round-robin 的
    // 「上次選中者」為該來源; GEMM 少一個 (輸出暫存已被佔用), oc 上限不變。prime 4: VEC 起點且不送 DMA / NOP
    // (放開回壓後只有 GEMM 與 VEC 競爭, 上次選中者為 VEC)
    ng = (prime == 0) ? NG : (prime >= 5) ? 0 : NG - 1;
    if (prime != 0) begin
      c = '0; c.id = 8'(200);
      case (prime)
        1: begin c.op = NPU_OP_DMA_LD; c.k = 16'd1; c.a = 26'((NG + 1) * STRIDE + 24); c.ext = $urandom(); end
        2, 6: c.op = NPU_OP_NOP;
        default: begin c.op = NPU_OP_VEC; c.k = 16'd2; c.a = 26'((NG + 1) * STRIDE); end
      endcase
      prog.push_back(c); prog_pos[200] = 0; ref_exec(c);
      in_batch = 1;
      for (int w = 0; w < 20000 && !(cpl_valid && cpl_id == 8'(200)); w++) @(posedge clk);
      if (!(cpl_valid && cpl_id == 8'(200))) begin $display("ERROR: fill_all prime %0d not in output register", prime); n_err++; end
    end
    for (int i = 0; i < NG + 4; i++) begin
      c = '0; c.id = 8'(i);
      if (i < NG) begin
        if (i >= ng) continue;
        c.op = NPU_OP_GEMM; c.k = 16'(K); c.sa = 1'($urandom()); c.sb = 1'($urandom());
        c.a = 26'(i * STRIDE); c.b = c.a + 26'(AL); c.c = c.b + 26'(AL);
      end else if (i == NG) begin
        if (prime == 4 || prime == 6) continue;
        c.op = NPU_OP_DMA_LD; c.k = 16'd2; c.a = 26'(NG * STRIDE); c.ext = $urandom();
      end else if (i == NG + 1) begin
        if (prime == 4 || prime == 5) continue;
        c.op = NPU_OP_NOP;
      end else begin c.op = NPU_OP_VEC; c.k = 16'd4; c.a = 26'((NG + 1) * STRIDE + i - NG - 2); end
      prog_pos[i] = prog.size(); prog.push_back(c); ref_exec(c);
    end
    in_batch = 1;
    // 等到所有槽都滿, 且 DMA / NOP / VEC 的完成都在待回報 (放開回壓後 4 個來源同時競爭)
    oc_exp = (prime == 4) ? 4 + NCU + 2 : (prime >= 5) ? 2 : 4 + NCU + 4;
    for (int w = 0; w < 20000 && !(int'(dut.oc) == oc_exp && (prime == 4 || (prime == 6 ? dut.nop_cv : (dut.dma_cv && (dut.nop_cv || prime == 5)))) && dut.vec_cv); w++) @(posedge clk);
    if (int'(dut.oc) != oc_exp) begin $display("ERROR: fill_all did not reach oc = %0d (oc = %0d)", oc_exp, dut.oc); n_err++; end
    p_cpl = 80;
    while (ncpl != prog.size()) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    if (longint'(perf_steps) != longint'(48'(ps0 + longint'(ng * K)))) begin $display("ERROR: perf_steps (fill_all) %0d != %0d + %0d", perf_steps, ps0, ng * K); n_err++; end
    n_cmds += prog.size();
  endtask

  int unsigned seed, nb, nc;
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("batches=%d", nb)) nb = 30;
    if (!$value$plusargs("cmds=%d", nc)) nc = 60;
    void'($urandom(seed));
    salt = $urandom();
    for (int b = 0; b < NB; b++) for (int d = 0; d < DEPTH; d++) dut.u_core.u_l2.mem[b][d] = {16{$urandom()}};
    for (int i = 0; i < LINES; i++) rl2[i] = l2_bd(i);
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("perf_dma=%d", fix_lat)) perf_dma();
    else for (int b = 0; b < int'(nb); b++) if (b % 10 == 3) begin fill_all((b / 10 + int'(seed)) % 5); if (b == 13) fill_all(5); if (b == 3) fill_all(6); end else batch((b % 5 == 0) ? 1 + ur(0, 3) : ur(nc / 2, nc), b == 1 && !$test$plusargs("no_big"));
    $display("NPUSS: N=%0d NCU=%0d NB=%0d seed=%0d batches=%0d cmds=%0d resets=%0d errors=%0d",
             N, NCU, NB, seed, n_batch, n_cmds, n_rst, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
