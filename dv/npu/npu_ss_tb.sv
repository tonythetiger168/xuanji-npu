// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss — CRV TB (每個產品組態各跑一次, 參數由 -G 指定)
//   每個 batch 產生一段隨機命令程式 (GEMM / DMA_LD / DMA_ST / NOP / VEC / KV / MOE, 含 K = 0、line 數 0、位址環繞),
//   VEC: 先以 DMA_LD 從外部記憶體載入向量程式, 緊接著 VEC 執行 (程式為 line 複製 / 填值 / 偶爾非法指令,
//   參考模型逐指令執行; 完整 ISA 在 npu_vec / npu_vec_eng 的單元驗證);
//   KV (v6.3.13): 參考模型維護 KV 頁管理的狀態 (區塊表 / 參考計數 / 空閒表), 跨 batch 保留, reset 時清空;
//   MOE: 先以 DMA_LD 從外部記憶體載入 top-k 結果, 緊接著 MOE (參考模型: docs/npu_llm.md 的規格; 完整情境在
//   npu_llm_eng 的單元驗證);
//   以「依序執行」的參考模型 (L2 + 外部記憶體) 求得預期結果。產生器追蹤自上一個等待旗標 (wt) 以來
//   所有命令的讀 / 寫範圍: 新命令與其衝突時必設 wt (另有隨機 wt) — 因此 DUT 的並行執行必須等同依序執行。
//   外部記憶體模型: 讀 / 寫 grant 隨機 (與 req 無關)、讀回應依序且延遲隨機 (0..L 拍後、可間斷);
//   部分 batch 隨機出現整段停擺 (100..1500 拍無 grant / 回應)。
//   batch 結束 (所有完成都已收到且 busy = 0) 時比對: 整個 L2 (後門) 與外部記憶體所有被寫過的 line、
//   每個 id 恰好完成一次、wt 命令的完成必在所有先前命令之後、perf_steps = 起始值 + 所有 GEMM 的 K。
//   每個 batch 隨機: 各埠 grant / 回壓機率、test_en、效能計數器起始值; 偶爾在 batch 中途 reset
//   (之後以 DUT 狀態重新同步參考模型)。
// plusargs: +seed=N  +batches=N (預設 30)  +cmds=N (每個 batch 的命令數上限, 預設 60)  +no_big (不跑長 DMA)
//           +cdv: coverage-driven — 依 dv/npu/npu_ss_e2e_sva.sv 的未命中 bin 調整每個 batch 的激勵, 收斂後停止
//           +perf_dma=L: DMA 吞吐量量測 (外部埠全速, 讀取延遲固定 L 拍)
//=============================================================================
module npu_ss_tb #(parameter int N = 8, parameter int NCU = 4, parameter int NB = 4, parameter int DEPTH = 1024,
                    parameter int OUT = 32, parameter int NDC = 2);
  import orca_pkg::*;
  localparam int QD = 8;                                  // 與 orca_npu_ss 預設一致
  localparam int LINES = NB * DEPTH, SPL = 64 / N, CL = (4 * N * N + 63) / 64;
  localparam int LB = $clog2(LINES);
  // LLM 引擎 (與 orca_npu_ss 預設一致)
  localparam int KV_NP = 256, KV_NSEQ = 16, KV_NSLOT = 64, KV_RW = 3, MOE_NEXP = 64, MOE_PW = 14, MOE_TW = 12;
  localparam int MOE_CAPMAX = (1 << (MOE_PW - $clog2(MOE_NEXP))) - 1, KV_RMAX = (1 << KV_RW) - 1;
  bit cdv;                                                // +cdv (見下方 cdv_bias)
  int focus = -1;                                         // 偏向的 op 類別 (-1: 無)
  int focus_sel [4] = '{0, 7, 13, 19};                    // gen_cmd 的選擇值: GEMM / DMA_LD / DMA_ST / NOP (4 VEC / 5 KV、MOE 另外)

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

  orca_npu_ss #(.N(N), .NCU(NCU), .NB(NB), .DEPTH(DEPTH), .QD(QD), .OUT(OUT), .NDC(NDC),
                .KV_NP(KV_NP), .KV_NSEQ(KV_NSEQ), .KV_NSLOT(KV_NSLOT), .KV_RW(KV_RW),
                .MOE_NEXP(MOE_NEXP), .MOE_PW(MOE_PW), .MOE_TW(MOE_TW)) dut (.*);

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
        default:                e = 1;                                           // 非法 op (TB 只產生 12..15; 11 = TOPK)
      endcase
    end
    if (e) vec_err_exp++;
  endfunction

  // 載入時解碼 (v6.3.13, DMA_LD 的 b[17:16]): 1 INT4 / 2 UINT4 / 3 MXFP4 (E2M1 × 2); 參考值以實數計算 E2M1 的定義
  function automatic logic [7:0] ref_dec4(int f, logic [3:0] x);
    real v;
    int e = int'(x[2:1]), m = int'(x[0]);
    if (f == 1) return 8'($signed(x));
    if (f == 2) return {4'd0, x};
    v = (e == 0) ? 0.5 * m : (2.0 ** (e - 1)) * (1.0 + 0.5 * m);   // E2M1: 偏移 1, 次正規 0 / 0.5
    v = v * 2.0;
    return x[3] ? 8'(-int'(v)) : 8'(int'(v));
  endfunction
  function automatic logic [511:0] ref_dec(int f, logic [511:0] d, int h);
    logic [511:0] o;
    for (int e = 0; e < 64; e++) o[e * 8 +: 8] = ref_dec4(f, d[h * 256 + e * 4 +: 4]);
    return o;
  endfunction
  function automatic int dma_fmt(npu_tile_cmd_t c); return (c.op == NPU_OP_DMA_LD) ? int'(c.b[17:16]) : 0; endfunction
  // 間接 (v6.3.13, b[18]): 位址表在 L2 line c 起 (每 line 16 項 u32), k 項, 每項 cw = (b[15:0] = 0 ? 1 : b[15:0]) 個 line
  function automatic int dma_cw(npu_tile_cmd_t c); return (c.b[18] && c.b[15:0] != 0) ? int'(c.b[15:0]) : 1; endfunction
  function automatic int dma_nl(npu_tile_cmd_t c); return int'(c.k) * dma_cw(c); endfunction      // 資料 line 數
  function automatic int dma_l2n(npu_tile_cmd_t c); return (dma_fmt(c) != 0) ? 2 * dma_nl(c) : dma_nl(c); endfunction
  function automatic logic [31:0] ind_ent(npu_tile_cmd_t c, int i);
    return rl2[l2i(c.c + 26'(i / 16))][(i % 16) * 32 +: 32];
  endfunction
  // 第 i 個資料 line 的外部位址; 間接的略過項回傳 0 並設 skip
  function automatic logic [31:0] dma_ext(npu_tile_cmd_t c, int i, output bit skip);
    skip = 0;
    if (!c.b[18]) return ext_of(c, i);
    begin
      logic [31:0] e = ind_ent(c, i / dma_cw(c));
      if (e == 32'hFFFF_FFFF) begin skip = 1; return '0; end
      return c.ext + e + 32'(i % dma_cw(c));
    end
  endfunction

  // ---------------- KV / MOE 參考模型 (docs/npu_llm.md) ----------------
  function automatic void ewr(logic [25:0] l, int w, logic [31:0] v); rl2[l2i(l)][32 * w +: 32] = v; endfunction
  function automatic void eset(logic [25:0] l, logic [511:0] v); rl2[l2i(l)] = v; endfunction
  bit kv_v [KV_NSEQ][KV_NSLOT];
  int kv_p [KV_NSEQ][KV_NSLOT];
  int kv_rc [KV_NP];
  int kv_fq [$];
  int kv_hw;
  int n_kv = 0, n_moe = 0;
  function automatic int kv_free(); return KV_NP - kv_hw + kv_fq.size(); endfunction
  function automatic void kv_reset();
    foreach (kv_v[s2, l2]) kv_v[s2][l2] = 0;
    foreach (kv_rc[p2]) kv_rc[p2] = 0;
    kv_fq.delete(); kv_hw = 0;
  endfunction
  function automatic void kv_unmap(int sq, int sl);
    int pg = kv_p[sq][sl];
    kv_v[sq][sl] = 0; kv_rc[pg]--;
    if (kv_rc[pg] == 0) kv_fq.push_back(pg);
  endfunction
  function automatic void kv_res(logic [25:0] l, int st, int data);
    logic [511:0] x = '0; x[31:0] = 32'(st); x[63:32] = 32'(data); x[95:64] = 32'(kv_free()); eset(l, x);
  endfunction
  function automatic int kv_sub(npu_tile_cmd_t c); return int'(c.b[2:0]); endfunction
  function automatic bit kv_bad(npu_tile_cmd_t c);
    int sub = kv_sub(c), seq = int'(c.c[7:0]), slot = int'(c.c[23:8]), src = int'(c.ext[7:0]);
    return seq >= KV_NSEQ || sub > 4 || (sub != 4 && slot >= KV_NSLOT) || (sub == 3 && src >= KV_NSEQ) ||
           (sub == 4 && int'(c.k) > KV_NSLOT);
  endfunction
  function automatic int kv_nl(npu_tile_cmd_t c); return (!kv_bad(c) && kv_sub(c) == 4) ? (int'(c.k) + 15) / 16 + 1 : 1; endfunction
  function automatic void kv_exec(npu_tile_cmd_t c);
    int sub = kv_sub(c), seq = int'(c.c[7:0]), slot = int'(c.c[23:8]), src = int'(c.ext[7:0]);
    int st = 0, data = 0, pg;
    n_kv++;
    if (kv_bad(c)) begin kv_res(c.a, 5, 0); return; end
    case (sub)
      0: if (kv_v[seq][slot]) st = 2; else if (kv_free() == 0) st = 1;
         else begin
           if (kv_fq.size() == 0) begin pg = kv_hw; kv_hw++; end else pg = kv_fq.pop_front();
           kv_v[seq][slot] = 1; kv_p[seq][slot] = pg; kv_rc[pg] = 1; data = pg;
         end
      1: if (!kv_v[seq][slot]) st = 3; else begin data = kv_p[seq][slot]; kv_unmap(seq, slot); end
      2: for (int l = 0; l < KV_NSLOT; l++) if (kv_v[seq][l]) begin kv_unmap(seq, l); data++; end
      3: if (kv_v[seq][slot]) st = 2; else if (!kv_v[src][slot]) st = 3; else if (kv_rc[kv_p[src][slot]] == KV_RMAX) st = 4;
         else begin pg = kv_p[src][slot]; kv_v[seq][slot] = 1; kv_p[seq][slot] = pg; kv_rc[pg]++; data = pg; end
      default: begin                                      // 4 MAP: 索引表 (頁 × mult, 未映射 0xFFFFFFFF) + 結果 line
        int hit = 0, kk = int'(c.k);
        for (int i = 0; i < kk; i++) begin
          int l = slot + i;
          bit h = l < KV_NSLOT && kv_v[seq][l];
          ewr(c.a + 26'(i / 16), i % 16, h ? 32'(kv_p[seq][l]) * c.ext : 32'hFFFF_FFFF);
          hit += int'(h);
        end
        for (int i = kk; i % 16 != 0; i++) ewr(c.a + 26'(i / 16), i % 16, 32'd0);
        kv_res(c.a + 26'((kk + 15) / 16), (hit == kk) ? 0 : 3, hit);
        return;
      end
    endcase
    kv_res(c.a, st, data);
  endfunction
  // MOE 欄位: k = T, b[4:0] = K, b[12:5] = nexp, b[13] = 無丟棄, ext[15:0] = cap, ext[31:16] = mult
  function automatic bit moe_bad(npu_tile_cmd_t c);
    int T = int'(c.k), K = int'(c.b[4:0]), ne = int'(c.b[12:5]), cap = int'(c.ext[15:0]);
    return K == 0 || K > 16 || ne == 0 || ne > MOE_NEXP || cap > MOE_CAPMAX || T == 0 || T >= (1 << MOE_TW);
  endfunction
  function automatic int moe_P(npu_tile_cmd_t c);
    return c.b[13] ? int'(c.k) * int'(c.b[4:0]) : int'(c.b[12:5]) * int'(c.ext[15:0]);
  endfunction
  // 輸出 line 數 (GL + CT + CN + 狀態); 參數不合法時只有狀態 line
  function automatic int moe_nl(npu_tile_cmd_t c);
    if (moe_bad(c)) return 1;
    return (moe_P(c) + 15) / 16 + (int'(c.k) * int'(c.b[4:0]) + 7) / 8 + (int'(c.b[12:5]) + 7) / 8 + 1;
  endfunction
  function automatic void moe_exec(npu_tile_cmd_t c);
    int n [MOE_NEXP], off [MOE_NEXP];
    int T = int'(c.k), K = int'(c.b[4:0]), ne = int'(c.b[12:5]), cap = int'(c.ext[15:0]);
    bit dl = c.b[13];
    int P, nval = 0, nerr = 0, gl, ct, cn, i = 0;
    logic [31:0] mult = 32'(c.ext[31:16]);
    n_moe++;
    if (moe_bad(c)) begin
      logic [511:0] x = '0; x[31:0] = 32'hFFFF_FFFF; eset(c.c, x); return;
    end
    P = moe_P(c);
    gl = (P + 15) / 16; ct = (T * K + 7) / 8; cn = (ne + 7) / 8;
    for (int l = 0; l < gl; l++) eset(c.c + 26'(l), '1);
    foreach (n[e]) begin n[e] = 0; off[e] = 0; end
    if (dl) begin
      for (int t = 0; t < T; t++) for (int j = 0; j < K; j++) begin
        logic [31:0] id = rl2[l2i(c.a + 26'(2 * t))][32 * j +: 32];
        if (id < 32'(ne)) n[id]++;
      end
      for (int e = 1; e < MOE_NEXP; e++) off[e] = off[e - 1] + n[e - 1];
      foreach (n[e]) n[e] = 0;
    end else foreach (off[e]) off[e] = e * cap;
    for (int t = 0; t < T; t++) for (int j = 0; j < K; j++) begin
      logic [31:0] id = rl2[l2i(c.a + 26'(2 * t))][32 * j +: 32];
      logic [31:0] w  = rl2[l2i(c.a + 26'(2 * t + 1))][32 * j +: 32];
      bit drop = 1;
      int pos = 0;
      if (id < 32'(ne)) begin
        int r = n[id];
        n[id]++; nval++;
        if (dl || r < cap) begin drop = 0; pos = off[id] + r; end
      end else nerr++;
      if (!drop) ewr(c.c + 26'(pos / 16), pos % 16, 32'(t) * mult);
      ewr(c.c + 26'(gl + i / 8), 2 * (i % 8), drop ? 32'hFFFF_FFFF : 32'(pos));
      ewr(c.c + 26'(gl + i / 8), 2 * (i % 8) + 1, drop ? 32'd0 : w);
      i++;
    end
    for (; i % 8 != 0; i++) begin ewr(c.c + 26'(gl + i / 8), 2 * (i % 8), 32'd0); ewr(c.c + 26'(gl + i / 8), 2 * (i % 8) + 1, 32'd0); end
    for (int e = 0; e < ne; e++) begin
      ewr(c.c + 26'(gl + ct + e / 8), 2 * (e % 8), 32'(dl ? n[e] : (n[e] > cap ? cap : n[e])));
      ewr(c.c + 26'(gl + ct + e / 8), 2 * (e % 8) + 1, 32'(off[e]));
    end
    for (int e = ne; e % 8 != 0; e++) begin ewr(c.c + 26'(gl + ct + e / 8), 2 * (e % 8), 32'd0); ewr(c.c + 26'(gl + ct + e / 8), 2 * (e % 8) + 1, 32'd0); end
    begin
      logic [511:0] x = '0;
      x[31:0] = {1'b0, 31'(nerr)}; x[63:32] = 32'(nval); x[95:64] = 32'(P);   // TB 不產生溢位 (T·K < 2^MOE_PW)
      eset(c.c + 26'(gl + ct + cn), x);
    end
  endfunction

  // 參考模型: 依序執行一個命令
  function automatic void ref_exec(npu_tile_cmd_t c);
    case (c.op)
      NPU_OP_DMA_LD: begin
        logic [511:0] d [$];                              // 先讀完所有來源 (位址表不會被本命令改寫: 產生器保證)
        int f = dma_fmt(c), nl = dma_nl(c);
        for (int i = 0; i < nl; i++) begin
          bit sk;
          logic [31:0] ea = dma_ext(c, i, sk);
          d.push_back(sk ? '0 : rerd(ea));
        end
        for (int i = 0; i < nl; i++) begin
          if (f == 0) rl2[l2i(c.a + 26'(i))] = d[i];
          else begin
            rl2[l2i(c.a + 26'(2 * i))]     = ref_dec(f, d[i], 0);
            rl2[l2i(c.a + 26'(2 * i + 1))] = ref_dec(f, d[i], 1);
          end
        end
      end
      NPU_OP_DMA_ST: for (int i = 0; i < dma_nl(c); i++) begin
        bit sk;
        logic [31:0] ea = dma_ext(c, i, sk);
        if (!sk) rext[ea] = rl2[l2i(c.a + 26'(i))];
      end
      NPU_OP_GEMM: if (c.k != 0) begin
        logic [31:0] C [];
        int nn, cl;                       // 迴圈上界用執行期變數: 常數上界會在每個內嵌點展開 N×N 份
        nn = N; cl = CL;
        C = new[N * N];
        for (int i = 0; i < nn; i++) for (int j = 0; j < nn; j++) begin
          longint acc = c.ext[0] ? longint'(rl2[l2i(c.c + 26'((i * N + j) / 16))][((i * N + j) % 16) * 32 +: 32]) : 0;   // ext[0]: C += A × B
          for (int kk = 0; kk < int'(c.k); kk++) begin
            logic [7:0] av = rl2[l2i(c.a + 26'(kk / SPL))][((kk % SPL) * N + i) * 8 +: 8];
            logic [7:0] bv = rl2[l2i(c.b + 26'(kk / SPL))][((kk % SPL) * N + j) * 8 +: 8];
            acc += (c.sa ? longint'($signed(av)) : longint'(av)) * (c.sb ? longint'($signed(bv)) : longint'(bv));
          end
          C[i * N + j] = 32'(acc);
        end
        for (int l = 0; l < cl; l++) for (int e = 0; e < 16; e++) rl2[l2i(c.c + 26'(l))][e * 32 +: 32] = C[l * 16 + e];
      end
      NPU_OP_VEC: vec_exec(c);
      NPU_OP_KV:  kv_exec(c);
      NPU_OP_MOE: moe_exec(c);
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
      NPU_OP_DMA_LD: begin
        for (int i = 0; i < dma_nl(c); i++) begin bit sk; logic [31:0] ea = dma_ext(c, i, sk); if (!sk) f.er.push_back(ea); end
        for (int i = 0; i < dma_l2n(c); i++) f.l2w.push_back(l2i(c.a + 26'(i)));
        if (c.b[18]) for (int l = 0; l < (int'(c.k) + 15) / 16; l++) f.l2r.push_back(l2i(c.c + 26'(l)));
      end
      NPU_OP_DMA_ST: begin
        for (int i = 0; i < dma_nl(c); i++) begin
          bit sk; logic [31:0] ea = dma_ext(c, i, sk);
          f.l2r.push_back(l2i(c.a + 26'(i)));
          if (!sk) f.ew.push_back(ea);
        end
        if (c.b[18]) for (int l = 0; l < (int'(c.k) + 15) / 16; l++) f.l2r.push_back(l2i(c.c + 26'(l)));
      end
      NPU_OP_GEMM: if (c.k != 0) begin
        for (int l = 0; l < (int'(c.k) + SPL - 1) / SPL; l++) begin f.l2r.push_back(l2i(c.a + 26'(l))); f.l2r.push_back(l2i(c.b + 26'(l))); end
        for (int l = 0; l < CL; l++) f.l2w.push_back(l2i(c.c + 26'(l)));
        if (c.ext[0]) for (int l = 0; l < CL; l++) f.l2r.push_back(l2i(c.c + 26'(l)));   // 累加: 讀舊的 C
      end
      NPU_OP_VEC: for (int i = 0; i < int'(c.k); i++) begin                  // 程式 line + 指令的 LD / ST
        npu_vec_pkg::npu_vec_ins_t x;
        x = vins(c.a, i);
        if (i % 4 == 0) f.l2r.push_back(l2i(c.a + 26'(i / 4)));
        if (x.op == npu_vec_pkg::VOP_LD) f.l2r.push_back(l2i(x.line));
        if (x.op == npu_vec_pkg::VOP_ST) f.l2w.push_back(l2i(x.line));
      end
      NPU_OP_KV: for (int l = 0; l < kv_nl(c); l++) f.l2w.push_back(l2i(c.a + 26'(l)));
      NPU_OP_MOE: begin
        if (!moe_bad(c)) for (int l = 0; l < 2 * int'(c.k); l++) f.l2r.push_back(l2i(c.a + 26'(l)));
        for (int l = 0; l < moe_nl(c); l++) f.l2w.push_back(l2i(c.c + 26'(l)));   // GL 讀 − 改 − 寫: 寫入範圍內
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
  int p_dec = 50;                                         // DMA_LD 使用解碼格式的機率 (%)
  function automatic logic [1:0] pick_fmt(); return rnd(p_dec) ? 2'(ur(1, 3)) : 2'd0; endfunction
  int unsigned n_sweep;                  // pick_n 的依序掃描位置 (seed 之後初始化)
  function automatic logic [15:0] pick_n();
    case (ur(0, 9))
      0:       return 16'd0;
      1:       return 16'd1;
      2, 3:    begin n_sweep++; return 16'(2 + n_sweep % (OUT - 1)); end   // 依序掃過 2..OUT (起點隨 seed): 每種長度都出現
                                                     // (2 / 10: KV / MOE 命令佔掉部分命令後, Pro 的 127 種長度仍能掃完)
      4:       return 16'(ur(2, OUT));
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
      c.b = rnd(50) ? '0 : 26'({10'($urandom()), 16'h0});   // 1D (ST 時 b 的高位元不影響; LD 的 b[17:16] 之後另選)
    end
  endfunction
  function automatic npu_tile_cmd_t gen_cmd(logic [7:0] id, int p_wt, int mode);
    npu_tile_cmd_t c;
    fp_t f;
    c = '0; c.id = id; c.ext = $urandom(); c.a = 26'($urandom()); c.b = 26'($urandom()); c.c = 26'($urandom());
    c.k = 16'($urandom()); c.sa = 1'($urandom()); c.sb = 1'($urandom());
    case ((focus >= 0 && focus < 4 && rnd(60)) ? focus_sel[focus] : mode == 1 ? (rnd(65) ? 0 : rnd(70) ? 19 : 7) : ur(0, 19))
      0, 1, 2, 3, 4, 5, 6: begin
        int al;
        c.op = NPU_OP_GEMM; c.k = pick_k(); al = (int'(c.k) + SPL - 1) / SPL;
        c.a = pick_l2(al); c.b = rnd(10) ? c.a : pick_l2(al);
        do c.c = pick_l2(CL);
        while (c.k != 0 && !(disjoint(l2i(c.c), CL, l2i(c.a), al) && disjoint(l2i(c.c), CL, l2i(c.b), al)));
      end
      7, 8, 9, 10, 11, 12: begin
        c.op = NPU_OP_DMA_LD; c.k = (mode == 1) ? 16'(ur(0, 3)) : pick_n(); c.ext = pick_ext(); dma_2d(c);
        c.b[17:16] = pick_fmt(); c.b[18] = 1'b0; c.a = pick_l2(dma_l2n(c));
      end
      13, 14, 15, 16, 17:  begin c.op = NPU_OP_DMA_ST; c.k = pick_n(); c.a = pick_l2(int'(c.k)); c.ext = pick_ext(); dma_2d(c); c.b[18] = 1'b0; end
      default: c.op = rnd(80) ? NPU_OP_NOP : npu_op_e'(7);           // 7 保留: 當 NOP
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
    if (!st) c.b[17:16] = pick_fmt();                     // 解碼 LD: L2 寫入 2 × 65535 個 line
    c.b[18] = 1'b0;
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
  // 間接 DMA: 回傳 {DMA_LD 位址表, 間接 LD / ST}; 位址表放外部記憶體 0xE800_0000 起 (pick_ext 會把 0xE... 改成 0xD...)
  int unsigned ind_ext = 32'hE800_0000;
  function automatic void gen_ind(logic [7:0] id0, logic [7:0] id1, int p_wt, output npu_tile_cmd_t c0, output npu_tile_cmd_t c1);
    int k, cw, nlist, p_skip;
    logic [25:0] pl;
    logic [31:0] base;
    logic [31:0] ent [$];
    c1 = '0; c1.id = id1; c1.op = rnd(55) ? NPU_OP_DMA_LD : NPU_OP_DMA_ST;
    k = rnd(10) ? 0 : rnd(15) ? ur(1, 3) : rnd(70) ? ur(4, 2 * OUT) : ur(2 * OUT, 6 * OUT);
    cw = rnd(50) ? 1 : rnd(70) ? ur(2, 4) : ur(5, 12);
    p_skip = rnd(30) ? 0 : rnd(20) ? 90 : ur(5, 40);       // 偶爾大半是略過 (未對應的 KV page / 被丟棄的 token)
    if (2 * k * cw > LINES / 2) k = LINES / 4 / cw;        // L2 資料範圍 (解碼時 × 2) <= L2 一半, 才放得下位址表
    base = pick_ext();
    if (base >= 32'hDF00_0000) base = base - 32'h0100_0000;
    for (int i = 0; i < k; i++)                           // 項值: 小範圍 (重複 / 重疊的項) 或較大的偏移
      ent.push_back(rnd(p_skip) ? 32'hFFFF_FFFF : rnd(50) ? 32'(ur(0, 3 * cw)) : 32'(ur(0, 4096) * cw));
    if (k > 0 && rnd(10)) ent[k - 1] = 32'hFFFF_FFFF;     // 最後一項略過
    nlist = (k + 15) / 16;
    for (int l = 0; l < nlist; l++) begin
      logic [511:0] d = {16{32'hFFFF_FFFF}};
      for (int j = 0; j < 16 && 16 * l + j < k; j++) d[32 * j +: 32] = ent[16 * l + j];
      if (rnd(30)) for (int j = 0; j < 16; j++) if (16 * l + j >= k) d[32 * j +: 32] = $urandom();   // 表尾以後的項不影響
      rext[ind_ext + 32'(l)] = d; emem[ind_ext + 32'(l)] = d;
    end
    pl = pick_l2(nlist);
    c0 = '0; c0.id = id0; c0.op = NPU_OP_DMA_LD; c0.k = 16'(nlist); c0.a = pl; c0.ext = ind_ext;
    ind_ext += 32'(nlist);
    c0 = mk_cmd(c0, p_wt);
    ref_exec(c0);                                         // 位址表進入參考 L2, 間接命令的 footprint 才能解讀
    c1.k = 16'(k); c1.ext = base; c1.c = pl;
    c1.b = {7'($urandom()), 1'b1, 2'd0, 16'(cw == 1 && rnd(50) ? 0 : cw)};
    if (c1.op == NPU_OP_DMA_LD) c1.b[17:16] = pick_fmt();
    // 資料的 L2 範圍不可蓋到位址表 (否則依序語意下位址表在讀完之前被改寫; 規格要求兩者不重疊)
    do c1.a = pick_l2(dma_l2n(c1)); while (!disjoint(l2i(c1.a), dma_l2n(c1), l2i(pl), nlist));
    c1 = mk_cmd(c1, p_wt);
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
      if (rnd(4)) begin x = '0; x.op = npu_vec_pkg::vop_e'(ur(12, 15)); p.push_back(x); end   // 非法指令
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

  // KV: 一個命令 (n = 1); MOE: {DMA_LD top-k 輸入, MOE} (n = 2)。top-k 輸入放外部記憶體 0xEC00_0000 起
  // (pick_ext 不會產生 0xE... 區域)。KV 偏向少數序列 (MAP 才有映射可轉換、頁會用完、參考計數會到上限)。
  int unsigned moe_ext = 32'hEC00_0000;
  function automatic npu_tile_cmd_t mk_kv(logic [7:0] id, int sub, int seq, int slot, int src, logic [25:0] a, int k, logic [31:0] mult);
    npu_tile_cmd_t c;
    c = '0; c.id = id; c.op = NPU_OP_KV; c.a = a; c.k = 16'(k);
    c.b = {23'($urandom()), 3'(sub)};                       // 未用的位元隨機
    c.c = {2'($urandom()), 16'(slot), 8'(seq)};
    c.ext = (sub == 4) ? mult : {24'($urandom()), 8'(src)};
    c.sa = 1'($urandom()); c.sb = 1'($urandom());
    return c;
  endfunction
  function automatic void gen_llm(logic [7:0] id0, logic [7:0] id1, int p_wt, output npu_tile_cmd_t c0, output npu_tile_cmd_t c1,
                                  output int n);
    if (rnd(55)) begin                                     // KV
      int r, sub, seq, slot, src, k;
      r = ur(0, 99);
      sub = (r < 35) ? 0 : (r < 55) ? 3 : (r < 70) ? 1 : (r < 88) ? 4 : (r < 93) ? 2 : ur(5, 7);
      seq = rnd(70) ? ur(0, 3) : ur(0, KV_NSEQ - 1); slot = rnd(60) ? ur(0, 15) : ur(0, KV_NSLOT - 1); src = ur(0, 3);
      if (sub == 3 || sub == 1) begin                      // 偏向有映射的 slot
        int cand [$];
        int e2;
        cand.delete();
        foreach (kv_v[s2, l2]) if (kv_v[s2][l2]) cand.push_back(s2 * KV_NSLOT + l2);
        if (cand.size() != 0 && rnd(80)) begin
          e2 = cand[ur(0, cand.size() - 1)];
          slot = e2 % KV_NSLOT;
          if (sub == 3) src = e2 / KV_NSLOT; else seq = e2 / KV_NSLOT;
        end
      end
      k = (sub == 4) ? (rnd(8) ? 0 : rnd(8) ? ur(KV_NSLOT + 1, 65535) : rnd(30) ? ur(1, 16) : ur(17, KV_NSLOT)) : int'($urandom() & 32'hFFFF);
      if (sub == 4 && rnd(20)) slot = rnd(50) ? ur(KV_NSLOT - 8, KV_NSLOT + 4) : 65535 - ur(0, 3);
      if (rnd(4)) seq = ur(KV_NSEQ, 255);
      if (rnd(4)) slot = ur(KV_NSLOT, 65535);
      if (rnd(4)) src = ur(KV_NSEQ, 255);
      c0 = mk_kv(id0, sub, seq, slot, src, '0, k, rnd(50) ? 32'd4 : $urandom());
      c0.a = pick_l2(kv_nl(c0));
      c0 = mk_cmd(c0, p_wt);
      c1 = '0;
      n = 1;
    end else begin                                         // MOE: 先載入 top-k
      int T, K, ne, cap, nin, hot, dm;
      bit dl;
      logic [25:0] pa;
      T = rnd(10) ? 1 : rnd(20) ? ur(17, 40) : ur(2, 16);
      K = rnd(20) ? 16 : rnd(20) ? 1 : ur(2, 8);
      ne = rnd(25) ? MOE_NEXP : rnd(30) ? ur(1, 8) : ur(1, MOE_NEXP);
      dl = 1'($urandom()); cap = rnd(20) ? ur(0, 1) : ur(2, 16);
      dm = ur(0, 2); hot = ur(0, ne - 1);
      nin = 2 * T;
      for (int t = 0; t < T; t++) begin
        logic [511:0] il, wl;
        for (int j = 0; j < 16; j++) begin
          int id;
          case (dm) 1: id = rnd(75) ? hot : ur(0, ne - 1); 2: id = hot; default: id = ur(0, ne - 1); endcase
          if (rnd(5)) id = ur(ne, 300);
          if (rnd(4)) id = 32'hFFFF_FFFF;                  // top-k 未用的槽
          il[32 * j +: 32] = 32'(id); wl[32 * j +: 32] = $urandom();
        end
        rext[moe_ext + 32'(2 * t)] = il; emem[moe_ext + 32'(2 * t)] = il;
        rext[moe_ext + 32'(2 * t + 1)] = wl; emem[moe_ext + 32'(2 * t + 1)] = wl;
      end
      pa = pick_l2(nin);
      c0 = '0; c0.id = id0; c0.op = NPU_OP_DMA_LD; c0.k = 16'(nin); c0.a = pa; c0.ext = moe_ext;
      moe_ext += 32'(nin);
      c0 = mk_cmd(c0, p_wt);
      ref_exec(c0);
      c1 = '0; c1.id = id1; c1.op = NPU_OP_MOE; c1.a = pa; c1.k = 16'(T);
      c1.b = {12'($urandom()), dl, 8'(ne), 5'(K)};
      c1.ext = {16'(rnd(30) ? 1 : rnd(10) ? ur(60000, 65535) : ur(2, 300)), 16'(cap)};
      c1.sa = 1'($urandom()); c1.sb = 1'($urandom());
      if (rnd(7)) case (ur(0, 5))                          // 參數不合法
        0: c1.b[4:0] = 5'd0;
        1: c1.b[4:0] = 5'(ur(17, 31));
        2: c1.b[12:5] = 8'd0;
        3: c1.b[12:5] = 8'(ur(MOE_NEXP + 1, 255));
        4: c1.ext[15:0] = 16'(ur(MOE_CAPMAX + 1, 65535));
        default: c1.k = rnd(50) ? 16'd0 : 16'(ur(1 << MOE_TW, 65535));
      endcase
      do c1.c = pick_l2(moe_nl(c1)); while (!disjoint(l2i(c1.c), moe_nl(c1), l2i(pa), nin));
      c1 = mk_cmd(c1, p_wt);
      n = 2;
    end
  endfunction

  // ---------------- 驅動 / 監看 (posedge 取樣, 非阻塞驅動) ----------------
  npu_tile_cmd_t prog [$];
  int pidx = 0, ncpl = 0, n_err = 0, n_rst = 0, n_batch = 0, n_cmds = 0;
  int cpl_pos [int];                     // id -> 完成順序
  int prog_pos [int];                    // id -> 程式中的位置
  int p_cmd = 80, p_cpl = 80, p_rgnt = 80, p_wgnt = 80, p_rv = 80, rlat = 4;
  int p_out = 0, outage = 0;                              // 外部記憶體整段停擺: 每拍以 p_out/10000 機率開始, 長 100..1500 拍
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
      rq.delete(); outage = 0;
    end else begin
      // 外部讀: grant → 依序回應 (延遲 0..rlat, 回應之間可間斷)
      if (m_rd_req && m_rd_gnt) begin
        automatic longint t = cyc + longint'(fix_lat >= 0 ? fix_lat : ur(0, rlat));
        if (t <= last_t) t = last_t + 1;
        last_t = t;
        rq.push_back('{erd(m_rd_addr), t});
      end
      if (outage > 0) outage--;
      else if (fix_lat < 0 && p_out != 0 && ur(0, 9999) < p_out) outage = ur(100, 1500);
      if (outage > 0) begin m_rd_rvalid <= 1'b0; m_rd_rdata <= {16{$urandom()}}; end
      else if (rq.size() != 0 && rq[0].t <= cyc && rnd(p_rv)) begin m_rd_rvalid <= 1'b1; m_rd_rdata <= rq.pop_front().d; end
      else begin m_rd_rvalid <= 1'b0; m_rd_rdata <= {16{$urandom()}}; end
      m_rd_gnt <= rnd(p_rgnt) && outage == 0;
      // 外部寫: grant 拍完成
      if (m_wr_req && m_wr_gnt) emem[m_wr_addr] = m_wr_data;
      m_wr_gnt <= rnd(p_wgnt) && outage == 0;
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
    kv_reset();                           // reset 清空 KV 頁管理
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

  // ---------------- coverage-driven 模式 (+cdv) ----------------
  // dut.u_e2e.hit[] (dv/npu/npu_ss_e2e_sva.sv) 是端到端 coverage 的命中位元; 每個 batch 前隨機挑一個未命中的 bin,
  // 依它調整命令組成 (focus: 偏向某一 op 類別) 與各埠壓力。全部命中後回到一般隨機。
  int cdv_closed_at = -1;
  function automatic int cdv_holes();
    int n = 0;
    for (int i = 0; i < $bits(dut.u_e2e.hit); i++) if (!dut.u_e2e.hit[i]) n++;
    return n;
  endfunction
  function automatic void cdv_bias(ref int ncmd, ref int mode, ref int p_wt);
    int holes [$];
    int h;
    for (int i = 0; i < $bits(dut.u_e2e.hit); i++) if (!dut.u_e2e.hit[i]) holes.push_back(i);
    if (holes.size() == 0) return;
    h = holes[ur(0, holes.size() - 1)];
    mode = 0;
    if (h < 18) begin                                      // op 類別 × 延遲
      focus = h / 3;
      unique case (h % 3)
        0: begin ncmd = ur(1, 3); p_cmd = 30; p_cpl = 100; rlat = 0; p_rgnt = 100; p_wgnt = 100; p_rv = 100; p_out = 0; p_wt = 0; end
        1: begin p_cpl = ur(40, 80); rlat = ur(1, 12); end
        default: begin p_cpl = ur(3, 10); rlat = ur(OUT, 3 * OUT); p_rgnt = ur(5, 30); p_wgnt = ur(5, 30); p_cmd = 100; end
      endcase
    end else if (h < 24) begin                             // 亂序完成: 前面放長命令、不設 wt
      focus = h - 18; p_wt = 0; p_cmd = 100; p_cpl = ur(50, 100); rlat = ur(OUT, 3 * OUT);
    end else if (h == 24) begin ncmd = ur(1, 3); p_cmd = 20; p_cpl = 100; end
    else if (h == 25) begin ncmd = ur(2, 6); p_cmd = 40; p_cpl = 100; end
    else begin                                             // 深佇列 / 回壓
      p_cmd = 100; p_cpl = ur(3, 10); p_wt = 0; ncmd = ur(nc / 2, nc);
    end
  endfunction

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
    p_out  = (ur(0, 4) == 0) ? ur(1, 20) : 0;                // 1/5 的 batch 有外部記憶體停擺 (DRAM refresh / NoC 壅塞)
    mode   = (ur(0, 3) == 0) ? 1 : 0;
    p_hot  = (mode == 1) ? 0 : 50;
    if (mode == 1) p_wt = 0;
    test_en = rnd(15);
    focus = -1;
    if (cdv) cdv_bias(ncmd, mode, p_wt);                         // coverage-driven: 偏向尚未命中的 bin
    if (big) begin p_rgnt = 100; p_wgnt = 100; p_rv = 100; p_out = 0; end   // 長 DMA: 外部埠全速
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
      if (!big && mode == 0 && i + 1 < ncmd && rnd(15)) begin                                    // DMA_LD 位址表 + 間接 DMA
        npu_tile_cmd_t c0, c1;
        gen_ind(ids[i], ids[i + 1], p_wt, c0, c1);
        prog.push_back(c0); prog_pos[int'(c0.id)] = i;
        prog.push_back(c1); prog_pos[int'(c1.id)] = i + 1;
        ref_exec(c1);
        i++;
        continue;
      end
      if (!big && mode == 0 && i + 1 < ncmd && rnd((focus == 5) ? 70 : 15)) begin             // KV, 或 DMA_LD top-k + MOE
        npu_tile_cmd_t c0, c1;
        int nl;
        gen_llm(ids[i], ids[i + 1], p_wt, c0, c1, nl);
        prog.push_back(c0); prog_pos[int'(c0.id)] = i;
        if (nl == 2) begin prog.push_back(c1); prog_pos[int'(c1.id)] = i + 1; ref_exec(c1); i++; end
        else ref_exec(c0);
        continue;
      end
      if (!big && mode == 0 && i + 1 < ncmd && rnd((focus == 4) ? 60 : 20)) begin             // DMA_LD 程式 + VEC
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
    longint t0, t_ld, t_st, t_dec;
    p_cmd = 100; p_cpl = 100; p_rgnt = 100; p_wgnt = 100; p_rv = 100; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); cpl_cyc.delete(); pidx = 0; ncpl = 0;
    c = '0; c.op = NPU_OP_DMA_LD; c.id = 8'd1; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h1000_0000;
    prog.push_back(c); prog_pos[1] = 0; ref_exec(c);
    c = '0; c.op = NPU_OP_DMA_LD; c.id = 8'd3; c.k = 16'd1024; c.a = 26'd0;  c.ext = 32'h3000_0000; c.wt = 1'b1;
    c.b[17:16] = 2'd3;                                    // MXFP4 解碼: 外部 1024 line → L2 2048 line
    prog.push_back(c); prog_pos[3] = 1; ref_exec(c);
    c = '0; c.op = NPU_OP_DMA_ST; c.id = 8'd2; c.k = 16'd2048; c.a = 26'd0;  c.ext = 32'h2000_0000; c.wt = 1'b1;
    prog.push_back(c); prog_pos[2] = 2; ref_exec(c);
    @(posedge clk); t0 = cyc; in_batch = 1;
    while (ncpl != 3) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    t_ld = cpl_cyc[1] - t0; t_dec = cpl_cyc[3] - cpl_cyc[1]; t_st = cpl_cyc[2] - cpl_cyc[3];
    $display("PERFDMA: OUT=%0d read_latency=%0d LD %0d lines %0d cycles (%.3f line/cycle)  ST %0d lines %0d cycles (%.3f line/cycle)  LD-MXFP4 %0d->%0d lines %0d cycles (%.3f L2 line/cycle)",
             OUT, fix_lat, 2048, t_ld, 2048.0 / real'(t_ld), 2048, t_st, 2048.0 / real'(t_st), 1024, 2048, t_dec, 2048.0 / real'(t_dec));
  endtask

  // directed: 完成輸出回壓期間填滿所有槽 (npu_core 佇列 + 每個 CU + 其輸出暫存 + tile 輸出暫存, 再加 DMA 與 NOP),
  //           與 LLM 引擎 (KV 命令), oc 達到上限 CQD + 4·NCU + 3 + NDC + 3 (界限是緊的); 再加兩個 VEC (程式以後門預先放進 L2):
  //           第二個等引擎忙碌、再等 VEC 完成槽; 之後放開回壓 (5 個來源同時待回報) 並照常比對
  task automatic fill_all(int prime = 3);   // prime: round-robin 起點 (0 GEMM / 1 DMA / 2 NOP / 3 VEC / 7 LLM /
                                            //        4 VEC, 只有 GEMM 競爭 / 5 VEC, 只有 DMA 與 VEC 競爭 / 6 NOP, 只有 NOP 與 VEC 競爭;
                                            //        4..6 不送 LLM 命令)
    localparam int K = 200, AL = (K + SPL - 1) / SPL, STRIDE = 2 * AL + CL;
    localparam int NG = 4 + 4 * NCU + 2; // npu_core 最多 4 + 4·NCU + 1 個, 另一個已在 tile 輸出暫存
    npu_tile_cmd_t c;
    int ng, oc_exp, nloop;
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
    ng = (prime == 0) ? NG : (prime == 5 || prime == 6) ? 0 : NG - 1;
    if (prime != 0) begin
      c = '0; c.id = 8'(200);
      case (prime)
        1: begin c.op = NPU_OP_DMA_LD; c.k = 16'd1; c.a = 26'((NG + 1) * STRIDE + 24); c.ext = $urandom(); end
        2, 6: c.op = NPU_OP_NOP;
        7: c = mk_kv(8'(200), 1, ur(0, KV_NSEQ - 1), ur(0, KV_NSLOT - 1), 0, 26'((NG + 1) * STRIDE + 33), 0, 0);   // FREE
        default: begin c.op = NPU_OP_VEC; c.k = 16'd2; c.a = 26'((NG + 1) * STRIDE); end
      endcase
      prog.push_back(c); prog_pos[200] = 0; ref_exec(c);
      in_batch = 1;
      for (int w = 0; w < 20000 && !(cpl_valid && cpl_id == 8'(200)); w++) @(posedge clk);
      if (!(cpl_valid && cpl_id == 8'(200))) begin $display("ERROR: fill_all prime %0d not in output register", prime); n_err++; end
    end
    // 迴圈上界用執行期變數: 常數上界時 Verilator 會展開迴圈 (X1 的 26 次展開讓產生的 C++ 達 85 MB, 編譯 > 30 分鐘)
    nloop = NG + NDC + 4;
    for (int i = 0; i < nloop; i++) begin
      c = '0; c.id = 8'(i);
      if (i < NG) begin
        if (i >= ng) continue;
        c.op = NPU_OP_GEMM; c.k = 16'(K); c.sa = 1'($urandom()); c.sb = 1'($urandom());
        c.a = 26'(i * STRIDE); c.b = c.a + 26'(AL); c.c = c.b + 26'(AL);
      end else if (i < NG + NDC) begin                     // 每個 DMA 通道一個
        if (prime == 4 || prime == 6) continue;
        c.op = NPU_OP_DMA_LD; c.k = 16'd2; c.a = 26'(NG * STRIDE + 2 * (i - NG)); c.ext = $urandom();
      end else if (i == NG + NDC) begin
        if (prime == 4 || prime == 5) continue;
        c.op = NPU_OP_NOP;
      end else if (i == NG + NDC + 1) begin                // LLM: KV ALLOC / FREE (結果 line 在 VEC 的範圍之後)
        if (prime >= 4 && prime <= 6) continue;
        c = mk_kv(8'(i), rnd(50) ? 0 : 1, ur(0, KV_NSEQ - 1), ur(0, KV_NSLOT - 1), 0, 26'((NG + 1) * STRIDE + 32), 0, 0);
      end else begin c.op = NPU_OP_VEC; c.k = 16'd4; c.a = 26'((NG + 1) * STRIDE + i - NG - NDC - 2); end
      prog_pos[i] = prog.size(); prog.push_back(c); ref_exec(c);
    end
    in_batch = 1;
    // 等到所有槽都滿, 且 DMA / NOP / VEC 的完成都在待回報 (放開回壓後 4 個來源同時競爭)
    oc_exp = (prime == 4) ? 4 + 4 * NCU + 2 : (prime == 5) ? 1 + NDC : (prime == 6) ? 2 : 4 + 4 * NCU + 4 + NDC;
    // GEMM 的完成也要已在等待 (src_v[0]; v6.3.12 起 CU 可持有多個命令, oc 到達上限時 GEMM 可能都還在計算)
    for (int w = 0; w < 20000 && !(int'(dut.oc) == oc_exp && (prime == 5 || prime == 6 || dut.src_v[0]) && (prime == 4 || (prime == 6 ? dut.nop_cv : (&dut.u_dma.cv && (dut.nop_cv || prime == 5)))) && dut.vec_cv && (prime == 4 || prime == 5 || prime == 6 || dut.llm_cv)); w++) @(posedge clk);
    if (int'(dut.oc) != oc_exp) begin $display("ERROR: fill_all did not reach oc = %0d (oc = %0d)", oc_exp, dut.oc); n_err++; end
    p_cpl = 80;
    while (ncpl != prog.size()) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    if (longint'(perf_steps) != longint'(48'(ps0 + longint'(ng * K)))) begin $display("ERROR: perf_steps (fill_all) %0d != %0d + %0d", perf_steps, ps0, ng * K); n_err++; end
    n_cmds += prog.size();
  endtask

  // directed: host L2 埠仲裁器的鎖定 (npu_l2_arb2)。先讓主端 X 單獨讀 L2 (使 round-robin 的上次選中者為 X), L2 閒置
  // 100 拍 (bank 進入 light sleep), 再送 X 的命令, 隔 gap 個 NOP 後送另一個主端 Y 的命令: X 的第一個讀取等 bank 喚醒
  // (LS_WAKE 拍沒有 grant), Y 在這段期間開始請求時 round-robin 偏向 Y, 必須靠鎖定保持 X (SVA c_rd_lock_* / a_rd_hold)。
  // 讀 (kind 0): X / Y = DMA_ST / VEC (取程式) 兩種順序; 寫 (kind 1): DMA_LD / VEC (ST 寫回) 兩種順序; gap = 0..5
  task automatic arb_lock();
    logic [511:0] d;
    npu_vec_pkg::npu_vec_ins_t x;
    npu_tile_cmd_t c;
    int nkind, nord, ngap;                 // 迴圈上界用執行期變數 (常數上界時 Verilator 展開迴圈, 每份都內嵌 ref_exec)
    n_batch++;
    nkind = 2; nord = 2; ngap = 5;
    p_cmd = 100; p_cpl = 100; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    // VEC 程式 (line 16): SPLAT v0, ST v0 → line 24, 25, 26 (連續寫入: VEC 連拿兩次 grant 時 DMA 的寫入必須先得到)
    d = '0;
    x = '0; x.op = npu_vec_pkg::VOP_SPLAT; x.vd = 3'd0; x.imm = $urandom(); d[0 +: 77] = x;
    for (int s = 1; s < 4; s++) begin x = '0; x.op = npu_vec_pkg::VOP_ST; x.va = 3'd0; x.line = 26'(23 + s); d[128 * s +: 77] = x; end
    l2_bd_wr(l2i(26'd16), d); rl2[l2i(26'd16)] = d;
    for (int kind = 0; kind < nkind; kind++)
    for (int ord = 0; ord < nord; ord++)
      for (int gap = 0; gap <= ngap; gap++) begin
        prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
        for (int t = 0; t < 3 + gap; t++) begin
          c = '0; c.id = 8'(210 + t);
          if (t == 1 + gap + 1 || t < 2) begin                  // t = 0, 1: 主端 X; t = gap + 2: 主端 Y
            if ((t < 2) == (ord == 0)) begin
              c.op = (kind == 0) ? NPU_OP_DMA_ST : NPU_OP_DMA_LD; c.k = 16'(ur(2, 6)); c.a = 26'(ur(0, 8) + 32 * kind); c.ext = pick_ext();
            end
            else begin c.op = NPU_OP_VEC; c.k = 16'd4; c.a = 26'd16; c.ext = $urandom(); c.b = 26'($urandom()); c.c = 26'($urandom()); end
          end else c.op = NPU_OP_NOP;
          prog_pos[210 + t] = prog.size(); prog.push_back(c); ref_exec(c);
          if (t == 0) begin                                    // 只送第一個, 完成後閒置 100 拍
            in_batch = 1;
            while (ncpl != 1) @(posedge clk);
            in_batch = 0;
            repeat (100) @(posedge clk);
          end
        end
        in_batch = 1;
        while (ncpl != prog.size()) @(posedge clk);
        in_batch = 0;
        for (int w = 0; busy && w < 1000; w++) @(posedge clk);
        check_state();
        n_cmds += prog.size();
      end
  endtask

  // directed: DMA 通道的埠選擇器鎖定 (npu_rr_lock; mutation rr_nolock 在 Nano 存活)。外部讀埠 grant 機率調低,
  // 讓通道在等待 grant 時另一個通道開始請求:
  //   偶數輪: LD A (→ 通道 0), 隔幾個 NOP 後 LD B (→ 通道 1): 通道 0 等待中、上次得主是 0, 通道 1 加入 (c_lock_0)
  //   奇數輪: 短 LD (→ 通道 0) 後長 LD (→ 通道 1), 再連續送短 LD: 通道 1 等待中、上次得主是 1, 通道 0 加入 (c_lock_1)
  // directed: K = 0 的 VEC 不得送進向量引擎 (只回報完成, L2 不變)。程式 line 100 含 SPLAT + ST → line 108:
  // 若 K = 0 的命令被執行, line 108 會被改寫 / 引擎的 SVA 失敗。之後再送一個 K = 2 的 VEC 確認同一段程式確實會寫。
  // 放在所有隨機 batch 之後 (不改變先前 batch 的隨機序列)。
  task automatic vec_k0();
    npu_tile_cmd_t c;
    logic [511:0] d;
    npu_vec_pkg::npu_vec_ins_t x;
    n_batch++;
    p_cmd = 100; p_cpl = 100; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    d = '0;
    x = '0; x.op = npu_vec_pkg::VOP_SPLAT; x.vd = 3'd0; x.imm = $urandom(); d[0 +: 77] = x;
    x = '0; x.op = npu_vec_pkg::VOP_ST; x.va = 3'd0; x.line = 26'd108; d[128 +: 77] = x;
    l2_bd_wr(l2i(26'd100), d); rl2[l2i(26'd100)] = d;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    for (int t = 0; t < 3; t++) begin
      c = '0; c.id = 8'(230 + t);
      if (t == 1) c.op = NPU_OP_NOP;
      else begin c.op = NPU_OP_VEC; c.k = (t == 0) ? 16'd0 : 16'd2; c.a = 26'd100; end
      prog_pos[230 + t] = prog.size(); prog.push_back(c); ref_exec(c);
    end
    in_batch = 1;
    while (ncpl != prog.size()) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    n_cmds += prog.size();
  endtask

  task automatic rr_lock();
    npu_tile_cmd_t c;
    int kq [$];                            // 每個命令: > 0 = DMA_LD 的 k, 0 = NOP
    int nrep, nq;
    n_batch++;
    nrep = 8;
    p_cmd = 100; p_cpl = 100; p_rgnt = 35; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    for (int r = 0; r < nrep; r++) begin
      prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
      kq.delete();
      if (r % 2 == 0) begin kq.push_back(8); for (int g = 0; g < r / 2; g++) kq.push_back(0); kq.push_back(4); end
      else begin kq.push_back(1); kq.push_back(24); for (int g = 0; g < 6; g++) kq.push_back(1); end
      nq = kq.size();
      for (int t = 0; t < nq; t++) begin
        c = '0; c.id = 8'(220 + t);
        if (kq[t] == 0) c.op = NPU_OP_NOP;
        else begin c.op = NPU_OP_DMA_LD; c.k = 16'(kq[t]); c.a = 26'(40 + 32 * t); c.ext = pick_ext(); end
        prog_pos[220 + t] = prog.size(); prog.push_back(c); ref_exec(c);
      end
      in_batch = 1;
      while (ncpl != prog.size()) @(posedge clk);
      in_batch = 0;
      for (int w = 0; busy && w < 1000; w++) @(posedge clk);
      check_state();
      n_cmds += prog.size();
    end
  endtask

  // directed: 長的間接 ST (項數接近 2^16, 位址表 4096 個 line 在 L2 折返): 涵蓋位址表計數器 (lf / nl) 的所有位元。
  // 先以 backdoor 把整個 L2 填成位址表: 大多是略過, 少數是小偏移 (外部位址落在 pick_ext 的安全範圍內)。
  // ST 只讀 L2, 所以資料範圍與位址表重疊沒有關係。
  task automatic big_ind();
    npu_tile_cmd_t c;
    n_batch++;
    p_cmd = 100; p_cpl = 100; p_rgnt = ur(60, 100); p_wgnt = ur(60, 100); p_rv = 100; fix_lat = -1; rlat = ur(1, 8); test_en = 0;
    for (int i = 0; i < LINES; i++) begin
      logic [511:0] d;
      for (int j = 0; j < 16; j++) d[32 * j +: 32] = rnd(97) ? 32'hFFFF_FFFF : 32'(ur(0, 4095));
      l2_bd_wr(i, d); rl2[i] = d;
    end
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    for (int t = 0; t < 2; t++) begin                     // 位址表 4095 個 line (0xFFF) 與 4096 個 line (0x1000)
      c = '0; c.id = 8'(230 + t); c.op = NPU_OP_DMA_ST; c.wt = 1'b1; c.k = 16'((t == 0) ? ur(65505, 65520) : ur(65521, 65535));
      c.a = 26'($urandom()); c.c = 26'($urandom()); c.ext = pick_ext(); c.b = {7'd0, 1'b1, 2'd0, 16'd0};
      if (c.ext >= 32'hDF00_0000) c.ext = c.ext - 32'h0100_0000;
      acc_clear(); acc_add(footprint(c));
      prog_pos[230 + t] = t; prog.push_back(c); ref_exec(c);
    end
    in_batch = 1;
    while (ncpl != prog.size()) @(posedge clk);
    in_batch = 0;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    n_cmds += prog.size();
  endtask

  // directed: 完成合併中 LLM / NOP 與另一個來源的競爭 (cg_cpl.cp_sel 的 pair bins)。完成輸出回壓期間先送 X (進入輸出暫存,
  // 成為 round-robin 的「上次選中者」), 再送 A、B, 等兩者的完成都在待回報後放開回壓: 第一個載入的必須是 X 之後的第一個。
  //   0: X = LLM, {GEMM, LLM} → GEMM    1: X = GEMM, {GEMM, LLM} → LLM
  //   2: X = VEC, {VEC, LLM}  → LLM     3: X = VEC, {NOP, VEC}   → NOP
  //   4: X = GEMM, {DMA, LLM} → DMA     5: X = DMA, {DMA, LLM}   → LLM     6: X = GEMM, {VEC, LLM} → VEC
  //   7: X = LLM, {LLM, LLM}  → 第二個 LLM 的完成佔住完成槽時, 第三個不得分派 (完成槽只有一格, llm_id 不可被覆蓋)
  task automatic llm_pairs();
    npu_tile_cmd_t c;
    logic [511:0] d;
    npu_vec_pkg::npu_vec_ins_t x;
    int nsc;
    logic [4:0] want;
    // VEC 程式 (line 100): SPLAT v0, ST v0 → line 108
    d = '0;
    x = '0; x.op = npu_vec_pkg::VOP_SPLAT; x.vd = 3'd0; x.imm = $urandom(); d[0 +: 77] = x;
    x = '0; x.op = npu_vec_pkg::VOP_ST; x.va = 3'd0; x.line = 26'd108; d[128 +: 77] = x;
    l2_bd_wr(l2i(26'd100), d); rl2[l2i(26'd100)] = d;
    nsc = 8;
    for (int sc = 0; sc < nsc; sc++) begin
      n_batch++;
      p_cmd = 100; p_cpl = 0; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
      prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
      for (int t = 0; t < 3; t++) begin
        int kind;                                          // 0 GEMM, 1 LLM, 2 VEC, 3 NOP, 4 DMA_LD
        int ks [8][3] = '{'{1, 0, 1}, '{0, 0, 1}, '{2, 2, 1}, '{2, 3, 2}, '{0, 4, 1}, '{4, 4, 1}, '{0, 2, 1}, '{1, 1, 1}};
        kind = ks[sc][t];
        c = '0; c.id = 8'(240 + t);
        case (kind)
          0: begin c.op = NPU_OP_GEMM; c.k = 16'(SPL); c.a = 26'(t * 24); c.b = 26'(t * 24 + 1); c.c = 26'(t * 24 + 2); end
          1: c = mk_kv(8'(240 + t), 1, ur(0, KV_NSEQ - 1), ur(0, KV_NSLOT - 1), 0, 26'(80 + t), 0, 0);
          2: begin c.op = NPU_OP_VEC; c.k = 16'd2; c.a = 26'd100; end
          4: begin c.op = NPU_OP_DMA_LD; c.k = 16'd2; c.a = 26'(120 + 2 * t); c.ext = pick_ext(); end
          default: c.op = NPU_OP_NOP;
        endcase
        prog_pos[240 + t] = prog.size(); prog.push_back(c); ref_exec(c);
        in_batch = 1;
        if (t == 0) begin                                  // X 進入輸出暫存
          for (int w = 0; w < 20000 && !(cpl_valid && cpl_id == 8'(240)); w++) @(posedge clk);
          if (!(cpl_valid && cpl_id == 8'(240))) begin $display("ERROR: llm_pairs %0d: X not in output register", sc); n_err++; end
        end
      end
      // sc 7: X、Y、Z 都是 LLM — X 佔住輸出暫存, Y 完成後佔住 LLM 完成槽, Z 必須等槽空出才能分派 (否則 llm_id 被覆蓋)
      want = (sc <= 1) ? 5'b10001 : (sc == 2 || sc == 6) ? 5'b11000 : (sc == 3) ? 5'b01100 : (sc == 7) ? 5'b10000 : 5'b10010;
      for (int w = 0; w < 20000 && dut.src_v != want; w++) @(posedge clk);
      if (dut.src_v != want) begin $display("ERROR: llm_pairs %0d: src_v %b != %b", sc, dut.src_v, want); n_err++; end
      p_cpl = 100;
      while (ncpl != prog.size()) @(posedge clk);
      in_batch = 0;
      for (int w = 0; busy && w < 1000; w++) @(posedge clk);
      check_state();
      n_cmds += prog.size();
    end
  endtask

  // directed: DMA 長度 2..OUT 各一個 (LD / ST 交替, 隨機格式 / 2D), 讓 cg_dma.cp_n 的每個長度 bin 確定出現
  // (Pro OUT = 128 時隨機產生容易漏掉幾個長度)
  task automatic len_sweep();
    npu_tile_cmd_t c;
    fp_t f;
    int nmax;
    n_batch++;
    nmax = OUT;                                            // 迴圈上界用執行期變數 (避免 Verilator 展開)
    p_cmd = 100; p_cpl = ur(60, 100); p_rgnt = ur(60, 100); p_wgnt = ur(60, 100); p_rv = ur(60, 100);
    fix_lat = -1; rlat = ur(1, 8); test_en = 0;
    // 每個長度一組 NDC 個命令, 組與組之間等 DMA 閒置: 通道全空時命令依序交給最低編號的空閒通道,
    // 所以組內第 i 個命令一定由通道 i 執行 → 每個通道的 cg_dma 都看到每個長度
    for (int k = 2; k <= nmax; k++) begin
      prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
      acc_clear();
      for (int ch = 0; ch < NDC; ch++) begin
        npu_tile_cmd_t c0;
        c0 = '0; c0.k = 16'(k); c0.id = 8'(prog.size());
        if (k % 2 == 0) begin c0.op = NPU_OP_DMA_LD; c0.b[17:16] = pick_fmt(); end
        else c0.op = NPU_OP_DMA_ST;
        if (rnd(30)) begin c0.b[15:0] = 16'(ur(1, 8)); c0.c = 26'(ur(0, 16)); end
        // 組內不等待 (wt = 0) 才會分給不同通道: 與前面的命令衝突時換位址重抽 (仍衝突就照 mk_cmd 等待, 正確性優先)
        for (int r = 0; r < 20; r++) begin
          c = c0; c.ext = pick_ext(); c.a = pick_l2(dma_l2n(c));
          f = footprint(c);
          if (ch == 0 || !conflicts(f)) break;
        end
        c = mk_cmd(c, 0);
        prog_pos[int'(c.id)] = prog.size(); prog.push_back(c); ref_exec(c);
      end
      in_batch = 1;
      while (ncpl != prog.size()) @(posedge clk);
      in_batch = 0;
      for (int w = 0; busy && w < 1000; w++) @(posedge clk);
      check_state();
      n_cmds += prog.size();
    end
  endtask

  // directed: 所有 DMA 通道的完成槽同時有效時才放行「一個」完成, 下一個命令補進空出的通道後再放行下一個。
  // 每次輸出時所有通道都有效, round-robin 依序輪到每個通道 → 會出現「較高編號的通道也有效、卻選較低編號」的情況
  // (一次放行全部時 rr 從 c_last 繞一圈又回到 c_last, NDC = 2 時永遠先選通道 1, 見不到這種情況)
  task automatic cpl_all();
    npu_tile_cmd_t c, c0;
    fp_t f;
    int nr;
    n_batch++;
    nr = 6 * NDC;
    p_cmd = 100; p_cpl = 0; p_rgnt = 100; p_wgnt = 100; p_rv = 100; fix_lat = -1; rlat = 2; test_en = 0;
    prog.delete(); prog_pos.delete(); cpl_pos.delete(); pidx = 0; ncpl = 0;
    acc_clear();
    for (int t = 0; t < nr; t++) begin
      c0 = '0; c0.id = 8'(240 + t); c0.op = NPU_OP_DMA_LD; c0.k = 16'd4;
      for (int r = 0; r < 20; r++) begin                    // 不等待 (wt = 0): 與前面的命令衝突時換位址重抽
        c = c0; c.ext = pick_ext(); c.a = pick_l2(dma_l2n(c));
        f = footprint(c);
        if (!conflicts(f)) break;
      end
      c = mk_cmd(c, 0);
      prog_pos[240 + t] = prog.size(); prog.push_back(c); ref_exec(c);
    end
    in_batch = 1;
    for (int w = 0; ncpl != prog.size(); w++) begin
      @(posedge clk);
      if (p_cpl == 0 && (dut.u_dma.cv == '1 || w > 2000 || prog.size() - ncpl < NDC)) begin p_cpl = 100; w = 0; end
      else if (p_cpl != 0 && cpl_valid && cpl_ready && prog.size() - ncpl > NDC) p_cpl = 0;   // 這拍握手, 下一拍 cpl_ready 落下: 只取走一個
    end
    in_batch = 0;
    p_cpl = 100;
    for (int w = 0; busy && w < 1000; w++) @(posedge clk);
    check_state();
    n_cmds += prog.size();
  endtask

  int unsigned seed, nb, nc;
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("batches=%d", nb)) nb = 30;
    if (!$value$plusargs("cmds=%d", nc)) nc = 60;
    void'($urandom(seed));
    salt = $urandom();
    kv_reset();
    n_sweep = $urandom();
    for (int b = 0; b < NB; b++) for (int d = 0; d < DEPTH; d++) dut.u_core.u_l2.mem[b][d] = {16{$urandom()}};
    for (int i = 0; i < LINES; i++) rl2[i] = l2_bd(i);
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
    // reset 清除每個 DMA 通道的控制狀態 (隨機上電時, 通道的 ICG 在 reset 期間必須開啟才會套用同步取樣的 reset)
    if (dut.u_dma.g_ch[0].u_dma.act || dut.u_dma.g_ch[0].u_dma.iss != '0 || dut.u_dma.g_ch[0].u_dma.wr != '0 ||
        dut.u_dma.g_ch[0].u_dma.n != '0 || dut.u_dma.g_ch[0].u_dma.f_cnt != '0 || dut.u_dma.g_ch[0].u_dma.lf != '0 ||
        dut.u_dma.g_ch[NDC-1].u_dma.act || dut.u_dma.g_ch[NDC-1].u_dma.iss != '0 || dut.u_dma.g_ch[NDC-1].u_dma.wr != '0 ||
        dut.u_dma.g_ch[NDC-1].u_dma.n != '0 || dut.u_dma.g_ch[NDC-1].u_dma.f_cnt != '0 || dut.u_dma.g_ch[NDC-1].u_dma.lf != '0) begin
      $display("ERROR: DMA channel state not cleared by reset"); n_err++;
    end
    cdv = $test$plusargs("cdv");
    if ($value$plusargs("perf_dma=%d", fix_lat)) perf_dma();
    else if (cdv) begin                                    // 跑到端到端 coverage 全部命中 (最多 nb 個 batch) 再多跑 5 個
      int extra = 5;
      for (int b = 0; b < int'(nb) && extra > 0; b++) begin
        batch(ur(nc / 2, nc));
        if (cdv_holes() == 0) begin if (cdv_closed_at < 0) cdv_closed_at = b + 1; extra--; end
      end
      $display("CDV: e2e bins %0d, holes %0d, closed after %0d batches", $bits(dut.u_e2e.hit), cdv_holes(), cdv_closed_at);
      for (int i = 0; i < $bits(dut.u_e2e.hit); i++) if (!dut.u_e2e.hit[i]) $display("CDV: hole bin %0d", i);
    end
    else for (int b = 0; b < int'(nb); b++) if (b == 2) arb_lock(); else if (b == 4 && !$test$plusargs("no_big")) big_ind(); else if (b == 6) len_sweep(); else if (b == 7) rr_lock(); else if (b == 9) llm_pairs(); else if (b == 8) cpl_all(); else if (b % 10 == 3) begin
      // fill_all 只有一個呼叫點 (每個呼叫點都會展開一份 task 與其內嵌的參考模型)
      int pl [$];
      int pr;
      pr = (b / 10 + int'(seed)) % 6;
      pl.delete(); pl.push_back(pr == 5 ? 7 : pr); if (b == 13) pl.push_back(5); if (b == 3) pl.push_back(6);
      foreach (pl[i]) fill_all(pl[i]);
    end else batch((b % 5 == 0) ? 1 + ur(0, 3) : ur(nc / 2, nc), b == 1 && !$test$plusargs("no_big"));
    vec_k0();
    $display("NPUSS: N=%0d NCU=%0d NB=%0d seed=%0d batches=%0d cmds=%0d (KV %0d MOE %0d) resets=%0d errors=%0d",
             N, NCU, NB, seed, n_batch, n_cmds, n_kv, n_moe, n_rst, n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
`ifdef WAVES
  // 波形 (以 --trace-fst +define+WAVES 編譯): +dump[=檔名] 寫 FST, 預設 waves.fst
  initial begin
    string fn;
    if ($test$plusargs("dump")) begin
      if (!$value$plusargs("dump=%s", fn)) fn = "waves.fst";
      $dumpfile(fn);
      $dumpvars(0, npu_ss_tb);
    end
  end
`endif
endmodule
