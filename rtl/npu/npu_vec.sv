// SPDX-License-Identifier: Apache-2.0
// Xuanji - NPU 向量核 (v6.3.10): 與矩陣核分開的 FP32 SIMD 單元, 做 transformer 中矩陣乘以外的運算
// (softmax、LayerNorm / RMSNorm、GELU / SiLU、INT32 GEMM 結果的反量化)。語意見 docs/npu_vec.md。
`include "orca_pkg.sv"

`include "npu_vec_pkg.sv"

// npu_vec: 單發射、依序; 每拍執行一個指令 (LD / ST 等到 L2 grant)。
//   指令以 valid / ready 交握, 等待期間指令須保持不變 (SVA 假設)。
//   L2 介面同 npu_gemm_cu: 讀為組合式 (rd_data 與 rd_gnt 同拍), 寫在 wr_gnt 拍完成; 等待時請求保持。
//   非法編碼 (op > 11、LD fn = 7、ST fn > 1、SFU fn > 5、TOPK fn > 1) 當成 NOP 接受並設定黏著的 err (reset 或 err_clr 清除)。
//   暫存器檔無 reset (資料暫存器); TB 開始時以 SPLAT 初始化。
// 歸約 (RSUM / RMAX) 以固定順序的樹: 每層 x[i] = f(x[2i], x[2i+1]), 結果廣播到 vd 每個 lane。
// TOPK (v6.3.14, LLM 取樣的 top-k / MoE 路由): 32 個 (值, 索引) = 候選 va (索引 imm + lane) 與目前的前 16 (vb, 索引表 vd),
//   取前 16 降序 (npu_topk 的 bitonic 網路); fn 0 寫值, fn 1 寫索引。
// 低功耗: 每個向量暫存器一個 clock gate, 只在寫入該暫存器的拍開啟 (test_en 強制開啟, 功能相同)。
module npu_vec
  import npu_vec_pkg::*;
#(parameter int NV = 8)
(
  input  logic clk, rst_n,
  input  logic test_en,
  input  logic         ins_valid,
  output logic         ins_ready,
  input  npu_vec_ins_t ins,
  output logic         err,
  input  logic         err_clr,                 // 清除 err (同拍又遇到非法指令時 err 仍為 1)
  // L2 主介面 (一個讀取埠 + 一個寫入埠)
  output logic         rd_req,
  output logic [25:0]  rd_line,
  input  logic         rd_gnt,
  input  logic [511:0] rd_data,
  output logic         wr_req,
  output logic [25:0]  wr_line,
  output logic [511:0] wr_data,
  input  logic         wr_gnt
);
  localparam int RW = $clog2(NV);
  logic [NV-1:0][VL-1:0][31:0] vr;                 // 向量暫存器檔
  wire  [VL-1:0][31:0] a = vr[RW'(ins.va)], b = vr[RW'(ins.vb)];
  wire  [VL-1:0][31:0] c = vr[RW'(ins.vd)];          // TOPK fn 1: 目前的索引表

  // ---------------- 解碼 ----------------
  wire illegal = (ins.op > VOP_TOPK) || (ins.op == VOP_LD && ins.fn == 3'd7) ||
                 (ins.op == VOP_ST && ins.fn > 3'd1) || (ins.op == VOP_SFU && ins.fn > 3'd5) ||
                 (ins.op == VOP_TOPK && ins.fn > 3'd1);
  wire is_ld = ins_valid && !illegal && ins.op == VOP_LD;
  wire is_st = ins_valid && !illegal && ins.op == VOP_ST;
  assign rd_req    = is_ld;
  assign rd_line   = ins.line;
  assign wr_req    = is_st;
  assign wr_line   = ins.line;
  assign ins_ready = is_ld ? rd_gnt : is_st ? wr_gnt : 1'b1;
  wire fire = ins_valid && ins_ready;
  wire we   = fire && !illegal && ins.op != VOP_NOP && ins.op != VOP_ST;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n)                err <= 1'b0;
    else if (fire && illegal)  err <= 1'b1;
    else if (err_clr)          err <= 1'b0;

  // ---------------- 載入: 格式轉換 ----------------
  wire [255:0] s16 = rd_data[256 * ins.part[0] +: 256];
  wire [127:0] s8  = rd_data[128 * ins.part[1:0] +: 128];
  wire [63:0]  s4  = rd_data[64 * ins.part +: 64];
  logic [VL-1:0][31:0] ld_v;
  // ---------------- 逐 lane 運算 ----------------
  logic [VL-1:0][31:0] add_v, mul_v, max_v, sfu_v;
  for (genvar i = 0; i < VL; i++) begin : g_lane
    logic [31:0] d_bf, d_i;
    logic [18:0] d_h;                              // FP32 高 19 bit (低 13 bit 為 0)
    logic [11:0] d_e4, d_mx, d_sc;                 // FP32 高 12 bit (低 20 bit 為 0)
    logic [10:0] d_e5;
    logic [9:0]  d_e2;
    fp_widen     #(.E(8), .MI(7), .MO(23))           u_bf (.a(s16[16*i +: 16]), .y(d_bf));
    fp_small_dec #(.E(5), .M(10), .BIAS(15), .SP(0)) u_h  (.x(s16[16*i +: 16]), .y(d_h));
    fp_small_dec #(.E(4), .M(3), .BIAS(7), .SP(1))   u_e4 (.x(s8[8*i +: 8]), .y(d_e4));
    fp_small_dec #(.E(5), .M(2), .BIAS(15), .SP(0))  u_e5 (.x(s8[8*i +: 8]), .y(d_e5));
    fp_small_dec #(.E(2), .M(1), .BIAS(1), .SP(2))   u_e2 (.x(s4[4*i +: 4]), .y(d_e2));
    // MX 格式: 元素解碼 (精確) 後乘上共用比例 2^(imm[7:0]-127)
    always_comb
      unique case (ins.fn)
        3'd4:    d_mx = d_e4;
        3'd5:    d_mx = {d_e5, 1'b0};
        default: d_mx = {d_e2, 2'b0};
      endcase
    fp_mx_scale  #(.M(3))                            u_sc (.x(d_mx), .scale(ins.imm[7:0]), .y(d_sc));
    fp_from_int                                      u_i  (.a(rd_data[32*i +: 32]), .y(d_i));
    always_comb
      unique case (ins.fn)
        3'd0:    ld_v[i] = rd_data[32*i +: 32];
        3'd1:    ld_v[i] = d_bf;
        3'd2:    ld_v[i] = {d_h, 13'd0};
        3'd3:    ld_v[i] = d_i;
        default: ld_v[i] = {d_sc, 20'd0};          // 4 .. 6 = MX (fn = 7 為非法, 不寫入)
      endcase
    // SUB = a + (-b)
    fp_add #(.E(8), .M(23)) u_add (.a(a[i]), .b({b[i][31] ^ (ins.op == VOP_SUB), b[i][30:0]}), .y(add_v[i]));
    fp_mul #(.E(8), .M(23)) u_mul (.a(a[i]), .b(b[i]), .y(mul_v[i]));
    fp_max                  u_max (.a(a[i]), .b(b[i]), .y(max_v[i]));
    sfu                     u_sfu (.op(ins.fn), .x(a[i]), .y(sfu_v[i]));
  end

  // ---------------- 歸約樹 ----------------
  // 堆積式排列: 節點 0..VL-1 = a, 節點 VL+k = f(節點 2k, 節點 2k+1), 根 = 節點 2VL-2 (與逐層兩兩合併同順序)
  logic [2*VL-2:0][31:0] ts, tm;
  assign ts[VL-1:0] = a;
  assign tm[VL-1:0] = a;
  for (genvar k = 0; k < VL - 1; k++) begin : g_t
    fp_add #(.E(8), .M(23)) u_s (.a(ts[2*k]), .b(ts[2*k+1]), .y(ts[VL+k]));
    fp_max                  u_m (.a(tm[2*k]), .b(tm[2*k+1]), .y(tm[VL+k]));
  end

  // ---------------- TOPK ----------------
  // 排序鍵 (64 bit, 無號比較, 大者在前) = {FP32 的 totalOrder 鍵, ~索引}: 值相同時索引小者較大。
  // totalOrder 鍵: 正數 (含 +0 / +inf / +NaN) 設最高位, 負數全部取反 → −NaN < −inf < … < −0 < +0 < … < +inf < +NaN。
  logic [VL-1:0][63:0] ka, kb, ky;
  logic [VL-1:0][31:0] tk_v, tk_i;
  for (genvar i = 0; i < VL; i++) begin : g_tk
    assign ka[i] = {a[i][31] ? ~a[i] : {1'b1, a[i][30:0]}, ~(ins.imm + 32'(i))};
    assign kb[i] = {b[i][31] ? ~b[i] : {1'b1, b[i][30:0]}, ~c[i]};
    assign tk_v[i] = ky[i][63] ? {1'b0, ky[i][62:32]} : ~ky[i][63:32];
    assign tk_i[i] = ~ky[i][31:0];
  end
`ifdef FORMAL_TOP_npu_vec
  (* anyseq *) logic [VL-1:0][63:0] f_ky;                // 控制證明: 排序網路以任意值替身 (網路本身見 formal/npu_topk)
  assign ky = f_ky;
`else
  npu_topk #(.N(VL), .W(64)) u_topk (.a(ka), .b(kb), .y(ky));
`endif

  // ---------------- 寫回 ----------------
  logic [VL-1:0][31:0] wd;
  always_comb
    unique case (ins.op)
      VOP_LD:   wd = ld_v;
      VOP_ADD,
      VOP_SUB:  wd = add_v;
      VOP_MUL:  wd = mul_v;
      VOP_MAX:  wd = max_v;
      VOP_SFU:  wd = sfu_v;
      VOP_SPLAT: wd = {VL{ins.imm}};
      VOP_RSUM: wd = {VL{ts[2*VL-2]}};
      VOP_TOPK: wd = ins.fn[0] ? tk_i : tk_v;
      default:  wd = {VL{tm[2*VL-2]}};              // RMAX (NOP / ST / 非法不寫入)
    endcase

  for (genvar r = 0; r < NV; r++) begin : g_reg
    logic gclk;
    logic [VL-1:0][31:0] q;
    clock_gate u_cg (.clk(clk), .en(we && ins.vd == 3'(r)), .test_en(test_en), .clk_out(gclk));
    always_ff @(posedge gclk)
      if (we && ins.vd == 3'(r)) q <= wd;
    assign vr[r] = q;
  end

  // ---------------- 儲存 ----------------
  logic [2*VL-1:0][15:0] st_bf;
  for (genvar i = 0; i < VL; i++) begin : g_st
    fp_narrow #(.E(8), .MI(23), .MO(7)) u_na (.a(a[i]), .y(st_bf[i]));
    fp_narrow #(.E(8), .MI(23), .MO(7)) u_nb (.a(b[i]), .y(st_bf[VL+i]));
  end
  assign wr_data = (ins.fn == 3'd0) ? 512'(a) : 512'(st_bf);
`ifdef FORMAL
  // ---------------- 形式驗證 (SymbiYosys, formal/npu_vec; 模擬時不編入) ----------------
  // 浮點元件以任意值替身取代 (formal/npu_vec/stubs.sv): 只證明控制。檢查器自己從埠解碼指令,
  // 不引用 RTL 的 illegal / is_ld / fire / we (否則與被檢查的邏輯循環)。
  logic f_past;
  initial f_past = 1'b0;
  always @(posedge clk) f_past <= 1'b1;
  initial assume (!rst_n);
  // 環境: 指令在 ready 之前保持 (reset 期間不約束)
  always @(posedge clk)
    if (f_past && rst_n && $past(rst_n) && $past(ins_valid && !ins_ready)) begin
      assume (ins_valid && ins == $past(ins));
    end
  wire [3:0] f_op = ins[76:73];
  wire [2:0] f_fn = ins[72:70];
  wire       f_ill = f_op > 4'd11 || (f_op == 4'd1 && f_fn == 3'd7) || (f_op == 4'd2 && f_fn > 3'd1) ||
                     (f_op == 4'd7 && f_fn > 3'd5) || (f_op == 4'd11 && f_fn > 3'd1);
  wire f_ld   = ins_valid && !f_ill && f_op == 4'd1;
  wire f_st   = ins_valid && !f_ill && f_op == 4'd2;
  wire f_fire = ins_valid && ins_ready;
  wire f_wr   = f_fire && !f_ill && f_op != 4'd0 && f_op != 4'd2;
  always @(*) begin
    // L2 請求只來自合法的 LD / ST, 位址為指令的 line; ready 跟隨對應的 grant, 其他指令一拍完成
    assert (rd_req == f_ld);
    assert (wr_req == f_st);
    if (f_ld) begin assert (rd_line == ins.line); end
    if (f_st) begin assert (wr_line == ins.line); end
    assert (ins_ready == (f_ld ? rd_gnt : f_st ? wr_gnt : 1'b1));
  end
  // err: reset 清除; 之後 = 前一拍 err 或「前一拍接受了非法指令」
  always @(posedge clk)
    if (f_past) begin
      if (!$past(rst_n) || !rst_n) begin assert (!err); end
      else begin assert (err == (($past(err) && !$past(err_clr)) || $past(f_fire && f_ill))); end
    end
  // 暫存器檔: 第 r 個暫存器只在前一拍接受了寫入它的指令時改變 (含 reset 期間; 暫存器檔無 reset)
  for (genvar r = 0; r < NV; r++) begin : g_f
    always @(posedge clk)
      if (f_past && vr[r] != $past(vr[r])) begin assert ($past(f_wr && ins.vd == 3'(r))); end
    always @(posedge clk) cover (f_past && $past(f_wr && ins.vd == 3'(r)));
  end
  always @(posedge clk) begin
    cover (f_past && rst_n && rd_req && !rd_gnt);
    cover (f_past && rst_n && err);
  end
`endif
endmodule : npu_vec

// npu_topk: 兩組各 N 個 W bit 鍵 (無號, 大者優先) 取前 N 個, 降序輸出。輸入不需要已排序。
//   a、b 各自以 bitonic 排序 (降序), c[i] = max(a_s[i], b_s[N-1-i]) 為含前 N 個的 bitonic 序列, 再 bitonic 合併 (降序)。
//   N = 16: 比較交換 2 × 80 + 16 + 32 = 208 個, 深度 10 + 1 + 4 = 15 (組合式, 與其他向量指令同拍)。
/* verilator lint_off DECLFILENAME */            // 與 npu_vec 同檔 (所有檔案清單不用改)
module npu_topk #(parameter int N = 16, parameter int W = 64) (
  input  logic [N-1:0][W-1:0] a, b,
  output logic [N-1:0][W-1:0] y
);
  localparam int LG = $clog2(N);
  localparam int NS = LG * (LG + 1) / 2;               // bitonic 排序的層數
  logic [N-1:0][W-1:0] as, bs, c;
  for (genvar m = 0; m < 2; m++) begin : g_s
    logic [NS:0][N-1:0][W-1:0] s;
    if (m == 0) begin : g_ia assign s[0] = a; end
    else begin : g_ib assign s[0] = b; end
    for (genvar p = 1; p <= LG; p++) begin : g_p          // 子序列長度 2^p
      for (genvar q = p - 1; q >= 0; q--) begin : g_q     // 比較距離 2^q
        localparam int ST = p * (p - 1) / 2 + (p - 1 - q);
        for (genvar i = 0; i < N; i++) begin : g_i
          if ((i ^ (1 << q)) > i) begin : g_cas
            localparam int L = i ^ (1 << q);
            localparam bit DN = ((i >> p) & 1) == 0;        // 這段降序 (大者在 i); 最後一層 (p = LG) 全部降序
            wire sw = DN ? (s[ST][L] > s[ST][i]) : (s[ST][i] > s[ST][L]);
            assign s[ST+1][i] = sw ? s[ST][L] : s[ST][i];
            assign s[ST+1][L] = sw ? s[ST][i] : s[ST][L];
          end
        end
      end
    end
    if (m == 0) begin : g_a assign as = s[NS]; end
    else begin : g_b assign bs = s[NS]; end
  end
  for (genvar i = 0; i < N; i++) begin : g_c
    assign c[i] = (as[i] > bs[N-1-i]) ? as[i] : bs[N-1-i];
  end
  logic [LG:0][N-1:0][W-1:0] t;                        // bitonic 合併 (降序)
  assign t[0] = c;
  for (genvar q = LG - 1; q >= 0; q--) begin : g_m
    localparam int ST = LG - 1 - q;
    for (genvar i = 0; i < N; i++) begin : g_i
      if ((i ^ (1 << q)) > i) begin : g_cas
        localparam int L = i ^ (1 << q);
        wire sw = t[ST][L] > t[ST][i];
        assign t[ST+1][i] = sw ? t[ST][L] : t[ST][i];
        assign t[ST+1][L] = sw ? t[ST][i] : t[ST][L];
      end
    end
  end
  assign y = t[LG];
`ifdef FORMAL_TOP_npu_topk
  // ---------------- 形式驗證 (formal/npu_topk; 組合式) ----------------
  // y[r] 是 2N 個輸入中排第 r 的鍵: 嚴格大於 y[r] 的輸入 <= r 個, 且大於或等於 y[r] 的輸入 >= r + 1 個 (重複的鍵也成立)。
  logic [N-1:0][$clog2(2*N+1)-1:0] f_gt, f_ge;
  always_comb
    for (int r = 0; r < N; r++) begin
      f_gt[r] = '0; f_ge[r] = '0;
      for (int j = 0; j < N; j++) begin
        f_gt[r] = f_gt[r] + (a[j] > y[r]) + (b[j] > y[r]);
        f_ge[r] = f_ge[r] + (a[j] >= y[r]) + (b[j] >= y[r]);
      end
    end
  // 一個形式驗證用的狀態位元 (ABC 的 BMC 不接受純組合式網路); 第 2 拍起檢查, 輸入每拍任意
  (* gclk *) wire f_clk;
  logic f_s;
  initial f_s = 1'b0;
  always @(posedge f_clk) f_s <= 1'b1;
  always @(*) if (f_s)
    for (int r = 0; r < N; r++) begin
      assert (32'(f_gt[r]) <= r && 32'(f_ge[r]) >= r + 1);
      if (r > 0) assert (y[r-1] >= y[r]);
    end
`endif
endmodule : npu_topk
/* verilator lint_on DECLFILENAME */
