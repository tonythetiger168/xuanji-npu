// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// orca_npu_ss_apb 整合測試 (導向): CPU 只經 APB 操作真的 orca_npu_ss —
//   讀 ID / CFG → 開中斷 → 下 NOP、GEMM K=0、DMA_LD (外部 → L2, 8 line)、DMA_ST (L2 → 外部另一區, 8 line, wt=1)
//   → 等 irq, 以 CPL_POP 取回 4 個完成 (id 集合) → 比對外部記憶體搬回的資料 → 讀 / 清除效能計數 → 未定義位址 PSLVERR
//   → 隨機 APB 讀寫 (不含 CMD_PUSH / PERF_CTRL; 只檢查 PSLVERR 與位址是否有定義一致) → reset。test_en 隨機切換。
// 外部記憶體模型: 讀 grant 隨機, 資料在 2 拍後依序回傳。
//=============================================================================
module orca_npu_ss_apb_tb;
  import orca_pkg::*;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  logic psel = 0, penable = 0, pwrite = 0, pready, pslverr, irq;
  logic [11:0] paddr = '0;
  logic [31:0] pwdata = '0, prdata;
  logic [3:0]  pstrb = '0;
  logic m_rd_req, m_rd_gnt, m_rd_rvalid = 0, m_wr_req, m_wr_gnt;
  logic [31:0] m_rd_addr, m_wr_addr;
  logic [511:0] m_rd_rdata = '0, m_wr_data;
  logic test_en = 0;
  orca_npu_ss_apb #(.TILE_ID(3)) dut (.clk, .rst_n, .test_en, .clk_aon(clk), .por_n(rst_n), .ret_save(1'b0), .ret_restore(1'b0), .psel, .penable, .pwrite, .paddr, .pwdata,
    .pstrb, .pready, .prdata, .pslverr, .irq, .m_rd_req, .m_rd_addr, .m_rd_gnt, .m_rd_rvalid, .m_rd_rdata,
    .m_wr_req, .m_wr_addr, .m_wr_data, .m_wr_gnt);

  // 外部記憶體
  logic [511:0] ext [logic [31:0]];
  logic [31:0] rq [$];
  logic g = 0;
  assign m_rd_gnt = m_rd_req && g;
  assign m_wr_gnt = m_wr_req && g;
  always @(posedge clk) begin
    g <= ($urandom_range(99) < 60);
    if ($urandom_range(99) < 5) test_en <= !test_en;
    if (m_rd_gnt) rq.push_back(m_rd_addr);
    if (m_wr_gnt) ext[m_wr_addr] = m_wr_data;
  end
  logic [31:0] rq_d [$];
  always @(posedge clk) begin                     // 2 拍延遲後依序回傳
    m_rd_rvalid <= 1'b0;
    if (rq_d.size() != 0) begin
      m_rd_rvalid <= 1'b1;
      m_rd_rdata  <= ext.exists(rq_d[0]) ? ext[rq_d[0]] : '0;
      void'(rq_d.pop_front());
    end
    if (rq.size() != 0) begin rq_d.push_back(rq[0]); void'(rq.pop_front()); end
  end

  int n_err = 0;
  task automatic apb(bit w, logic [11:0] a, logic [31:0] d, output logic [31:0] r, output bit e);
    @(negedge clk); psel = 1; penable = 0; pwrite = w; paddr = a; pwdata = d; pstrb = w ? 4'hF : 4'h0;
    @(negedge clk); penable = 1;
    #1 r = prdata; e = pslverr;
    @(negedge clk); psel = 0; penable = 0;
  endtask
  task automatic wr(logic [11:0] a, logic [31:0] d);
    logic [31:0] r; bit e;
    apb(1, a, d, r, e);
    if (e) begin n_err++; $display("ERROR: write %h slverr", a); end
  endtask
  function automatic logic [31:0] ctl(int op, bit wt, int id, int k);
    return {16'(k), 8'(id), 2'd0, 1'(op >> 2), 1'b0, 1'b0, wt, 2'(op)};   // OP[2] 在 bit 5
  endfunction
  task automatic push(int op, bit wt, int id, int k, logic [31:0] e, logic [25:0] a);
    wr(12'h020, ctl(op, wt, id, k)); wr(12'h024, e); wr(12'h028, {6'd0, a}); wr(12'h034, 0);
  endtask

  initial begin
    logic [31:0] r; bit e;
    int got [$];
    for (int i = 0; i < 8; i++) ext[32'h100 + i] = {16{32'hA5000000 + i}};
    begin                                                      // 向量程式 (ext 0x500): LD r0 ← L2 0x40; 非法; ST r0 → L2 0x60
      npu_vec_pkg::npu_vec_ins_t x0, x1, x2;
      x0 = '0; x0.op = npu_vec_pkg::VOP_LD; x0.vd = 3'd0; x0.line = 26'h40;
      x1 = '0; x1.op = npu_vec_pkg::vop_e'(4'd13);
      x2 = '0; x2.op = npu_vec_pkg::VOP_ST; x2.va = 3'd0; x2.line = 26'h60;
      ext[32'h500] = '0; ext[32'h500][76:0] = x0; ext[32'h500][128 +: 77] = x1; ext[32'h500][256 +: 77] = x2;
    end
    repeat (3) @(posedge clk); rst_n = 1;
    apb(0, 12'h000, 0, r, e); if (r !== 32'h584E_0100) begin n_err++; $display("ERROR: ID %h", r); end
    apb(0, 12'h004, 0, r, e); if (r !== {8'd10, 8'd4, 8'd4, 8'd8}) begin n_err++; $display("ERROR: CFG0 %h", r); end
    apb(0, 12'h008, 0, r, e); if (r !== {8'd4, 8'd3, 8'd32, 8'd8}) begin n_err++; $display("ERROR: CFG1 %h", r); end
    wr(12'h01C, 32'h1);                                        // IRQ_ENABLE.CPL
    push(NPU_OP_NOP, 0, 8'h11, 0, 0, 0);
    push(NPU_OP_GEMM, 0, 8'h12, 0, 0, 0);                      // K = 0: 只回報完成
    push(NPU_OP_DMA_LD, 0, 8'h13, 8, 32'h100, 26'h40);
    push(NPU_OP_DMA_ST, 1, 8'h14, 8, 32'h200, 26'h40);          // wt: 等 DMA_LD 完成
    push(NPU_OP_DMA_LD, 1, 8'h15, 1, 32'h500, 26'h80);          // 載入向量程式
    push(NPU_OP_VEC,    1, 8'h16, 3, 0, 26'h80);                // 執行 (3 個指令)
    push(NPU_OP_DMA_ST, 1, 8'h17, 1, 32'h300, 26'h60);          // 結果存回外部
    for (int t = 0; t < 4000 && got.size() < 7; t++) begin
      if (irq) begin
        apb(0, 12'h040, 0, r, e);                              // CPL_POP
        if (r[31]) got.push_back(int'(r[7:0]));
      end else @(negedge clk);
    end
    got.sort();
    if (got.size() != 7 || got[0] != 'h11 || got[1] != 'h12 || got[2] != 'h13 || got[3] != 'h14 || got[4] != 'h15 ||
        got[5] != 'h16 || got[6] != 'h17) begin
      n_err++; $display("ERROR: completions %p", got);
    end
    if (irq) begin n_err++; $display("ERROR: irq still high"); end
    for (int i = 0; i < 8; i++)
      if (!ext.exists(32'h200 + i) || ext[32'h200 + i] !== {16{32'hA5000000 + i}}) begin
        n_err++; $display("ERROR: ext[%h]", 32'h200 + i);
      end
    if (!ext.exists(32'h300) || ext[32'h300] !== ext[32'h100]) begin n_err++; $display("ERROR: VEC copy result"); end
    apb(0, 12'h018, 0, r, e); if (r[3] !== 1'b1) begin n_err++; $display("ERROR: VEC_ERR not set (%h)", r); end
    wr(12'h018, 32'h8);
    apb(0, 12'h018, 0, r, e); if (r[3] !== 1'b0) begin n_err++; $display("ERROR: VEC_ERR not cleared"); end
    apb(0, 12'h014, 0, r, e);                                  // STATUS: 閒置、FIFO 空
    if (r[3:0] !== 4'b1000 || r[11:8] !== 4'd4) begin n_err++; $display("ERROR: STATUS %h", r); end
    wr(12'h050, 32'h1);                                        // 清除效能計數
    apb(0, 12'h054, 0, r, e); if (r !== 0) begin n_err++; $display("ERROR: PERF_BUSY_LO %h", r); end
    apb(0, 12'h400, 0, r, e); if (!e || r !== 0) begin n_err++; $display("ERROR: undefined address"); end
    apb(0, 12'h018, 0, r, e); if (r[2] !== 1'b1) begin n_err++; $display("ERROR: BUS_ERR not set"); end
    wr(12'h018, 32'h4);
    apb(0, 12'h018, 0, r, e); if (r[2] !== 1'b0) begin n_err++; $display("ERROR: BUS_ERR not cleared"); end
    for (int i = 0; i < 3000; i++) begin                     // 隨機存取 (含高位址、paddr[1:0])
      logic [11:0] a = (i % 4 == 0) ? 12'($urandom) : 12'($urandom_range(24) * 4 + $urandom_range(3));
      bit def;
      if (a[11:2] == 10'h00D || a[11:2] == 10'h014) a = 12'h01C;
      def = a[11:2] inside {10'h000, 10'h001, 10'h002, 10'h005, 10'h006, 10'h007, 10'h008, 10'h009, 10'h00A, 10'h00B,
                            10'h00C, 10'h00D, 10'h010, 10'h011, 10'h014, 10'h015, 10'h016, 10'h017, 10'h018};
      apb($urandom_range(1), a, $urandom, r, e);
      if (e === def) begin n_err++; $display("ERROR: random access %h slverr=%0d", a, e); end
    end
    rst_n = 0; repeat (2) @(posedge clk); rst_n = 1;
    apb(0, 12'h018, 0, r, e); if (r !== 0) begin n_err++; $display("ERROR: IRQ_STATUS after reset %h", r); end
    $display("NPUSSAPB: completions=%0d errors=%0d", got.size(), n_err);
    if (n_err == 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
