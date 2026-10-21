// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// npu_ss_regs TB: 受約束隨機 APB4 交易 (已定義 / 未定義位址、隨機 PSTRB 與 paddr[1:0]、連續或間隔的交易) 對
// TB 內的暫存器參考模型 (逐拍) 比對: 讀資料、PSLVERR、irq、送往 orca_npu_ss 的命令 (順序與內容)、cpl_ready、perf_wr。
// orca_npu_ss 端以模型代替: cmd_ready 隨機, 完成以隨機 id 產生並保持到 cpl_ready, 效能計數器隨機變化, busy 隨機。
// 另有中途 reset。plusargs: +seed=N  +cycles=N (預設 200000)
//=============================================================================
module npu_ss_regs_tb;
  import orca_pkg::*;
  localparam int CQ = 4;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  logic psel = 0, penable = 0, pwrite = 0, pready, pslverr, irq;
  logic [11:0] paddr = '0;
  logic [31:0] pwdata = '0, prdata;
  logic [3:0]  pstrb = '0;
  logic cmd_valid, cmd_ready = 0, cpl_valid = 0, cpl_ready, ss_busy = 0, perf_wr, vec_err = 0;
  npu_tile_cmd_t cmd;
  logic [7:0]  cpl_id = '0;
  logic [47:0] perf_busy = '0, perf_steps = '0;
  logic test_en = 0;                                        // DFT: 隨機強制開啟 ICG (功能必須相同)
  always @(posedge clk) if ($urandom_range(99) == 0) test_en <= !test_en;
  npu_ss_regs #(.TILE_ID(3), .N(8), .NCU(4), .NB(4), .DEPTH(1024), .QD(8), .OUT(32), .CQ(CQ)) dut (.*);

  // ---------------- 參考模型 ----------------
  localparam int A_ID = 'h000, A_CFG0 = 'h004, A_CFG1 = 'h008, A_STATUS = 'h014, A_IRQ_ST = 'h018, A_IRQ_EN = 'h01C,
                 A_CMD_CTL = 'h020, A_CMD_EXT = 'h024, A_CMD_A = 'h028, A_CMD_B = 'h02C, A_CMD_C = 'h030,
                 A_CMD_PUSH = 'h034, A_CPL_POP = 'h040, A_CPL_PEEK = 'h044, A_PERF_CTRL = 'h050,
                 A_PB_LO = 'h054, A_PB_HI = 'h058, A_PS_LO = 'h05C, A_PS_HI = 'h060;
  int defined [19] = '{A_ID, A_CFG0, A_CFG1, A_STATUS, A_IRQ_ST, A_IRQ_EN, A_CMD_CTL, A_CMD_EXT, A_CMD_A, A_CMD_B,
                       A_CMD_C, A_CMD_PUSH, A_CPL_POP, A_CPL_PEEK, A_PERF_CTRL, A_PB_LO, A_PB_HI, A_PS_LO, A_PS_HI};
  logic [31:0] m_ctl, m_ext;
  logic [25:0] m_a, m_b, m_c;
  logic [3:0]  m_ien;
  logic        m_ovf, m_berr, m_verr;
  logic [15:0] m_shb, m_shs;
  npu_tile_cmd_t m_q [$];

  function automatic bit is_def(int a);
    foreach (defined[i]) if (defined[i] == a) return 1;
    return 0;
  endfunction
  function automatic logic [31:0] mrg(logic [31:0] o, logic [31:0] n, logic [3:0] s);
    for (int i = 0; i < 4; i++) if (s[i]) o[8*i +: 8] = n[8*i +: 8];
    return o;
  endfunction
  function automatic npu_tile_cmd_t m_stg();
    npu_tile_cmd_t c;
    c.op = npu_op_e'({m_ctl[5], m_ctl[1:0]}); c.wt = m_ctl[2]; c.sa = m_ctl[3]; c.sb = m_ctl[4]; c.id = m_ctl[15:8];
    c.k = m_ctl[31:16]; c.ext = m_ext; c.a = m_a; c.b = m_b; c.c = m_c;
    return c;
  endfunction
  function automatic logic [31:0] m_read(int a);
    case (a)
      A_ID:       return 32'h584E_0100;
      A_CFG0:     return {8'd10, 8'd4, 8'd4, 8'd8};
      A_CFG1:     return {8'(CQ), 8'd3, 8'd32, 8'd8};
      A_STATUS:   return {20'd0, 4'(CQ - m_q.size()), 4'd0, m_q.size() == 0, m_q.size() == CQ, cpl_valid,
                          ss_busy || m_q.size() != 0};
      A_IRQ_ST:   return {28'd0, m_verr, m_berr, m_ovf, cpl_valid};
      A_IRQ_EN:   return {28'd0, m_ien};
      A_CMD_CTL:  return m_ctl;
      A_CMD_EXT:  return m_ext;
      A_CMD_A:    return {6'd0, m_a};
      A_CMD_B:    return {6'd0, m_b};
      A_CMD_C:    return {6'd0, m_c};
      A_CPL_POP,
      A_CPL_PEEK: return {cpl_valid, 23'd0, cpl_valid ? cpl_id : 8'd0};
      A_PB_LO:    return perf_busy[31:0];
      A_PB_HI:    return {16'd0, m_shb};
      A_PS_LO:    return perf_steps[31:0];
      A_PS_HI:    return {16'd0, m_shs};
      default:    return '0;
    endcase
  endfunction
  task automatic m_reset();
    m_ctl = '0; m_ext = '0; m_a = '0; m_b = '0; m_c = '0; m_ien = '0; m_ovf = 0; m_berr = 0; m_verr = 0; m_shb = '0; m_shs = '0;
    m_q.delete();
  endtask

  int n_err = 0, n_rd = 0, n_wr = 0, n_push = 0, n_pop = 0, n_cpl = 0, n_ovf = 0, n_slverr = 0, n_rst = 0;
  task automatic err(string s);
    n_err++; if (n_err < 10) $display("MISMATCH %0t %s", $time, s);
  endtask

  // 每拍: negedge 檢查組合輸出並推進模型 (代表即將到來的 posedge), posedge 之後換新的輸入
  task automatic check_and_step();
    bit acc = psel && penable, wr = acc && pwrite, rd = acc && !pwrite;
    int a = {paddr[11:2], 2'b00};
    bit hit = is_def(a);
    logic [31:0] exp_rd = rd ? (hit ? m_read(a) : '0) : '0;
    bit e_cv = m_q.size() != 0;
    bit e_pwr = wr && a == A_PERF_CTRL && pstrb[0] && pwdata[0];
    if (pready !== 1'b1) err("pready");
    if (pslverr !== (acc && !hit)) err($sformatf("pslverr a=%h", a));
    if (prdata !== exp_rd) err($sformatf("prdata a=%h got %h exp %h", a, prdata, exp_rd));
    if (cmd_valid !== e_cv) err("cmd_valid");
    if (e_cv && cmd !== m_q[0]) err($sformatf("cmd got %h exp %h", cmd, m_q[0]));
    if (cpl_ready !== (rd && a == A_CPL_POP)) err("cpl_ready");
    if (irq !== |({m_verr, m_berr, m_ovf, cpl_valid} & m_ien)) err("irq");
    if (perf_wr !== e_pwr) err("perf_wr");
    if (!rst_n) return;
    // 模型轉移
    if (rd) n_rd++;
    if (wr) n_wr++;
    if (acc && !hit) n_slverr++;
    if (rd && a == A_PB_LO) m_shb = perf_busy[47:32];
    if (rd && a == A_PS_LO) m_shs = perf_steps[47:32];
    if (wr && a == A_CMD_PUSH) begin
      if (m_q.size() == CQ) begin m_ovf = 1; n_ovf++; end   // 滿: 丟棄 (以 pop 之前的數量判斷)
    end
    if (e_cv && cmd_ready) begin void'(m_q.pop_front()); n_pop++; end
    if (wr && a == A_CMD_PUSH && !(m_q.size() + (e_cv && cmd_ready) == CQ)) begin m_q.push_back(m_stg()); n_push++; end
    if (wr && a == A_IRQ_ST && pstrb[0]) begin
      if (pwdata[1]) m_ovf = 0;
      if (pwdata[2]) m_berr = 0;
      if (pwdata[3]) m_verr = 0;
    end
    if (vec_err) m_verr = 1;
    if (acc && !hit) m_berr = 1;
    if (wr) case (a)
      A_CMD_CTL: m_ctl = mrg(m_ctl, pwdata, pstrb) & 32'hFFFF_FF3F;
      A_CMD_EXT: m_ext = mrg(m_ext, pwdata, pstrb);
      A_CMD_A:   m_a = 26'(mrg({6'd0, m_a}, pwdata, pstrb));
      A_CMD_B:   m_b = 26'(mrg({6'd0, m_b}, pwdata, pstrb));
      A_CMD_C:   m_c = 26'(mrg({6'd0, m_c}, pwdata, pstrb));
      A_IRQ_EN:  m_ien = 4'(mrg({28'd0, m_ien}, pwdata, pstrb));
      default: ;
    endcase
  endtask

  // ---------------- 刺激 ----------------
  typedef enum {P_IDLE, P_SETUP, P_ACCESS} ph_t;
  ph_t ph = P_IDLE;
  bit  stall = 0;
  function automatic logic [11:0] rnd_addr();
    int k = $urandom_range(99);
    logic [11:0] a;
    if (k < 80) a = 12'(defined[$urandom_range(18)]);
    else if (k < 95) a = 12'($urandom_range(31) * 4);                   // 定義區附近 (含空洞)
    else a = 12'($urandom);
    return a | 12'($urandom_range(3) * (k % 2));                       // paddr[1:0] 應被忽略
  endfunction
  function automatic logic [31:0] rnd_data(logic [11:0] a);
    int k = $urandom_range(99);
    if ({a[11:2], 2'b00} == A_IRQ_ST || {a[11:2], 2'b00} == A_PERF_CTRL || {a[11:2], 2'b00} == A_IRQ_EN)
      return (k < 50) ? 32'($urandom_range(15)) : $urandom;
    return (k < 10) ? '0 : (k < 20) ? '1 : $urandom;
  endfunction
  task automatic drive();
    // 先處理取決於本拍 (剛完成的 posedge 之前) APB 存取的回應, 再換 APB 訊號
    if ($urandom_range(999) < 3) stall = !stall;                     // 長時間不收命令的階段 (讓 FIFO 滿)
    cmd_ready = !stall && ($urandom_range(99) < 40);
    if (cpl_valid && cpl_ready) cpl_valid = 0;
    if (!cpl_valid && $urandom_range(99) < 20) begin cpl_valid = 1; cpl_id = 8'($urandom); n_cpl++; end
    ss_busy = ($urandom_range(99) < 50);
    vec_err = ($urandom_range(99) < 2);
    if (perf_wr) begin perf_busy = '0; perf_steps = '0; end
    else if ($urandom_range(99) < 30) begin perf_busy = {16'($urandom), 32'($urandom)}; perf_steps = {16'($urandom), 32'($urandom)}; end
    // APB: IDLE → SETUP → ACCESS (PREADY 恆 1: 一拍) → 連續交易直接 SETUP 或回 IDLE
    case (ph)
      P_IDLE, P_ACCESS: if ($urandom_range(99) < 60) begin
        ph = P_SETUP; psel = 1; penable = 0; pwrite = $urandom_range(1); paddr = rnd_addr();
        pwdata = rnd_data(paddr); pstrb = !pwrite ? 4'h0 : ($urandom_range(99) < 70) ? 4'hF : 4'($urandom);
      end else begin ph = P_IDLE; psel = 0; penable = 0; end
      P_SETUP: begin ph = P_ACCESS; penable = 1; end
    endcase
  endtask

  int cycles = 200000, seed = 1;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("cycles=%d", cycles));
    void'($urandom(seed));
    m_reset();
    repeat (3) @(posedge clk);
    #1 rst_n = 1;
    for (int c = 0; c < cycles; c++) begin
      @(negedge clk);
      check_and_step();
      @(posedge clk); #1;
      if ($urandom_range(9999) < 5) begin                                    // 中途 reset
        rst_n = 0; ph = P_IDLE; psel = 0; penable = 0; cpl_valid = 0; m_reset(); n_rst++;
        @(negedge clk); check_and_step(); @(posedge clk); #1 rst_n = 1;
      end
      drive();
    end
    $display("REGS: cycles=%0d rd=%0d wr=%0d push=%0d pop=%0d cpl=%0d ovf=%0d slverr=%0d resets=%0d errors=%0d",
             cycles, n_rd, n_wr, n_push, n_pop, n_cpl, n_ovf, n_slverr, n_rst, n_err);
    if (n_err == 0 && n_push > 100 && n_ovf > 0 && n_slverr > 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
