// SPDX-License-Identifier: Apache-2.0
// npu_ss_dma 形式驗證 wrapper (SymbiYosys; 只用埠訊號 + 環境模型, 不改 RTL)
//   環境: 命令只在 idle 送入; 外部讀回應只在「已 grant 未回應」時出現 (依序); 其餘輸入任意。
//   性質 (對所有輸入序列證明):
//     帳目: 已 grant 的讀 <= n、寫 <= 讀 (LD 為已回應的讀)、在途 <= OUT
//     位址: 第 k 個讀 / 寫的位址 = 基底 + k
//     協定: 請求等待 grant 時保持 (位址 / 資料不變); done 恰在第 n 個寫入之後
//     資料: 任選一個 index i (anyconst), 第 i 個寫出的資料 = 第 i 個讀入的資料 (→ 所有 index 皆正確)
module npu_ss_dma_fv (input clk);
  localparam int OUT = 4;
  (* anyseq *) logic        cmd_valid, cmd_st;
  (* anyseq *) logic [31:0] cmd_ext;
  (* anyseq *) logic [25:0] cmd_l2;
  (* anyseq *) logic [15:0] cmd_n;
  (* anyseq *) logic        h_rd_gnt, h_wr_gnt, m_rd_gnt, m_wr_gnt, m_rd_rvalid_any;
  (* anyseq *) logic [511:0] h_rd_data, m_rd_rdata;
  (* anyconst *) logic [15:0] idx;              // 被追蹤資料的 line index
  logic rst_n = 1'b0;
  always @(posedge clk) rst_n <= 1'b1;

  logic cmd_ready, done, busy;
  logic h_rd_req, h_wr_req, m_rd_req, m_wr_req, m_rd_rvalid;
  logic [25:0] h_rd_line, h_wr_line;
  logic [31:0] m_rd_addr, m_wr_addr;
  logic [511:0] h_wr_data, m_wr_data;

  npu_ss_dma #(.OUT(OUT)) dut (.*);

  // ---- 環境模型: 目前命令與計數 ----
  logic        st; logic [31:0] ext; logic [25:0] l2; logic [16:0] n;
  logic [16:0] rd_g, rsp, wr_g;                 // 已 grant 的讀、已回應的讀 (LD)、已 grant 的寫
  logic        act_m;                           // 模型: 命令進行中
  logic        have; logic [511:0] tracked;     // 第 idx 個讀入的資料
  wire rd_fire = st ? (h_rd_req && h_rd_gnt) : (m_rd_req && m_rd_gnt);
  wire wr_fire = st ? (m_wr_req && m_wr_gnt) : (h_wr_req && h_wr_gnt);
  // 讀回應只在有未回應的讀時出現
  assign m_rd_rvalid = m_rd_rvalid_any && act_m && !st && (rsp < rd_g);
  wire in_fire = st ? rd_fire : m_rd_rvalid;    // 資料進入 DMA (ST: L2 讀 grant 拍; LD: 回應拍)
  wire [511:0] in_data = st ? h_rd_data : m_rd_rdata;
  wire [16:0]  in_idx  = st ? rd_g : rsp;

  always @(posedge clk) begin
    if (!rst_n) begin
      act_m <= 0; rd_g <= 0; rsp <= 0; wr_g <= 0; have <= 0; n <= 0; st <= 0; ext <= 0; l2 <= 0;
    end else begin
      if (cmd_valid && cmd_ready) begin
        act_m <= 1; st <= cmd_st; ext <= cmd_ext; l2 <= cmd_l2; n <= {1'b0, cmd_n};
        rd_g <= 0; rsp <= 0; wr_g <= 0; have <= 0;
      end else begin
        if (done) act_m <= 0;
        if (rd_fire) rd_g <= rd_g + 1;
        if (m_rd_rvalid) rsp <= rsp + 1;
        if (wr_fire) wr_g <= wr_g + 1;
        if (in_fire && in_idx == {1'b0, idx}) begin have <= 1; tracked <= in_data; end
      end
    end
  end

  // ---- 性質 ----
  logic past_v = 1'b0;
  always @(posedge clk) past_v <= rst_n;
  always @(*) if (rst_n) begin
    // 模型與 DUT 的狀態一致 (busy = 命令進行中)
    a_busy:  assert (busy == act_m && cmd_ready == !act_m);
    a_dir:   assert (!(st && act_m && (m_rd_req || h_wr_req)) && !(!st && act_m && (h_rd_req || m_wr_req)));
    a_idle:  assert (act_m || !(h_rd_req || h_wr_req || m_rd_req || m_wr_req));
    if (act_m) begin
      a_acct:  assert (wr_g <= (st ? rd_g : rsp) && rsp <= rd_g && rd_g <= n);
      a_outst: assert (rd_g - wr_g <= OUT);
      a_done:  assert (done == (wr_g == n));
      if (m_rd_req) a_mra: assert (m_rd_addr == ext + rd_g[15:0]);
      if (h_rd_req) a_hra: assert (h_rd_line == l2 + rd_g[15:0]);
      if (m_wr_req) a_mwa: assert (m_wr_addr == ext + wr_g[15:0]);
      if (h_wr_req) a_hwa: assert (h_wr_line == l2 + wr_g[15:0]);
      // 有額度就發讀 (吞吐量)
      if (rd_g != n && rd_g - wr_g < OUT) a_issue: assert (st ? h_rd_req : m_rd_req);
      // 資料完整: 第 idx 個寫出 = 第 idx 個讀入
      if (wr_g == {1'b0, idx} && (st ? m_wr_req : h_wr_req)) begin
        a_have: assert (have);
        a_data: assert ((st ? m_wr_data : h_wr_data) == tracked);
      end
    end
  end
  // 請求保持
  always @(posedge clk) if (past_v && rst_n && $past(rst_n)) begin
    if ($past(m_rd_req && !m_rd_gnt)) a_mrd_hold: assert (m_rd_req && m_rd_addr == $past(m_rd_addr));
    if ($past(m_wr_req && !m_wr_gnt)) a_mwr_hold: assert (m_wr_req && m_wr_addr == $past(m_wr_addr) && m_wr_data == $past(m_wr_data));
    if ($past(h_rd_req && !h_rd_gnt)) a_hrd_hold: assert (h_rd_req && h_rd_line == $past(h_rd_line));
    if ($past(h_wr_req && !h_wr_gnt)) a_hwr_hold: assert (h_wr_req && h_wr_line == $past(h_wr_line) && h_wr_data == $past(h_wr_data));
    if ($past(done)) a_done_pulse: assert (!done);
  end
  // 可達性
  always @(*) if (rst_n) begin
    c_ld_done: cover (act_m && !st && done && n == 6);
    c_st_done: cover (act_m && st && done && n == 6);
    c_full:    cover (act_m && rd_g - wr_g == OUT && rd_g != n);
  end
endmodule
