// SPDX-License-Identifier: Apache-2.0
//=============================================================================
// 浮點函式庫 TB: FP32 / BF16 的 fp_add、fp_mul 與 fp_narrow (FP32 → BF16)、fp_widen (BF16 → FP32),
//   逐筆讀入 dv/fp/gen_vectors.py 產生的向量 (期望值來自位元精確參考模型 fp_ref.py) 並比對。
// plusargs: +vec=<目錄> (預設 build/fp_vec)
//=============================================================================
module fp_tb;
  logic [31:0] a32, b32, add32, mul32, n32;
  logic [15:0] a16, b16, add16, mul16, c16, w16;
  logic [31:0] c32w;
  fp_add #(.E(8), .M(23)) u_add32 (.a(a32), .b(b32), .y(add32));
  fp_mul #(.E(8), .M(23)) u_mul32 (.a(a32), .b(b32), .y(mul32));
  fp_add #(.E(8), .M(7))  u_add16 (.a(a16), .b(b16), .y(add16));
  fp_mul #(.E(8), .M(7))  u_mul16 (.a(a16), .b(b16), .y(mul16));
  fp_narrow #(.E(8), .MI(23), .MO(7)) u_narrow (.a(n32), .y(c16));
  fp_widen  #(.E(8), .MI(7),  .MO(23)) u_widen  (.a(w16), .y(c32w));

  string dir;
  int n_err = 0, n_vec = 0;

  task automatic run2(string file, int w, int op);   // op: 0 add32, 1 mul32, 2 add16, 3 mul16
    int fd, r; logic [31:0] x, y, e, got;
    fd = $fopen({dir, "/", file}, "r");
    if (fd == 0) begin $display("ERROR: cannot open %s/%s", dir, file); n_err++; return; end
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%h %h %h\n", x, y, e);
      if (r != 3) break;
      if (w == 32) begin a32 = x; b32 = y; end else begin a16 = x[15:0]; b16 = y[15:0]; end
      #1;
      case (op) 0: got = add32; 1: got = mul32; 2: got = {16'h0, add16}; default: got = {16'h0, mul16}; endcase
      n_vec++;
      if (got !== e) begin
        if (n_err < 20) $display("MISMATCH %s a=%h b=%h dut=%h ref=%h", file, x, y, got, e);
        n_err++;
      end
    end
    $fclose(fd);
  endtask

  task automatic run1(string file, int op);          // op: 0 narrow, 1 widen
    int fd, r; logic [31:0] x, e, got;
    fd = $fopen({dir, "/", file}, "r");
    if (fd == 0) begin $display("ERROR: cannot open %s/%s", dir, file); n_err++; return; end
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%h %h\n", x, e);
      if (r != 2) break;
      if (op == 0) n32 = x; else w16 = x[15:0];
      #1;
      got = (op == 0) ? {16'h0, c16} : c32w;
      n_vec++;
      if (got !== e) begin
        if (n_err < 20) $display("MISMATCH %s a=%h dut=%h ref=%h", file, x, got, e);
        n_err++;
      end
    end
    $fclose(fd);
  endtask

  initial begin
    if (!$value$plusargs("vec=%s", dir)) dir = "build/fp_vec";
    run2("add_fp32.hex", 32, 0);
    run2("mul_fp32.hex", 32, 1);
    run2("add_bf16.hex", 16, 2);
    run2("mul_bf16.hex", 16, 3);
    run1("cvt_fp32_bf16.hex", 0);
    run1("cvt_bf16_fp32.hex", 1);
    $display("FP: vectors=%0d errors=%0d", n_vec, n_err);
    if (n_err == 0 && n_vec > 0) $display("TB PASS"); else $display("TB FAIL");
    $finish;
  end
endmodule
