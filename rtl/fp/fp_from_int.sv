// SPDX-License-Identifier: Apache-2.0
// Xuanji 共用浮點函式庫: 有號 INT32 → FP32, round-to-nearest-even (GEMM 的 INT32 累加結果轉浮點)。
//   0 → +0; |a| <= 2^31 不會溢位。
module fp_from_int (
  input  logic [31:0] a,
  output logic [31:0] y
);
  wire        s = a[31];
  wire [31:0] m = s ? -a : a;                      // -2^31 的絕對值 2^31 以無號表示
  logic [4:0]  lz;
  always_comb begin
    lz = '0;
    for (int i = 0; i < 32; i++) if (m[i]) lz = 5'(31 - i);
  end
  // 前導 1 移到 bit 31 後的 bit 30..1 (bit 31 是隱藏位元; bit 0 恆為 0: 只有 m = 2^31 時 lz = 0, 其 bit 0 = 0)
  wire [30:1] n  = 30'((m << lz) >> 1);
  wire        up = n[7] && ((|n[6:1]) || n[8]);    // RNE: guard && (sticky || lsb)
  wire [23:0] q  = {1'b0, n[30:8]} + 24'(up);      // q[23]: 捨入進位到下一個 2 的冪 (尾數變 0)
  wire [7:0]  e  = 8'(158 - lz) + 8'(q[23]);       // 127 + 31 - lz
  assign y = (m == '0) ? 32'd0 : {s, e, q[22:0]};
endmodule : fp_from_int
