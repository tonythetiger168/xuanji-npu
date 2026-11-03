// SPDX-License-Identifier: Apache-2.0
// 形式驗證用替身: clock_gate 直通 (證明的是功能; 時脈閘控的致能是否涵蓋所有更新, 由模擬 (test_en 大多為 0,
// 逐拍參考模型) 與 mutation 檢查)。
module clock_gate (input logic clk, en, test_en, output logic clk_out);
  assign clk_out = clk;
endmodule
