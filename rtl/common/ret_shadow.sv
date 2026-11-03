// SPDX-License-Identifier: Apache-2.0
// Xuanji - retention 影子暫存器 (v6.3.11)。
// 模型化 retention flop 的「保持」部分: 放在常開電源上, 以常開時脈與上電 reset (por_n) 運作; PMU 的 ret_save 一拍
// 把主暫存器的值存入, 斷電期間保持。主暫存器在 PMU 的 ret_restore 拍載入 q (由使用者在主暫存器的 always_ff 中寫)。
// UPF (rtl/soc/orca_v63_soc.upf) 以 -exclude_elements 把本模組的 instance 留在常開域 (不另用 set_retention)。
module ret_shadow #(
  parameter int W = 1
)(
  input  logic         clk_aon, por_n,
  input  logic         save,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  always_ff @(posedge clk_aon or negedge por_n)
    if (!por_n)    q <= '0;
    else if (save) q <= d;
endmodule : ret_shadow
