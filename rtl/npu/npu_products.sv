// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 產品組態 (v6.3.8)
`include "orca_pkg.sv"

// 三款產品 = 同一份 npu_core RTL 的三組參數 (見 docs/npu.md 的效能與驗證結果)。
//   峰值 INT8 = N² × NCU MAC/拍 × 2 op; 例如 1 GHz 時 Nano 32 GOPS / X1 512 GOPS / Pro 4.1 TOPS
//   (時脈為假設值, 尚未做合成 / 時序收斂)。
package npu_products_pkg;
  typedef struct packed { int unsigned n, ncu, nb, depth; } npu_cfg_t;
  localparam npu_cfg_t X1_NANO = '{n: 4,  ncu: 1, nb: 2,  depth: 512};    // 16 MAC/拍,   L2 64 KiB
  localparam npu_cfg_t X1      = '{n: 8,  ncu: 4, nb: 4,  depth: 1024};   // 256 MAC/拍,  L2 256 KiB
  localparam npu_cfg_t X1_PRO  = '{n: 16, ncu: 8, nb: 16, depth: 1024};   // 2048 MAC/拍, L2 1 MiB
  // NPU Subsystem DMA 在途上限 (orca_npu_ss 的 OUT): 外部讀取延遲 L 拍時 DMA 吞吐量 = min(1, OUT / (L + 3)) line/拍
  localparam int X1_NANO_TILE_OUT = 16, X1_TILE_OUT = 32, X1_PRO_TILE_OUT = 64;
endpackage
