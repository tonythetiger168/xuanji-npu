// SPDX-License-Identifier: Apache-2.0
// Xuanji (原 ORCA v6.3) - NPU 產品組態 (v6.3.8)
`include "orca_pkg.sv"

// 四款產品 = 同一份 npu_core RTL 的三組參數 (見 docs/npu.md 的效能與驗證結果)。
//   峰值 INT8 = N² × NCU MAC/拍 × 2 op; 例如 1 GHz 時 Nano 32 GOPS / X1 512 GOPS / Pro 4.1 TOPS / Max 49 TOPS
//   (時脈為假設值, 尚未做合成 / 時序收斂)。
package npu_products_pkg;
  typedef struct packed { int unsigned n, ncu, nb, depth; } npu_cfg_t;
  localparam npu_cfg_t X1_NANO = '{n: 4,  ncu: 1, nb: 2,  depth: 512};    // 16 MAC/拍,   L2 64 KiB
  localparam npu_cfg_t X1      = '{n: 8,  ncu: 4, nb: 4,  depth: 1024};   // 256 MAC/拍,  L2 256 KiB
  localparam npu_cfg_t X1_PRO  = '{n: 16, ncu: 8, nb: 16, depth: 1024};   // 2048 MAC/拍, L2 1 MiB
  localparam npu_cfg_t X1_MAX  = '{n: 32, ncu: 24, nb: 32, depth: 1024};  // 24576 MAC/拍, L2 2 MiB (手機 / AI PC)
  // NPU Subsystem DMA 在途上限 (orca_npu_ss 的 OUT): 外部讀取延遲 L 拍時 DMA 吞吐量 = min(1, OUT / (L + 3)) line/拍
  // v6.3.13: Nano 16 → 32、X1 32 → 64、Pro 64 → 128 (DRAM 級延遲約 100 拍時: 解碼 LD 只需外部 0.5 line/拍, X1 以上
  // 都能讓 L2 寫入滿速; COPY 時 Pro / Max 接近 1 line/拍)。FIFO = OUT × 64 B, 合成時應對應到 SRAM。
  localparam int X1_NANO_TILE_OUT = 32, X1_TILE_OUT = 64, X1_PRO_TILE_OUT = 128, X1_MAX_TILE_OUT = 128;
endpackage
