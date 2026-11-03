# NPU Subsystem 改善計劃

依據 2024–2026 年公開的 NPU / TPU / LPU 架構 (來源見文末) 制定。所有效能數字都是廠商或論文自行報告,
部分來源為二手且彼此有出入, 只作為設計方向參考; Xuanji 尚未合成, 效能只能以模擬週期 × 假設時脈估算。

## 全球現況重點

| 類別 | 代表作 | 關鍵設計 |
|---|---|---|
| 資料中心 TPU | Google TPU v7 Ironwood: 2 TensorCore + 4 SparseCore, 192 GB HBM (7.4 TB/s), FP8 ≈ 2 × BF16 | 大型 systolic MXU、FP8 雙倍速率、獨立稀疏 / embedding 引擎 |
| 資料中心 NPU | Huawei Ascend 910B/C/D (Da Vinci): Cube 16×16×16, Cube (AIC) 與 Vector (AIV) 分核 | 矩陣核與向量核分開、各自排程 |
| 資料流 | Tenstorrent Blackhole (Tensix: 5 RISC-V、1.5 MB L1、無 cache、2D NoC); Groq LPU (編譯器靜態排程、權重在 SRAM) | 明確的 DMA 搬移與編譯器排程 |
| AI PC NPU | AMD XDNA 2 (4×8 AI Engine tile, 每 tile 64 KB + 每欄 512 KB MemTile, BFP16); Intel NPU 4; Qualcomm Hexagon; Apple M4 | 兩層 scratchpad、區塊浮點、40–50 TOPS 級 |
| MCU 邊緣 | Arm Ethos-U85 (128–2048 MAC, INT8 權重 / INT16 activation, 原生 MATMUL) | 小型 NPU 也原生支援 transformer 運算子 |
| 數字格式 | OCP MX (MXFP8 / MXFP4, 32 元素共用 E8M0); NVIDIA NVFP4 | micro-scaling 成為標準; MXFP4 跨廠商可攜 |
| LLM decode | P3-LLM (W4A8KV4); Oaken (ISCA 2025, KV 解量化在 DMA) | decode 受頻寬限制: 權重 / KV 量化、搬移時解壓縮 |
| 影像邊緣 NPU | Chips&Media WAVE-N (v2): 主打超解析 (SESR) 與物件偵測 (YOLO), 高 TOPS 下保留 FP16; 宣稱「近零頻寬」, 搭配 feature map 壓縮 IP、frame buffer 壓縮 (Cframe) 與 WAVE6 codec / AI ISP 組成影像子系統; 未公布 TOPS / 製程 / 面積 | 影像專用運算、line / tile 串流 (特徵圖不回 DRAM)、特徵圖壓縮、與 codec / ISP 直接串接 |

## TPU / LPU 的定位

不另做 TPU 或 LPU 晶片; 兩者是架構風格。Xuanji NPU 已是 systolic (TPU 類), 改為吸收兩者的特點:
- TPU: FP8 雙倍速率 (#19)、大陣列 (N=32 X1-Max #28, 資料中心 N=64/128 X3 #30)、SparseCore 式稀疏 / embedding 引擎 (X3)。
- LPU: **權重常駐低延遲 decode 模式** — 權重固定放在 L2 / 本地 scratchpad、編譯器靜態排程 (固定延遲)、大 L2 選項
  (例如 8–32 MiB) 讓小型 LLM / VLM 全部在晶片上。不採用「全部 SRAM、數百晶片串接」: 只適合少數雲端場景。

## 現況差距 (2026-10-28 更新)

| 面向 | 現況 | 差距 |
|---|---|---|
| 數字格式 | 矩陣核 INT8 → INT32; 向量核 FP16 / BF16 / MXFP8 / MXFP4 → FP32 (#19) | 矩陣核無 FP8 / BF16 MAC、無 INT4 權重 |
| 運算單元 | 矩陣核 + 向量 / SFU 核 (#19: softmax / norm / GELU / SiLU kernel 已量測) | attention 融合、MoE 尚未實作 (#20、#25) |
| 影像運算 | 只有 GEMM (conv 需在外部展開) | 無 conv / depthwise / pixel shuffle / 上採樣路徑 |
| 頻寬 | 每個命令從 L2 / 外部記憶體讀寫完整張量 | 無 line-buffer / tile 融合 (多層 conv 中間特徵圖留在 SRAM)、無特徵圖壓縮 |
| 影像子系統 | ISP / JPEG / H.264 / H.265 在 IMS (`docs/ims_interface.md`), 以記憶體交換 | 無 codec / ISP → NPU 的串流輸入 (YUV) |
| 記憶體 | 共享 L2 | 無計算單元本地 scratchpad, A / B 無法跨命令重用 |
| 小 K 效率 | 單一波次 K=64 時 Pro 46%; 連續命令 (v6.3.12 命令重疊) Pro 91.7% | Max (N=32) 小 K 受 C 寫回限制 (K=64 連續命令 34.8%) |
| 稀疏 | 無 | 無 2:4 結構化稀疏、無 embedding 引擎 |
| 資料搬移 | 單一循序 DMA | 無 2D/3D stride、gather/scatter、搬移時解壓縮 |
| 陣列 | N ≤ 16 | 40 TOPS 以上需 N=32 或更多 CU |
| 軟體 | 無 | 無 tiling / 排程編譯器 |
| 外部記憶體 | NPU Subsystem 的外部記憶體埠在 SoC 中接死 | DMA 命令在 SoC 中無法完成 (SoC 整合 #26) |
| 影像 benchmark | GEMM 利用率、向量 kernel | 無 SESR / YOLO 端到端數字 |

## 計劃 (每階段都走同一套驗證: 參考模型 + CRV + SVA + coverage 100% + mutation + 形式驗證 + 效能量測)

### 階段 1 — 讓 transformer 可以執行
1. 向量 / SFU 核 (與矩陣核分開): exp、recip、rsqrt、max / sum / reduce → softmax、LayerNorm / RMSNorm、GELU / SiLU;
   矩陣核輸出直接送向量核 (不繞回 L2)。
2. FP16 / BF16 → FP32 與 MXFP8 / MXFP4 (FP8 = 2 × BF16 速率、MXFP4 再 × 2); 保留 INT8。
3. 驗收: transformer block (attention + FFN) 與 PyTorch 參考誤差; MX 格式與 OCP 參考實作位元一致。

### 階段 2 — 記憶體階層與資料搬移
4. 計算單元本地 scratchpad (L2 改為第二層), A / B 跨命令重用。
5. 多維 DMA: 2D / 3D stride (tiling、im2col)、gather / scatter (MoE)、搬移時解量化 / 解壓縮 (INT4 權重、KV)。
6. 命令重疊: CU 輸出 C 時預取下一命令的 A / B; 目標 Pro K=64 利用率 46% → 75% 以上 (`npu_core_tb +perf_k`)。
   **完成 (v6.3.12)**: npu_mma tile 管線化 + CU 輸入端 / 寫回器分開 (最多持有 3 個命令) + npu_core tag / done 握手;
   連續命令 Pro K=64 = 91.7% (32 輪), 83.4% (8 輪); 同一 TB 量測 v6.3.11 為 48.0%。

### 階段 2V — 影像資料流 (對標 Chips&Media WAVE-N; 併入 #33 之後的 #41–#44)
7V. 影像運算子: conv (直接卷積與 im2col 兩種, 3×3 / 1×1 / stride 2)、depthwise conv (向量核或獨立小引擎)、
    pixel shuffle / 上採樣與 concat (以 DMA 重排實作)、Leaky ReLU / SiLU / sigmoid (向量核已有)。
8V. line-buffer / tile 融合: 多層 conv 以列 (line) 或 tile 為單位串流, 中間特徵圖留在本地 scratchpad / L2,
    只有網路輸入與輸出經過外部記憶體; 編譯器決定融合的層數與 tile 大小。
9V. 特徵圖壓縮: 無損 (delta + bit-packing, 零值 / 小值區塊短碼), DMA 寫出時壓縮、讀入時解壓縮;
    格式與 c1p 的 RFC / FBC 對齊的可能性待評估 (需 c1p repo)。
10V. 影像子系統串接: 在 IMS 合約 (`docs/ims_interface.md`) 增加 AXI-Stream YUV 輸入 (codec / ISP → NPU,
    不經 DRAM); 修改合約須與 IMS repo 同步。
11V. Vision 產品組態: Nano-Vision / X1-Vision (line-buffer 大小、depthwise 引擎、壓縮器), 各組態分別收斂。
12V. 驗收: SESR (×2 / ×4) 與 YOLOv8n 端到端, 與 PyTorch / numpy 參考的誤差 (SESR: PSNR 差; YOLO: 輸出張量誤差與
    偵測框一致); 量測每幀週期數、每幀外部記憶體位元組數 (融合 / 壓縮前後對比)。尚未合成: 不宣稱每瓦效能;
    GB/s 只能以「每拍位元組 × 假設時脈」換算並註明假設。

### 階段 3 — LLM decode
7. INT4 / MXFP4 權重量化 decode 路徑。
8. KV cache 4-bit 量化 (DMA 內解量化), 搭配 attention (#20)。
9. 權重常駐低延遲 decode 模式 (LPU 式): 權重鎖在晶片上 + 靜態排程。
10. 驗收: 每 token DRAM 流量、tokens/s、每 token 延遲變異 (固定排程下應接近 0); 與全 BF16 比較。

### 階段 4 — 擴展與稀疏
11. N=32 陣列與 X1-Max (#28): 每拍 2 個 k 步的資料, L2 bank 加倍。
12. 2:4 結構化稀疏 (權重存非零值 + 索引, MAC 吞吐量 × 2)。
13. 稀疏 / embedding 引擎 (SparseCore 式; X3)。

### 階段 5 — 編譯器
14. tiling (GEMM / conv / attention)、命令產生、雙緩衝與**靜態排程 / 固定延遲**模式;
    輸出 `npu_tile_cmd_t` 命令串, 在 RTL 上端到端執行。

## 順序 (與 SIS 計劃 docs/sis_plan.md 合併)

| 順序 | 內容 | 任務 | 狀態 |
|---|---|---|---|
| 1 | 共用 FP + SFU 函式庫 (NPU 與 SIS 共用) | #21 | 完成 |
| 2 | 向量核 + FP16 / BF16 + MX 格式 | #19 | 單元與引擎已驗證; 剩各組態回歸與文件 |
| 3 | X1-Max (N=32) | #28 | 剩文件 (`docs/npu.md`) |
| 4 | 本地 scratchpad + 多維 DMA (含 conv / im2col、pixel shuffle 重排) + 命令重疊 | #33 | 2D DMA、命令重疊完成; 剩 scratchpad、gather / scatter、3D |
| 5 | 影像運算子 (conv / depthwise) + line-buffer / tile 融合 | #41 | |
| 6 | 特徵圖壓縮 (DMA 內壓縮 / 解壓縮) | #42 | |
| 7 | Vision 產品組態 (Nano-Vision / X1-Vision) | #43 | |
| 8 | SESR / YOLOv8n benchmark (週期、外部記憶體位元組數、誤差) | #44 | 需要命令串 (手寫或 #36) |
| 9 | IMS 合約: AXI-Stream YUV 輸入 (與 IMS repo 同步) | #45 | |
| 10 | Attention + KV cache 量化 | #20 | |
| 11 | INT4 / MXFP4 decode + 權重常駐低延遲模式 | #34 | |
| 12 | MoE (gather / scatter) | #25 | |
| 13 | 2:4 稀疏 | #35 | |
| 14 | 編譯器 (含靜態排程、conv 融合) | #36 | |

影像線 (5–9) 與雲端 LLM 線 (10–13) 市場不重疊, WAVE-N 不影響 LLM 線的規劃。

## 來源

- TPU: [Google Cloud TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x) · [Google blog: Ironwood](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) · [TPU v2→Ironwood](https://arxiv.org/pdf/2606.15870) · [XPU.pub](https://xpu.pub/2025/04/16/google-ironwood/)
- Ascend: [SGEMM-cube](https://arxiv.org/pdf/2507.23387) · [DaVinci](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf) · [Ascend 910C](https://www.emergentmind.com/topics/huawei-ascend-910c-npus)
- Tenstorrent: [RISC-V guide](https://docs.tenstorrent.com/tt-vscode-toolkit/riscv-guide/) · [TT-Metal architecture](https://docs.tenstorrent.com/tt-quietbox2-guide/builder-hacker/01-tt-metal-architecture/) · [Blackhole deck](https://speakerdeck.com/tenstorrent_japan/blackhole-architecture)
- Groq: [LPU architecture](https://scour.ing/@matmat/p/https://groq.com/lpu-architecture) · [deepgains](https://deepgains.substack.com/p/custom-ai-silicon-emerging-challengers) · [naddod](https://www.naddod.com/pt/ai-insights/deep-dive-into-nvidia-groq-3-lpu-a-new-choice-for-ai-inference)
- AMD XDNA: [GEMM on Ryzen AI NPUs](https://arxiv.org/pdf/2512.13282) · [FlashAttention on XDNA](https://arxiv.org/pdf/2609.21264) · [AMD XDNA](https://www.amd.com/en/technologies/xdna.html)
- Client NPU: [NPUs for Inference](https://arxiv.org/html/2609.13166) · [localaimaster](https://localaimaster.com/blog/npu-tops-list)
- Arm Ethos-U85: [Arm product page](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85) · [EE Times](https://www.eetimes.com/arm-brings-transformers-to-iot-devices/)
- MX / FP4: [MXFP4](https://zeroentropy.dev/concepts/mxfp4/) · [NVFP4](https://zeroentropy.dev/concepts/nvfp4/) · [PyTorch MXFP8 / NVFP4](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)
- 影像 NPU: [Chips&Media WAVE-N 生態系 (2026/02)](https://www.edge-ai-vision.com/2026/02/chipsmedia-accelerates-wave-n-ecosystem-redefining-the-future-of-next-generation-customized-npus/) · [SemiWiki: Chips&Media CEO 訪談](https://semiwiki.com/ip/chipsmedia/368201-ceo-interview-with-steve-kim-of-chipsmedia/) (廠商自述, 未公布 TOPS / 面積)
- LLM decode: [P3-LLM](https://arxiv.org/pdf/2511.06838) · [Oaken](https://jongse-park.github.io/files/paper/2025-isca-oaken.pdf) · [SAW-INT4](https://arxiv.org/pdf/2604.19157)
