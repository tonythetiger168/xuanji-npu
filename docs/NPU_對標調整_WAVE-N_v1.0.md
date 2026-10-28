# ORCA NPU 目標調整：對標 Chips&Media WAVE-N（v1.0）

日期：2026-10-28（Asia/Taipei）。狀態：目標調整提案，尚未實作。
依據：專案的 WAVE-N 對標報告（2026-10-28），以及 main `3d66ab0` 的 README、`accelerators/`、
`docs/TV_STB_IPCAM_IMPROVEMENT_PLAN_2026-10-26.md`，
並參考 `codex/edge-npu-soc-design-2026-10-26` 分支（尚未合併）的 edge 整合草案。
本文件只調整「邊緣影像」這條線的技術目標與驗收方式，不改任何價格、成本或財務數字，
雲端 Roc／Garuda 系列也不在調整範圍。

## 1. 為什麼要調整

WAVE-N 是影像專用的邊緣 NPU，主打超解析（SESR）和物件偵測（YOLO），
賣點是「同面積同功耗快 6–10 倍」、FP16 畫質、以及搭配 feature map 壓縮後「頻寬接近零」，
並和 WAVE6 codec、AI ISP 打包成影像子系統賣給相機、IPCAM、車用、手機客戶。
它沒有公開 TOPS、製程與面積。

ORCA 的 edge 層級（Kestrel／Falcon／Hawk）目前寫成通用 AI 加輕量 NLP，
沒有影像專用路徑、沒有 line-buffer 資料流、沒有 feature map 壓縮、
沒有和 codec 直接串接，benchmark 也只有 ResNet 和 LLM。
這幾項正是 WAVE-N 的核心賣點，所以邊緣影像市場上 ORCA 目前沒有可比的說法。

## 2. 現況（誠實盤點）

| 項目 | 現況 | 證據 |
|---|---|---|
| Kestrel RTL | `orca_npu_v5.sv` 285 行，8 MAC | `accelerators/v5_0_kestrel/rtl/` |
| Falcon RTL | `orca_npu_v5_1.sv` 195 行 | `accelerators/v5_1_falcon/rtl/` |
| Hawk RTL | `orca_npu_v5_2.sv` 285 行 | `accelerators/v5_2_hawk/rtl/` |
| 實際算力 | Kestrel 8 MAC × 1 GHz × 2 = 16 GOPS，是標稱 1 TOPS 的 1/62.5 | edge 整合草案的核算 |
| 影像運算子 | 無 depthwise、pixel shuffle、上採樣專用路徑 | grep `depthwise`／`pixel_shuffle` 無結果 |
| 串流介面 | 無 AXI4-Stream video 輸入 | grep `tvalid`／`axis` 無結果 |
| 壓縮 | 只有 `orca_pkg` 的 `sparse_mask` 一行註解 | `rtl/common/orca_pkg.sv:348` |
| 影像 benchmark | 無 | README 只列 ResNet-50 與 LLM |

結論：邊緣影像這條線基本上要從頭設計；
現有 edge RTL 可以當控制面和 MAC 單元的起點，但算力與資料流都不足以支撐 README 的標稱。

## 3. 調整後的定位

- 新增 **ORCA-V（Vision）** 變體，只出現在 Kestrel 與 Falcon 兩個層級：
  **Kestrel-V**（IPCAM、門鈴、低功耗相機）與 **Falcon-V**（車用環景、機器人、多路 NVR）。
  Hawk 維持 Transformer 路線，Phoenix 以上不變。
- Vision 變體的承諾是「每瓦 fps」與「每次推論 DRAM bytes」，
  不再以峰值 TOPS 作為主要宣稱，因為 WAVE-N 本身也不比 TOPS。
- 和 c1p-codec、xuanji-ims（ISP）組成影像子系統，
  對應 TV／STB／IPCAM 總計畫中的 U-CAM SKU 與 P2 的 AI-SR、AI-ISP 項目。

## 4. 調整後的技術目標

### 4.1 影像運算子（P0）

Vision 變體必須原生支援下列運算，不能回落到 CPU：

- 標準 conv：1×1、3×3、5×5，stride 1/2，dilation 1/2。
- depthwise 3×3 與 5×5（獨立 datapath，避免 systolic 陣列利用率掉到 1/C）。
- pixel shuffle／depth-to-space（SESR 輸出端）、nearest 與 bilinear 2× 上採樣（YOLO neck）。
- concat、elementwise add、max-pool 5×5（SPPF）、SiLU／ReLU／LeakyReLU（LUT 或分段線性）。
- 資料格式：INT8 per-channel 量化為主，另提供 **FP16 路徑給超解析**，回應 WAVE-N 的畫質訴求。
  驗收：SESR 在 INT8 下相對 FP32 參考的 PSNR 下降不超過 0.1 dB，超過就必須走 FP16。

### 4.2 Line-buffer／tile-fusion 資料流（P0）

- 多層 conv 以「行」或「tile」為單位融合，中間特徵圖留在片上 SRAM，不回 DRAM。
- line buffer 大小公式：融合 K 層 3×3 conv、寬 W、通道 C（INT8）時約需 `2·K·W·C` bytes。
  例：W=640、C=64、K=4 約 320 KB；W=1920、C=16、K=6（SESR 類）約 360 KB。
  Kestrel-V 先以 256–512 KB SRAM 為設計假設，Falcon-V 以 1–2 MB 為假設，
  實際值由 G1 的 cycle model 決定。
- 不能整層融合的網路（YOLO 的 neck、跨尺度 concat）用 tile 切分加 halo 處理，
  編譯器輸出每個 tile 的 DRAM 讀寫量。
- 驗收指標：**每次推論 DRAM bytes ≤ 權重 + 輸入 + 輸出 + 20%**（無壓縮時），
  用來直接回應 WAVE-N 的「近零頻寬」說法。

### 4.3 Feature map 壓縮（P1）

- 必須溢出到 DRAM 的特徵圖，以及權重，走無損壓縮：
  以 32×4 或 64×1 block 做 delta（對左鄰或上鄰）加 bit-packing，附 block header。
- 和 c1p-codec 計畫中的 RFC／FBC（lossless reference-frame compression）共用 block 格式與
  encoder／decoder 核心，只在 tile 形狀和 header 欄位上做參數化。兩邊各寫各的會重複工作。
- 驗收：YOLOv8n 溢出特徵圖平均壓縮比 ≥ 1.5×；權重壓縮比另外報告；
  壓縮器吞吐不低於 NPU DMA 頻寬，不造成停頓。

### 4.4 與 c1p-codec 直接串接（P1）

- NPU 增加 AXI4-Stream video 輸入埠，遵循總計畫的 P0 共用合約：
  SOF（`tuser[0]`）、EOL（`tlast`）、有效 lane、NV12／P010 packing、backpressure、跨時脈 FIFO。
- 前處理放在 NPU 入口：YUV→RGB、縮放到模型輸入尺寸、正規化，
  免得 c1p 解碼輸出先寫 DRAM、CPU 再讀回來做前處理。
- 兩種模式：
  1. **直通**：c1p 解碼輸出直接送 NPU，同時照常寫 frame buffer 給顯示。
  2. **旁路**：NPU 從 DRAM 讀 frame（含 c1p 的 RFC 壓縮格式，可直接解壓讀入）。
- 輸出：偵測結果走 descriptor ring 到 DRAM；超解析結果以 AXI4-Stream 送給 PRISM scaler 或顯示路徑。
- c1p 目前沒有 inter 預測、4K60 也未達成；串接驗收先用 HEVC intra 串流與 pattern source 做。

### 4.5 Benchmark 與報告格式（P0）

每個 Vision 配置都必須報告下面這張表，數字來自 cycle model 或 RTL 模擬，
並註明頻率、SRAM、工具版本；未經合成與 STA 的數字標成「估算」：

| 模型 | 輸入 | 指標 |
|---|---|---|
| SESR-M5（×2） | 960×540 → 1920×1080 | fps、PSNR（相對 FP32）、DRAM GB/s、fps/W（估算） |
| SESR-M5（×2） | 1920×1080 → 3840×2160 | 同上（Falcon-V） |
| YOLOv8n | 640×640 | fps、mAP 差異（相對 FP32）、DRAM GB/s、fps/W（估算） |
| MobileNetV2 | 224×224 | 和既有 ResNet 結果銜接，作為通用基準 |

算力需求參考：YOLOv8n 約 8.7 GFLOPs／幀，30 fps 需約 0.26 TOPS 實際算力；
以 50% 利用率估，Kestrel-V 至少需要約 256 MAC @ 1 GHz。
現有 Kestrel 只有 8 MAC，所以 Kestrel-V 的 MAC 陣列需要擴大約 32 倍，這點要明確寫進規格。

## 5. 不變的部分

- 雲端 Roc／Roc-C／Roc-D／Roc-E／Garuda 的定位、規格與對標平頭哥、寒武紀的內容不變；
  WAVE-N 不做 LLM，市場不重疊。
- Hawk 維持 Transformer 硬加速路線。
- README 與報告中的所有價格、成本、財務數字不變。
- 共用 ISA 與編譯器方向不變；Vision 變體以 MMIO offload 加 descriptor 執行，
  和 edge 整合草案的首代策略一致，不要求 custom instruction。

## 6. 里程碑（對齊總計畫的 G0–G3）

| 階段 | 內容 | 完成條件 |
|---|---|---|
| G0 | 匯出 SESR-M5、YOLOv8n、MobileNetV2 的 ONNX 與 INT8／FP16 參考輸出；建 layer-by-layer cycle 與 DRAM 流量模型 | 模型與 golden 可重現，baseline（不融合）數字出表 |
| G1 | 決定 Kestrel-V／Falcon-V 的 MAC 數與 SRAM；凍結 AXI4-Stream 輸入與 descriptor 合約；和 c1p 對齊壓縮 block 格式 | 規格文件與兩個 repo 的介面審查通過 |
| G2 | RTL：depthwise／pixel shuffle／上採樣單元、line-buffer 控制器、壓縮器；每個單元有 scoreboard、SVA、coverage | 單元 TB 全通過，結果與 golden 位元一致 |
| G3 | 端到端：c1p 解碼 → NPU YOLOv8n → 結果；pattern → NPU SESR → 顯示 | 第 4.5 節表格以 RTL 模擬數字填滿 |
| G4 | 合成、STA、功耗估算（開源 PDK），補 fps/W | 未做完不得宣稱 PPA |

## 7. 不宣稱的事項

- 不宣稱勝過 WAVE-N 的「6–10 倍」；WAVE-N 沒有公開數字，只能比較我們自己量到的 fps、DRAM GB/s 與畫質。
- 不宣稱 FP16 畫質等同 WAVE-N v2；只報告我們相對 FP32 參考的 PSNR／mAP 差異。
- 預先驗證模型與軟體生態（WAVE-N 的 AI All-in-One 聯盟）短期無法對等，
  先交付上述三個模型的完整編譯與驗證流程。
- 未完成合成、STA、板級驗證前，所有 fps/W 都是估算。

## 來源

- 專案文件：Chips&Media WAVE-N 對標 orca-v63（2026-10-28）、Chips&Media 競品報告（c1p 目標調整）
- [WAVE-N 生態系（2026/02）](https://www.edge-ai-vision.com/2026/02/chipsmedia-accelerates-wave-n-ecosystem-redefining-the-future-of-next-generation-customized-npus/)
- [SemiWiki CEO 訪談](https://semiwiki.com/ip/chipsmedia/368201-ceo-interview-with-steve-kim-of-chipsmedia/)
- [Ultralytics YOLOv8 模型表（YOLOv8n 8.7 GFLOPs）](https://docs.ultralytics.com/models/yolov8/)
