# Xuanji-X1 NPU Subsystem: 三款產品組態與驗證

檔案: `rtl/npu/` (RTL)、`dv/npu/` (驗證)。舊 AI tile 的模組 (cluster / systolic / attention / HBM3 / NoC 介面卡 / KV cache / MoE router)
已移到 `rtl/legacy/ai/`, **未經驗證**, 不屬於 NPU Subsystem。

同一份 RTL (`rtl/npu/npu_core.sv`) 以參數產生三款產品。每款都有獨立的驗證收斂
(CRV + SVA + coverage 100% + mutation + 效能量測), CI job `npu (nano / x1 / pro)`。

## 產品組態

| 產品 | N×N 陣列 | 計算單元 (CU) | L2 bank × 容量 | 峰值 MAC/拍 | 峰值 INT8 @ 1 GHz* | 量測 MAC 利用率 (K=4000) | 對標 (市場層級, 見 STATUS_AND_ROADMAP §4b) |
|---|---|---|---|---|---|---|---|
| **Xuanji-X1-Nano** | 4×4 | 1 | 2 × 32 KiB = 64 KiB | 16 | 32 GOPS | 99.7% (15.9 MAC/拍) | MCU / IoT 的 AI 加速 (關鍵字偵測、感測器融合) |
| **Xuanji-X1** | 8×8 | 4 | 4 × 64 KiB = 256 KiB | 256 | 512 GOPS | 99.2% (253.9 MAC/拍) | 邊緣視覺 / 小型模型推論 |
| **Xuanji-X1-Pro** | 16×16 | 8 | 16 × 64 KiB = 1 MiB | 2048 | 4.1 TOPS | 98.1% (2010 MAC/拍) | 邊緣 AI / 機器人 (往 Ascend 310 級 16 TOPS 的路上) |

\* 時脈是假設值: **尚未做合成、時序收斂與面積 / 功耗評估**。TOPS = MAC/拍 × 2 × 時脈 × 利用率 (只計 INT8 GEMM)。
利用率是 RTL 模擬量測 (`npu_core_tb +perf`: 每個 CU 一個 K=4000 的命令同時執行), 不是推論。
CI 以 95% 作為效能回歸門檻 (`dv/scripts/npu_cfg.sh`)。

短 K 時固定開銷 (start、讀第一對 line、排空 2N−1 拍、輸出 N 列) 佔比上升, 陣列越大越明顯。
單一波次 (`npu_core_tb +perf +perf_k=K`, 每個 CU 一個命令同時執行; 開銷無法被下一個命令遮蓋):

| K | X1-Nano | X1 | X1-Pro |
|---|---|---|---|
| 64 | 82.1% | 65.3% | 46.0% |
| 256 | 94.8% | 88.0% | 77.6% |
| 1024 | 98.7% | 96.7% | 93.2% |
| 4000 | 99.7% | 99.2% | 98.1% |

(v6.3.11 加入 L2 light sleep 後重新量測; 之前 X1 為 64.6 / 88.6 / 96.8 / 99.1%、Pro 為 46.4 / 77.8 / 93.4 / 98.2%)

連續命令 (v6.3.12 命令重疊, 見「架構 / 命令重疊」): `+perf_rounds=R` 把同一組 NCU 個命令連續送 R 輪
(完成時只記錄 id, 結束後才讀回 C 比對, 不以 host 讀取佔用 L2 或回壓完成)。前一個命令排空 / 輸出 / 寫回時下一個命令
已在預取與餵入, 固定開銷大部分被遮蓋。同一個 TB 量測 v6.3.11 RTL (不重疊) 作為對照:

| K | 輪數 R | X1-Nano | X1 | X1-Pro | v6.3.11 X1 / Pro (同 TB) |
|---|---|---|---|---|---|
| 64 | 8 | 93.4% | 89.8% | 83.4% | 65.9% / 47.6% |
| 64 | 32 | 95.0% | 94.0% | **91.7%** | 66.0% / 48.0% |
| 256 | 8 | 98.3% | 97.2% | 95.1% | 88.7% / 78.4% |
| 256 | 32 | 98.7% | 98.4% | 97.6% | 88.9% / 78.8% |
| 1024 | 8 | 99.6% | 99.3% | 98.7% | — |
| 1024 | 32 | 99.7% | 99.6% | 99.3% | — |

(單一波次的數字不變: 第一個命令的開銷沒有前一個命令可以重疊。)

X1-Max (N=32, NCU=24, NB=32; `npu_cfg max`; v6.3.12 量測, 同一 TB):

| K | 單一波次 | 連續 8 輪 |
|---|---|---|
| 64 | 23.7% | 34.8% |
| 256 | 53.7% | 87.0% |
| 958 (slot 上限) | 78.5% | 93.6% |

N=32 的小 K 受 C 寫回限制: 一個 tile 的 C 是 64 個 line (4 KiB), K=64 時與 A + B 的輸入量相同, 寫回器每列寫 2 個
line, 24 個 CU 共用 32 個 bank 的單一寫入埠 — 寫回時間比計算長, 重疊遮不住。功能: `npu_core_tb` (seed 1, 450 個命令)
errors = 0。

X1-Max 的 NPU coverage 收斂 (v6.3.12, 本機; `run_npu_cov.sh max 4 200` 的全部 23 個執行, 以 `RESUME=1 NOREPORT=1 ONLY=...`
分成多個行程並行, 最後 `REPORT_ONLY=1` 合併): line 100% (51)、branch 100% (106)、expr 100% (686)、toggle 100% (623 802,
3 個 [C] 豁免)、covergroup 100% (243, 6 個 [C] 豁免)、SVA cover 100% (50); 豁免見 `dv/npu/waivers_max.txt`。
core TB 每個 seed (200 命令) 約 2 小時 20 分、約 3.9 GB 記憶體 (4 個同時執行會超過 15 GB 而被 OOM 終止), 所以 CI 只做
Max 的建置 + smoke (`npu-max-smoke`), 收斂在本機分段執行。效能: K=958 單一波次 81.0% (門檻 78%)、K=64 22.1%、K=256 53.8%
(K=1024 受 slot 上限 958 限制, 與 K=958 相同)。Max 的 mutation 尚未執行 (每個 mutant 重建約 20 分鐘)。

**尚未具備** (列為下一步, 不對外宣稱): BF16 / FP8、稀疏、卷積 / attention 專用資料流、SoC 的 APB 互連與 NoC / DRAM 連接
(NPU Subsystem 在 SoC 頂層的命令與外部記憶體埠目前接死)、編譯器 / runtime。
CPU 的設定 / 命令介面已有: APB4 暫存器 `npu_ss_regs` + 包裝 `orca_npu_ss_apb` (暫存器表與驗證見 `docs/apb.md`)。

## NPU Subsystem (`orca_npu_ss`)

npu_core 加上命令前端與 DMA, 成為可以直接下命令的子系統 (由舊 AI tile 重建; 原 `orca_v63_ai_tile`)。

```
命令 ──► 命令佇列 (QD=8) ──► 依序分派 ──┬─► npu_core (GEMM, 多個 CU 同時執行)
                                       ├─► npu_ss_dma (外部記憶體 ⇄ L2, 64 B line) ─┐ npu_l2_arb2: 共用
                                       ├─► npu_vec_eng (VEC: 向量程式)  ──────────┘ npu_core 的 host L2 埠
                                       ├─► npu_llm_eng (KV / MOE 表格) ──────────── (第二個 npu_l2_arb2)
                                       └─► NOP (wt=1 時即 fence)
                    完成合併 (round-robin, 輸出暫存保持) ◄┘──► cpl_valid + id, irq
```

| 命令 | 動作 |
|---|---|
| `GEMM` | C = A × B (INT8 → INT32, 資料已在 L2); ext[0] = 1 (v6.3.14) 時 **C += A × B** (累加, 見下「GEMM 累加」); K = 0 時只回報完成 |
| `DMA_LD` / `DMA_ST` | 外部記憶體 → L2 / L2 → 外部記憶體, k 個 line (k = 0 時立即完成); 位址模數環繞。2D (v6.3.11): b[15:0] = 每列 line 數 (0 = 1D), c = 外部列距 (line) — 第 i 個 line 的外部位址 = ext + (i / cols) × c + i % cols, L2 端連續。載入時解碼 (v6.3.13, 只對 LD): b[17:16] = 0 COPY / 1 INT4 / 2 UINT4 / 3 MXFP4。間接 gather / scatter (v6.3.13): b[18] = 1, c = 位址表的 L2 line, b[15:0] = 每項 line 數; 見下方「DMA 引擎 npu_dma」 |
| `VEC` (op 4) | 執行 L2 line `a` 起的 `k` 個向量指令 (每 line 4 個; `docs/npu_vec.md`); k = 0 時只回報完成; 遇到非法指令時 `vec_err` 一拍 |
| `KV` (op 5, v6.3.13) | paged KV cache 頁管理 (ALLOC / FREE / FREE_SEQ / SHARE) 與 MAP (區塊表 → 間接 DMA 的位址表); 欄位與結果見 `docs/npu_llm.md` |
| `MOE` (op 6, v6.3.13) | 從 L2 的 top-k 結果產生 MoE gather 位址表、combine 表、各 expert 數量; 見 `docs/npu_llm.md` |
| `NOP` (op 3; 保留的 op 7 同 NOP) | 只回報完成 |

- **等待旗標 `wt`**: 先前所有命令完成 (結果已寫入, 完成已移入輸出暫存) 後才分派。沒有 `wt` 的命令可與先前的命令並行
  (例如 GEMM 執行時 DMA 載入下一批資料); 軟體須保證並行命令的 L2 / 外部區域不重疊, 否則設 `wt`。
- 完成順序不保證與命令順序相同。
- **DMA 吞吐量**: 每拍 1 line (64 B); 外部讀取延遲 L 拍時為 min(1, OUT / (L + 3)) line/拍, OUT = 在途上限
  (v6.3.13: Nano 32 / X1 64 / Pro 128 / Max 128; 之前 16 / 32 / 64 / 128)。量測 (`npu_ss_tb +perf_dma=L`) 見下表。
  解碼 LD 每個外部 line 寫 2 個 L2 line, 外部只需 0.5 line/拍就能讓 L2 寫入滿速。

### DMA 引擎 `npu_dma` (v6.3.13)

由 `npu_ss_dma` 複製並加上載入時解碼; `npu_ss_dma` 保持不變 (SIS 子系統使用)。介面多一個 `cmd_fmt` (orca_npu_ss 接 `b[17:16]`)。
主流 LLM 權重以 4 bit 發佈 (INT4 / MXFP4), decode 階段受 DRAM 頻寬限制: 在 DMA 內展開成 INT8 讓 GEMM 不必改,
外部位元組數減半。

| fmt | 格式 | 每個 4 bit 元素 → INT8 |
|---|---|---|
| 0 | COPY | 原樣搬移 (外部 1 line → L2 1 line) |
| 1 | INT4 | 符號延伸: −8..7 |
| 2 | UINT4 | 補 0: 0..15 (非對稱量化; zero point 由軟體處理) |
| 3 | MXFP4 (OCP E2M1) | 值 × 2: ±{0, 1, 2, 3, 4, 6, 8, 12} (精確; −0 → 0) |

- 打包順序: 第 2i 個元素 = byte i 的低 4 bit (OCP MX)。外部 line j 的元素 0..63 → L2 `a + 2j`, 64..127 → `a + 2j + 1`; k 是外部 line 數。
- MX 的區塊比例 (E8M0, 每 32 個元素) 與 INT4 的 group scale 不在 DMA 套用: per-channel scale 由向量核在 GEMM 之後反量化;
  每 32 個 K 一個比例的 MX 區塊比例列為後續工作 (GEMM 端)。MXFP4 解出的值是真值 × 2, 反量化時比例要多除 2。
- ST 忽略 fmt (原樣搬移)。2D 與解碼可同時使用 (外部端 2D, L2 端連續)。
- 驗證: 形式證明 (k-induction, DW = 64, 檢查器以 E2M1 的定義另寫參考解碼) + 形式 mutation 25 / 25 (含間接 12 個);
  subsystem CRV 的參考模型以實數計算 E2M1, LD 有 50% 機率隨機選解碼格式 (含 2D、長 DMA 65535 line → L2 131070 line、位址環繞)。
- **間接 gather / scatter** (`b[18] = 1`, 給 paged KV cache 與 MoE token 分派):
  位址表放在 L2 (line `c` 起), 每項 u32 小端, 一個 line 16 項。k = 項數, 每項搬 cw = (b[15:0] = 0 ? 1 : b[15:0]) 個 line。
  第 i 項的值 E: 外部端 `ext + E .. ext + E + cw − 1`, L2 端連續 (`a + i × cw`, 解碼時 × 2)。LD = gather, ST = scatter。
  總 line 數 k × cw 須 <= 65535 (計數器 16 bit; 較長的 gather 由軟體切成多個命令)。
  E = `0xFFFF_FFFF` 為略過: LD 對該項的 L2 line 寫 0 (不讀外部), ST 不寫外部。
  位址表經 L2 讀埠讀入 2 line 緩衝並預取下一個 line; ST 時與資料讀取共用讀埠 (位址表優先, 請求保持到 gnt)。
  解碼 (b[17:16]) 可與間接同時使用 (例: 從分頁的 INT4 權重 / KV 區塊 gather 並展開成 INT8)。
  LD 的略過項要等先前發出的外部讀取都回來才進入 FIFO (保持順序, 不需要重排緩衝), 所以略過密集時吞吐量會下降。
  形式證明 (DW = 64, 每 line 2 項, 假設 cw <= 2) 另外追蹤一個任意項的位址表讀取與外部位址;
  subsystem CRV 隨機產生位址表 (含略過、cw 1..12、與解碼同時), 先以一般 DMA_LD 把位址表載入 L2 再執行間接命令。
  另有一個 directed 測試: 項數 65505..65535 的間接 ST (位址表 4095 / 4096 個 line, 在 L2 折返), 涵蓋位址表計數器的所有位元。
  形式證明只含安全性質: 「位址表最後一個 line 沒讀」這類只會停住的錯誤由 subsystem TB 的 watchdog 抓 (mutation `ind_nl`)。
  mutation: 形式 12 個 (`ind_*`, 其中 `ind_nohold` 以 BMC 16 拍), subsystem 15 個 (含命令欄位接錯)。
- **多通道** (`npu_dma_mc`, v6.3.13): orca_npu_ss 的參數 `NDC` (預設 2, 所有產品組態) 個 `npu_dma` 通道, 不同通道的 DMA 命令同時執行
  (例: 載入下一層權重的同時 gather KV cache 或寫回結果)。命令交給編號最小的空閒通道 (閒置且完成槽空); 每個通道一個完成槽,
  round-robin 回報。通道之間沒有順序保證, 與其他 wt = 0 的命令相同: 同時執行的 DMA 存取區域須不重疊, 否則以 wt 分隔。
  L2 讀 / L2 寫 / 外部讀 / 外部寫各一個 `npu_rr_lock` (round-robin, 等待 grant 時鎖定); 外部讀回應依請求順序,
  以深度 NDC × OUT 的 tag FIFO 送回發出的通道。每個通道各有 OUT 項 FIFO 與一個 ICG。
  形式: `npu_rr_lock` (NM = 3: one-hot、只選請求者、鎖定、確切 round-robin、每個主端最多等 NM − 1 次) 與 `npu_dma_mc`
  (通道為替身; 命令分派、四個埠的位址 / 資料屬於得主、任選一個外部讀取追蹤其回應送回發出者、完成 id) 皆 k-induction 證明;
  `orca_npu_ss` 的形式以 npu_dma_mc 替身 (最多 NDC 個未回報命令) 重新證明。subsystem SVA 另外掛在每個通道與 npu_dma_mc 上。
- **低功耗**: 每個 DMA 通道閒置時停止時脈 (ICG); 向量引擎閒置時停止取指 / 控制的時脈, 每個向量暫存器各有 ICG; npu_core 內每個 CU、每列另有 ICG。

### NPU Subsystem 驗證

```
bash dv/scripts/run_npu_ss_cov.sh <nano|x1|pro|all> [seeds] [batches]   # coverage 收斂 + DMA 吞吐量
bash dv/scripts/run_npu_ss_mutation.sh <nano|x1|pro|all>                # 檢查器 mutation 測試
```

| 元件 | 內容 |
|---|---|
| TB `dv/npu/npu_ss_tb.sv` | 每個 batch 一段隨機命令程式 (含 K = 0、line 數 0、L2 2^26 / 外部 2^32 位址環繞); 產生器追蹤自上一個 `wt` 以來的讀 / 寫範圍, 衝突必設 `wt` → DUT 必須等同依序執行。外部記憶體模型: grant 隨機 (與 req 無關)、讀回應依序且延遲隨機 (含 > OUT)。比對整個 L2 (後門) 與外部記憶體、每個 id 恰好完成一次、`wt` 命令的完成在所有先前命令之後、`perf_steps`。另有: GEMM / NOP / DMA 突發 (填滿 npu_core 佇列)、65535 line 長 DMA、fill_all (完成回壓時填滿所有槽)、中途 reset、隨機 `test_en` |
| SVA `dv/npu/npu_ss_sva.sv` | 33 個 assertion: 佇列 / 分派 (`a_wt`、`a_no_stall`)、未完成數上限 (`a_oc_bound`, 界限可達)、完成合併 (保持、不遺失、公平 ≤ 2 次搶先)、DMA 帳目 / 吞吐量 (`a_issue`) / 請求保持、閘控安全; 24 個 cover |
| covergroup | 命令類型 × `wt`、分派時未完成數、每種來源組合下的選擇、佇列深度、DMA 長度、在途數 / FIFO 深度 |
| 範圍 | 量測 tile 自身 (`orca_npu_ss`、`npu_dma_mc`、`npu_dma`、`npu_rr_lock`、`clock_gate`); npu_core 子樹由上面的 NPU flow 收斂 |

### NPU Subsystem 結果 (每款 6 seed × 30 batch + 2 次隨機上電 + DMA 吞吐量量測; 豁免 0 項; v6.3.12 重新收斂)

| 組態 | DMA OUT | line | branch | expr | toggle | covergroup | SVA cover | mutation |
|---|---|---|---|---|---|---|---|---|
| X1-Nano | 16 | 100% (15) | 100% (76) | 100% (157) | 100% (39 032) | 100% (110) | 100% (44) | 38 / 38 |
| X1 | 32 | 100% (15) | 100% (76) | 100% (157) | 100% (55 426) | 100% (158) | 100% (44) | 38 / 38 |
| X1-Pro | 64 | 100% (15) | 100% (76) | 100% (157) | 100% (88 204) | 100% (254) | 100% (44) | 38 / 38 |

mutation (`run_npu_ss_mutation.sh`, 20 batch + 隨機上電一次): v6.3.12 三款 38 / 38 (含 2D DMA 3 個、`arb_nolock_w`);
2D DMA 加入前為 35 / 35。加入向量引擎 (#19) 後原本
有兩個 pattern 失效、數個存活, 修正: L2 仲裁器 (`npu_l2_arb2`) 加只看埠的 SVA (下游請求保持、round-robin 公平) — 原本
「等待中換主端」與「固定優先權」只有形式證明抓得到; TB 的 VEC 程式加連續 ST / 連續 LD 變體, 與 DMA 爭用 host L2 埠
(Pro 的固定優先權 mutant 在 12 batch 內沒有爭用, 預設改為 20 batch); `busy` 拿掉 `v_busy` 是等價 mutant (形式不變式
v_busy ⇒ oc ≠ 0)。Pro 的 completion 競爭 bin「VEC 待回報時選中 DMA」在隨機中沒有出現, `fill_all` 加上只有 DMA 與 VEC
競爭的情境 (prime 5)。2D DMA 新增的 mutant (列多一個 line、忽略列距、ST 的外部位址跟著讀端前進、位址高位遺失) 見
`docs/npu.md`「形式驗證」與回歸紀錄。

DMA 讀取吞吐量 (LD 2048 line, 外部埠全速; 寫入 ST 在所有情況為 0.998 line/拍):

| 讀取延遲 L | X1-Nano (OUT 16) | X1 (OUT 32) | X1-Pro (OUT 64) |
|---|---|---|---|
| 0 | 0.996 | 0.996 | 0.996 |
| OUT − 3 | 0.990 (L=13) | 0.982 (L=29) | 0.946 (L=61) |
| 2 × OUT | 0.455 (L=32) | 0.472 (L=64) | 0.451 (L=128) |
| 100 | 0.151 | 0.300 | 0.593 |

量測值接近 min(1, OUT / (L + 3)) (OUT 大時略低: Pro L=61 為 0.946)。v6.3.12 重新量測; 與 v6.3.11 文件中的數字
(Pro 0.967 / 0.481 / 0.609 等) 的差異來自 2D DMA 之後的 RTL — 以命令重疊前的 RTL (同一 TB) 量測 Pro L=61 / 100 同為
0.946 / 0.593, 命令重疊不影響 DMA。 DRAM 級延遲 (約 100 拍) 下要接近滿速需要更大的 OUT 或多個 DMA 通道。

### NPU Subsystem 驗證中找到的問題

| 問題 | 種類 |
|---|---|
| DMA 在途上限 8 時, 讀取延遲 16 拍吞吐量只剩 0.42 line/拍 → OUT 改為依產品 16 / 32 / 64 | 設計 (效能量測發現) |
| 未完成數上限原本高估 (每個 CU 只會「執行中」或「待回報」其一) → 改為緊的界限, SVA 證明可達 | 設計 |
| 計數器寬度大於其界限 (在途數、未完成數) → 依界限縮小 | coverage → 改寫 RTL (不豁免) |
| K = 0 的 GEMM 會違反 CU 的 K ≥ 1 假設 → tile 以 NOP 處理 | 設計審查 |
| 外部埠 grant 未以 req 限定 | 設計審查 (TB 以與 req 無關的隨機 grant 驗證) |
| mutation: 固定優先權的完成合併在 X1-Nano 存活 (「最多被搶先 2 次」太寬) → 加入確切 round-robin assertion `a_rr` | 檢查器 |
| DMA 時脈關閉時發生非同步 reset, 以閘控時脈取樣的 SVA 誤報 → DMA SVA 改用 tile 的未閘控時脈 | 檢查器 |
| TB: 結束時 `wait (!busy)` 不受 watchdog 保護、48 bit 計數器基準值以有號數比較 | TB |
| **v6.3.12 命令重疊後 npu_core 容量變成 CQD + 4·NCU + 1 (每個 CU 持有 3 個命令 + 1 個待回報), 但 tile 的未完成數 `oc` 的寬度 / 上限仍依舊容量 (X1: 上限 12、4 bit, 實際可到 24 → 環繞, 等待旗標失效)** → `CCAP` 依新容量; 形式證明沒抓到, 因為 npu_core 替身的契約仍是舊容量 (替身契約必須隨被替換模組的介面一起改) | 模擬 SVA `a_oc_bound` (X1 隨機測試) |
| v6.3.12: `fill_all` 以 `oc` 到達上限判斷「所有槽都滿」, 命令重疊後 GEMM 可能都還在 CU 內計算 (沒有完成在等) → 四方競爭的 bin 沒有出現 | coverage (`cg_cpl.cp_sel.all`) → 另等 GEMM 的完成 |
| mutation `arb_nolock` (host L2 埠仲裁器等待中換主端) 在 X1 / Nano 存活: 需要「一個主端等 grant 時另一個開始請求, 且 round-robin 偏向後者」 → 加入埠層級 cover `c_rd_lock_*` / `c_wr_lock_*` (coverage 要求 100%) 與 directed `arb_lock` (以 L2 light sleep 喚醒製造等待); 另加寫入方向的 `arb_nolock_w` | mutation |

## 形式驗證 (SymbiYosys)

```
bash dv/scripts/run_formal.sh            # 證明 (k-induction) + cover
bash dv/scripts/run_formal_mutation.sh   # 形式 mutation: 植入錯誤的 RTL 必須無法證明
```

流程: sv2v (`-DFORMAL -E Assert`, 保留 assertion) → Yosys 0.33 → SymbiYosys (`smtbmc`, z3 ≥ 4.12)。
性質與歸納不變式寫在 RTL 的 `` `ifdef FORMAL `` 區塊 (模擬時不編入)。

| 區塊 | 參數 | 性質 | 結果 |
|---|---|---|---|
| `npu_ss_dma` | OUT = 4, DW = 8 (計數器為實際 16 bit) | 計數順序 (寫 ≤ 讀 ≤ n)、在途 ≤ OUT、FIFO 佔用 / 指標、done 恰在第 n 個寫入、有額度就發讀、請求等待 grant 時保持、done 為單拍; **資料完整**: 任選一個 line index (anyconst), 從輸入埠記錄的資料必須等於寫出的資料 | k-induction 證明通過 (深度 5); cover 3 / 3 可達; 形式 mutation 7 / 7 無法證明 |
| `npu_dma` (v6.3.13) | OUT = 4, DW = 64 | `npu_ss_dma` 的性質 + 載入時解碼 (檢查器另寫 E2M1 參考) + 間接: 任選一項追蹤其位址表讀取、值與外部位址、略過項 (假設 cw ≤ 2) | k-induction 證明通過; cover 可達; 形式 mutation 25 / 25 |
| `npu_rr_lock` (v6.3.13) | NM = 3 | one-hot、只選請求者、等待 grant 時保持、確切 round-robin、每個主端最多被搶先 NM − 1 次 | k-induction 證明通過; cover 可達 |
| `npu_dma_mc` (v6.3.13) | NDC = 2, OUT = 4, DW = 8; 通道為契約替身 (`formal/npu_dma_mc/stubs.sv`) | 命令交給最小的空閒通道、四個共用埠的 grant 只給請求者且位址 / 資料屬於得主、任選第 k 個外部讀取其回應送回發出的通道、回應只送給有在途讀取的通道、完成 id = 該通道接受的命令 id | k-induction 證明通過; cover 可達 |
| `orca_npu_ss` (前端) | QD = 4, NCU = 2; npu_core / DMA / ICG 以契約替身取代 (`formal/orca_npu_ss/stubs.sv`) | 命令守恆 (接受 = 佇列中 + 已分派; 已分派 = 未完成 + 已載入; 已載入 = 已輸出 + 輸出暫存中)、未完成數 = core 內 + DMA (v6.3.13: 最多 NDC 個) + NOP 且 ≤ 上限、等待旗標 (oc = 0 才分派)、可分派時必分派、K = 0 不送 core、NOP / VEC 完成槽的 id 不被覆蓋、DMA 的完成 id 原樣輸出、完成合併選一個有效來源且為**確切 round-robin** (檢查器自己記錄上次選中者)、輸出等待時保持、core 的 id 原樣輸出、busy / irq (DMA 時脈閘控在 npu_dma_mc 內, 由 subsystem SVA `a_gate` 檢查) | k-induction 證明通過 (深度 6); cover 4 / 4 可達; 形式 mutation 14 / 14 無法證明 |
| `npu_l2_xbar` | NM = 4, NW = 3, NB = 2, DEPTH = 2, LDW = 8 (新增參數 LDW, 預設 512); light sleep LS_IDLE = 3、LS_WAKE = 2 | grant 必有請求、每個 bank 每拍最多一讀一寫 grant、醒著時**確切 round-robin** 且 work-conserving (檢查器從 gnt 埠記錄得主)、等待 ≤ NM − 1 + LS_WAKE + 1 拍; light sleep: 檢查器由請求埠重算每個 bank 的睡眠 / 喚醒狀態, `bank_ls` 相符、睡眠 / 喚醒中不給 grant; **資料**: 任選一個 line (anyconst), 讀到的必是最後寫入的資料 | k-induction 證明通過 (深度 10); 拿掉輔助不變式的 ABC BMC 30 拍通過; cover 5 / 5 可達; 形式 mutation 13 / 13 無法證明 (含 light sleep 5) |
| `npu_core` (控制) | QD = 4, NCU = 2; CU / xbar / ICG 以契約替身取代 (`formal/npu_core/stubs.sv`; CU 的契約以 assume 寫在 npu_core 的 FORMAL 區塊) | 命令守恆 (接受 − 輸出 = 未完成 = 佇列 + 各 CU 持有 + 待回報槽 + 輸出暫存) 且上限 QD + 4·NCU + 1 為緊; 輸出的 id 必是尚未完成的命令 (不捏造、不重複; host 契約: id 不與未完成者重複); 檢查器從埠記錄每個 CU 持有的 tag 佇列, 待回報槽的 tag = CU 完成握手帶回的 tag; 分派最多一個 CU、只給可接受命令的 CU、優先完全閒置者中編號最小者、否則可接受者中編號最小者、可分派時必分派; done_ready = 待回報槽空; 輸出等待時保持; **完成仲裁公平**: 待回報期間被略過 ≤ NCU − 1 次; busy / cmd_ready | k-induction 證明通過 (深度 4, 約 17 分鐘); cover 6 / 6 可達; 形式 mutation 11 / 11 無法證明 |

替身只遵守介面契約 (npu_core: 容量 CQD + 4·NCU + 1 (v6.3.12)、只在有未回報命令時回報、等待時保持; DMA: idle 才接受、忙碌中某拍 done;
CU (v6.3.12): 最多持有 3 個命令、閒置時可接受、busy = 持有命令、done 依受理順序帶回 tag 並保持到 done_ready; xbar: 只 grant 有請求者)。被替身取代的區塊本身都另有證明或模擬驗證:
DMA、xbar、npu_core 見上表; CU (`npu_gemm_cu`) 與 `npu_mma` 的運算由 NPU flow (CRV + coverage 100% + mutation) 驗證。
時脈閘控在 `npu_core` 證明中以直通替身表示, 其安全性由模擬 SVA (未閘控時脈取樣) 與 `orca_npu_ss` 的 DMA 閘控性質涵蓋。

DW / LDW = 8 的資料完整性推廣到 512: DMA 與 xbar 不運算資料, 只依位置搬移, 每個 bit 獨立。

形式 mutation 中 npu_core 為 11 / 11 (v6.3.12 命令重疊後重寫; 其他區塊見上表)。只有證明實際失敗 (FAIL / UNKNOWN) 才算抓到;
逾時或工具錯誤記為 INCONCLUSIVE 並使流程失敗。xbar 原本的 `gnt_no_req` 是等價 mutant (grant 得主必是請求者, 拿掉 req 條件不改行為),
已換成 `gnt_all`。

形式驗證過程中發現的檢查器問題 (都已修正, 記錄以免重犯):
- sv2v 預設刪除所有 assertion: 證明「通過」但沒有檢查任何性質 → `-E Assert`, 且轉出的設計沒有 assert 時流程失敗。
- 資料完整性原本記錄內部的 `push_d`, 與被檢查的邏輯循環 (mutation `pushd_wrong` 存活) → 改為從輸入埠記錄。
- round-robin 性質原本相對於 RTL 自己的 `last` 指標 (固定優先權的 mutant `cpl_fixed` 存活) → 檢查器自己從 `sel` 記錄上次選中者。
- `if (c) assert (x); else ...` 的 `else` 會被當成 assert 的失敗動作 → 一律加 `begin`/`end`。
- 只看埠的 wrapper 以 PDR 證明不收斂 (frame 5 需 620 s) → 不變式移入 RTL, 改用 k-induction。
- Ubuntu 的 z3 4.8.12 在歸納步驟當掉 → z3 5.3.1 (pip) 數秒完成。
- SymbiYosys 的 git main 版與 Yosys 0.33 的 witness 格式不相容 → 使用 `yosys-0.33` tag。
- 函式內讀取模組訊號: Yosys 不正確處理 (npu_core 第一版的距離函式) → 一律以引數傳入。
- 索引運算式 `q[QW'(int'(rd) + s)]` 經 sv2v 後是有號窄值, `q[rd + s]` 則被乘法的 32 位元語境放寬而不環繞 → 索引先存入 QW 位元變數。
- npu_core 第一版的 id 模型假設依序完成 (完成實際上不依序) → 改為 host 契約「id 不與未完成者重複」+ 兩兩相異不變式。
- 公平性性質的歸納需要輔助不變式 (已略過次數 + 與上次得主的循環距離 ≤ NCU)。
- 512 bit 資料讓 xbar 證明數分鐘不收斂 → 新增 LDW 參數以 8 bit 證明。
- 形式 mutation 腳本原本把「不是 PASS」都算抓到 (逾時也算) → 只有 FAIL / UNKNOWN 算。
- v6.3.12 npu_core: CU 可持有 3 個命令後, 未完成上限變成 QD + 4·NCU + 1; NCU = 3 時需要 5 bit id 空間, 計數與兩兩相異
  (pigeonhole 型) 的歸納 z3 40 分鐘、ABC PDR frame 4 需 534 s 都不收斂 → 證明改用 NCU = 2 (上限 13, 4 bit id 空間;
  id 寬度由參數推得); 函式 `f_ok` 原本讀取模組訊號 `f_os` (重犯上面那一條) → 改為引數傳入。

## 架構

```
host 命令 ──► 命令佇列 (QD=4) ──► 分派 (優先閒置 CU, 其次可重疊的 CU) ──► CU 0 .. NCU-1 ──► 完成佇列 ──► host
                                                                    │ 讀 A/B、寫 C
host L2 埠 ─────────────────────────────────────────────► npu_l2_xbar (NB bank, 每 bank 1R1W, round-robin)
```

| 模組 | 檔案 | 說明 |
|---|---|---|
| npu_mma | `rtl/npu/npu_mma.sv` | N×N output-stationary systolic array, INT8 (A / B 各自可選有號 / 無號) × INT8 → INT32; tile 管線化 (v6.3.12) |
| npu_gemm_cu | `rtl/npu/npu_gemm_cu.sv` | 一個命令 = 從 L2 讀 A / B → npu_mma → 寫回 C; 2 項 line buffer 預取; 命令重疊, 最多持有 3 個命令 (v6.3.12) |
| npu_l2_xbar | `rtl/npu/npu_l2_xbar.sv` | NB bank (line 交錯) 的共用 L2 + crossbar, 讀 / 寫各自 round-robin |
| npu_core | `rtl/npu/npu_core.sv` | 命令佇列、分派、完成佇列 (輸出暫存)、效能計數器 (可寫入)、CU 時脈閘控 |
| 產品參數 | `rtl/npu/npu_products.sv`, `dv/scripts/npu_cfg.sh` | 三款組態 |

### 命令與資料版面 (64 B line, SPL = 64/N 個 k 步 / line)

| 資料 | 位址 |
|---|---|
| A[N][K] | `a_line*64 + k*N + i` (第 k 欄連續 N byte) |
| B[K][N] | `b_line*64 + k*N + j` (第 k 列連續 N byte) |
| C[N][N] | `c_line*64 + (i*N + j)*4`, INT32 列優先; 溢位時模 2^32 環繞 (不飽和) |

K = 1..65535。完成順序不保證 (以 id 對應); host 不得在命令執行期間存取其 A / B / C 區域。

### GEMM 累加 (v6.3.14, `ext[0]` / `npu_core.cmd_acc`)

C += A × B: 寫回器在寫每個 C line 之前先讀出 L2 中舊的 line, 逐 32 bit 相加 (模 2^32, 與累加器相同) 後寫回。
用途: **K 切塊** (K 超過 L2 能放的 A / B, 或 MX 區塊縮放前的部分和: 同一個 C 分多個命令累加)、殘差相加 (C 先放殘差)。
- 讀舊 C 走 CU 的讀取埠 0 (與取線器共用; N = 32 時是 A 埠): 寫回器優先, 但取線器已發出、尚未拿到 grant 的請求先完成
  (L2 介面的保持規則, 兩邊的請求都不會被撤回)。每個 C line 多一次讀取: N = 8 的 C 佔 4 個 line, 一個 K = 512 的命令
  讀 128 個 A / B line, 增加約 3%。
- 命令重疊照舊 (讀舊 C 與下一個命令的取線可同時進行; 只是共用埠 0)。軟體規則不變: 同一個 C 的累加命令與前一個命令
  重疊時須設 `wt` (C 區不可與在途命令的區域重疊)。
- 驗證: `npu_gemm_cu_tb` / `npu_core_tb` / `npu_ss_tb` 約 1/3 ~ 1/2 的 GEMM 為累加 (舊 C = 隨機值); SVA: 寫回器讀的
  是正要寫的 line、每個累加寫回之前剛讀過同一個 line、非累加不讀、取線器未完成的請求不被搶走; mutation 7 個
  (不加舊值、旗標取錯命令、搶走請求、不讀舊值、把取線器的 grant 當自己的、A / B 切換錯、讀錯位址)。

### npu_mma 資料流

第 i 列的 A 經 i 級 skew 延遲線從左邊進入, 第 j 欄的 B 經 j 級延遲線從上方進入, 兩者在
t = k + i + j 同時抵達 PE[i][j]。因為同時抵達, 每個 PE 只需要一個 valid token (SVA `a_tok_diag` 檢查此前提)。
空拍不產生 token。流程: start → K 次輸入握手 → 排空 2N−1 拍 → 逐列輸出 (out_valid / out_ready)。

tile 管線化 (v6.3.12): 輸入端 (start / 輸入握手) 與輸出端 (排空 / 輸出) 是兩個 FSM。token 帶旗標
{first, last, a 有號, b 有號} 經過 skew 與陣列: first 讓 PE 以本步乘積取代累加值 (不需要全陣列清除),
last 讓 PE 把累加結果存入結果暫存 `res` (輸出從 `res` 讀, 不受下一個 tile 影響)。最後一步餵入後輸入端立刻閒置
(`in_idle`), 下一個 tile 可在前一個排空 / 輸出時開始; 唯一的限制是下一個 tile 的最後一步要等前一個輸出完
(否則 `res` 會被覆蓋) — 此時 `in_ready` 對 `in_last` 拉低。SVA: 同一步 k 的旗標在陣列中一致 (`a_fl_diag`)、
輸出期間陣列內沒有屬於本 tile 的 last token (`a_drained`)、輸出期間 `res` 不變 (`a_res_hold`)。

### 命令重疊 (npu_gemm_cu / npu_core, v6.3.12)

CU 的輸入端 (取線器 + 餵入器) 與寫回器分開: 最後一個 k 步餵入後, CU 立刻可接受下一個命令並開始預取, 寫回器同時把
前一個 tile 的 C 寫回。命令的 {C line, tag} 依序經過三段: 輸入端 → 等待輸出 → 寫回器, 所以一個 CU 最多持有 3 個命令,
依受理順序完成。命令的 id 當作 tag 隨命令進入 CU, 完成時以 `done` (保持到 `done_ready`) + `done_tag` 帶回;
npu_core 的待回報槽空時才接受 (`done_ready = !cpl_pend`), 所以兩個完成不會互相覆蓋。分派優先給完全閒置的 CU,
沒有時才給輸入端閒置、前一個命令仍在輸出的 CU (否則所有命令都會重疊到 CU 0)。npu_core 的外部介面不變。

### 預取 (npu_gemm_cu)

取線器與餵入器解耦: 2 項 {A line, B line} 緩衝, 有空槽就先讀 A 再讀 B, 餵入器每拍從緩衝頭取一個 k 步。
每 SPL 步只需 2 次讀取, 只要 L2 grant 足夠, npu_mma 每拍都有輸入 (前一版不預取時約 SPL/(SPL+2))。

## 低功耗

| 層級 | 機制 |
|---|---|
| CU | 每個 CU 一個 ICG: 閒置且沒有分派 / done 脈衝時整個 CU 停止時脈 (SVA `a_gate_safe`: 關閉時必在 idle、無 L2 請求) |
| npu_mma 列 | 每列一個 ICG: 本列左邊緣有 token 進入或本列仍有 token 才開 (`a_gate_safe` 檢查關閉時本列與上一列都沒有 token) |
| 資料路徑 | operand isolation: skew 延遲線、PE 暫存、line buffer 只在有效拍更新 |
| L2 | bank 陣列只在寫入 grant 時寫; 實體 SRAM 的 CE 取自該 bank 的讀 / 寫 grant |
| L2 light sleep (v6.3.11) | 每個 bank 連續 `LS_IDLE` (預設 64) 拍沒有讀 / 寫請求 → SRAM light sleep (`bank_ls`: 保留內容、不可存取); 有請求指向睡眠中的 bank 時喚醒, `LS_WAKE` (預設 2) 拍內不給 grant (主端的 req 保持到 gnt)。`LS_IDLE = 0` 不睡 |

L2 light sleep 量測 (`npu_core_tb` 的 `NPUCORE_LS` / `PERF_LS`: 睡眠或喚醒中的 bank-拍 / 全部 bank-拍):

| 組態 | 隨機命令 (4 seed) | 長 GEMM (K = 4000) 期間 | K = 4000 利用率 (不睡 → 睡) |
|---|---|---|---|
| X1-Nano | 15.1–52.5% | 0.0% | 99.7% (不變) |
| X1 | 15.1–34.0% | 0.0% | 99.1% → 99.2% |
| X1-Pro | 10.8–17.8% | 0.0% | 98.2% → 98.1% |

`LS_IDLE` 起初取 32: Pro (16 bank) 的長 GEMM 中 bank 偶有 32 拍沒被存取, 喚醒延遲讓利用率掉到 92.9% (效能門檻 95%
失敗); 64 與 256 都是 98.0–98.1%, 取 64。只量 bank 處於睡眠的比例, 尚未合成, 不換算功耗。

功能不依賴 clock gating: TB 隨機設定 `test_en` (強制開啟所有 ICG), 兩種情況的結果都必須與參考模型一致。
SoC 的 AI 電源域 (時脈閘控、隔離、電源開關、retention、斷電模擬) 見 `docs/low_power.md`。

## 驗證

```
bash dv/scripts/run_npu_cov.sh <nano|x1|pro|all> [seeds] [count]     # coverage 收斂 (含效能門檻)
bash dv/scripts/run_npu_mutation.sh <nano|x1|pro|all>                # 檢查器 mutation 測試
```

| 元件 | 檔案 | 內容 |
|---|---|---|
| npu_mma TB | `dv/npu/npu_mma_tb.sv` | 隨機 K (1..80)、有號性、資料樣式 (隨機 / 邊界值 / 稀疏 / 全 0xFF / 單位矩陣)、空拍與回壓、忙碌中的 start、垃圾 in_valid、隨機 test_en、中途 reset; 整數 GEMM 參考模型 |
| npu_gemm_cu TB | `dv/npu/npu_gemm_cu_tb.sv` | L2 行為模型 + 隨機 grant 延遲; K 含 32768..65535 與 65535 (累加器環繞); 隨機 line 位址 (含高位元); 檢查 C 以外沒有被寫 |
| npu_core TB | `dv/npu/npu_core_tb.sv` | producer / consumer 並行: host 寫 A / B、讀回 C 與 CU 的 L2 存取競爭; burst 送命令 (佇列滿); 長時間不收完成 (全部 CU 同時待回報); 隨機 id; abort burst (佇列滿時 reset); 效能計數器寫入 (含接近 2^48); +long (K ≥ 32768, 位址超出容量折返, 後門參考值); +perf (利用率門檻); directed round-robin 折返 |
| SVA | `dv/npu/npu_*_sva.sv` (bind) | 握手協定與保持、token 對齊 / 排空足夠、預取緩衝帳 (`a_line_acct`)、xbar 每 bank 每拍最多一讀一寫 / work-conserving / 有界等待、分派到編號最小的空閒 CU、完成輸出保持、時脈閘控安全 |
| 報告 | `dv/scripts/cov_report.py --by-module` | 模組層級合併 (同一模組在不同 TB / instance 的同一點合併, generate 範圍仍分開; covergroup bin 原樣保留) |
| 豁免 | `dv/npu/waivers*.txt` | 共用 0 項; 組態專屬 (參數造成的結構常數): Nano 3 項、X1 0 項、Pro 1 項 |

### 結果 (每款 4 seed × 200 命令 + 2 次隨機上電 + perf + long; v6.3.12 命令重疊後重新收斂)

| 組態 | line | branch | expr | toggle | covergroup | SVA cover | 豁免 | mutation |
|---|---|---|---|---|---|---|---|---|
| X1-Nano | 100% (49) | 100% (106) | 100% (228) | 100% (32 690) | 100% (142) | 100% (45) | 3 [C] | 51 / 51 |
| X1 | 100% (52) | 100% (108) | 100% (328) | 100% (67 894) | 100% (155) | 100% (47) | 0 | 53 / 53 |
| X1-Pro | 100% (51) | 100% (106) | 100% (650) | 100% (163 308) | 100% (175) | 100% (47) | 1 [C] (3 個 toggle 點) | 52 / 52 |

mutant 數因組態而異: NCU=1 時「多 CU 同拍計數」與「不優先閒置 CU」不適用 (等價), N=16 時「C 列放錯位置」不適用 (每列一個 line)。
v6.3.12 新增 tile 管線化 (`res_no_last`、`no_last_stall`、`fl_no_skew`、`first_stuck`)、命令重疊 (`cu_pend_tag`、`cu_take_early`、
`cu_done_noack`、`cu_lr`、`cu_take_cmd`、`cu_busy_pend`) 與 npu_core 握手 (`core_dack`、`core_idle_1st`) 的 mutant;
`core_icg_rst` 是等價 mutant 已移除 (理由見腳本)。

L2 SRAM 陣列 (`u_l2.mem`) 超過 toggle 寬度上限, 不量 toggle (儲存陣列, 以資料比對驗證), 報告中明確列出。

### 驗證過程中找到並修正的問題

| 問題 | 發現方式 |
|---|---|
| npu_mma 列時脈致能只看「進入」本列的 token, 最後一個 token 停在本列時時鐘就關了, 下一列重複累加 | 送測前自我審查 (mutation `gate_no_own` 確認 TB 抓得到) |
| A / B 各一個 token 是多餘的 (兩者恆等) | mutation 存活 → 合併, 少 N² 個 flop |
| 完成輸出以組合邏輯 round-robin 選擇: 等待 cpl_ready 期間另一個 CU 完成可能讓 cpl_id 改變 | 寫 SVA `a_cpl_hold` 時 → 改為輸出暫存 |
| **CU 的 done 脈衝當拍仍被視為空閒: 同拍分派新命令, 其 id 蓋掉尚未回報的完成 (舊命令永遠不回報)** | burst 測試 (佇列有等待命令時才會發生) → `a_no_disp_done` |
| round-robin 遮罩、狀態編碼中的恆定位元 | coverage → 改寫 RTL (不豁免) |
| mutation 測試的漏洞: xbar 讀 / 寫改成固定優先權時, 16 bank 的 Pro 很少發生 bank 衝突而存活 | 加入 host bank hammer 與 directed 寫入公平性測試 |
| v6.3.12: CU SVA `a_no_rd_wr` (CU 不會同拍讀與寫) 是舊版「命令完全串行」的性質; 命令重疊後讀下一個 tile 與寫回前一個本來就同拍 → 改為 cover `c_rd_wr` | 新 RTL 第一次模擬 |
| v6.3.12: 效能量測的 consumer 每個完成都以 host 埠讀回 C, 且隨機長時間不收完成 → 連續命令量測被 TB 自己卡住 (Pro K=64 只量到 25%) → 量測時只記錄 id, 結束後才比對 | 效能量測 |

### 驗證環境本身的問題 (記錄供參考)

- Verilator 5.052: 含 `fork ... disable fork` 的 process 不受 `$urandom(seed)` 播種影響 (不同 seed 結果相同) → 改用 `process::self().srandom()`。
- 以 `// Verilator:` 開頭的註解會被當成 pragma (`BADVLTPRAGMA`)。
- SVA 若以閘控時脈取樣, 非同步 reset 落在兩個取樣沿之間時 `$past` 會跨過 reset → 加入 `rst_seen` 停用。
- `initial` 內帶初值的區域變數是 static, 在讀取 plusargs 之前就求值。
- mutant 讓完成 id 錯誤時 TB 會永遠等待: 加入 watchdog (以命令送出 / 完成為進度)。
- v6.3.12 再犯: CU TB 主迴圈內 `int k = pick_k(t);` 是 static, 只求值一次 (所有命令 K 相同, 重疊從未發生) → 宣告與賦值分開;
  TB 另要求「重疊 / 同時持有 3 個命令」的次數 > 0 才算 PASS。
- Verilator 的 expression coverage 不計 `always_ff` 內 `for` 迴圈 (迴圈變數索引) 的運算式 (恆為 0 次) → 改寫成 generate。
- Verilator 中 rst_n 從 0 開始時沒有 negedge 事件: 非同步 reset 只在 reset 期間有時脈沿時生效, 所以「ICG 在 reset 期間
  不開」的 mutant 是否被抓到取決於隨機上電狀態 (硬體上非同步 reset 不需要時脈, 是等價 mutant) → CU 的 `core_icg_rst` 移除。
- Verilator 會展開常數上界的迴圈, 並把 timing 程序呼叫的函式在每個呼叫點內嵌: NPU SS TB 新加的 `arb_lock` (2 × 2 × 6 次的
  常數迴圈) 讓參考模型 `ref_exec` 被內嵌上百份, X1 主 initial 的 coroutine 單一 C++ 檔 85 MB、編譯超過 30 分鐘 → 迴圈上界改用
  執行期變數 (含 `ref_exec` 內的 N×N 迴圈)、`fill_all` 只留一個呼叫點 → 6.6 MB (`no_inline_task` 不支援引用模組變數的函式)。
- Verilator 預設把產生的 C++ 併成「建置 job 數」個編譯單元: Max 的 538 MB C++ 在 JOBS=1 時變成單一編譯單元, 本機與
  CI runner 的編譯器都被 OOM 終止 → 一律 `--output-groups 64` (-j 4 峰值約 9.4 GB, 約 20 分鐘)。
