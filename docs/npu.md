# Xuanji-X1 NPU: 三款產品組態與驗證

同一份 RTL (`rtl/ai/npu_core.sv`) 以參數產生三款產品。每款都有獨立的驗證收斂
(CRV + SVA + coverage 100% + mutation + 效能量測), CI job `npu (nano / x1 / pro)`。

## 產品組態

| 產品 | N×N 陣列 | 計算單元 (CU) | L2 bank × 容量 | 峰值 MAC/拍 | 峰值 INT8 @ 1 GHz* | 量測 MAC 利用率 (K=4000) | 對標 (市場層級, 見 STATUS_AND_ROADMAP §4b) |
|---|---|---|---|---|---|---|---|
| **Xuanji-X1-Nano** | 4×4 | 1 | 2 × 32 KiB = 64 KiB | 16 | 32 GOPS | 99.7% (15.9 MAC/拍) | MCU / IoT 的 AI 加速 (關鍵字偵測、感測器融合) |
| **Xuanji-X1** | 8×8 | 4 | 4 × 64 KiB = 256 KiB | 256 | 512 GOPS | 99.1% (253.8 MAC/拍) | 邊緣視覺 / 小型模型推論 |
| **Xuanji-X1-Pro** | 16×16 | 8 | 16 × 64 KiB = 1 MiB | 2048 | 4.1 TOPS | 98.2% (2010 MAC/拍) | 邊緣 AI / 機器人 (往 Ascend 310 級 16 TOPS 的路上) |

\* 時脈是假設值: **尚未做合成、時序收斂與面積 / 功耗評估**。TOPS = MAC/拍 × 2 × 時脈 × 利用率 (只計 INT8 GEMM)。
利用率是 RTL 模擬量測 (`npu_core_tb +perf`: 每個 CU 一個 K=4000 的命令同時執行), 不是推論。
CI 以 95% 作為效能回歸門檻 (`dv/scripts/npu_cfg.sh`)。

短 K 時固定開銷 (start / 排空 2N−1 拍 / 輸出 N 列) 佔比上升; 例如 X1 每個命令約 3N + 4 拍的開銷,
K = 64 時利用率約 K / (K + 28) ≈ 70%。實際模型需要編譯器把 GEMM 切成大 K 的 tile。

**尚未具備** (列為下一步, 不對外宣稱): BF16 / FP8、稀疏、卷積 / attention 專用資料流、DMA 與 CPU 介面
(目前 host 介面是 TB 驅動的命令 / L2 埠)、AI tile 整合、編譯器 / runtime。

## 架構

```
host 命令 ──► 命令佇列 (QD=4) ──► 分派 (編號最小的空閒 CU) ──► CU 0 .. NCU-1 ──► 完成佇列 ──► host
                                                                    │ 讀 A/B、寫 C
host L2 埠 ─────────────────────────────────────────────► npu_l2_xbar (NB bank, 每 bank 1R1W, round-robin)
```

| 模組 | 檔案 | 說明 |
|---|---|---|
| npu_mma | `rtl/ai/pe/npu_mma.sv` | N×N output-stationary systolic array, INT8 (A / B 各自可選有號 / 無號) × INT8 → INT32 |
| npu_gemm_cu | `rtl/ai/cluster/npu_gemm_cu.sv` | 一個命令 = 從 L2 讀 A / B → npu_mma → 寫回 C; 2 項 line buffer 預取 |
| npu_l2_xbar | `rtl/ai/memory/npu_l2_xbar.sv` | NB bank (line 交錯) 的共用 L2 + crossbar, 讀 / 寫各自 round-robin |
| npu_core | `rtl/ai/npu_core.sv` | 命令佇列、分派、完成佇列 (輸出暫存)、效能計數器 (可寫入)、CU 時脈閘控 |
| 產品參數 | `rtl/ai/npu_products.sv`, `dv/scripts/npu_cfg.sh` | 三款組態 |

### 命令與資料版面 (64 B line, SPL = 64/N 個 k 步 / line)

| 資料 | 位址 |
|---|---|
| A[N][K] | `a_line*64 + k*N + i` (第 k 欄連續 N byte) |
| B[K][N] | `b_line*64 + k*N + j` (第 k 列連續 N byte) |
| C[N][N] | `c_line*64 + (i*N + j)*4`, INT32 列優先; 溢位時模 2^32 環繞 (不飽和) |

K = 1..65535。完成順序不保證 (以 id 對應); host 不得在命令執行期間存取其 A / B / C 區域。

### npu_mma 資料流

第 i 列的 A 經 i 級 skew 延遲線從左邊進入, 第 j 欄的 B 經 j 級延遲線從上方進入, 兩者在
t = k + i + j 同時抵達 PE[i][j]。因為同時抵達, 每個 PE 只需要一個 valid token (SVA `a_tok_diag` 檢查此前提)。
空拍不產生 token。流程: start → K 次輸入握手 → 排空 2N−1 拍 → 逐列輸出 (out_valid / out_ready)。

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

功能不依賴 clock gating: TB 隨機設定 `test_en` (強制開啟所有 ICG), 兩種情況的結果都必須與參考模型一致。
UPF / power domain (電源開關、隔離、retention) 尚未做。

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

### 結果 (每款 4 seed × 200 命令 + perf + long)

| 組態 | line | branch | expr | toggle | covergroup | SVA cover | 豁免 | mutation |
|---|---|---|---|---|---|---|---|---|
| X1-Nano | 100% (34) | 100% (86) | 100% (164) | 100% (25 814) | 100% (133) | 100% (30) | 3 [C] | 見下 |
| X1 | 100% (37) | 100% (88) | 100% (252) | 100% (50 522) | 100% (144) | 100% (32) | 0 | 見下 |
| X1-Pro | 100% (36) | 100% (86) | 100% (510) | 100% (112 726) | 100% (159) | 100% (32) | 1 [C] | 見下 |

L2 SRAM 陣列 (`u_l2.mem`) 超過 toggle 寬度上限, 不量 toggle (儲存陣列, 以資料比對驗證), 報告中明確列出。

### 驗證過程中找到並修正的問題

| 問題 | 發現方式 |
|---|---|
| npu_mma 列時脈致能只看「進入」本列的 token, 最後一個 token 停在本列時時鐘就關了, 下一列重複累加 | 送測前自我審查 (mutation `gate_no_own` 確認 TB 抓得到) |
| A / B 各一個 token 是多餘的 (兩者恆等) | mutation 存活 → 合併, 少 N² 個 flop |
| 完成輸出以組合邏輯 round-robin 選擇: 等待 cpl_ready 期間另一個 CU 完成可能讓 cpl_id 改變 | 寫 SVA `a_cpl_hold` 時 → 改為輸出暫存 |
| **CU 的 done 脈衝當拍仍被視為空閒: 同拍分派新命令, 其 id 蓋掉尚未回報的完成 (舊命令永遠不回報)** | burst 測試 (佇列有等待命令時才會發生) → `a_no_disp_done` |
| round-robin 遮罩、狀態編碼中的恆定位元 | coverage → 改寫 RTL (不豁免) |

### 驗證環境本身的問題 (記錄供參考)

- Verilator 5.052: 含 `fork ... disable fork` 的 process 不受 `$urandom(seed)` 播種影響 (不同 seed 結果相同) → 改用 `process::self().srandom()`。
- 以 `// Verilator:` 開頭的註解會被當成 pragma (`BADVLTPRAGMA`)。
- SVA 若以閘控時脈取樣, 非同步 reset 落在兩個取樣沿之間時 `$past` 會跨過 reset → 加入 `rst_seen` 停用。
- `initial` 內帶初值的區域變數是 static, 在讀取 plusargs 之前就求值。
