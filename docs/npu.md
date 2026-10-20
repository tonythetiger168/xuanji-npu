# Xuanji-X1 NPU Subsystem: 三款產品組態與驗證

檔案: `rtl/npu/` (RTL)、`dv/npu/` (驗證)。舊 AI tile 的模組 (cluster / systolic / attention / HBM3 / NoC 介面卡 / KV cache / MoE router)
已移到 `rtl/legacy/ai/`, **未經驗證**, 不屬於 NPU Subsystem。

同一份 RTL (`rtl/npu/npu_core.sv`) 以參數產生三款產品。每款都有獨立的驗證收斂
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

短 K 時固定開銷 (start、讀第一對 line、排空 2N−1 拍、輸出 N 列) 佔比上升, 陣列越大越明顯。
量測 (`npu_core_tb +perf +perf_k=K`, 每個 CU 一個命令同時執行):

| K | X1-Nano | X1 | X1-Pro |
|---|---|---|---|
| 64 | 82.1% | 64.6% | 46.4% |
| 256 | 94.8% | 88.6% | 77.8% |
| 1024 | 98.7% | 96.8% | 93.4% |
| 4000 | 99.7% | 99.1% | 98.2% |

實際模型需要編譯器把 GEMM 切成大 K 的 tile; 改善小 K 的利用率 (例如前一個命令輸出時預取下一個命令) 是下一步。

**尚未具備** (列為下一步, 不對外宣稱): BF16 / FP8、稀疏、卷積 / attention 專用資料流、CPU 命令介面與 NoC / DRAM 連接
(NPU Subsystem 在 SoC 頂層的命令與外部記憶體埠目前接死)、編譯器 / runtime。

## NPU Subsystem (`orca_npu_ss`)

npu_core 加上命令前端與 DMA, 成為可以直接下命令的子系統 (由舊 AI tile 重建; 原 `orca_v63_ai_tile`)。

```
命令 ──► 命令佇列 (QD=8) ──► 依序分派 ──┬─► npu_core (GEMM, 多個 CU 同時執行)
                                       ├─► npu_ss_dma (外部記憶體 ⇄ L2, 64 B line)
                                       └─► NOP (wt=1 時即 fence)
                    完成合併 (round-robin, 輸出暫存保持) ◄┘──► cpl_valid + id, irq
```

| 命令 | 動作 |
|---|---|
| `GEMM` | C = A × B (INT8 → INT32, 資料已在 L2); K = 0 時只回報完成 |
| `DMA_LD` / `DMA_ST` | 外部記憶體 → L2 / L2 → 外部記憶體, k 個 line (k = 0 時立即完成); 位址模數環繞 |
| `NOP` | 只回報完成 |

- **等待旗標 `wt`**: 先前所有命令完成 (結果已寫入, 完成已移入輸出暫存) 後才分派。沒有 `wt` 的命令可與先前的命令並行
  (例如 GEMM 執行時 DMA 載入下一批資料); 軟體須保證並行命令的 L2 / 外部區域不重疊, 否則設 `wt`。
- 完成順序不保證與命令順序相同。
- **DMA 吞吐量**: 每拍 1 line (64 B); 外部讀取延遲 L 拍時為 min(1, OUT / (L + 3)) line/拍, OUT = 在途上限
  (Nano 16 / X1 32 / Pro 64)。量測 (`npu_ss_tb +perf_dma=L`) 見下表。
- **低功耗**: DMA 閒置時停止時脈 (ICG); npu_core 內每個 CU、每列另有 ICG。

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
| 範圍 | 量測 tile 自身 (`orca_npu_ss`、`npu_ss_dma`、`clock_gate`); npu_core 子樹由上面的 NPU flow 收斂 |

### NPU Subsystem 結果 (每款 6 seed × 30 batch + DMA 吞吐量量測; 豁免 0 項)

| 組態 | DMA OUT | line | branch | expr | toggle | covergroup | SVA cover | mutation |
|---|---|---|---|---|---|---|---|---|
| X1-Nano | 16 | 100% (13) | 100% (48) | 100% (106) | 100% (29 868) | 100% (96) | 100% (24) | 23 / 23 |
| X1 | 32 | 100% (13) | 100% (48) | 100% (106) | 100% (46 260) | 100% (144) | 100% (24) | 23 / 23 |
| X1-Pro | 64 | 100% (13) | 100% (48) | 100% (106) | 100% (79 036) | 100% (240) | 100% (24) | 23 / 23 |

DMA 讀取吞吐量 (LD 2048 line, 外部埠全速; 寫入 ST 在所有情況為 0.998 line/拍):

| 讀取延遲 L | X1-Nano (OUT 16) | X1 (OUT 32) | X1-Pro (OUT 64) |
|---|---|---|---|
| 0 | 0.996 | 0.996 | 0.996 |
| OUT − 3 | 0.990 (L=13) | 0.982 (L=29) | 0.967 (L=61) |
| 2 × OUT | 0.455 (L=32) | 0.474 (L=64) | 0.481 (L=128) |
| 100 | 0.155 | 0.309 | 0.609 |

量測值符合 min(1, OUT / (L + 3)); DRAM 級延遲 (約 100 拍) 下要接近滿速需要更大的 OUT 或多個 DMA 通道。

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

DW = 8 的資料完整性推廣到 512: DMA 不運算資料, 只依位置搬移, 每個 bit 獨立。

形式驗證過程中發現的檢查器問題 (都已修正, 記錄以免重犯):
- sv2v 預設刪除所有 assertion: 證明「通過」但沒有檢查任何性質 → `-E Assert`, 且轉出的設計沒有 assert 時流程失敗。
- 資料完整性原本記錄內部的 `push_d`, 與被檢查的邏輯循環 (mutation `pushd_wrong` 存活) → 改為從輸入埠記錄。
- 只看埠的 wrapper 以 PDR 證明不收斂 (frame 5 需 620 s) → 不變式移入 RTL, 改用 k-induction。
- Ubuntu 的 z3 4.8.12 在歸納步驟當掉 → z3 5.3.1 (pip) 數秒完成。
- SymbiYosys 的 git main 版與 Yosys 0.33 的 witness 格式不相容 → 使用 `yosys-0.33` tag。

## 架構

```
host 命令 ──► 命令佇列 (QD=4) ──► 分派 (編號最小的空閒 CU) ──► CU 0 .. NCU-1 ──► 完成佇列 ──► host
                                                                    │ 讀 A/B、寫 C
host L2 埠 ─────────────────────────────────────────────► npu_l2_xbar (NB bank, 每 bank 1R1W, round-robin)
```

| 模組 | 檔案 | 說明 |
|---|---|---|
| npu_mma | `rtl/npu/npu_mma.sv` | N×N output-stationary systolic array, INT8 (A / B 各自可選有號 / 無號) × INT8 → INT32 |
| npu_gemm_cu | `rtl/npu/npu_gemm_cu.sv` | 一個命令 = 從 L2 讀 A / B → npu_mma → 寫回 C; 2 項 line buffer 預取 |
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
| X1-Nano | 100% (34) | 100% (86) | 100% (164) | 100% (25 814) | 100% (133) | 100% (30) | 3 [C] | 33 / 33 |
| X1 | 100% (37) | 100% (88) | 100% (252) | 100% (50 522) | 100% (143) | 100% (32) | 0 | 34 / 34 |
| X1-Pro | 100% (36) | 100% (86) | 100% (510) | 100% (112 726) | 100% (159) | 100% (32) | 1 [C] | 33 / 33 |

mutant 數因組態而異: NCU=1 時「多 CU 同拍計數」不適用, N=16 時「C 列放錯位置」不適用 (每列一個 line)。

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

### 驗證環境本身的問題 (記錄供參考)

- Verilator 5.052: 含 `fork ... disable fork` 的 process 不受 `$urandom(seed)` 播種影響 (不同 seed 結果相同) → 改用 `process::self().srandom()`。
- 以 `// Verilator:` 開頭的註解會被當成 pragma (`BADVLTPRAGMA`)。
- SVA 若以閘控時脈取樣, 非同步 reset 落在兩個取樣沿之間時 `$past` 會跨過 reset → 加入 `rst_seen` 停用。
- `initial` 內帶初值的區域變數是 static, 在讀取 plusargs 之前就求值。
- mutant 讓完成 id 錯誤時 TB 會永遠等待: 加入 watchdog (以命令送出 / 完成為進度)。
