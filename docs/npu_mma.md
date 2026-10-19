# NPU 運算核心: npu_mma (INT8 GEMM systolic tile) 與 npu_gemm_cu

`rtl/ai/pe/npu_mma.sv` — N×N output-stationary systolic array, `C[N][N] (INT32) = A[N][K] × B[K][N]`,
A / B 各自可選有號或無號 INT8。預設 N = 8 (64 MAC/拍)。

## 為何重寫

原本的 `npu_pe` / `npu_systolic` 無法算出 GEMM:

- weight-stationary 模式下權重暫存器從未載入, 乘積恆為 0
- activation 與權重沒有 skew, 部分和輸入被忽略
- BF16 / FP16 / FP32 是佔位實作, FP4 case 重複, `partial_sum_in[63:32]` 超出範圍

`npu_mma` 取代其運算核心, `npu_gemm_cu` 把它包成「從 L2 讀 A / B、寫回 C」的命令單元。
**兩者都尚未接入 AI tile**: tile 的 cluster 輸入仍接 0, TDB 寫入與 DMA descriptor 仍接常數,
GSCU 命令沒有資料路徑。接入 (以 npu_gemm_cu 取代 npu_cu, L2 埠與 attention engine 仲裁) 是下一步。

## 介面與流程

| 階段 | 拍數 | 說明 |
|---|---|---|
| start | 1 | `busy=0` 時有效; 清除累加器, 鎖存 `a_signed` / `b_signed` |
| 輸入 | K 次握手 | `in_valid` / `in_ready`; 每次給 A 的一欄 `a_vec[i]=A[i][k]` 與 B 的一列 `b_vec[j]=B[k][j]`, 最後一次 `in_last=1`; 可插空拍 |
| 排空 | 2N−1 | 最後一個 token 抵達 PE[N−1][N−1] |
| 輸出 | N 次握手 | `out_valid` / `out_ready`, 每次一列 (`out_row`, `out_vec[j]=C[row][j]`) |

`busy` 期間的 `start` 會被忽略。每個 tile 的固定開銷約 3N 拍, K 步的利用率約為 K / (K + 3N)。

## 資料流

- 第 i 列的 A 經 i 級 skew 延遲線從左邊進入, 第 j 欄的 B 經 j 級延遲線從上方進入, 兩者在
  t = k + i + j 同時抵達 PE[i][j]。
- 因為同時抵達, 每個 PE 只需要 **一個** valid token (`tok_q`)。它同時標記 `a_q` / `b_q` 有效,
  SVA `a_tok_diag` 檢查這個前提 (PE[r][c−1] 與 PE[r−1][c] 的 token 永遠一致)。
- 空拍不產生 token, 不會被累加。

## 低功耗

- **列級 clock gating**: 每列一個 ICG (`rtl/common/clock_gate.sv`)。致能 = 左邊緣有 token 進入、本列仍有
  token, 或 `start` 清除累加器。上方傳入的 token 必與本列 token 相同, 不必列入。
  `a_gate_safe` 檢查時鐘關閉時該列與上一列都沒有 token。
- **operand isolation**: skew 延遲線與 PE 的資料暫存只在有 token 時更新, 空拍不翻轉資料位元。
- 功能不依賴 clock gating: TB 每個 tile 隨機設定 `test_en` (強制開啟 ICG), 兩種情況結果都必須與參考模型一致。

## npu_gemm_cu (`rtl/ai/cluster/npu_gemm_cu.sv`)

命令: A / B / C 的 64 B line 位址、K、A / B 有號性; 完成時 `done` 一拍脈衝。

| 資料 | L2 版面 (N = 8) |
|---|---|
| A | `a_line*64 + k*N + i = A[i][k]` (第 k 欄連續 N byte, 每個 line 8 個 k 步) |
| B | `b_line*64 + k*N + j = B[k][j]` |
| C | `c_line*64 + (i*N + j)*4 = C[i][j]` (INT32 列優先, 每個 line 2 列; 溢位時模 2^32 環繞) |

L2 介面與 `npu_l2_sram` 相同 (組合式讀取), 另加 `rd_gnt` / `wr_gnt` 以便與 attention engine / DMA 仲裁。
每 8 個 k 步讀 2 個 line (A / B), 不重疊, 因此陣列利用率約 8/10;
要做到接近 100%, 需要 line buffer 雙緩衝 (下一版)。

## 驗證

`bash dv/scripts/run_npu_cov.sh [seeds] [tiles]` 與 `bash dv/scripts/run_npu_mutation.sh` (CI job `npu-coverage`)。
兩個 TB 分別量測; npu_mma 在 CU 內部的 cover 點不計入 (CU 不會產生「忙碌中 start」等情境, 其斷言在 CU 內仍啟用)。

| 元件 | 檔案 |
|---|---|
| CRV TB + 參考模型 | `dv/npu/npu_mma_tb.sv`: 隨機 K (1..80)、有號 / 無號組合、資料樣式 (隨機 / 邊界值 / 稀疏 / 全 0xFF / 單位矩陣)、隨機空拍與回壓、忙碌中的 start 與未就緒時的 in_valid (給垃圾資料)、隨機 `test_en`、運算中途 reset; 以整數 GEMM 逐列比對 |
| SVA / cover / covergroup | `dv/npu/npu_mma_sva.sv` (bind): 握手協定、token 對齊、排空足夠、輸出期間累加器不變、閘控安全; K × 有號性、運算元邊界值 × 有號性、結果符號 / 量級 |
| CU TB | `dv/npu/npu_gemm_cu_tb.sv`: L2 行為模型 + 隨機 grant 延遲; 隨機 K (含 >= 32768 的長命令與累加器溢位)、隨機 line 位址 (含高位元); 比對寫回的 C, 並檢查 C 區以外沒有被寫 |
| CU SVA | `dv/npu/npu_gemm_cu_sva.sv`: 等待 grant 時請求保持、讀寫互斥、done 語意、餵資料時 npu_mma 必就緒; K 分類與 grant 等待 covergroup |

結果 (每個 TB 4 seed, 無豁免):

| 類型 | 覆蓋率 |
|---|---|
| line / branch / expr / toggle | 100% / 100% / 100% / 100% |
| covergroup / SVA cover | 100% / 100% |
| mutation | 20 / 20 killed (npu_mma 13, npu_gemm_cu 7) |

開發過程中修正的設計問題:

- 第一版的列時脈致能只看「進入」本列的 token。最後一個 token 停在本列暫存器時時鐘就關了, token 卡住,
  讓下一列持續重複累加 (自我審查時發現, 送測前已修正; mutation `gate_no_own` 確認 TB 抓得到這類錯誤)。
- A / B 各一個 token 是多餘的 (mutation 測試中「只看 A token」的 mutant 存活, 證明兩者恆等),
  因此合併成一個, 少 N² = 64 個 flop。
