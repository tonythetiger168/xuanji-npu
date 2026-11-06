# Xuanji NPU paged KV cache 管理 (`rtl/npu/npu_kv_pager.sv`)

LLM 推論的 KV cache 頁管理, 模型與 vLLM PagedAttention 相同: KV 以「區塊」(BS 個 token) 為單位存放在實體頁,
每個序列有一張區塊表 (slot → 頁)。issue #2 P0 的第一項; 取代舊的 `rtl/legacy/ai/cloud/npu_kv_cache.sv` 的頁管理。

**狀態**: 單元已驗證 (CRV + scoreboard + 後門不變式、SVA、coverage 100%、mutation 26 / 26), 已加入 CI 與 `make lint`。
**尚未**: 接進 NPU Subsystem (命令 / DMA 位址轉換)、KV 量化、attention 資料流 (見「後續」)。
尚未合成, 不宣稱頻率 / 面積 / 功耗。

## 舊 RTL 的問題 (`dv/npu_kv/legacy_kv_repro_tb.sv`, 依介面語意的 directed 測試)

| # | 問題 | 重現 |
|---|---|---|
| 1 | 配發從 `free_ptr` 往後連續取, 不看 `page_used`、不回繞也不回收 | 16 頁: A = 8 頁、B = 4 頁、釋放 B、再配 8 頁 → 得到 [12, 20): 超出 16 頁並與仍在使用的 A 重疊 |
| 2 | `alloc_num_pages` 與頁索引同寬 | 一次配 16 頁 (4 bit) 被截成 0 |
| 3 | 沒有 double-free 檢查, 已用計數下溢 | 釋放從未配發的頁兩次 → `free_pages` = 18 > 16 |
| 4 | 頁內索引用 `kv_offset[5:0]` (0..63), 但一頁只有 BLOCK_SIZE = 16 列 | offset 20 寫入覆蓋了 offset 4 |
| 5 | 區塊表 slot 用 `kv_offset[15:6]` (64 token / slot), 與 BLOCK_SIZE = 16 不一致 | token 16 查 slot 0, 查不到 slot 1 的頁 |

另外: 區塊表宣告為 65536 × 64 (無法實作)、多頁配發後只能逐頁釋放。`dv/scripts/run_npu_kv.sh` 每次都會執行這個 TB
並印出結果 (預期 FAIL; 舊 RTL 保留在 `rtl/legacy` 不修改)。

## 介面

參數: NP 頁 / NSEQ 序列 / NSLOT 每序列區塊數 / BS 每區塊 token 數 (都是 ≥ 2 的 2 的冪)、RW 參考計數寬度、POSW token 位置寬度。

| 命令 (`cmd_op`) | 運算元 | 動作 | `rsp_data` |
|---|---|---|---|
| `KV_ALLOC` | seq, slot | 配一個空閒頁, 參考計數 = 1 | 頁 |
| `KV_FREE` | seq, slot | 解除映射, 參考計數 −1, 歸零時頁回到空閒表 | 頁 |
| `KV_FREE_SEQ` | seq | 逐 slot 解除整個序列 (NSLOT 拍, `busy`) | 解除的 slot 數 |
| `KV_SHARE` | seq, slot, src_seq | (seq, slot) 指向 (src_seq, slot) 的同一頁, 參考計數 +1 (prefix 共享 / beam fork) | 頁 |

狀態 `rsp_st`: `OK` / `NOMEM` (沒有空閒頁) / `MAPPED` (目標已有映射) / `UNMAPPED` (要釋放或共享來源沒有映射) /
`REFMAX` (參考計數已達 2^RW − 1)。錯誤的命令不改變任何狀態, `rsp_data` = 0。
釋放只經由區塊表, 所以**不會 double free** (未映射的 slot 回 `UNMAPPED`)。

位址轉換 (`tr_*`): 每拍一個、延遲 1 拍、不回壓; 看到的是本拍控制命令執行之前的區塊表。
slot = pos / BS、offset = pos % BS、KV line = 頁 × BS + offset; pos ≥ NSLOT × BS → `tr_oor`; 未命中時 `tr_line` = 0。

空閒表: 從未配發的頁由水位 `hw` 依序配發, 釋放的頁進 FIFO, 配發優先用 FIFO; 重置後不需要初始化週期。
儲存: 區塊表有效位元是 flop (需重置), 頁號 / 參考計數 / FIFO 只在有效時讀, 可對應 SRAM。
低功耗: 沒有命令 / 回應 / 轉換時 ICG 關閉時脈。

## 驗證 (`dv/scripts/run_npu_kv.sh`, `run_npu_kv_mutation.sh`)

| 項目 | 內容 |
|---|---|
| 參考模型 | TB 內: 區塊表、參考計數、水位 + FIFO (配發順序由規格決定, 逐位元比對回應與轉換結果) |
| 不變式 (每拍, 後門) | 每個頁恰好在「空閒 (水位以上或 FIFO 中)」或「被映射」其一; 參考計數 = 指向它的映射數; `free_cnt` 與模型相同 → 無重疊配發、無遺失、不超界 |
| 情境 | 均衡 / 配發壓力 (耗盡) / 共享鏈 (到上限) / FREE_SEQ 密集 / 大量釋放 (FIFO 滿); 回應回壓三種; 轉換 pos 偏向範圍末端、剛好超出、最大值; 隨機 test_en; FREE_SEQ 中與回應等待中 reset; 隨機上電狀態 |
| 組態 | 小 (NP 16 / NSEQ 4 / NSLOT 8 / BS 4 / RW 2) 4 seed + 隨機上電 2 次; 預設 (NP 64 / NSEQ 8 / NSLOT 16 / BS 16 / RW 3) 2 seed; 每個 40000 拍 |
| SVA | 水位 / FIFO 帳、配發只在有空閒頁、推入時 FIFO 不滿、每拍最多一個區塊表操作、FREE_SEQ 期間不受理 / 沒有回應、回應等待時保持、錯誤回應資料為 0、hit / oor 互斥、轉換延遲 1 拍、閘控安全、free_cnt 增減 |
| coverage | line / branch / expr / toggle 100% (無豁免); covergroup 23 / 23 (命令 × 狀態全部合法組合、FREE_SEQ 解除 0 / 部分 / 全部、轉換 miss / hit / oor / 邊界、空閒 0 / 全部、參考計數 1 / 共享 / 上限); SVA cover 9 / 9 |
| mutation | 26 / 26: 各狀態檢查拿掉 (6)、回收條件 / 不回收 / 不用 FIFO / FIFO 指標 / 水位 / free_cnt (6)、參考計數 (2)、FREE_SEQ 少一個 slot / 計數 / 用錯 seq (3)、ready (2)、錯誤回應資料、轉換範圍 / slot 位元 / offset / 未命中位址 (4)、ICG 致能 (2) |

驗證過程中: TB 第一版在 `initial` 內以 `rsp_t r = mcmd(...)` 宣告並賦值 — static 變數只求值一次, 模型從未更新
(與 `docs/npu.md` 記錄過的同一類錯誤); 命令保持邏輯原本以 `cmd_ready` 判斷, FREE_SEQ 期間一直保持同一個命令,
mutation `sweep_seq` (FREE_SEQ 用到後來的 seq 輸入) 與 `icg_sweep` 因此存活 → 改為以「上一拍是否被接受」判斷。

## 後續

- 接進 NPU Subsystem: 命令佇列與 DMA 的位址轉換 (DMA 以 (seq, pos) 存取 KV, 由 `tr_*` 換成 line)。
- 區塊表放 L2 (或可由 DMA 間接讀取), 讓 gather 模式一次搬多頁, 不需要 CPU 每頁下一個命令。
- 每頁的量化中繼資料 (scale / zero), 預留 INT4 / FP8 KV (K 依 channel、V 依 token 分組; issue #2 / `docs/npu_plan.md` 階段 3)。
- 環形 / sliding window 回收, 以及 MLA 這類壓縮後共用的單一 latent KV。
- Attention: 新 NPU 上以 GEMM (QKᵀ、PV) + 向量核 softmax 組成, 對照 numpy 參考做端到端測試 (遮罩、GQA、prefill / decode)。
  舊的 `rtl/legacy/ai/attention/npu_attn_engine.sv` 標示 BF16 但以 INT16 整數乘法計算 BF16 位元, 是行為佔位, 不修改。
