# Xuanji NPU MoE dispatch (`rtl/npu/npu_moe_disp.sv`)

Mixture-of-Experts 層的 dispatch 表產生器。它把 router 的 top-k 結果 (每個 token K 個 {expert, weight}) 換成「每一項
在 dispatch 緩衝中的位置」與「每個 expert 的數量 / 起點」。資料搬移由 DMA 間接模式 (gather / scatter) 負責,
expert 的計算是一般的 GEMM, 合併 (combine) 是向量核的加權和; 這個單元只算位置, 不碰資料。

**狀態**: 單元已驗證 (CRV + scoreboard、SVA、coverage 100%、mutation), 已加入 CI 與 `make lint`。
**尚未**: NPU Subsystem 命令 `NPU_OP_MOE` (op 6): 從 L2 讀 top-k 結果、把位置寫成間接 DMA 的索引表
(格式見下)。依賴向量核 top-k 指令與 DMA 間接模式 (另一個 PR 系列, `rtl/npu/npu_dma.sv`)。
尚未合成, 不宣稱頻率 / 面積 / 功耗。

舊的 `rtl/legacy/ai/cloud/npu_moe_router.sv` (issue #2 P0 第 4 項: scores / dispatch 依賴即時輸入、沒有鎖存契約、
沒有 top-k 邊界與回壓) 保留在 `rtl/legacy` 不修改; 新 NPU 的 router = GEMM (logits) + 向量核 softmax / top-k。

## MoE 層在新 NPU 上的資料流

```
x (T × d) ──GEMM──► logits (T × E) ──向量核 softmax + top-k──► {expert, weight} × K / token
                                                                 │
                                         npu_moe_disp ◄──────────┘   位置 pos、各 expert {n, off}
                                                 │
            DMA 間接 gather: x[token] → 緩衝[pos]  (每個 expert 的 token 連續)
                                                 │
            每個 expert e: GEMM (n[e] 列) — FFN₁ → 向量核 SiLU / GELU → FFN₂
                                                 │
            DMA 間接 scatter + 向量核: y[token] += weight × 緩衝[pos]
```

## 介面

| 埠 | 說明 |
|---|---|
| `cmd_*` | 命令: `cmd_dropless` 模式、`cmd_cap` 每 expert 容量 (容量模式)、`cmd_nexp` 本命令的 expert 數 (1..NEXP) |
| `in_*` | 項目: `in_exp` (多 2 bit, 可表達超出範圍的 id)、`in_tok`、`in_w` (原樣傳遞, 例如 BF16)、`in_last` 結束一趟 |
| `out_*` | 依輸入順序: `out_pos`、`out_exp`、`out_tok`、`out_w`、`out_drop`、`out_err` |
| `cnt_*` | 輸入結束後, 依 expert 順序 `cmd_nexp` 筆 {expert, n, off} |
| `done` / `ovf` / `busy` | 最後一筆數量被接受後一拍 / 本命令項目數超過 2^PW − 1 / 命令執行中 |

兩種模式:

| 模式 | expert e 的區域 | 第 r 個指向 e 的項目 | 輸入趟數 |
|---|---|---|---|
| 容量 (GShard / Switch) | [e × cap, (e + 1) × cap) | pos = e × cap + r; r ≥ cap → drop (pos = 0) | 1 |
| 無丟棄 (MegaBlocks 式) | [off[e], off[e] + n[e]), off = 數量的前綴和 | pos = off[e] + r | 2 (第一趟只計數) |

expert ≥ `cmd_nexp` 的項目: `err` = `drop` = 1, 不計入任何 expert、不計入項目數上限。容量模式 `cnt_n` = min(數量, cap)。
命令開始時以 `cmd_nexp` 拍清除計數; 輸出暫存一級 (`in_ready` = 輸出暫存可用)。低功耗: 閒置時 ICG 關閉。

## 驗證 (`dv/scripts/run_npu_moe.sh`, `run_npu_moe_mutation.sh`)

| 項目 | 內容 |
|---|---|
| 參考模型 | TB 內依規格算出每一項的 pos / drop / err 與各 expert {n, off}, 逐項比對 |
| 額外檢查 | 非 drop 的 pos 兩兩相異且落在該 expert 的區域內 (dispatch 是單射, 不會有兩個 token 寫到同一列) |
| 情境 | 兩種模式; cap = 0 / 1 / 中間 / 最大; nexp = 1 / 中間 / 全部; 項目數 1 / 少 / 多 / 剛好上限 / 超過上限; expert 分布均勻 / 偏斜 (熱門 expert) / 全部同一個; 5% 超出範圍的 id; 輸入空拍、輸出與數量回壓; 隨機 test_en; 命令中途 reset; 隨機上電狀態 |
| 組態 | 小 (NEXP 8 / PW 9) 4 seed × 300 命令 + 隨機上電 2 次; 大 (NEXP 64 / PW 14) 1 seed × 100 命令 |
| SVA | 命令只在閒置時接受、輸入只在計數 / 輸出趟接受、輸出與數量等待時保持、drop ⇒ pos = 0、err ⇒ drop、無丟棄模式不 drop 有效項目、done 時序、閘控安全 |
| coverage | line / branch / expr / toggle 100% (無豁免); covergroup 28 / 28; SVA cover 6 / 6 |
| mutation | 19 / 19 (見腳本): 範圍檢查、容量比較、兩種模式的位置、前綴和、清除、數量 min / 起點、少一筆、drop 位置、err、輸出暫存、溢位、cap 鎖存、ICG |

驗證過程中: TB 的 `ndrop` / `nbad` 在 `initial` 的迴圈內宣告並給初值 (static, 只初始化一次) → 跨命令累加, 溢位期望錯誤;
同一類錯誤已記錄在 `docs/npu.md` 與 `docs/npu_kv.md`。RTL: 命令開始時清除 `ovf` 的同一拍, 溢位檢查又以上一個命令的
計數設回 1 (後寫的指派優先) → 溢位檢查只在命令執行中。mutation: 讓「計數少一筆」的 mutant 使命令永遠不結束,
TB 沒有 watchdog 而卡住 → 加入每個命令的拍數上限; `ovf_bad` (超出範圍的 id 也計入上限) 存活 → 加入「總數超過上限、
扣掉錯誤 id 後在上限內」的情境。

## 後續

- `NPU_OP_MOE` 命令: 讀 top-k 輸出 (兩個 line: 16 × u32 索引、16 × FP32 值, 未用的槽索引 0xFFFFFFFF),
  寫出每個 expert 的 gather 索引表 (u32 LE, 每 line 16 項, 項目 = token 的外部 line 偏移) 與 combine 用的權重表。
- 端到端: 一個 MoE 層 (router + top-k + dispatch + expert FFN + combine) 對照 numpy。
- 共享 expert (DeepSeek 式) 與 expert 負載統計 (load balancing loss 的硬體計數)。
