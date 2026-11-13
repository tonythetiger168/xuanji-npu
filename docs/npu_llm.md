# Xuanji NPU LLM 引擎 (`rtl/npu/npu_llm_eng.sv`): KV / MOE 命令

NPU Subsystem 給 LLM 推論用的兩個命令 (op 5 `NPU_OP_KV`、op 6 `NPU_OP_MOE`; op 7 保留)。引擎內含 paged KV cache
頁管理 `npu_kv_pager` (`docs/npu_kv.md`) 與 MoE dispatch `npu_moe_disp` (`docs/npu_moe.md`), 只讀寫 L2 中的小表格;
KV 資料與 token activation 的搬移由 DMA 間接模式 (`rtl/npu/npu_dma.sv`, 另一個 PR 系列) 依這些表格執行。

**狀態**: 引擎已獨立驗證 (CRV + L2 模型 + 每個命令後比對整個 L2、SVA、coverage 100%、mutation)。
已接進 `orca_npu_ss` (v6.3.13): KV / MOE 命令依序分派到引擎 (一次一個; 引擎閒置且完成槽空時), 完成合併多一個來源
(5 個來源 round-robin), host L2 埠以第二個 `npu_l2_arb2` 與 DMA / VEC 共用; 子系統 CRV 的參考模型含 KV 頁管理狀態與
MOE dispatch, 形式證明 (orca_npu_ss 前端) 含 LLM 引擎的契約替身。
端到端 (paged attention 的 KV cache 與 MoE 層, 對照 numpy) 見文末。**尚未**: 合成; MoE expert GEMM 直接使用收集結果 (見文末限制)。

子系統參數 (orca_npu_ss): `KV_NP` 256 頁、`KV_NSEQ` 16 序列、`KV_NSLOT` 64 區塊 / 序列、`KV_BS` 16 token / 區塊、
`KV_RW` 3 (共享上限 7)、`MOE_NEXP` 64 expert、`MOE_PW` 14 (每個命令 < 16384 個項目)、`MOE_TW` 12 (每個命令 < 4096 個 token)。

## 與其他單元的介面 (與 npu_dma 系列約定)

| 項目 | 格式 |
|---|---|
| 間接 DMA 索引表 | L2 line c 起, u32 little-endian, 每 line 16 項 (第 i 項在 line c + i/16 的 bit 32·(i%16)); 項目 E = 外部 line 偏移 (相對命令的 ext), 每項搬 cw = b[15:0] 個 line (0 = 1; 外部 ext+E .. ext+E+cw−1, L2 端連續); **0xFFFFFFFF = 略過** (LD: L2 端寫 0、不讀外部; ST: 不寫外部)。KV_MAP 的 mult 與 MOE 的 mult 通常等於 cw (每頁 / 每 token 的 line 數) |
| top-k 輸出 (向量核) | 每個 token 兩個 line: 16 × u32 expert id、16 × FP32 weight (由大到小); 未用的槽 id = 0xFFFFFFFF |
| KV 量化 | 每頁一個 scale 表項 (外部記憶體); `KV_MAP` 以 mult = 1 產生 scale 表的索引, 以 mult = 每頁 line 數產生資料的索引; 兩者都用間接 DMA 搬, 解量化在向量核 |

## KV 命令 (op 5)

| 欄位 | 內容 |
|---|---|
| b[2:0] | 子命令: 0 ALLOC、1 FREE、2 FREE_SEQ、3 SHARE、4 MAP (5..7 → BADARG) |
| c[7:0] / c[23:8] | seq / slot |
| ext | SHARE: 來源序列 (ext[7:0]); MAP: mult (項目 = 頁 × mult) |
| k | MAP: slot 數 (≤ NSLOT) |
| a | 輸出 line |

- ALLOC / FREE / FREE_SEQ / SHARE: 結果 line 寫到 a — w0 狀態、w1 資料 (頁 / 解除的 slot 數)、w2 free_cnt。
- MAP: slot .. slot + k − 1 的索引表寫到 a 起 (有映射 = 頁 × mult; 沒有或 slot ≥ NSLOT = 0xFFFFFFFF; 最後一個
  line 未用的字組 = 0), 之後一個結果 line: w0 = 全部有映射 ? OK : UNMAPPED、w1 有映射數、w2 free_cnt。
- 狀態: 0 OK、1 NOMEM、2 MAPPED、3 UNMAPPED、4 REFMAX、5 BADARG (seq ≥ NSEQ、slot ≥ NSLOT、SHARE 來源 ≥ NSEQ、
  MAP 的 k > NSLOT; 只寫結果 line, 不改變頁管理狀態)。

## MOE 命令 (op 6)

| 欄位 | 內容 |
|---|---|
| k | token 數 T (1 .. 2^TW − 1) |
| a | top-k 輸入: token t 在 line a + 2t (id) 與 a + 2t + 1 (weight) |
| b | b[4:0] = K (1..16), b[12:5] = nexp (1..NEXP), b[13] = 無丟棄模式 |
| ext | ext[15:0] = 每 expert 容量 cap (< 2^(PW − log2 NEXP)), ext[31:16] = mult (token 的外部 line 偏移 = token × mult) |
| c | 輸出區起點 |

輸出 (從 line c 起依序):

| 區 | 大小 | 內容 |
|---|---|---|
| GL gather 索引表 | ⌈P/16⌉ lines, P = 無丟棄 ? T·K : nexp·cap | 先全部填 0xFFFFFFFF; 每個沒被 drop 的項目 GL[pos] = token × mult |
| CT combine 表 | ⌈T·K/8⌉ lines | 依輸入順序每項 {pos, weight}; drop = {0xFFFFFFFF, 0} |
| CN 各 expert | ⌈nexp/8⌉ lines | 每 expert {n, off} |
| 狀態 line | 1 line | w0 = {ovf, 錯誤 id 數}, w1 = 有效項目數, w2 = P |

參數不合法 → 只在 line c 寫狀態 line (w0 = 0xFFFFFFFF)。T·K > 2^PW − 1 → ovf = 1 (位置以 PW bit 回繞, 無效)。
GL 依 pos 寫入 (讀 − 改 − 寫), 其他區依序累積成整個 line 再寫。

## 驗證 (`dv/scripts/run_npu_llm.sh`, `run_npu_llm_mutation.sh`)

| 項目 | 內容 |
|---|---|
| L2 模型 | 稀疏 line 陣列, 未寫過的 line 讀到依位址決定的內容; 讀 / 寫 grant 隨機延遲 (每個命令 20–100%) |
| 參考模型 | TB 內重寫 KV 頁管理與 MoE dispatch 的規格, 算出每個命令應寫的 line; 每個命令後比對 L2 的完整內容 (DUT 不可寫到其他 line) |
| 情境 | 開頭 directed: 序列 0 配 16 頁、slot 0 共享到參考計數上限、MAP 整個序列; 之後隨機: 全部子命令與非法參數, MAP 跨多個 line / k = 0 / k > NSLOT / slot 接近 65535, MOE 兩種模式、K = 1..16、T 到上限、偏斜分布、未用槽與超出範圍的 id、溢位、非法參數; 位址遍及整個 26 bit (含回繞); 欄位中未用的位元隨機; 隨機 test_en; 命令中途 reset; 隨機上電狀態 |
| 組態 | NP 16 / NSEQ 4 / NSLOT 32 / BS 4 / RW 2 / NEXP 16 / PW 9 / TW 6; 4 seed × 400 命令 + 隨機上電 2 次 |
| SVA | L2 請求保持到 grant、讀寫不同時、命令只在閒置時接受、done 單拍、閘控安全; 引擎依序使用子模組 (pager / dispatch 在被使用時必定就緒) |
| coverage | npu_llm_eng 的 line / branch / expr / toggle 100% (無豁免); covergroup 24 / 24; SVA cover 5 / 5 (子模組在各自的腳本收斂) |
| mutation | 23 / 23 (見腳本): 參數檢查、結果狀態 / 位置、MAP 的 mult / 未映射值 / 範圍 / 多 line / 狀態、MOE 的 T 上限 / GL 初值與數量 / weight line / 讀 − 改 − 寫 / mult / drop 權重 / 最後 combine line / 下一個 token / 第二趟 / CN / 溢位旗標、ICG |

驗證過程中:
- TB 第一版的 fork 中, 「不 reset」分支是空的 `if`, 立即結束 → `join_any` 立刻返回, 每個命令還沒執行就比對。
- 又一次 static 區域變數: `initial` 內 `int r = ur(0, 99)` 只求值一次, 子命令永遠相同 (MAP 從未執行, coverage 才發現);
  `int c2 [$]` 跨迭代累積。
- reset 只維持半拍 (不跨過取樣沿) 時 SVA 誤報請求保持 → TB 的 reset 跨過一個取樣沿。
- mutation `icg` (閘控漏掉 done 拍) 原本存活: TB 在 done 之後的下一拍就送下一個命令, done 被命令的時脈清掉 →
  命令之間加入隨機閒置。`map_range` 的原版 (命中不看 slot 是否在範圍內) 是等價 mutant (超出範圍時沒有送轉換,
  tr_hit 本來就是 0) → RTL 拿掉多餘的條件, mutant 改為「超出範圍時仍送轉換」(slot 回繞)。
- RTL: 寬度依合法命令的界限重新決定 (MAP 的 k ≤ NSLOT、MOE 的 T < 2^TW), toggle coverage 才能收斂;
  原本 32 bit 的計數器高位永遠不動。

## 端到端 (`dv/scripts/run_npu_llm_e2e.sh`, `dv/npu_llm/llm_e2e.py`)

在 `orca_npu_ss` 上依序執行一段命令程式 (情境檔由 numpy 產生, TB `dv/npu_llm/npu_llm_e2e_tb.sv` 只負責執行與輸出外部記憶體),
期望值由 numpy 獨立計算。外部記憶體的讀 / 寫 grant 與完成 ready 隨機。

| 情境 | 命令流程 | 比對 |
|---|---|---|
| paged attention (KV cache) | KV ALLOC 序列 0 的 5 頁; SHARE 序列 1 共享前 3 頁 (prefix); ALLOC 序列 1 自己的 2 頁; DMA_LD 連續的 K (GEMM 的 B 運算元 layout, 1 頁 = N 個 token) → KV MAP (mult = 每頁 line 數) → 間接 DMA_ST 把 K 散到外部記憶體的頁; decode: MAP 序列 1 的 6 個 slot (最後一個未映射) → 間接 DMA_LD gather 回 L2 → 每頁一個 GEMM 算 Q·K^T; 最後 FREE_SEQ / FREE / ALLOC (FIFO 重複使用) 與錯誤命令 | 外部記憶體中每頁的內容、16 個 KV 結果 line、3 個位址表與其結果 line、gather 結果 (未映射的頁 = 0)、注意力分數 (int32, numpy matmul) |
| MoE 層 | numpy router (偏斜, 每 token top-2 / 8 expert, softmax 權重 FP32, 含一個超出範圍的 id) → DMA_LD top-k → MOE (容量模式 cap 5, 與無丟棄模式各一次) → 間接 DMA_LD 依 GL 收集 token | GL / CT / CN / 狀態 line (以 numpy 另外實作的規格); 收集結果; MoE 層輸出: 以硬體的 CT + 收集結果 + 每個 expert 的運算 (numpy, f_e(x) = (e + 1)·x) 算出 y_t, 與 numpy 直接計算的 Σ_k w_k·f_e(x_t) (容量模式排除被丟棄的項) 完全相同 |

nano / x1 / pro 各 3 個 seed 全部通過 (CI: x1 3 個 seed)。

**限制 (後續工作)**: 每個 expert 的 GEMM 還不能直接用收集結果 — GEMM 的 A / B 運算元 layout 是「一個 line 放 SPL 個
k × N 列」交錯排列, 以 token 為單位 (一個 token = 連續的 line) gather 出來的資料不是這個 layout。需要在 DMA 加一個
「token 列 → GEMM 運算元」的轉置格式, 或以向量核轉置; 目前 expert 的運算在 numpy 端。paged attention 不受影響: KV 頁
直接以 B 運算元 layout 存放 (1 頁 = N 個 token)。
