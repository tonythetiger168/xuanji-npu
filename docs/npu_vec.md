# Xuanji NPU 向量核 (`rtl/npu/npu_vec.sv`)

與矩陣核 (npu_core 的 GEMM CU) 分開的 FP32 SIMD 單元, 做 transformer 中矩陣乘以外的運算: softmax、LayerNorm / RMSNorm、
GELU / SiLU、INT32 GEMM 結果的反量化, 以及 BF16 / FP16 / MXFP8 / MXFP4 的載入轉換。
計劃見 `docs/npu_plan.md` 階段 1 (#19); 浮點元件來自共用函式庫 `rtl/fp` (`docs/fp.md`)。

**狀態**: 單元已驗證 (位元一致、coverage、mutation、形式證明、kernel 精度與拍數, 見下); 已經由向量引擎 `npu_vec_eng`
接進 NPU Subsystem (`VEC` 命令, 與 DMA 共用 host L2 埠; 見「向量引擎與 NPU Subsystem 整合」)。
**尚未**: 矩陣核輸出直送向量核 (目前經 L2)、MX 編碼 (FP32 → MX)、管線化 / 雙發射。
尚未合成, 不宣稱頻率 / 面積 / 功耗; 目前每個指令是單拍組合式運算 (含 16 個 SFU 與 15 + 15 個歸約節點)。

## 架構

```
指令 (valid / ready, 77 bit) ──► 解碼 ──┬─ LD: L2 讀 (等 grant) ─► 格式轉換 ×16 ─┐
                                       ├─ ADD / SUB / MUL / MAX / SFU ×16 lane ──┼─► 寫回 vd (8 個向量暫存器, 每個 ICG)
                                       ├─ RSUM / RMAX: 固定順序歸約樹 ─► 廣播 ──┤
                                       ├─ SPLAT imm ─────────────────────────────┘
                                       └─ ST: FP32 或 BF16 (兩個暫存器 → 一個 line) ─► L2 寫 (等 grant)
```

- 16 lane × FP32 = 一個 64 B L2 line; 8 個向量暫存器 (參數 NV)。單發射、依序, 每拍一個指令; LD / ST 等到 L2 grant。
- L2 介面與 `npu_gemm_cu` 相同 (讀為組合式, rd_data 與 rd_gnt 同拍; 寫在 wr_gnt 拍完成; 等待時請求保持),
  之後可直接成為 `npu_l2_xbar` 的一個主端。
- 非法編碼當成 NOP 接受, 設定黏著的 `err` (只有 reset 清除)。暫存器檔沒有 reset (資料暫存器), 跨 reset 保持。
- **低功耗**: 每個向量暫存器一個 `clock_gate`, 只在寫入該暫存器的拍開啟; `test_en` 強制開啟時功能相同
  (TB 隨機切換 test_en 驗證)。

## 指令 (`npu_vec_pkg::npu_vec_ins_t`)

`{op[3:0], fn[2:0], vd[2:0], va[2:0], vb[2:0], part[2:0], imm[31:0], line[25:0]}`

| op | 指令 | 動作 |
|---|---|---|
| 0 | NOP | — |
| 1 | LD vd, line | 依 fn 的格式取 line 中第 part 組 16 個元素 → FP32 |
| 2 | ST va (, vb), line | fn 0: va 的 16 個 FP32; fn 1: BF16 (va → 元素 0..15, vb → 16..31, RNE) |
| 3 / 4 / 5 | ADD / SUB / MUL | 逐 lane, `fp_add` / `fp_mul` (SUB = a + (−b)) |
| 6 | MAX | 逐 lane, `fp_max` |
| 7 | SFU vd, va | fn 0 recip、1 rsqrt、2 exp2、3 exp、4 sinπ、5 cosπ (`sfu`) |
| 8 | SPLAT vd, imm | 每個 lane = imm (原始位元) |
| 9 / 10 | RSUM / RMAX vd, va | va 的 16 個 lane 歸約 (樹: 每層 x[i] = f(x[2i], x[2i+1])), 結果廣播到 vd 每個 lane |

LD 格式 (fn): 0 FP32、1 BF16、2 FP16、3 INT32、4 MXFP8 E4M3、5 MXFP8 E5M2、6 MXFP4 E2M1。
16 / 8 / 4 bit 格式一個 line 有 32 / 64 / 128 個元素, part 選第幾組 16 個。MX 格式的共用比例 (E8M0) 在 imm[7:0]:
OCP MX 一個區塊 32 個元素, 一個 FP8 line 是兩個區塊, 軟體以 part 0–1 / 2–3 分別給比例 (FP4 一個 line 四個區塊)。

## 數值語意

函式庫一致的 FTZ / DAZ (`docs/fp.md`), 加上:
- **MX / FP16 解碼精確** (`fp_small_dec`): 元素的次正規數照規格解碼 (E2M1 的 0.5、FP16 的 2^-24 等), 正規化後
  成為 FP32 正規數; E4M3 只有 S.1111.111 是 NaN (最大值 448), E5M2 / FP16 為 IEEE 式 (inf / NaN), E2M1 無特殊值。
  乘上比例 2^(X−127) (`fp_mx_scale`) 是精確的, 只判斷範圍: 超過 FP32 → ±inf, 小於最小正規數 → ±0 (保留符號);
  X = 0xFF → NaN。
- **INT32 → FP32** (`fp_from_int`): RNE, 0 → +0。
- **BF16 載入** 同 `fp_widen` (DAZ); **BF16 儲存** 同 `fp_narrow` (RNE)。
- **MAX**: 任一 NaN → 標準 qNaN; 次正規數當 ±0; max(+0, −0) = +0。
- **歸約順序固定** (樹), 所以結果與參考模型位元一致, 也與執行時序無關。

## 驗證 (`dv/scripts/run_vec.sh`, `run_vec_mutation.sh`, `run_formal.sh npu_vec`)

- **參考模型** `dv/vec/vec_ref.py`: 指令編碼、格式轉換、逐 lane 運算 (沿用 `dv/fp/fp_ref.py` / `sfu_ref.py`) 與歸約;
  TB 逐指令比對寫入暫存器的值、寫入 L2 的資料與 err。
- **受約束隨機** `dv/vec/gen_vec_vectors.py`: 各格式的特殊值 / 邊界 / 隨機位元的 L2 內容, 隨機指令 (含 2% 非法編碼、
  中途 reset), L2 grant 隨機 (70%)、指令間隨機空拍、test_en 隨機切換; 沒有 grant 時 L2 讀資料為垃圾。
  seed 1 另含**窮舉**: 256 個 FP8 碼 × 256 個比例 (E4M3 / E5M2)、FP4 碼 × 256 個比例、全部 65536 個 16 bit 碼 (BF16 / FP16)。
- **SVA** `dv/vec/npu_vec_sva.sv`: L2 交握、ready、err、每個暫存器只在被寫入的拍改變 (clock gating 安全); covergroup:
  每個 op / LD 格式 / ST 格式 / SFU 函式 / 每種寬度的每個 part / MX 比例類別 / 每個目的暫存器 / 每類非法編碼。
- **形式證明** (`formal/npu_vec`, k-induction, 浮點元件換成任意值替身, NV = 2): 檢查器自己從指令埠解碼 —
  L2 請求只來自合法 LD / ST 且位址正確、ready 跟隨 grant、err 的下一拍值、暫存器只在接受了寫入它的指令後改變。
  形式 mutation 7 / 7 (不等 grant、ST 寫暫存器、err 不黏著、非法 ST 當合法、位址錯、寫入所有暫存器、非法 ST 發出寫入)。

### 結果

| 項目 | 結果 |
|---|---|
| 位元一致 | 4 seed (22,624 + 3 × 約 5,980 個指令) + 9 個 kernel, 0 錯誤 |
| coverage | line / branch / expr / toggle 100% (唯一豁免: BF16 擴展補上的 16 個低位恆為 0 [C]); covergroup 57 / 57、SVA cover 5 / 5 |
| mutation | 49 / 49: 單元 39 (解碼 / 非法編碼、ready、ST 寫回、err、part 選擇、SUB、格式選擇、歸約節點、clock gate、BF16 儲存、MX / FP16 解碼、比例、INT32 捨入、max) + 向量引擎 `npu_vec_eng` 10 (命令保持、slot、取指、PC、韌體寫入、完成、err 清除、ready、clock gate、L2 grant) |
| 形式證明 | prove PASS、cover PASS; 形式 mutation 7 / 7 |

### Kernel (`dv/vec/vec_kernels.py`, D = 1024 個 FP32, 參考: float64 numpy 的數學定義, 輸入為同一組 FP32 值)

RTL 與參考模型位元一致; 拍數為 `+perf` (grant 全給、無空拍) 的實測。PyTorch 不在環境中, 以 numpy float64 作數學參考。

| kernel | 程式 | 誤差 (門檻) | 指令 = 拍數 | 元素 / 拍 |
|---|---|---|---|---|
| softmax (logits σ = 8) | 3 趟: max → exp 與總和 → × 1/總和 | 最大相對誤差 2.0e-6 (1e-5) | 645 | 1.59 |
| LayerNorm (μ = 5, σ = 2) | 2 趟平均 / 置中變異數, rsqrt | 最大誤差 / max\|y\| 1.3e-7 (1e-5) | 906 | 1.13 |
| RMSNorm | 平方和, rsqrt | 1.1e-7 (1e-5) | 520 | 1.97 |
| GELU (tanh 近似) | x · sigmoid(2c(x + 0.044715x³)) (代數恆等) | 最大相對誤差 6.6e-7 (1e-5) | 643 | 1.59 |
| SiLU | x · 1/(1 + e^-x) | 1.4e-7 (1e-5) | 450 | 2.28 |
| INT32 × 比例 → BF16 | 反量化 GEMM 輸出 | 3.8e-3 (BF16 半 ulp 2^-8) | 161 | 6.4 |
| MXFP8 E4M3 / E5M2、MXFP4 E2M1 解碼 | 對 OCP 元素值表 × 2^(X−127) | 0 (精確) | 32 | 8 / 8 / 16 |

softmax 的誤差主要來自 x − max 的 FP32 捨入 (|x − max| 達 60 時 ulp ≈ 4e-6, 經 exp 放大為相對誤差), 是 FP32 本身的限制。

吞吐量受單發射限制 (每拍一個指令, LD / ALU / ST 輪流): 雙發射 (LD / ST 與運算並行) 或把常見組合 (x − m 後 exp、
乘加) 合併成一個指令, 是提升元素 / 拍的方向 (尚未實作)。

### 驗證中的發現

- 第一版解碼器把 MX 比例乘在每個格式的解碼器內: FP16 (比例固定 1) 的比例輸入是常數、各格式的 SP 參數產生恆為真 / 假的
  運算式, coverage 出現大量結構性缺口 → 改為「精確元素解碼 (窄輸出, 不輸出恆為 0 的低位) + 共用的 MX 比例級」,
  各格式的特殊值規則以 generate 分支實作, 結構性豁免從數百項降到 1 項。
- INT32 → FP32 正規化後的最低位恆為 0 (只有 −2^31 會不移位, 其最低位為 0) → 從資料路徑移除。

## 向量引擎與 NPU Subsystem 整合 (`rtl/npu/npu_vec_eng.sv`, `rtl/npu/npu_l2_arb2.sv`)

- **VEC 命令** (`npu_tile_cmd_t.op = 4`, 命令的 opcode 擴為 3 bit): 執行 L2 line `a` 起的 `k` 個指令。程式版面: 一個 64 B line
  放 4 個指令, 第 j 個在 bit [128j +: 77]。k = 0 當 NOP; 保留的 op 5–7 也當 NOP。軟體通常先以 `DMA_LD` 把程式從外部記憶體
  載入 L2, 再下 `VEC` (wt = 1 或範圍不重疊)。程式不可被自己的 ST 改寫 (預取可能已讀走舊內容)。
- **取指**: 兩個 line 的緩衝, 執行一個 line 時預取下一個; 與 npu_vec 共用一個 L2 讀取埠 — 取指只在 npu_vec 沒有讀取請求時
  開始, 一旦發出就保持到 grant (L2 介面規則: 請求與位址保持)。
- **錯誤**: 命令開始時清除 npu_vec 的 err (`err_clr`); 命令完成時若遇過非法指令, `orca_npu_ss.vec_err` 一拍 →
  APB `IRQ_STATUS.VEC_ERR` (黏著, W1C)。
- **共用 host 埠** (`npu_l2_arb2`): DMA 與向量引擎輪流 (兩者都請求時 round-robin; 選中者等待 grant 期間鎖定, 下游的請求 /
  位址 / 資料保持)。
- **低功耗**: 引擎閒置時停止取指 / 控制的時脈; npu_vec 用未閘控的時脈 (只有 err 一個未閘控的 flop, 向量暫存器各有 ICG),
  也讓 npu_vec 的 SVA 以未閘控時脈取樣。

### 整合的驗證

| 項目 | 內容 / 結果 |
|---|---|
| 引擎 (`dv/vec/npu_vec_eng_tb.sv`, `run_vec.sh`) | 4 seed × 60 個命令 (程式長度 1–200, 每個 seed 一個 65,280–65,535 個指令的長命令; 程式位置任意 26 bit, 含跨越 0x3FFFFFF 回繞; 3% 非法指令), 隨機 grant (50–100%), 命令之間偶爾 reset; 結果 (每個命令的 derr、最後的 L2 資料區) 與 `vec_ref.py` 位元一致 |
| 引擎 SVA (`dv/vec/npu_vec_eng_sva.sv`) | 只在閒置時接受命令、done 一次、L2 讀 / 寫請求保持到 grant、時脈關閉時沒有活動; cover: 取指與 LD 競爭、預取、取指等待中 LD 也要讀、指令用完等取指 |
| 引擎形式證明 (`formal/npu_vec_eng`, npu_vec 為契約替身) | 任選序號 i: 第 i 個交給 npu_vec 的指令 = 程式第 i 個字 (從埠記錄取指 grant 的資料); 恰好 n 個後 done; 讀取請求保持 |
| 仲裁器形式證明 (`formal/npu_l2_arb2`) | grant 互斥且只給請求者、下游位址 / 資料屬於拿到 grant 者、下游請求保持、被插隊一次的主端下一次必得 grant |
| NPU Subsystem (`npu_ss_tb`) | 隨機程式中 12% 為「DMA_LD 程式 + VEC」(程式: line 複製 / 填值 / 偶爾非法指令; 參考模型依序執行並計算預期的 vec_err 次數); 導向測試: 兩個 VEC 背靠背 (第二個等引擎忙碌、再等 VEC 完成槽), 4 個完成來源同時待回報 |
| APB 整合 (`orca_npu_ss_apb_tb`) | 只經 APB: DMA_LD 程式 → VEC (LD / 非法 / ST) → DMA_ST 存回, 比對資料與 IRQ_STATUS.VEC_ERR 的設定 / W1C 清除 |

驗證整合時發現: (1) 預取請求在沒拿到 grant 時, 若下一拍 npu_vec 也要讀, 原本會撤回請求 (違反 L2 介面的保持規則) →
加上 f_pend 鎖定; (2) npu_vec 的 SVA 以引擎的閘控時脈取樣時, 時脈關閉期間的 reset 造成 `a_err_stk` 誤報 → npu_vec 改用
未閘控時脈; (3) 引擎 SVA 原本假設命令保持到接受, 但 orca_npu_ss 在 VEC 完成槽被佔用時會撤回 `g_vec` (合法) → 移除該假設;
(4) 仲裁器的形式檢查與環境假設在 orca_npu_ss 的證明中也被編入 (子模組的假設約束了上層) → 改以 `FORMAL_TOP_<block>`
只在自己是證明頂層時編入。
