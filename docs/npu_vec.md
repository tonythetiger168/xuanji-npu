# Xuanji NPU 向量核 (`rtl/npu/npu_vec.sv`)

與矩陣核 (npu_core 的 GEMM CU) 分開的 FP32 SIMD 單元, 做 transformer 中矩陣乘以外的運算: softmax、LayerNorm / RMSNorm、
GELU / SiLU、INT32 GEMM 結果的反量化, 以及 BF16 / FP16 / MXFP8 / MXFP4 的載入轉換。
計劃見 `docs/npu_plan.md` 階段 1 (#19); 浮點元件來自共用函式庫 `rtl/fp` (`docs/fp.md`)。

**狀態**: 單元本身已驗證 (位元一致、coverage、mutation、形式證明、kernel 精度與拍數, 見下)。
**尚未**: 接到 npu_core / orca_npu_ss (命令與 L2 埠)、矩陣核輸出直送向量核、MX 編碼 (FP32 → MX)、管線化。
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
