#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# npu_vec 的 kernel 程式 (transformer 的非矩陣運算) 與驗收:
#   1. vec_ref.py 執行程式 → TB 向量檔 (RTL 必須位元一致, 由 run_vec.sh 執行)。
#   2. 參考模型的輸出與 float64 (numpy) 的數學定義比較, 誤差須在門檻內 (輸入為同一組 FP32 值)。
# kernel: softmax、LayerNorm、RMSNorm、GELU (tanh 近似)、SiLU、INT32 → FP32 反量化 + BF16 輸出、MXFP8 / MXFP4 解碼。
# 用法: vec_kernels.py <輸出目錄> [D=1024]
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vec_ref as V

VL = V.VL
X, TMP, OUT, GAM, BET = 0x1000, 0x2000, 0x3000, 0x4000, 0x5000


def f2u(f): return struct.unpack('<I', struct.pack('<f', float(f)))[0]
def u2f(u): return struct.unpack('<f', struct.pack('<I', u))[0]


def put(mem, base, vals):
    """FP32 向量 → 連續 line。"""
    for l in range(len(vals) // VL):
        mem[base + l] = sum(f2u(vals[l * VL + i]) << (32 * i) for i in range(VL))


def get(mem, base, n):
    return np.array([u2f((mem.get(base + l, 0) >> (32 * i)) & 0xFFFFFFFF) for l in range(n // VL) for i in range(VL)],
                    dtype=np.float64)


E = V.enc
def splat(r, f): return E('SPLAT', vd=r, imm=f2u(f))


def softmax(L):
    p = [E('SPLAT', vd=0, imm=0xFF800000)]                                     # -inf
    for l in range(L):
        p += [E('LD', vd=1, line=X + l), E('MAX', vd=0, va=0, vb=1)]
    p += [E('RMAX', vd=0, va=0), E('SPLAT', vd=2, imm=0)]
    for l in range(L):
        p += [E('LD', vd=1, line=X + l), E('SUB', vd=1, va=1, vb=0), E('SFU', fn=3, vd=1, va=1),
              E('ADD', vd=2, va=2, vb=1), E('ST', va=1, line=TMP + l)]
    p += [E('RSUM', vd=2, va=2), E('SFU', fn=0, vd=2, va=2)]
    for l in range(L):
        p += [E('LD', vd=1, line=TMP + l), E('MUL', vd=1, va=1, vb=2), E('ST', va=1, line=OUT + l)]
    return p


def norm(L, D, eps, rms):
    p = [E('SPLAT', vd=0, imm=0), splat(7, 1.0 / D)]
    if not rms:
        for l in range(L):
            p += [E('LD', vd=1, line=X + l), E('ADD', vd=0, va=0, vb=1)]
        p += [E('RSUM', vd=0, va=0), E('MUL', vd=0, va=0, vb=7)]               # 平均
    p += [E('SPLAT', vd=2, imm=0)]
    for l in range(L):
        p += [E('LD', vd=1, line=X + l)]
        if not rms:
            p += [E('SUB', vd=1, va=1, vb=0), E('ST', va=1, line=TMP + l)]     # 置中後存起來 (兩趟算法)
        p += [E('MUL', vd=3, va=1, vb=1), E('ADD', vd=2, va=2, vb=3)]
    p += [E('RSUM', vd=2, va=2), E('MUL', vd=2, va=2, vb=7), splat(3, eps), E('ADD', vd=2, va=2, vb=3),
          E('SFU', fn=1, vd=2, va=2)]                                           # 1 / sqrt(var + eps)
    for l in range(L):
        p += [E('LD', vd=1, line=(X if rms else TMP) + l), E('MUL', vd=1, va=1, vb=2),
              E('LD', vd=3, line=GAM + l), E('MUL', vd=1, va=1, vb=3)]
        if not rms:
            p += [E('LD', vd=3, line=BET + l), E('ADD', vd=1, va=1, vb=3)]
        p += [E('ST', va=1, line=OUT + l)]
    return p


C2 = 2.0 * np.sqrt(2.0 / np.pi)


def gelu(L):
    # GELU_tanh(x) = 0.5·x·(1 + tanh(c·(x + 0.044715·x³))) = x · sigmoid(2c·(x + 0.044715·x³)) (代數恆等)
    # -z = x·(K0 + K1·x²), sigmoid(z) = 1 / (1 + e^(-z))
    p = [splat(5, -C2), splat(6, -C2 * 0.044715), splat(7, 1.0)]
    for l in range(L):
        p += [E('LD', vd=1, line=X + l), E('MUL', vd=2, va=1, vb=1), E('MUL', vd=2, va=2, vb=6),
              E('ADD', vd=2, va=2, vb=5), E('MUL', vd=2, va=2, vb=1), E('SFU', fn=3, vd=2, va=2),
              E('ADD', vd=2, va=2, vb=7), E('SFU', fn=0, vd=2, va=2), E('MUL', vd=1, va=1, vb=2),
              E('ST', va=1, line=OUT + l)]
    return p


def silu(L):
    p = [splat(6, -1.0), splat(7, 1.0)]
    for l in range(L):
        p += [E('LD', vd=1, line=X + l), E('MUL', vd=2, va=1, vb=6), E('SFU', fn=3, vd=2, va=2),
              E('ADD', vd=2, va=2, vb=7), E('SFU', fn=0, vd=2, va=2), E('MUL', vd=1, va=1, vb=2),
              E('ST', va=1, line=OUT + l)]
    return p


def dequant(L, s):
    # GEMM 的 INT32 結果 × 比例 → BF16 (兩個 INT32 line → 一個 BF16 line)
    p = [splat(7, s)]
    for l in range(0, L, 2):
        p += [E('LD', fn=3, vd=1, line=X + l), E('MUL', vd=1, va=1, vb=7), E('LD', fn=3, vd=2, line=X + l + 1),
              E('MUL', vd=2, va=2, vb=7), E('ST', fn=1, va=1, vb=2, line=OUT + l // 2)]
    return p


def mx(L, fn, scales):
    # MX 區塊 (32 個元素共用一個 E8M0 比例) 解碼成 FP32: FP8 一個 line 64 個元素 (4 個 part), FP4 128 個 (8 個 part)
    np_ = 4 if fn != 6 else 8
    p = []
    for l in range(L):
        for q in range(np_):
            p += [E('LD', fn=fn, vd=1, part=q, imm=scales[(l * np_ + q) // 2], line=X + l),
                  E('ST', va=1, line=OUT + l * np_ + q)]
    return p


# ---------------- MX 參考 (OCP MX v1.0 的元素值表, 與 vec_ref 的位元解碼獨立) ----------------
def e2m1_val(c):
    v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][c & 7]
    return -v if c & 8 else v


def fp8_val(c, e4m3):
    s = -1.0 if c & 0x80 else 1.0
    if e4m3:
        ex, m = (c >> 3) & 15, c & 7
        if ex == 15 and m == 7:
            return np.nan
        return s * (m / 8 * 2.0 ** -6 if ex == 0 else (1 + m / 8) * 2.0 ** (ex - 7))
    ex, m = (c >> 2) & 31, c & 3
    if ex == 31:
        return np.nan if m else s * np.inf
    return s * (m / 4 * 2.0 ** -14 if ex == 0 else (1 + m / 4) * 2.0 ** (ex - 15))


def main():
    out = sys.argv[1]
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    L = D // VL
    rng = np.random.default_rng(7)
    ok = True
    res = []

    def go(name, prog, mem0, check):
        nonlocal ok
        mem = dict(mem0)
        recs, v = V.run(prog, mem)
        assert v.err == 0
        V.write_tb(os.path.join(out, 'k_' + name), mem0, recs)
        err, lim, what = check(mem)
        good = err <= lim
        ok &= good
        res.append(name)
        print('KERNEL %s D=%d ins=%d %s=%.3g (limit %.3g) %s' % (name, D, len(prog), what, err, lim,
                                                                   'PASS' if good else 'FAIL'))

    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    # softmax: logits 量級 ±30 (含大負數 → exp 下溢成 0)
    x = f32(rng.normal(0, 8, D))
    m0 = {}; put(m0, X, x)
    def ck_sm(mem):
        y = get(mem, OUT, D)
        r = np.exp(x - x.max()); r /= r.sum()
        big = r > 1e-30
        return np.max(np.abs(y[big] - r[big]) / r[big]), 1e-5, 'max_rel_err'
    go('softmax', softmax(L), m0, ck_sm)
    # LayerNorm / RMSNorm: 非零平均 (兩趟算法避免相消), gamma / beta
    x = f32(rng.normal(5, 2, D)); g = f32(rng.normal(1, 0.1, D)); bt = f32(rng.normal(0, 0.1, D))
    m0 = {}; put(m0, X, x); put(m0, GAM, g); put(m0, BET, bt)
    eps = 1e-5
    def ck_ln(mem):
        y = get(mem, OUT, D)
        mu = x.mean(); var = ((x - mu) ** 2).mean()
        r = (x - mu) / np.sqrt(var + float(np.float32(eps))) * g + bt
        return np.max(np.abs(y - r)) / np.max(np.abs(r)), 1e-5, 'max_err/max|y|'
    go('layernorm', norm(L, D, eps, False), m0, ck_ln)
    def ck_rms(mem):
        y = get(mem, OUT, D)
        r = x / np.sqrt((x ** 2).mean() + float(np.float32(eps))) * g
        return np.max(np.abs(y - r)) / np.max(np.abs(r)), 1e-5, 'max_err/max|y|'
    go('rmsnorm', norm(L, D, eps, True), m0, ck_rms)
    # GELU / SiLU: x 在 [-12, 12] (含飽和區)
    x = f32(np.concatenate([rng.uniform(-12, 12, D // 2), rng.normal(0, 1, D // 2)]))
    m0 = {}; put(m0, X, x)
    def ck_gelu(mem):
        y = get(mem, OUT, D)
        r = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        return np.max(np.abs(y - r) / np.maximum(np.abs(r), 1e-3)), 1e-5, 'max_rel_err'
    go('gelu', gelu(L), m0, ck_gelu)
    def ck_silu(mem):
        y = get(mem, OUT, D)
        r = x / (1 + np.exp(-x))
        return np.max(np.abs(y - r) / np.maximum(np.abs(r), 1e-3)), 1e-5, 'max_rel_err'
    go('silu', silu(L), m0, ck_silu)
    # INT32 → FP32 × 比例 → BF16
    xi = rng.integers(-2 ** 31, 2 ** 31, D, dtype=np.int64)
    xi[:4] = [-2 ** 31, 2 ** 31 - 1, 0, (1 << 24) + 1]
    m0 = {X + l: sum((int(xi[l * VL + i]) & 0xFFFFFFFF) << (32 * i) for i in range(VL)) for l in range(L)}
    sc = 2.0 ** -20 * 1.37
    def ck_dq(mem):
        y = np.array([V.fp_ref.classify((mem[OUT + l // 2] >> (16 * i)) & 0xFFFF, V.fp_ref.BF16)[2]
                      for l in range(0, L, 2) for i in range(2 * VL)], dtype=np.float64)
        r = xi.astype(np.float64) * float(np.float32(sc))
        return np.max(np.abs(y - r) / np.maximum(np.abs(r), 1e-30)), 2 ** -8, 'max_rel_err'   # BF16: 2^-8
    go('dequant', dequant(L, sc), m0, ck_dq)
    # MXFP8 (E4M3 / E5M2) / MXFP4 (E2M1): 對 OCP 元素值表 × 2^(X-127) 精確比對 (結果在 FP32 正規範圍內)
    for name, fn, nel in (('mx_e4m3', 4, 64), ('mx_e5m2', 5, 64), ('mx_e2m1', 6, 128)):
        LL = max(1, D // nel // 4)
        codes = rng.integers(0, 1 << (8 if nel == 64 else 4), LL * nel)
        if fn == 4:
            codes[codes == 0x7F] = 0x7E; codes[codes == 0xFF] = 0xFE         # 不放 NaN (另由隨機測試涵蓋)
        if fn == 5:
            codes[(codes & 0x7C) == 0x7C] &= 0x7B                            # 不放 inf / NaN
        w = 8 if nel == 64 else 4
        m0 = {X + l: sum(int(codes[l * nel + i]) << (w * i) for i in range(nel)) for l in range(LL)}
        scales = [int(s) for s in rng.integers(110, 145, LL * nel // 32)]
        def ck_mx(mem, codes=codes, scales=scales, fn=fn, nel=nel, LL=LL):
            y = get(mem, OUT, LL * nel)
            val = np.array([e2m1_val(c) if fn == 6 else fp8_val(c, fn == 4) for c in codes])
            r = val * np.array([2.0 ** (scales[i // 32] - 127) for i in range(LL * nel)])
            return float(np.max(np.abs(y - r))), 0.0, 'max_abs_err'
        go(name, mx(LL, fn, scales), m0, ck_mx)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
