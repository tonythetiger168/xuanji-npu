#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SFU (special function unit) 的演算法參考模型 — 位元精確 (只用整數運算, 與 rtl/fp/sfu_*.sv 相同),
# 以及對雙精度參考的 ULP 誤差量測。
#
# 作法 (GPU SFU 常見): 範圍縮減到 [1, 2) (或 [0, 1)) 的有效位元 / 小數, 以前 S 個位元查表取得一段的二次多項式
# 係數, 對該段內的偏移 t 計算 y = c0 + c1·t + c2·t² (定點), 正規化後 RNE 捨入成 FP32。
#   recip(x) = 1/x, rsqrt(x) = 1/sqrt(x), exp2(x) = 2^x, exp(x) = 2^(x·log2 e) (x·log2 e 以定點高精度計算)
# 語意同 fp_add (DAZ / FTZ / 標準 qNaN); 特殊值見各函式。
# 用法: sfu_ref.py tables <輸出 .svh>      產生 RTL 係數表
#       sfu_ref.py check [n]               量測誤差 (recip / rsqrt 窮舉所有尾數; exp2 / exp 隨機 n 個)
import math, sys
import numpy as np

S = 7                       # 查表索引位元數 (2^7 = 128 段)
F = 30                      # 多項式結果的小數位元 (y 以 2^-F 為單位)
C0W, C1W, C2W = 31, 25, 17  # 係數寬度: c0 無號 (恆正, < 2^31), c1 / c2 有號 — fit() 檢查
QNAN = 0x7FC00000


def u2f(u):
    return np.frombuffer(np.uint32(u).tobytes(), dtype=np.float32)[0]


# ---------------- 係數 ----------------
def fit(func, lo, hi, n=S):
    """把 [lo, hi) 分成 2^n 段, 每段以 Chebyshev 節點最小平方擬合二次多項式 (t ∈ [0, 1) 為段內偏移)。
    回傳 (c0, c1, c2) 整數表: y·2^F ≈ c0 + c1·t + c2·t² (c1, c2 已依 t 的定點寬度調整, 見 eval_poly)。"""
    seg = (hi - lo) / (1 << n)
    out = []
    k = np.arange(16)
    tn = 0.5 - 0.5 * np.cos((2 * k + 1) * np.pi / 32)            # [0, 1) 的 Chebyshev 節點
    for i in range(1 << n):
        x = lo + (i + tn) * seg
        y = np.array([func(v) for v in x])
        A = np.stack([np.ones_like(tn), tn, tn * tn], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append((int(round(c[0] * 2 ** F)), int(round(c[1] * 2 ** F)), int(round(c[2] * 2 ** F))))
    for c0, c1, c2 in out:
        assert 0 <= c0 < (1 << C0W), c0
        assert -(1 << (C1W - 1)) <= c1 < (1 << (C1W - 1)), c1
        assert -(1 << (C2W - 1)) <= c2 < (1 << (C2W - 1)), c2
    return out


TABLES = {
    'recip':   (lambda m: 1.0 / m, 1.0, 2.0),
    'rsqrt_e': (lambda m: 1.0 / math.sqrt(m), 1.0, 2.0),          # 指數為偶數: m ∈ [1, 2)
    'rsqrt_o': (lambda m: 1.0 / math.sqrt(m), 2.0, 4.0),          # 指數為奇數: 2m ∈ [2, 4)
    'exp2':    (lambda f: 2.0 ** f, 0.0, 1.0),
    # sinπ / cosπ: g(r) = sin(πr) / (4r), r ∈ [0, 0.5) (g ∈ (0.5, 0.786]; 除以 4 讓 y 落在 [2^29, 2^31))
    'sinpi':   (lambda r: (math.sin(math.pi * r) / (4 * r)) if r > 0 else math.pi / 4, 0.0, 0.5),
}
_cache = {}


def table(name):
    if name not in _cache:
        f, lo, hi = TABLES[name]
        _cache[name] = fit(f, lo, hi)
    return _cache[name]


def eval_poly(name, idx, t, tw):
    """t: 段內偏移 (tw 位元無號定點, t / 2^tw ∈ [0, 1))。整數運算 (與 RTL 相同的截斷位置):
       p1 = (c1 · t) >> tw;  t2 = (t · t) >> tw;  p2 = (c2 · t2) >> tw;  y = c0 + p1 + p2  (單位 2^-F)"""
    c0, c1, c2 = table(name)[idx]
    p1 = (c1 * t) >> tw
    t2 = (t * t) >> tw
    p2 = (c2 * t2) >> tw
    y = c0 + p1 + p2
    assert (1 << 29) <= y < (1 << 31), y              # RTL 依此只做兩種正規化 (y 的最高位在第 29 或 30 位)
    return y


def round_fp32(y, k):
    """y: 正的定點值 (單位 2^-F), 實際值 = y · 2^(k - F)。RNE 成 FP32 (FTZ / 溢位 → inf)。回傳 (exp 欄位, 尾數) 或特殊。"""
    nb = y.bit_length()
    sh = nb - 24                                   # 保留 24 個有效位元
    if sh > 0:
        q, rem = y >> sh, y & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        if q == (1 << 24):
            q >>= 1
            sh += 1
    else:
        q = y << (-sh)
    e = k - F + sh + 23                            # 值 = q · 2^(k - F + sh), q ∈ [2^23, 2^24)
    if e > 127:
        return 0x7F800000
    if e < -126:
        return 0
    return ((e + 127) << 23) | (q & 0x7FFFFF)


def sfu_recip(x):
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0xFF:
        return QNAN if fr else (s << 31)           # 1/inf = ±0
    if ex == 0:
        return (s << 31) | 0x7F800000              # 1/±0 = ±inf (DAZ)
    idx, t = fr >> (23 - S), fr & ((1 << (23 - S)) - 1)
    y = eval_poly('recip', idx, t, 23 - S)
    return (s << 31) | round_fp32(y, 127 - ex)     # 1/(m·2^(ex-127)) = (1/m)·2^(127-ex)


def sfu_rsqrt(x):
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0xFF:
        return QNAN if (fr or s) else 0            # rsqrt(+inf) = +0; rsqrt(-inf) = NaN
    if ex == 0:
        return (s << 31) | 0x7F800000              # rsqrt(±0) = ±inf
    if s:
        return QNAN
    e = ex - 127
    idx, t = fr >> (23 - S), fr & ((1 << (23 - S)) - 1)
    if e % 2 == 0:
        y = eval_poly('rsqrt_e', idx, t, 23 - S)   # 1/sqrt(m) · 2^(-e/2)
        k = -(e // 2)
    else:
        y = eval_poly('rsqrt_o', idx, t, 23 - S)   # 1/sqrt(2m) · 2^(-(e-1)/2)
        k = -((e - 1) // 2)
    return round_fp32(y, k)


XF = 32                                            # exp2 / exp 的定點小數位元
LOG2E = round(math.log2(math.e) * (1 << 48))       # log2(e) · 2^48
LOG2E_TZ = (LOG2E & -LOG2E).bit_length() - 1      # 尾端 0 的個數: RTL 用 LOG2E >> TZ 並少移 TZ 位 (結果相同)


def to_fixed(x):
    """|x| < 256 的 FP32 → 有號定點 (XF 位元小數, 截斷)。回傳 None 表示超出範圍 (由呼叫端處理)。"""
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0:
        return 0
    m = (1 << 23) | fr
    sh = ex - 127 - 23 + XF                        # 值 · 2^XF = m · 2^sh
    v = (m << sh) if sh >= 0 else (m >> (-sh))
    return -v if s else v


def exp2_fixed(v):
    """v: 2^x 的指數 x (有號定點, XF 位元小數)。"""
    n = v >> XF                                    # floor
    f = v & ((1 << XF) - 1)
    idx, t = f >> (XF - S), f & ((1 << (XF - S)) - 1)
    y = eval_poly('exp2', idx, t, XF - S)          # 2^f ∈ [1, 2)
    return round_fp32(y, n)


def sfu_exp2(x):
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0xFF:
        return QNAN if fr else (0 if s else 0x7F800000)
    if ex >= 127 + 7:                              # |x| >= 128
        return 0 if s else 0x7F800000
    return exp2_fixed(to_fixed(x))


def sfu_exp(x):
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0xFF:
        return QNAN if fr else (0 if s else 0x7F800000)
    if ex >= 127 + 7:                              # |x| >= 128: e^x 溢位或 FTZ
        return 0 if s else 0x7F800000
    v = to_fixed(x)
    w = (v * LOG2E) >> 48                          # x · log2(e), 定點 XF 位元小數
    return exp2_fixed(w)


RF = 48                                            # sinπ / cosπ 範圍縮減的定點小數位元


def sfu_trig(x, cos):
    """sinπ(x) (cos = 0) / cosπ(x) (cos = 1)。範圍縮減在 48 位元小數的定點上精確進行 (|x| < 2^23, 指數 >= -25):
    u = |x| mod 2 (cos 先加 0.5), u >= 1 → 變號且 u -= 1, u > 0.5 → u = 1 - u, 得 r ∈ [0, 0.5];
    結果 = 4 · g(r) · r (g = sin(πr) / (4r) 查表), r 保持全精度 → 結果的相對精度不受 r 很小影響。"""
    s, ex, fr = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    if ex == 0xFF:
        return QNAN
    if ex == 0:
        return 0x3F800000 if cos else (s << 31)        # cosπ(0) = 1, sinπ(±0) = ±0
    e, m = ex - 127, (1 << 23) | fr
    if e >= 23:                                        # 整數: sinπ = ±0 (x 的符號), cosπ = (-1)^x
        if not cos:
            return s << 31
        odd = (e == 23) and (m & 1)
        return 0xBF800000 if odd else 0x3F800000
    if e < -25:                                        # |x| < 2^-25
        if cos:
            return 0x3F800000                          # 1 - (πx)²/2 捨入為 1
        rr, er, neg = m, 23 - e, s                     # r = m · 2^-(23 - e) (不需縮減)
    else:
        v = (m << (e + 25)) & ((1 << (RF + 1)) - 1)    # |x| · 2^48 mod 2^49 (= |x| mod 2)
        neg = 0 if cos else s                          # cosπ 為偶函數
        if cos:
            v = (v + (1 << (RF - 1))) & ((1 << (RF + 1)) - 1)
        if v >= (1 << RF):
            neg ^= 1
            v -= 1 << RF
        if v > (1 << (RF - 1)):
            v = (1 << RF) - v
        if v == 0:
            return 0 if cos else (s << 31)             # sinπ(整數) = ±0 (x 的符號); cosπ(n + 1/2) = +0
        if v == (1 << (RF - 1)):
            return (neg << 31) | 0x3F800000            # r = 1/2: ±1
        rr, er = v, RF
    # g 查表: r · 256 的整數部分為索引, 小數部分取 25 位元為 t
    rs = rr << (RF - er) if er <= RF else rr >> (er - RF)   # r · 2^48 (極小的 r 只用來查表)
    idx, t = rs >> 40, (rs >> 15) & ((1 << 25) - 1)
    g = eval_poly('sinpi', idx, t, 25)                 # g · 2^30
    return (neg << 31) | round_fp32(g * rr, 2 - er)    # 4 · g · r


def sfu_sinpi(x):
    return sfu_trig(x, 0)


def sfu_cospi(x):
    return sfu_trig(x, 1)


FUNCS = {'recip': (sfu_recip, lambda v: 1.0 / v), 'rsqrt': (sfu_rsqrt, lambda v: 1.0 / math.sqrt(v)),
         'exp2': (sfu_exp2, lambda v: 2.0 ** v), 'exp': (sfu_exp, math.exp),
         'sinpi': (sfu_sinpi, lambda v: _sinpi(v)), 'cospi': (sfu_cospi, lambda v: _sinpi(v + 0.5) if abs(v) < 2 ** 52 else 1.0)}


def _sinpi(v):
    """雙精度參考 (先精確縮減到 [-1, 1] 再乘 π, 避免 sin(π·v) 的大引數誤差)。"""
    from fractions import Fraction
    q = Fraction(v) % 2
    if q > 1:
        q -= 2
    return math.sin(math.pi * float(q)) if abs(q) not in (Fraction(1), Fraction(0)) else 0.0


def ulp_err(got, x, ref):
    """got (FP32 位元) 相對精確值 ref(x) 的誤差, 以 got 所在 binade 的 FP32 ulp 為單位。"""
    g = float(u2f(got))
    try:
        r = ref(float(u2f(x)))
    except OverflowError:
        r = math.inf
    if r == 0 or not math.isfinite(r) or abs(r) < 2 ** -126:
        return 0.0 if (g == 0 or not math.isfinite(g)) else 1e9
    if not math.isfinite(g) or g == 0:
        return 1e9 if abs(r) < 2 ** 127 * (2 - 2 ** -23) else 0.0
    e = math.floor(math.log2(abs(r)))
    return abs(g - r) / 2 ** (e - 23)


def check(n):
    rng = np.random.default_rng(1)
    worst = {}
    # recip / rsqrt: 誤差只與尾數 (rsqrt 再加指數奇偶) 有關 → 對兩個指數窮舉所有尾數
    for name in ('recip', 'rsqrt'):
        fn, ref = FUNCS[name]
        w = 0.0
        for ex in ((127, 128) if name == 'rsqrt' else (127,)):
            for fr in range(1 << 23):
                x = (ex << 23) | fr
                w = max(w, ulp_err(fn(x), x, ref))
        worst[name] = w
    for name in ('exp2', 'exp'):
        fn, ref = FUNCS[name]
        w = 0.0
        lim = 127.0 if name == 'exp2' else 87.0
        xs = np.concatenate([rng.uniform(-lim, lim, n), rng.uniform(-1, 1, n // 4),
                             rng.uniform(-2 ** -10, 2 ** -10, n // 8)]).astype(np.float32)
        for v in xs:
            x = int(np.frombuffer(v.tobytes(), dtype=np.uint32)[0])
            w = max(w, ulp_err(fn(x), x, ref))
        worst[name] = w
    return worst


def write_tables(path):
    W = C0W + C1W + C2W
    with open(path, 'w') as f:
        f.write('// SPDX-License-Identifier: Apache-2.0\n')
        f.write(f'// 由 dv/fp/sfu_ref.py tables 產生 — 請勿手改。S = {S}, F = {F}, 係數 (c0, c1, c2) 寬度 {C0W} / {C1W} / {C2W}\n')
        lr = LOG2E >> LOG2E_TZ
        f.write(f"localparam int SFU_LOG2E_SH = {48 - LOG2E_TZ};\n")
        f.write(f"localparam logic [{lr.bit_length() - 1}:0] SFU_LOG2E = {lr.bit_length()}'d{lr};   // round(log2(e) · 2^48) >> {LOG2E_TZ}\n")
        for name in TABLES:
            f.write(f'localparam logic [{W - 1}:0] SFU_TAB_{name.upper()} [{1 << S}] = \'{{\n')
            ent = [f"  {{{C0W}'h{c0 & ((1 << C0W) - 1):0{C0W // 4}x}, {C1W}'h{c1 & ((1 << C1W) - 1):x}, {C2W}'h{c2 & ((1 << C2W) - 1):x}}}"
                   for c0, c1, c2 in table(name)]
            f.write(',\n'.join(ent) + '\n};\n')


if __name__ == '__main__':
    if sys.argv[1] == 'tables':
        write_tables(sys.argv[2])
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        for k, v in check(n).items():
            print(f'{k:6s} max error {v:.3f} ulp')
