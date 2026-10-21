#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Xuanji 共用浮點函式庫的位元精確參考模型 (以整數 / 有理數精確運算, 不依賴主機浮點)。
#
# 格式 (E, M): 1 bit 符號 + E bit 指數 (bias = 2^(E-1) - 1) + M bit 尾數。FP32 = (8, 23), BF16 = (8, 7)。
# 語意 (AI 加速器常見的 FTZ / DAZ, 與 rtl/fp/*.sv 相同; 細節見 docs/fp.md):
#   - 輸入 DAZ: 指數 0 (含次正規數) 一律當作 ±0。
#   - 指數全 1: 尾數 0 = ±inf, 否則 NaN。任何 NaN 輸入、inf - inf、0 × inf → 標準 qNaN (0, 全 1, 尾數最高位 1)。
#   - 捨入: round-to-nearest-even, 先以無限指數範圍把精確值捨入到 M+1 個有效位元,
#     再判斷範圍: 指數 > emax → ±inf; 指數 < emin (結果小於最小正規數) → ±0 (FTZ, 保留符號)。
#   - 加法的零符號: 精確和為 0 時 → +0, 但 (-0) + (-0) = -0。乘法: 符號 = 兩輸入符號 XOR。
import sys
from fractions import Fraction

FP32 = (8, 23)
BF16 = (8, 7)


def fields(x, fmt):
    e, m = fmt
    s = (x >> (e + m)) & 1
    ex = (x >> m) & ((1 << e) - 1)
    fr = x & ((1 << m) - 1)
    return s, ex, fr


def pack(s, ex, fr, fmt):
    e, m = fmt
    return (s << (e + m)) | (ex << m) | fr


def qnan(fmt):
    e, m = fmt
    return pack(0, (1 << e) - 1, 1 << (m - 1), fmt)


def inf(s, fmt):
    e, m = fmt
    return pack(s, (1 << e) - 1, 0, fmt)


def classify(x, fmt):
    """回傳 ('zero'|'norm'|'inf'|'nan', sign, value as Fraction or None)"""
    e, m = fmt
    s, ex, fr = fields(x, fmt)
    bias = (1 << (e - 1)) - 1
    if ex == (1 << e) - 1:
        return ('nan' if fr else 'inf'), s, None
    if ex == 0:
        return 'zero', s, Fraction(0)              # DAZ
    v = Fraction((1 << m) | fr, 1 << m) * (Fraction(2) ** (ex - bias))
    return 'norm', s, (-v if s else v)


def round_to(v, sign_if_zero, fmt):
    """把精確有理數 v 依 RNE 捨入到格式 fmt (FTZ, 溢位成 inf)。"""
    e, m = fmt
    bias = (1 << (e - 1)) - 1
    emin, emax = 1 - bias, (1 << e) - 2 - bias
    if v == 0:
        return pack(sign_if_zero, 0, 0, fmt)
    s = 1 if v < 0 else 0
    a = -v if v < 0 else v
    # 找指數 k: 2^k <= a < 2^(k+1)
    k = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** k > a:
        k -= 1
    elif Fraction(2) ** (k + 1) <= a:
        k += 1
    # 縮放到 [2^m, 2^(m+1)) 後 RNE 捨入成整數
    t = a * (Fraction(2) ** (m - k))
    q, r = divmod(t.numerator, t.denominator)
    rem = Fraction(r, t.denominator)
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (q & 1)):
        q += 1
    if q == (1 << (m + 1)):                         # 捨入進位到下一個 2 的冪
        q >>= 1
        k += 1
    if k > emax:
        return inf(s, fmt)
    if k < emin:
        return pack(s, 0, 0, fmt)                    # FTZ
    return pack(s, k + bias, q & ((1 << m) - 1), fmt)


def fp_add(a, b, fmt=FP32):
    ca, sa, va = classify(a, fmt)
    cb, sb, vb = classify(b, fmt)
    if ca == 'nan' or cb == 'nan':
        return qnan(fmt)
    if ca == 'inf' and cb == 'inf':
        return inf(sa, fmt) if sa == sb else qnan(fmt)
    if ca == 'inf':
        return inf(sa, fmt)
    if cb == 'inf':
        return inf(sb, fmt)
    v = va + vb
    zero_sign = 1 if (ca == 'zero' and cb == 'zero' and sa and sb) else 0
    if v == 0 and not (ca == 'zero' and cb == 'zero'):
        zero_sign = 0                                # x + (-x) = +0 (RNE)
    return round_to(v, zero_sign, fmt)


def fp_mul(a, b, fmt=FP32):
    ca, sa, va = classify(a, fmt)
    cb, sb, vb = classify(b, fmt)
    s = sa ^ sb
    if ca == 'nan' or cb == 'nan':
        return qnan(fmt)
    if (ca == 'inf' and cb == 'zero') or (ca == 'zero' and cb == 'inf'):
        return qnan(fmt)
    if ca == 'inf' or cb == 'inf':
        return inf(s, fmt)
    return round_to(va * vb, s, fmt)


def cvt(x, src, dst):
    """格式轉換 (例如 FP32 → BF16 捨入, BF16 → FP32 精確)。"""
    c, s, v = classify(x, src)
    if c == 'nan':
        return qnan(dst)
    if c == 'inf':
        return inf(s, dst)
    return round_to(v, s, dst)


if __name__ == '__main__':
    # 自我檢查: 與主機 IEEE float32 比對 (只比對兩者語意相同的情況: 輸入與結果都不是次正規數, 結果不是 NaN)
    import random, struct
    def f2u(f): return struct.unpack('<I', struct.pack('<f', f))[0]
    def u2f(u): return struct.unpack('<f', struct.pack('<I', u))[0]
    random.seed(1)
    bad = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for _ in range(n):
        a, b = random.getrandbits(32), random.getrandbits(32)
        for op, ref in (('add', fp_add), ('mul', fp_mul)):
            fa, fb = u2f(a), u2f(b)
            if any(((x >> 23) & 0xFF) == 0 for x in (a, b)):
                continue
            import numpy as np
            with np.errstate(all='ignore'):
                h = np.float32(fa) + np.float32(fb) if op == 'add' else np.float32(fa) * np.float32(fb)
            hu = f2u(float(h))
            if ((hu >> 23) & 0xFF) == 0 or h != h:
                continue
            r = ref(a, b)
            if r != hu:
                bad += 1
                if bad < 5:
                    print(f'{op} {a:08x} {b:08x} ref={r:08x} host={hu:08x}')
    print('self-check mismatches:', bad)
    sys.exit(1 if bad else 0)
