#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 產生浮點函式庫的測試向量 (位元精確期望值來自 fp_ref.py): 隨機位元、相近指數 (大量相消)、捨入平手、
# 溢位、FTZ 邊界、特殊值 (±0 / 次正規 / ±inf / NaN / 最大 / 最小正規) 的所有組合。
# 輸出: <dir>/{add,mul}_{fp32,bf16}.hex 每行 "a b y"; cvt_fp32_bf16.hex / cvt_bf16_fp32.hex 每行 "a y"
# 用法: gen_vectors.py <dir> [seed] [n_random]
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_ref import FP32, BF16, fp_add, fp_mul, cvt, pack


def specials(fmt):
    e, m = fmt
    emax = (1 << e) - 1
    v = []
    for s in (0, 1):
        v += [pack(s, 0, 0, fmt), pack(s, 0, 1, fmt), pack(s, 0, (1 << m) - 1, fmt),       # ±0, 次正規
              pack(s, 1, 0, fmt), pack(s, 1, (1 << m) - 1, fmt),                            # 最小正規
              pack(s, (1 << (e - 1)) - 1, 0, fmt), pack(s, (1 << (e - 1)) - 1, 1, fmt),     # 1.0, 1 + ulp
              pack(s, emax - 1, (1 << m) - 1, fmt), pack(s, emax - 1, 0, fmt),             # 最大
              pack(s, emax, 0, fmt), pack(s, emax, 1, fmt), pack(s, emax, 1 << (m - 1), fmt)]  # inf, sNaN, qNaN
    return v


def rnd_norm(fmt, r, elo=1, ehi=None):
    e, m = fmt
    ehi = (1 << e) - 2 if ehi is None else ehi
    return pack(r.getrandbits(1), r.randint(elo, ehi), r.getrandbits(m), fmt)


def pairs(fmt, r, n):
    e, m = fmt
    emax = (1 << e) - 2
    bias = (1 << (e - 1)) - 1
    sp = specials(fmt)
    out = [(x, y) for x in sp for y in sp]
    for _ in range(n):
        kind = r.randint(0, 9)
        if kind <= 2:                                   # 任意位元
            out.append((r.getrandbits(e + m + 1), r.getrandbits(e + m + 1)))
        elif kind == 3:                                 # 指數相近 (相消 / 對齊)
            a = rnd_norm(fmt, r)
            ea = (a >> m) & ((1 << e) - 1)
            eb = min(emax, max(1, ea + r.randint(-3, 3)))
            out.append((a, pack(r.getrandbits(1), eb, r.getrandbits(m), fmt)))
        elif kind == 4:                                 # 近乎相等的反號數 (大量相消, 含精確相消)
            a = rnd_norm(fmt, r)
            d = r.choice([0, 1, 2, 3, r.getrandbits(m)])
            b = (a ^ (1 << (e + m))) + (d if r.getrandbits(1) else -d)
            out.append((a, b & ((1 << (e + m + 1)) - 1)))
        elif kind == 5:                                 # 指數差很大 (只剩 sticky)
            a = rnd_norm(fmt, r, elo=m + 5)
            ea = (a >> m) & ((1 << e) - 1)
            out.append((a, pack(r.getrandbits(1), r.randint(1, ea - (m + 3)), r.getrandbits(m), fmt)))
        elif kind == 6:                                 # 乘積溢位 / FTZ 邊界
            hi = r.getrandbits(1)
            ea = r.randint(bias + bias // 2, emax) if hi else r.randint(1, bias // 2)
            eb = (2 * bias - ea + r.randint(-2, 2)) if r.getrandbits(1) else ((emax + bias - ea + r.randint(-2, 2)) if hi else (bias + 1 - ea + r.randint(-2, 2)))
            eb = min(emax, max(1, eb))
            out.append((pack(r.getrandbits(1), ea, r.getrandbits(m), fmt), pack(r.getrandbits(1), eb, r.getrandbits(m), fmt)))
        elif kind == 7:                                 # 捨入平手: b = a 的半個 ulp (加) / 尾數低位多 0 (乘)
            a = rnd_norm(fmt, r, elo=m + 3)
            ea = (a >> m) & ((1 << e) - 1)
            b = pack(r.getrandbits(1), ea - m - 1, r.choice([0, 0, 1]), fmt)
            out.append((a, b))
        elif kind == 8:                                 # 尾數全 1: 加上略大於半個 ulp 的同號數 → 捨入進位到指數
            sa = r.getrandbits(1)
            ea = r.randint(m + 3, emax)
            a = pack(sa, ea, (1 << m) - 1, fmt)
            b = pack(sa, ea - m - 1, r.getrandbits(m) | 1, fmt) if r.getrandbits(1) else rnd_norm(fmt, r)
            out.append((a, b))
        else:                                           # 小整數 (乘法平手常見)
            out.append((pack(0, bias + r.randint(0, 4), r.getrandbits(3) << (m - 3), fmt),
                        pack(r.getrandbits(1), bias + r.randint(0, 4), r.getrandbits(3) << (m - 3), fmt)))
    # 乘法捨入進位到 2.0 (有效位元乘積在 [1, 2) 且捨入後變成 2): a 的尾數接近全 1, b 接近 1
    for _ in range(max(20, n // 200)):
        ma = (1 << (m + 1)) - 1 - r.randint(0, 3)
        for j in range(0, 64):
            mb = (1 << m) + j
            p = ma * mb
            if p >> (2 * m + 1):
                break
            q, rest = p >> m, p & ((1 << m) - 1)
            if q == (1 << (m + 1)) - 1 and (rest >> (m - 1)) & 1:
                ea = r.randint(bias - 20, bias + 20)
                out.append((pack(r.getrandbits(1), ea, ma & ((1 << m) - 1), fmt), pack(r.getrandbits(1), bias, mb & ((1 << m) - 1), fmt)))
                break
    return out


def main():
    d = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    r = random.Random(seed)
    os.makedirs(d, exist_ok=True)
    for name, fmt in (('fp32', FP32), ('bf16', BF16)):
        w = (sum(fmt) + 1 + 3) // 4
        ps = pairs(fmt, r, n)
        for op, fn in (('add', fp_add), ('mul', fp_mul)):
            with open(f'{d}/{op}_{name}.hex', 'w') as f:
                for a, b in ps:
                    f.write(f'{a:0{w}x} {b:0{w}x} {fn(a, b, fmt):0{w}x}\n')
    with open(f'{d}/cvt_fp32_bf16.hex', 'w') as f:
        xs = specials(FP32) + [r.getrandbits(32) for _ in range(n)]
        xs += [pack(r.getrandbits(1), r.randint(1, 254), (r.getrandbits(7) << 16) | (1 << 15) | (r.getrandbits(1) * r.getrandbits(15)), FP32) for _ in range(n // 4)]   # 平手附近
        xs += [pack(r.getrandbits(1), r.choice([253, 254]), (0x7F << 16) | r.getrandbits(16), FP32) for _ in range(200)]  # 捨入溢位成 inf
        for a in xs:
            f.write(f'{a:08x} {cvt(a, FP32, BF16):04x}\n')
    with open(f'{d}/cvt_bf16_fp32.hex', 'w') as f:
        xs = list(range(1 << 16))                       # BF16 → FP32: 窮舉 (打亂順序, 讓每個位元兩個方向都翻轉)
        r.shuffle(xs)
        for a in xs:
            f.write(f'{a:04x} {cvt(a, BF16, FP32):08x}\n')


if __name__ == '__main__':
    main()
