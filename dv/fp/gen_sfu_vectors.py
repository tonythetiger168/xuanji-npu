#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SFU 測試向量: 每行 "op x y" (y 為 sfu_ref.py 的位元精確結果)。特殊值、查表段邊界、範圍邊界、隨機。
# 用法: gen_sfu_vectors.py <檔案> [seed] [n]
import os, random, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfu_ref as R

def f2u(f): return struct.unpack('<I', struct.pack('<f', f))[0]

def main():
    out, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    r = random.Random(seed)
    fns = [R.sfu_recip, R.sfu_rsqrt, R.sfu_exp2, R.sfu_exp, R.sfu_sinpi, R.sfu_cospi]
    sp = []
    for s in (0, 1):
        for ex in (0, 1, 2, 126, 127, 128, 129, 133, 134, 253, 254, 255):
            for fr in (0, 1, 0x400000, 0x7FFFFF, 0x7F0000, 0x00FFFF, 0x010000):
                sp.append((s << 31) | (ex << 23) | fr)
    xs = []
    for op in range(6):
        for x in sp: xs.append((op, x))
    # sinπ / cosπ: 整數、半整數、零點附近、極小 / 極大的引數 (縮減邊界)
    for op in (4, 5):
        for v in [0.5, 1.0, 1.5, 2.0, 0.25, 0.75, 3.0, -0.5, -1.0, -2.5, 8388607.0, 8388608.0, 8388609.0, 4194303.5,
                  1 + 2 ** -23, 0.5 - 2 ** -25, 0.5 + 2 ** -24, 1 - 2 ** -24, 2 ** -25, 2 ** -26, 2 ** -24, 1e-38, 2 ** -100]:
            for sg in (1, -1):
                xs.append((op, f2u(sg * v)))
    for _ in range(n):
        op = r.randint(0, 5)
        k = r.randint(0, 5)
        if k == 0:
            x = r.getrandbits(32)
        elif op <= 1:
            # 段邊界: 索引任意, 段內偏移 t = 0 / 最大 / 隨機
            idx = r.getrandbits(7); t = r.choice([0, 0xFFFF, r.getrandbits(16)])
            ex = r.randint(1, 254) if k < 4 else r.choice([1, 2, 253, 254])
            x = (r.getrandbits(1) if op == 0 else (0 if k != 5 else 1)) << 31 | ex << 23 | idx << 16 | t
        elif op >= 4:
            v = r.choice([r.uniform(-4, 4), r.uniform(-1e7, 1e7), r.uniform(-1e-3, 1e-3), r.uniform(-2 ** -20, 2 ** -20),
                          r.randint(-1000, 1000) * 0.5 + r.choice([0, 2 ** -20, -2 ** -20, 2 ** -10]),
                          r.uniform(-2 ** 23, 2 ** 23), r.uniform(-2 ** 24, 2 ** 24)])
            x = f2u(v) if r.getrandbits(3) else (r.getrandbits(1) << 31 | r.randint(1, 160) << 23 | r.getrandbits(23))
        else:
            lim = 128.0 if op == 2 else 89.0
            v = r.choice([r.uniform(-lim, lim), r.uniform(-1, 1), r.uniform(-1e-6, 1e-6),
                          float(r.randint(-130, 130)) + r.choice([0, 1e-7, -1e-7, 0.5]),
                          r.uniform(126, 128) * r.choice([1, -1]), r.uniform(87, 89) * r.choice([1, -1])])
            x = f2u(v)
        xs.append((op, x))
    with open(out, 'w') as f:
        for op, x in xs:
            f.write(f'{op:x} {x:08x} {fns[op](x):08x}\n')

if __name__ == '__main__':
    main()
