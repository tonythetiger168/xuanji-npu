#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# npu_vec 的受約束隨機測試: 隨機 L2 內容 (各格式的特殊值 / 邊界 / 隨機位元) + 隨機指令串 (含非法編碼與 reset),
# 期望值由 vec_ref.py 逐指令計算。輸出 <base>.mem / <base>.ins (格式見 vec_ref.write_tb)。
# 用法: gen_vec_vectors.py <base> [seed] [n_ins]
import os, random, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vec_ref as V

def f2u(f): return struct.unpack('<I', struct.pack('<f', f))[0]

FP32_SP = [0, 0x80000000, 1, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
           0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF, 0x3F000000, 0x40000000, 0x4B000000,
           0x3F7FFFFF, 0x3F800001, 0x42FE0000, 0x43000000, 0xC3000000, 0x3E800000, 0x41200000]
INT32_SP = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, (1 << 25) + 2, (1 << 25) + 6,
            0xFF000001, 0x00FFFFFF, 0x01000001, 0x7FFFFFC0, 0x7FFFFF80, 0x80000001]


def rnd_word(r):
    k = r.randint(0, 4)
    if k == 0:
        return r.getrandbits(32)
    if k == 1:
        return r.choice(FP32_SP)
    if k == 2:
        return f2u(r.uniform(-8, 8)) if r.random() < 0.5 else f2u(r.gauss(0, 100))
    if k == 3:
        return (r.getrandbits(1) << 31) | (r.randint(100, 154) << 23) | r.getrandbits(23)
    x = r.random()
    if x < 0.3:
        return r.randint(-4, 4) & 0xFFFFFFFF          # 小整數 (INT32 → FP32 指數 127 附近)
    return r.choice(INT32_SP) if x < 0.6 else (r.getrandbits(1) << 31) | r.getrandbits(r.randint(1, 31))


def rnd_line(r):
    k = r.randint(0, 3)
    if k == 0:
        return r.getrandbits(512)
    if k == 1:                                       # 32 bit 元素
        return sum(rnd_word(r) << (32 * i) for i in range(16))
    if k == 2:                                       # 16 bit 元素: BF16 / FP16 的特殊值與隨機
        sp = [0, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x007F, 0x0080,
              0x3F80, 0x7F7F, 0x7F80, 0x7FC0, 0x8001, 0x83FF]
        return sum((r.choice(sp) if r.random() < 0.4 else r.getrandbits(16)) << (16 * i) for i in range(32))
    sp8 = [0, 0x80, 0x01, 0x07, 0x08, 0x7E, 0x7F, 0xFF, 0x7C, 0xFC, 0x7D, 0x03, 0x04, 0x38, 0x3C]
    return sum((r.choice(sp8) if r.random() < 0.4 else r.getrandbits(8)) << (8 * i) for i in range(64))


def main():
    base = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    r = random.Random(seed)
    lines = [r.getrandbits(26) for _ in range(12)] + [0, (1 << 26) - 1] + list(range(100, 118))
    mem0 = {ln: rnd_line(r) for ln in lines}
    mem = dict(mem0)
    scales = [0, 1, 2, 100, 120, 126, 127, 128, 133, 140, 200, 240, 250, 253, 254, 255]
    prog = [V.enc('SPLAT', vd=i, imm=rnd_word(r)) for i in range(V.NV)]
    ops = (['LD'] * 6 + ['ST'] * 3 + ['ADD', 'SUB', 'MUL'] * 2 + ['MAX', 'SFU', 'SFU', 'SPLAT', 'RSUM', 'RMAX', 'NOP'])
    for _ in range(n):
        x = r.random()
        if x < 0.004:
            prog.append(V.RESET)
            continue
        if x < 0.008:
            prog.append(V.ERRCLR)
            continue
        if x < 0.02:                                 # 非法編碼
            k = r.randint(0, 3)
            op, fn = [(r.randint(11, 15), r.getrandbits(3)), (1, 7), (2, r.randint(2, 7)), (7, r.randint(6, 7))][k]
            prog.append(V.enc(op, fn=fn, vd=r.getrandbits(3), va=r.getrandbits(3), vb=r.getrandbits(3),
                              part=r.getrandbits(3), imm=r.getrandbits(32), line=r.choice(lines)))
            continue
        op = r.choice(ops)
        fn = {'LD': r.randint(0, 6), 'ST': r.randint(0, 1), 'SFU': r.randint(0, 5)}.get(op, r.getrandbits(3))
        imm = rnd_word(r) if op == 'SPLAT' else (r.choice(scales) | (r.getrandbits(24) << 8) if r.random() < 0.8
                                                 else r.getrandbits(32))
        ln = r.choice(lines) if r.random() < 0.95 else r.getrandbits(26)
        prog.append(V.enc(op, fn=fn, vd=r.getrandbits(3), va=r.getrandbits(3), vb=r.getrandbits(3),
                          part=r.getrandbits(3), imm=imm, line=ln))
    if seed == 1:                                    # 窮舉: 所有 FP8 / FP4 碼 × 所有比例, 所有 16 bit 碼 (BF16 / FP16)
        ex = {}
        for k in range(4):                           # 4 line × 64 byte = 256 個 FP8 碼; FP4: 每 line 128 個 nibble
            ex[0x200000 + k] = sum(((64 * k + i) & 0xFF) << (8 * i) for i in range(64))
        for k in range(2048):                        # 2048 line × 32 = 65536 個 16 bit 碼
            ex[0x210000 + k] = sum((32 * k + i) << (16 * i) for i in range(32))
        mem0.update(ex)
        mem.update(ex)
        for sc in range(256):
            for k in range(4):
                for p in range(4):
                    prog.append(V.enc('LD', fn=4 + (p + sc) % 2, vd=r.getrandbits(3), part=p, imm=sc, line=0x200000 + k))
                    prog.append(V.enc('LD', fn=4 + (p + sc + 1) % 2, vd=r.getrandbits(3), part=p, imm=sc,
                                      line=0x200000 + k))
            prog.append(V.enc('LD', fn=6, vd=r.getrandbits(3), part=sc % 8, imm=sc, line=0x200000))
        for k in range(2048):
            for p in range(2):
                prog.append(V.enc('LD', fn=1 + (k + p) % 2, vd=r.getrandbits(3), part=p, line=0x210000 + k))
                prog.append(V.enc('LD', fn=1 + (k + p + 1) % 2, vd=r.getrandbits(3), part=p, line=0x210000 + k))
    recs, _ = V.run(prog, mem)
    V.write_tb(base, mem0, recs)
    print('vec: %d instructions, %d lines' % (len(recs), len(mem0)))


if __name__ == '__main__':
    main()
