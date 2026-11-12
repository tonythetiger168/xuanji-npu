#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# npu_vec_eng 的測試: 一串 VEC 命令, 每個命令的程式放在 L2 (每 line 4 個指令, 第 i 個在 bit [128i +: 77]),
# 執行結果 (暫存器檔跨命令保持、每個命令的 err、最後的 L2 內容) 由 vec_ref.py 計算。
# 輸出: <base>.mem (初始 L2, 含程式), <base>.cmd ("line n derr"), <base>.exp (最後的資料區 line)
# 用法: gen_eng_vectors.py <base> [seed] [命令數]
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vec_ref as V
import gen_vec_vectors as G

M26 = (1 << 26) - 1


def main():
    base = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    r = random.Random(seed)
    lines = [r.getrandbits(26) for _ in range(8)] + list(range(100, 116)) + [M26 - i for i in range(4)]
    mem0 = {ln: G.rnd_line(r) for ln in lines}
    ops = ['LD'] * 6 + ['ST'] * 3 + ['ADD', 'SUB', 'MUL'] * 2 + ['MAX', 'SFU', 'SPLAT', 'RSUM', 'RMAX', 'NOP', 'TOPK']
    cmds, progs = [], []
    used = set(lines)
    first = True
    for c in range(nc):
        n = r.choice([1, 2, 3, 4, 5, 7, 8, 9, r.randint(1, 60), r.randint(60, 200)])
        big = (c == nc // 2)                      # 一個很長的命令 (cmd_n 的高位元; 多半是 NOP)
        if big:
            n = 65535 - r.randint(0, 255)          # cmd_n 的高 8 位元全為 1
        prog = []
        if first:                                 # 第一個命令先初始化所有暫存器
            prog = [V.enc('SPLAT', vd=i, imm=G.rnd_word(r)) for i in range(V.NV)]
            n = max(n, V.NV)
            first = False
        while len(prog) < n:
            if big and r.random() < 0.98:
                prog.append(V.enc('NOP'))
                continue
            if r.random() < 0.03:
                prog.append(V.enc(r.randint(12, 15), vd=r.getrandbits(3)))
                continue
            op = r.choice(ops)
            fn = {'LD': r.randint(0, 6), 'ST': r.randint(0, 1), 'SFU': r.randint(0, 5), 'TOPK': r.randint(0, 1)}.get(op, r.getrandbits(3))
            imm = G.rnd_word(r) if op == 'SPLAT' else r.choice([120, 126, 127, 128, 135, 255]) | (r.getrandbits(24) << 8)
            prog.append(V.enc(op, fn=fn, vd=r.getrandbits(3), va=r.getrandbits(3), vb=r.getrandbits(3),
                              part=r.getrandbits(3), imm=imm, line=r.choice(lines)))
        np_ = (n + 3) // 4
        while True:                                # 程式位置: 任意 26 bit (含跨越 0x3FFFFFF 回繞), 不與資料 / 其他程式重疊
            pl = r.choice([r.getrandbits(26), (M26 - r.randint(0, 3)) & M26, r.getrandbits(26) & ~0xFFFF])
            span = {(pl + i) & M26 for i in range(np_)}
            if not (span & used):
                break
        used |= span
        for i in range(0, n, 4):
            mem0[(pl + i // 4) & M26] = sum(prog[i + j] << (128 * j) for j in range(min(4, n - i)))
        cmds.append((pl, n))
        progs.append(prog)
    mem = {k: v for k, v in mem0.items()}
    v = V.Vec()
    res = []
    for (pl0, n), prog in zip(cmds, progs):
        v.err = 0
        for ins in prog:
            st = {}
            f = V.dec(ins)
            if V.legal(f) and V.OPN[f['op']] == 'ST':
                v.step(ins, st)
                mem.update(st)
            else:
                v.step(ins, mem)
        res.append((pl0, n, v.err))
    with open(base + '.mem', 'w') as fm:
        for ln, d in sorted(mem0.items()):
            fm.write('%07x %0128x\n' % (ln, d))
    with open(base + '.cmd', 'w') as fc:
        for pl0, n, e in res:
            fc.write('%07x %d %d\n' % (pl0, n, e))
    with open(base + '.exp', 'w') as fe:
        for ln in sorted(set(lines)):
            fe.write('%07x %0128x\n' % (ln, mem.get(ln, 0)))
    print('eng: %d commands, %d instructions' % (len(cmds), sum(n for _, n in cmds)))


if __name__ == '__main__':
    main()
