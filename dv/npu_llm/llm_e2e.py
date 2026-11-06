#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# NPU Subsystem 端到端 (KV / MOE 命令 + 間接 DMA + GEMM), 期望值以 numpy 計算:
#   paged : paged attention 的 KV cache — KV ALLOC / SHARE (prefix 共享) 配頁, KV MAP 產生位址表, 間接 DMA ST 把連續的 K
#           散到外部記憶體的頁, 再以 MAP + 間接 DMA LD 把序列 1 的頁 (含共享頁與一個未映射的 slot) gather 回 L2,
#           每頁一個 GEMM 算 Q·K^T; 最後 FREE_SEQ / FREE / ALLOC 驗證空閒表 (FIFO 重複使用)。
#           比對: 外部記憶體中每頁的內容、各 KV 命令的結果 line、位址表、gather 結果、注意力分數 (int32, numpy matmul)。
#   moe   : MoE 層 — router (numpy) 的 top-k (每 token 2 個 expert, softmax 權重 FP32), DMA 載入後以 MOE 命令產生
#           dispatch 表 (容量模式與無丟棄模式各一次), 再以間接 DMA LD 依 gather 表把 token 收集到各 expert 的連續區。
#           比對: 表格 (以 numpy 另外實作的規格) 與收集結果; 並以硬體的 combine 表 + 收集結果 + 每個 expert 的運算
#           (numpy) 算出 MoE 層輸出, 與直接以 numpy 算的 y_t = Σ_k w_k · f_e(x_t) 比較 (容量模式時排除被丟棄的項)。
# 用法: llm_e2e.py <TB 執行檔> <工作目錄> [N=8] [seed=1]
import os, struct, subprocess, sys
import numpy as np

FF = 0xFFFFFFFF
OP_GEMM, OP_LD, OP_ST, OP_NOP, OP_VEC, OP_KV, OP_MOE = range(7)


def line_from_words(ws):
    return sum((int(w) & FF) << (32 * i) for i, w in enumerate(ws))


def words(v, n=16):
    return [(v >> (32 * i)) & FF for i in range(n)]


def line_from_bytes(bs):
    return sum((int(b) & 0xFF) << (8 * i) for i, b in enumerate(bs))


def f2u(f):
    return struct.unpack('<I', struct.pack('<f', float(f)))[0]


def u2f(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


class Scn:
    def __init__(s):
        s.ext, s.cmds, s.outs, s.nid = {}, [], [], 0

    def cmd(s, op, ext=0, a=0, b=0, c=0, k=0, sa=0, sb=0, wt=1):
        s.cmds.append((op, wt, s.nid, ext & FF, a, b, c, k, sa, sb))
        s.nid += 1

    def kv(s, sub, seq, slot, out, src=0, k=0, mult=0):
        s.cmd(OP_KV, ext=(mult if sub == 4 else src), a=out, b=sub, c=seq | (slot << 8), k=k)

    def out(s, name, addr, n):
        s.outs.append((name, addr, n))

    def write(s, fn):
        with open(fn, 'w') as f:
            for a, v in sorted(s.ext.items()):
                f.write('E %x %0128x\n' % (a, v))
            for c in s.cmds:
                f.write('C ' + ' '.join('%x' % x for x in c) + '\n')
            for n, a, k in s.outs:
                f.write('O %s %x %x\n' % (n, a, k))


def run(tb, wd, name, scn, seed):
    sf, of = os.path.join(wd, name + '.scn'), os.path.join(wd, name + '.out')
    scn.write(sf)
    r = subprocess.run([tb, '+scn=' + sf, '+out=' + of, '+seed=%d' % seed], capture_output=True, text=True)
    log = r.stdout + r.stderr
    open(os.path.join(wd, name + '.log'), 'w').write(log)
    if 'TB PASS' not in log or 'Assertion failed' in log or '%Error' in log:
        print(log[-2000:])
        raise SystemExit('%s: TB FAIL' % name)
    res = {}
    for ln in open(of):
        n, i, v = ln.split()
        res.setdefault(n, {})[int(i)] = int(v, 16)
    cyc = [l for l in log.splitlines() if l.startswith('E2E: done')]
    return res, (cyc[0] if cyc else '')


class Chk:
    def __init__(s):
        s.n, s.bad = 0, 0

    def eq(s, what, got, exp):
        s.n += 1
        if got != exp:
            s.bad += 1
            if s.bad <= 20:
                print('MISMATCH %s: got %s exp %s' % (what, got if not isinstance(got, int) else '%x' % got,
                                                    exp if not isinstance(exp, int) else '%x' % exp))


# ------------------------------------------------------------------------------------------------------------
# paged attention
# ------------------------------------------------------------------------------------------------------------
def paged(tb, wd, N, rng, seed, chk):
    SPL = 64 // N
    D = 8 * SPL                       # head dim (K of the GEMM): 8 lines per operand
    LP = D // SPL                     # lines per page (= per GEMM B operand); page = N tokens
    CL = (4 * N * N + 63) // 64
    KV_NP = 256
    KV_BASE, X0, X1, QX = 0x10000000, 0x20000000, 0x20001000, 0x20002000
    O_RES, O_LST, O_G, O_SC = 0x30000000, 0x30001000, 0x30002000, 0x30003000
    RES, KD, KD1, LST0, LST1, LST2, G, QA, SC = 0x000, 0x100, 0x140, 0x180, 0x190, 0x1A0, 0x200, 0x300, 0x340
    n0, n1own = 5, 2                  # 序列 0: 5 頁; 序列 1: 共享前 3 頁 + 自己 2 頁; gather 6 個 slot (最後一個未映射)
    K0 = rng.integers(-128, 128, size=(n0 * N, D))
    K1 = rng.integers(-128, 128, size=(n1own * N, D))
    Q = rng.integers(-128, 128, size=(N, D))

    def bpage(K, p):                  # 一頁 (N 個 token) 的 B 運算元 layout: line l 的 byte (kk%SPL)*N + j = K[pN+j][kk]
        out = []
        for l in range(LP):
            bs = [0] * 64
            for kk in range(l * SPL, (l + 1) * SPL):
                for j in range(N):
                    bs[(kk % SPL) * N + j] = K[p * N + j][kk]
            out.append(line_from_bytes(bs))
        return out

    def aop(A):                       # A 運算元 layout: line l 的 byte (kk%SPL)*N + i = A[i][kk]
        out = []
        for l in range(LP):
            bs = [0] * 64
            for kk in range(l * SPL, (l + 1) * SPL):
                for i in range(N):
                    bs[(kk % SPL) * N + i] = A[i][kk]
            out.append(line_from_bytes(bs))
        return out

    s = Scn()
    for p in range(n0):
        for l, v in enumerate(bpage(K0, p)):
            s.ext[X0 + p * LP + l] = v
    for p in range(n1own):
        for l, v in enumerate(bpage(K1, p)):
            s.ext[X1 + p * LP + l] = v
    for l, v in enumerate(aop(Q)):
        s.ext[QX + l] = v
    # 頁管理參考 (FIFO 空閒表 + 水位, 與規格相同)
    bt, rc, fq, hw = {}, {}, [], [0]
    res = []                          # 每個 KV 命令的結果 line (word 0..2)

    def free_cnt():
        return KV_NP - hw[0] + len(fq)

    def alloc(seq, slot):
        if fq:
            pg = fq.pop(0)
        else:
            pg = hw[0]; hw[0] += 1
        bt[(seq, slot)] = pg; rc[pg] = 1
        return pg

    def unmap(seq, slot):
        pg = bt.pop((seq, slot)); rc[pg] -= 1
        if rc[pg] == 0:
            fq.append(pg)
        return pg

    r = 0
    for slot in range(n0):
        s.kv(0, 0, slot, RES + r); res.append((0, alloc(0, slot), free_cnt())); r += 1
    for slot in range(3):
        s.kv(3, 1, slot, RES + r, src=0)
        pg = bt[(0, slot)]; bt[(1, slot)] = pg; rc[pg] += 1
        res.append((0, pg, free_cnt())); r += 1
    for slot in range(3, 3 + n1own):
        s.kv(0, 1, slot, RES + r); res.append((0, alloc(1, slot), free_cnt())); r += 1
    s.kv(0, 1, 0, RES + r); res.append((2, 0, free_cnt())); r += 1          # 已有映射 → MAPPED
    s.cmd(OP_LD, ext=X0, a=KD, k=n0 * LP)
    s.cmd(OP_LD, ext=X1, a=KD1, k=n1own * LP)
    s.cmd(OP_LD, ext=QX, a=QA, k=LP)
    # prefill: 序列 0 全部、序列 1 自己的頁, MAP (mult = 每頁 line 數) + 間接 ST (每項 LP 個 line)
    s.kv(4, 0, 0, LST0, k=n0, mult=LP)
    s.cmd(OP_ST, ext=KV_BASE, a=KD, b=(1 << 18) | LP, c=LST0, k=n0)
    s.kv(4, 1, 3, LST1, k=n1own, mult=LP)
    s.cmd(OP_ST, ext=KV_BASE, a=KD1, b=(1 << 18) | LP, c=LST1, k=n1own)
    # decode: 序列 1 的 slot 0..5 (slot 5 未映射 → 略過, L2 端寫 0) gather 到 G, 每頁一個 GEMM
    ng = 3 + n1own + 1
    s.kv(4, 1, 0, LST2, k=ng, mult=LP)
    s.cmd(OP_LD, ext=KV_BASE, a=G, b=(1 << 18) | LP, c=LST2, k=ng)
    for p in range(ng):
        s.cmd(OP_GEMM, a=QA, b=G + p * LP, c=SC + p * CL, k=D, sa=1, sb=1)
    # 釋放: FREE_SEQ 0 (頁 3, 4 歸還; 0..2 仍被序列 1 引用), FREE (1, 4) (頁 6), ALLOC (2, 0) → FIFO 第一個 (頁 3)
    s.kv(2, 0, 0, RES + r)
    nfree = 0
    for slot in range(n0):
        unmap(0, slot); nfree += 1
    res.append((0, nfree, free_cnt())); r += 1
    s.kv(1, 1, 4, RES + r); pg = unmap(1, 4); res.append((0, pg, free_cnt())); r += 1
    s.kv(0, 2, 0, RES + r); res.append((0, alloc(2, 0), free_cnt())); r += 1
    s.kv(1, 3, 7, RES + r); res.append((3, 0, free_cnt())); r += 1          # 未映射 → UNMAPPED
    s.kv(0, 99, 0, RES + r); res.append((5, 0, free_cnt())); r += 1         # seq 超出範圍 → BADARG
    # 結果搬到外部記憶體
    s.cmd(OP_ST, ext=O_RES, a=RES, k=r)
    s.cmd(OP_ST, ext=O_LST, a=LST0, k=0x22)
    s.cmd(OP_ST, ext=O_G, a=G, k=ng * LP)
    s.cmd(OP_ST, ext=O_SC, a=SC, k=ng * CL)
    s.out('pages', KV_BASE, 7 * LP); s.out('res', O_RES, r); s.out('lst', O_LST, 0x22)
    s.out('g', O_G, ng * LP); s.out('sc', O_SC, ng * CL)
    out, cyc = run(tb, wd, 'paged', s, seed)
    # ---- 比對 ----
    page_tok = {}                     # 實體頁 → 該頁的 N 個 token 的 K
    for p in range(n0):
        page_tok[p] = K0[p * N:(p + 1) * N]
    for p in range(n1own):
        page_tok[5 + p] = K1[p * N:(p + 1) * N]
    for pg, Kp in page_tok.items():
        exp = bpage(Kp, 0)
        for l in range(LP):
            chk.eq('page %d line %d' % (pg, l), out['pages'][pg * LP + l], exp[l])
    for i, (st, data, fc) in enumerate(res):
        chk.eq('KV result %d' % i, words(out['res'][i], 3), [st, data, fc])
    # 位址表: LST0 = 頁 0..4 × LP; LST1 = 頁 5, 6; LST2 = 0, 1, 2, 5, 6, 未映射
    def lst(entries):
        return line_from_words(entries + [0] * (16 - len(entries)))
    chk.eq('LST0', out['lst'][0], lst([p * LP for p in range(n0)]))
    chk.eq('LST0 result', words(out['lst'][1], 3), [0, n0, KV_NP - 7])
    chk.eq('LST1', out['lst'][0x10], lst([5 * LP, 6 * LP]))
    chk.eq('LST1 result', words(out['lst'][0x11], 3), [0, 2, KV_NP - 7])
    chk.eq('LST2', out['lst'][0x20], lst([0, LP, 2 * LP, 5 * LP, 6 * LP, FF]))
    chk.eq('LST2 result', words(out['lst'][0x21], 3), [3, 5, KV_NP - 7])
    seq1 = [page_tok[0], page_tok[1], page_tok[2], page_tok[5], page_tok[6], np.zeros((N, D), dtype=np.int64)]
    for p in range(ng):
        exp = bpage(seq1[p], 0)
        for l in range(LP):
            chk.eq('gather page %d line %d' % (p, l), out['g'][p * LP + l], exp[l])
        S = (Q.astype(np.int64) @ seq1[p].astype(np.int64).T).astype(np.int64)    # [N 個 query, N 個 token]
        flat = [int(x) & FF for x in S.reshape(-1)]
        for l in range(CL):
            chk.eq('scores page %d line %d' % (p, l), out['sc'][p * CL + l], line_from_words(flat[l * 16:(l + 1) * 16]))
    print('paged: %d KV commands, %d pages gathered, %s' % (len(res) + 3, ng, cyc))


# ------------------------------------------------------------------------------------------------------------
# MoE
# ------------------------------------------------------------------------------------------------------------
def moe_tables(ids, w, T, K, ne, cap, dl, mult):
    """dispatch 表的獨立實作 (numpy): 回傳 GL (P 項), CT [(pos, wbits)], CN [(n, off)], status (w0, w1, w2)。"""
    ids = np.asarray(ids, dtype=np.int64)[:, :K]
    valid = ids < ne
    counts = np.array([np.sum(ids[valid] == e) for e in range(ne)], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(counts)[:-1]]) if dl else np.arange(ne) * cap
    P = T * K if dl else ne * cap
    GL = [FF] * P
    CT = []
    rank = np.zeros(ne, dtype=np.int64)
    for t in range(T):
        for j in range(K):
            e = int(ids[t, j])
            if e < ne and (dl or rank[e] < cap):
                pos = int(off[e] + rank[e]); GL[pos] = (t * mult) & FF; CT.append((pos, w[t][j]))
            else:
                CT.append((FF, 0))
            if e < ne:
                rank[e] += 1
    CN = [(int(counts[e]) if dl else int(min(counts[e], cap)), int(off[e])) for e in range(ne)]
    st = (int(np.sum(~valid)), int(np.sum(valid)), P)
    return GL, CT, CN, st


def moe(tb, wd, N, rng, seed, chk):
    T, K, E, cap, D = 24, 2, 8, 5, 64
    XB, TKB = 0x40000000, 0x40001000
    O1, O2, OG1, OG2 = 0x50000000, 0x50001000, 0x50002000, 0x50003000
    TK, C1, XG1, C2, XG2 = 0x400, 0x480, 0x500, 0x600, 0x700
    X = rng.integers(-128, 128, size=(T, D))
    logits = rng.normal(size=(T, E))
    logits[:, 3] += 1.5                                   # 偏斜: expert 3 熱門 (容量模式時會丟棄)
    top = np.argsort(-logits, axis=1, kind='stable')[:, :K]
    sel = np.take_along_axis(logits, top, axis=1)
    wt = np.exp(sel - sel.max(axis=1, keepdims=True)); wt = (wt / wt.sum(axis=1, keepdims=True)).astype(np.float32)
    ids16 = np.full((T, 16), FF, dtype=np.int64); ids16[:, :K] = top
    if T > 5:
        ids16[5, 1] = 77                                  # 超出範圍的 id (錯誤項, drop)
    wbits = [[f2u(wt[t][j]) for j in range(K)] for t in range(T)]
    s = Scn()
    for t in range(T):
        s.ext[XB + t] = line_from_bytes(X[t])
        s.ext[TKB + 2 * t] = line_from_words(list(ids16[t]))
        s.ext[TKB + 2 * t + 1] = line_from_words(wbits[t] + [0] * (16 - K))
    s.cmd(OP_LD, ext=TKB, a=TK, k=2 * T)
    nl = {}
    for (dl, C, XG, O, OG) in ((0, C1, XG1, O1, OG1), (1, C2, XG2, O2, OG2)):
        P = T * K if dl else E * cap
        n = (P + 15) // 16 + (T * K + 7) // 8 + (E + 7) // 8 + 1
        nl[dl] = (P, n)
        s.cmd(OP_MOE, a=TK, b=K | (E << 5) | (dl << 13), c=C, ext=cap | (1 << 16), k=T)
        s.cmd(OP_LD, ext=XB, a=XG, b=(1 << 18) | 1, c=C, k=P)      # gather: 每項 1 個 line, 略過項寫 0
        s.cmd(OP_ST, ext=O, a=C, k=n)
        s.cmd(OP_ST, ext=OG, a=XG, k=P)
        s.out('tab%d' % dl, O, n); s.out('xg%d' % dl, OG, P)
    out, cyc = run(tb, wd, 'moe', s, seed)
    ref_ids = [list(ids16[t]) for t in range(T)]
    ref_w = [wbits[t] + [0] * (16 - K) for t in range(T)]
    W = [np.full((D,), e + 1, dtype=np.float64) for e in range(E)]       # expert e: f_e(x) = (e + 1) · x (逐元素)
    for dl in (0, 1):
        P, n = nl[dl]
        GL, CT, CN, st = moe_tables(ref_ids, ref_w, T, K, E, cap, dl, 1)
        tab = out['tab%d' % dl]
        gw = [w for l in range((P + 15) // 16) for w in words(tab[l])][:P]
        chk.eq('moe%d GL' % dl, gw, GL)
        b = (P + 15) // 16
        ct = [w for l in range((T * K + 7) // 8) for w in words(tab[b + l])][:2 * T * K]
        chk.eq('moe%d CT' % dl, [(ct[2 * i], ct[2 * i + 1]) for i in range(T * K)], CT)
        b += (T * K + 7) // 8
        cn = words(tab[b])
        chk.eq('moe%d CN' % dl, [(cn[2 * e], cn[2 * e + 1]) for e in range(E)], CN)
        chk.eq('moe%d status' % dl, words(tab[b + 1], 3), list(st))
        xg = out['xg%d' % dl]
        for p in range(P):
            exp = line_from_bytes(X[GL[p]]) if GL[p] != FF else 0
            chk.eq('moe%d gather %d' % (dl, p), xg[p], exp)
        # MoE 層輸出: 硬體的 combine 表 + 收集結果 vs numpy 直接計算
        rank = np.zeros(E, dtype=np.int64)
        for t in range(T):
            y_ref = np.zeros(D); y_hw = np.zeros(D)
            for j in range(K):
                e = int(ref_ids[t][j])
                if e < E:
                    keep = dl or rank[e] < cap
                    rank[e] += 1
                    if keep:
                        y_ref += float(wt[t][j]) * (W[e] * X[t])
                pos, wb = ct[2 * (t * K + j)], ct[2 * (t * K + j) + 1]
                if pos != FF:
                    xb = np.array([((xg[pos] >> (8 * i)) & 0xFF) for i in range(D)], dtype=np.int64)
                    xb = np.where(xb >= 128, xb - 256, xb)
                    ex = next(ee for ee in range(E) if CN[ee][1] <= pos < CN[ee][1] + (CN[ee][0] if dl else cap))
                    y_hw += float(u2f(wb)) * (W[ex] * xb)
            chk.eq('moe%d y[%d]' % (dl, t), np.array_equal(y_hw, y_ref), True)
        print('moe (%s): T=%d K=%d E=%d P=%d valid=%d dropped=%d' % ('dropless' if dl else 'capacity %d' % cap, T, K, E, P,
              st[1], sum(1 for c in CT if c[0] == FF)))
    print('moe: %s' % cyc)


def main():
    tb, wd = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(wd, exist_ok=True)
    rng = np.random.default_rng(seed)
    chk = Chk()
    paged(tb, wd, N, rng, seed, chk)
    moe(tb, wd, N, rng, seed, chk)
    print('E2E: %d checks, %d mismatches' % (chk.n, chk.bad))
    print('E2E PASS' if chk.bad == 0 else 'E2E FAIL')
    sys.exit(0 if chk.bad == 0 else 1)


if __name__ == '__main__':
    main()
