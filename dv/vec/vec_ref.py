#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# NPU 向量核 (rtl/npu/npu_vec.sv) 的位元精確參考模型: 指令編碼、格式轉換 (BF16 / FP16 / INT32 / MXFP8 / MXFP4)、
# 逐 lane FP32 運算 (rtl/fp 的 fp_add / fp_mul / sfu) 與固定順序的樹狀歸約。語意見 docs/npu_vec.md。
import os, sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fp'))
import fp_ref                                    # noqa: E402
import sfu_ref                                   # noqa: E402

VL = 16                                          # lane 數 (一個 L2 line = 16 × FP32)
NV = 8                                           # 向量暫存器數
QNAN = 0x7FC00000

# 指令 (npu_vec_ins_t, 77 bit): {op[3:0], fn[2:0], vd[2:0], va[2:0], vb[2:0], part[2:0], imm[31:0], line[25:0]}
OPS = dict(NOP=0, LD=1, ST=2, ADD=3, SUB=4, MUL=5, MAX=6, SFU=7, SPLAT=8, RSUM=9, RMAX=10)
OPN = {v: k for k, v in OPS.items()}
LD_FMT = dict(FP32=0, BF16=1, FP16=2, INT32=3, E4M3=4, E5M2=5, E2M1=6)    # LD 的 fn
ST_FMT = dict(FP32=0, BF16=1)                                            # ST 的 fn
SFU_FN = dict(RECIP=0, RSQRT=1, EXP2=2, EXP=3, SINPI=4, COSPI=5)          # SFU 的 fn
INS_W = 77


def enc(op, fn=0, vd=0, va=0, vb=0, part=0, imm=0, line=0):
    op = OPS[op] if isinstance(op, str) else op
    v = op
    for w, x in ((3, fn), (3, vd), (3, va), (3, vb), (3, part), (32, imm), (26, line)):
        assert 0 <= x < (1 << w)
        v = (v << w) | x
    return v


def dec(v):
    f = {}
    for name, w in (('line', 26), ('imm', 32), ('part', 3), ('vb', 3), ('va', 3), ('vd', 3), ('fn', 3), ('op', 4)):
        f[name] = v & ((1 << w) - 1)
        v >>= w
    return f


def legal(f):
    op, fn = f['op'], f['fn']
    if op > 10:
        return False
    if op == 1:
        return fn <= 6
    if op == 2:
        return fn <= 1
    if op == 7:
        return fn <= 5
    return True


# ---------------- 格式轉換 ----------------
# 小格式 (OCP MX v1.0 / IEEE FP16): (E, M, bias, 特殊值)
#   'ieee': 指數全 1 = inf / NaN; 'e4m3': 只有 S.1111.111 = NaN (無 inf); 'none': 無 inf / NaN (E2M1)
SMALL = {'FP16': (5, 10, 15, 'ieee'), 'E4M3': (4, 3, 7, 'e4m3'), 'E5M2': (5, 2, 15, 'ieee'), 'E2M1': (2, 1, 1, 'none')}
ELEM_W = {'FP32': 32, 'INT32': 32, 'BF16': 16, 'FP16': 16, 'E4M3': 8, 'E5M2': 8, 'E2M1': 4}


def dec_small(x, fmt, scale):
    """小格式元素 × 2^(scale-127) → FP32 (次正規元素精確解碼; 結果 FTZ / 溢位成 inf)。scale = 0xFF → NaN。"""
    e, m, bias, sp = SMALL[fmt]
    s = (x >> (e + m)) & 1
    ex = (x >> m) & ((1 << e) - 1)
    fr = x & ((1 << m) - 1)
    if scale == 0xFF:
        return QNAN
    if sp == 'ieee' and ex == (1 << e) - 1:
        return QNAN if fr else ((s << 31) | 0x7F800000)
    if sp == 'e4m3' and ex == (1 << e) - 1 and fr == (1 << m) - 1:
        return QNAN
    if ex == 0:
        v = Fraction(fr, 1 << m) * Fraction(2) ** (1 - bias)
    else:
        v = Fraction((1 << m) | fr, 1 << m) * Fraction(2) ** (ex - bias)
    v *= Fraction(2) ** (scale - 127)
    return fp_ref.round_to(-v if s else v, s, fp_ref.FP32)


def dec_elem(x, fmt, scale):
    if fmt == 'FP32':
        return x
    if fmt == 'BF16':
        return fp_ref.cvt(x, fp_ref.BF16, fp_ref.FP32)
    if fmt == 'INT32':
        v = x - (1 << 32) if x >> 31 else x
        return fp_ref.round_to(Fraction(v), 0, fp_ref.FP32)
    if fmt == 'FP16':
        return dec_small(x, fmt, 127)
    return dec_small(x, fmt, scale)


def ld_lanes(line_val, fmt, part, scale):
    w = ELEM_W[fmt]
    per = 512 // w
    base = 0 if w == 32 else (part % (per // VL)) * VL
    return [dec_elem((line_val >> (w * (base + i))) & ((1 << w) - 1), fmt, scale) for i in range(VL)]


def st_line(va, vb, fmt):
    if fmt == 'FP32':
        return sum(va[i] << (32 * i) for i in range(VL))
    el = [fp_ref.cvt(x, fp_ref.FP32, fp_ref.BF16) for x in va + vb]
    return sum(el[i] << (16 * i) for i in range(2 * VL))


# ---------------- 運算 ----------------
def fp_max(a, b):
    """max: 任一 NaN → qNaN; DAZ (次正規 = 同號的 0); +0 > -0; 結果為較大者 (0 以 ±0 表示)。"""
    ca, sa, va = fp_ref.classify(a, fp_ref.FP32)
    cb, sb, vb = fp_ref.classify(b, fp_ref.FP32)
    if ca == 'nan' or cb == 'nan':
        return QNAN
    def key(c, s, v):
        if c == 'inf':
            return (-1 if s else 1, 0)
        return (0, v)
    def bits(x, c, s):
        return (s << 31) if c == 'zero' else x
    ka, kb = key(ca, sa, va), key(cb, sb, vb)
    if ka > kb:
        return bits(a, ca, sa)
    if kb > ka:
        return bits(b, cb, sb)
    if ca == 'zero':
        return (sa & sb) << 31                   # max(+0, -0) = +0
    return a


SFU_F = [sfu_ref.sfu_recip, sfu_ref.sfu_rsqrt, sfu_ref.sfu_exp2, sfu_ref.sfu_exp, sfu_ref.sfu_sinpi, sfu_ref.sfu_cospi]


def tree(v, f):
    """固定順序的樹狀歸約: 每層 x[i] = f(x[2i], x[2i+1])。"""
    while len(v) > 1:
        v = [f(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    return v[0]


class Vec:
    def __init__(self):
        self.r = [[0] * VL for _ in range(NV)]
        self.err = 0

    def step(self, ins, mem):
        """執行一個指令; mem: dict line → 512 bit 整數 (未寫過的 line 為 0)。"""
        f = dec(ins)
        if not legal(f):
            self.err = 1
            return
        op, fn = OPN[f['op']], f['fn']
        a, b = self.r[f['va']], self.r[f['vb']]
        res = None
        if op == 'LD':
            fmt = [k for k, v in LD_FMT.items() if v == fn][0]
            res = ld_lanes(mem.get(f['line'], 0), fmt, f['part'], f['imm'] & 0xFF)
        elif op == 'ST':
            fmt = [k for k, v in ST_FMT.items() if v == fn][0]
            mem[f['line']] = st_line(a, b, fmt)
        elif op == 'ADD':
            res = [fp_ref.fp_add(a[i], b[i]) for i in range(VL)]
        elif op == 'SUB':
            res = [fp_ref.fp_add(a[i], b[i] ^ 0x80000000) for i in range(VL)]
        elif op == 'MUL':
            res = [fp_ref.fp_mul(a[i], b[i]) for i in range(VL)]
        elif op == 'MAX':
            res = [fp_max(a[i], b[i]) for i in range(VL)]
        elif op == 'SFU':
            res = [SFU_F[fn](a[i]) for i in range(VL)]
        elif op == 'SPLAT':
            res = [f['imm']] * VL
        elif op == 'RSUM':
            res = [tree(list(a), fp_ref.fp_add)] * VL
        elif op == 'RMAX':
            res = [tree(list(a), fp_max)] * VL
        if res is not None:
            self.r[f['vd']] = res


# ---------------- 執行與 TB 向量檔 ----------------
RESET = 'RESET'                                  # 程式中的 reset 記錄 (err 清除; 暫存器檔無 reset, 保持)


def run(prog, mem):
    """執行程式 (指令整數或 RESET); 回傳 TB 記錄 [(ins, kind, err, exp)]:
    kind 0 = 不寫入 (NOP), 1 = 寫暫存器 (exp = 寫入值 512 bit), 2 = 寫 L2 (exp = wr_data), 3 = 非法, 4 = reset。"""
    v, recs = Vec(), []
    for ins in prog:
        if ins == RESET:
            v.err = 0
            recs.append((0, 4, 0, 0))
            continue
        f = dec(ins)
        st = {}
        if legal(f) and OPN[f['op']] == 'ST':
            v.step(ins, st)
            mem[f['line']] = st[f['line']]
            recs.append((ins, 2, v.err, st[f['line']]))
            continue
        v.step(ins, mem)
        if not legal(f):
            recs.append((ins, 3, v.err, 0))
        elif OPN[f['op']] == 'NOP':
            recs.append((ins, 0, v.err, 0))
        else:
            recs.append((ins, 1, v.err, sum(x << (32 * i) for i, x in enumerate(v.r[f['vd']]))))
    return recs, v


def write_tb(base, mem0, recs):
    with open(base + '.mem', 'w') as fm:
        for ln, d in sorted(mem0.items()):
            fm.write('%07x %0128x\n' % (ln, d))
    with open(base + '.ins', 'w') as fi:
        for ins, k, e, x in recs:
            fi.write('%020x %d %d %0128x\n' % (ins, k, e, x))
