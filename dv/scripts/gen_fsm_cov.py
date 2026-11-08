#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FSM coverage 產生器 (pyslang): 從 RTL 抽出每個有限狀態機的狀態與轉移 (arc), 產生 bind 用的 SVA cover 模組。

Verilator 沒有 FSM coverage, 這裡以靜態分析補上:
  狀態變數 V = `case (V)` 的條件, 且 V 在該 case 的 item 內被指派 (直接 `V <= C`, 或經由 next-state
  變數 N 再以 `V <= N` 更新)。狀態集合 = V 的 enum 型態的全部成員 (非 enum 時: case 標籤 ∪ 指派的常數)。
  arc  = 每個 case item 標籤 L 對 item 內指派給 V / N 的每個常數 C 產生 L -> C; item 內有不指派的路徑時
         另加 L -> L (停留; 若 case 前有 `N = C0` 的預設值則改為 L -> C0); case 以外 (非 reset 分支) 的
         指派視為任何狀態 -> C。
產生的 cover 名稱: fsmS_<V>_<S>、fsmA_<V>_<A>_to_<B> (cov_report.py 依前綴歸類為 fsm_state / fsm_arc)。
靜態上存在但實際不可達的狀態 / arc 會以未覆蓋出現, 必須以形式證明 (dv/scripts/prove_unreachable.sh) 後才可豁免。

用法: gen_fsm_cov.py [-I dir]... [--json out.json] [--only MOD,...] -o out.sv files...
"""
import argparse
import json
import os
import re
import sys

from pyslang import ast, syntax

RST_RX = re.compile(r'\brst\w*|\breset\w*|\bclr_n\b', re.I)


def const_of(e):
    """常數值 (int) 或 None。"""
    try:
        cv = e.constant
        if cv is not None and cv.value is not None:
            return int(cv.value)
    except Exception:
        pass
    sym = getattr(e, 'symbol', None)
    if sym is not None and hasattr(sym, 'value') and isinstance(sym, (ast.EnumValueSymbol, ast.ParameterSymbol)):
        try:
            return int(sym.value.value)
        except Exception:
            return None
    return None


CONT = {}     # 目前模組的連續指派: 變數 / net 名稱 -> RHS 運算式 (含已例化的 generate 區塊)


def rhs_values(e, depth=0):
    """RHS 可能的常數值集合; 含非常數時回傳 None。經由連續指派 (assign x = 常數 / 條件式) 追溯。"""
    if isinstance(e, ast.ConditionalExpression):
        a, b = rhs_values(e.left, depth), rhs_values(e.right, depth)
        return None if a is None or b is None else a | b
    if isinstance(e, ast.ConversionExpression):
        return rhs_values(e.operand, depth)
    c = const_of(e)
    if c is not None:
        return {c}
    sym = getattr(e, 'symbol', None)
    if sym is not None and depth < 8 and sym.name in CONT:
        return rhs_values(CONT[sym.name], depth + 1)
    return None


def named_var(e):
    while isinstance(e, ast.ConversionExpression):
        e = e.operand
    if isinstance(e, ast.NamedValueExpression) and isinstance(e.symbol, ast.VariableSymbol):
        return e.symbol
    return None


def stmts_of(s):
    if isinstance(s, ast.BlockStatement):
        return stmts_of(s.body)
    if isinstance(s, ast.StatementList):
        return list(s.list)
    return [s]


class Item:
    def __init__(self):
        self.vals = set()      # 指派的常數
        self.unknown = False   # 有非常數的指派


def scan(stmt, targets, it):
    """收集 stmt 內對 targets 的指派 (到 it); 回傳 must-assign (所有路徑都指派)。"""
    must = False
    for s in stmts_of(stmt):
        if isinstance(s, ast.ExpressionStatement) and isinstance(s.expr, ast.AssignmentExpression):
            v = named_var(s.expr.left)
            if v is not None and v.name in targets:
                vals = rhs_values(s.expr.right)
                if vals is None:
                    rv = named_var(s.expr.right)
                    if rv is not None and rv.name in targets:      # N = V: 停留
                        it.vals.add('self')
                    else:
                        it.unknown = True
                else:
                    it.vals |= vals
                must = True
        elif isinstance(s, ast.ConditionalStatement):
            m1 = scan(s.ifTrue, targets, it)
            m2 = scan(s.ifFalse, targets, it) if s.ifFalse is not None else False
            must = must or (m1 and m2)
        elif isinstance(s, ast.CaseStatement):
            ms = [scan(i.stmt, targets, it) for i in s.items]
            md = scan(s.defaultCase, targets, it) if s.defaultCase is not None else False
            must = must or (all(ms) and md)
        elif isinstance(s, (ast.BlockStatement, ast.StatementList)):
            must = scan(s, targets, it) or must
        elif isinstance(s, ast.TimedStatement):
            must = scan(s.stmt, targets, it) or must
        elif isinstance(s, (ast.ForLoopStatement, ast.ForeverLoopStatement, ast.WhileLoopStatement)):
            scan(s.body, targets, it)
    return must


class Fsm:
    def __init__(self, var, mod):
        self.var = var; self.mod = mod
        self.states = {}       # value -> name
        self.arcs = set()      # (a, b)
        self.notes = []
        self.clk = None; self.rst = None; self.rst_active_low = True
        self.enum = False


def clock_reset(pb):
    """always_ff 的時脈與非同步 reset (名稱, active-low?)。"""
    b = pb.body
    if not isinstance(b, ast.TimedStatement):
        return None, None, True
    t = str(b.timing.syntax)
    clk = re.search(r'posedge\s+(\w+)', t)
    rst = re.search(r'negedge\s+(\w+)', t)
    if rst:
        return clk and clk.group(1), rst.group(1), True
    rp = re.search(r'posedge\s+(\w+)\s+or\s+posedge\s+(\w+)', t)
    if rp:
        return rp.group(1), rp.group(2), False
    # 同步 reset: 第一個 if 的條件
    body = stmts_of(b.stmt)
    if body and isinstance(body[0], ast.ConditionalStatement):
        cond = str(body[0].conditions[0].expr.syntax).strip()
        m = RST_RX.search(cond)
        if m:
            return clk and clk.group(1), m.group(0), cond.startswith('!') or cond.startswith('~')
    return clk and clk.group(1), None, True


def analyze_body(body, only):
    mod = body.definition.name if hasattr(body, 'definition') else body.name
    if only and mod not in only:
        return []
    CONT.clear()

    def cont(n):
        if isinstance(n, ast.InstanceSymbol):
            return ast.VisitAction.Skip
        if isinstance(n, ast.GenerateBlockSymbol) and n.isUninstantiated:
            return ast.VisitAction.Skip
        if isinstance(n, ast.ContinuousAssignSymbol):
            a = n.assignment
            if isinstance(a, ast.AssignmentExpression) and isinstance(a.left, ast.NamedValueExpression):
                CONT[a.left.symbol.name] = a.right
        return ast.VisitAction.Advance
    body.visit(cont)
    return analyze_scope(body, mod, '')


def analyze_scope(body, mod, prefix):
    """body 內 (不含子 instance) 的 FSM; generate 區塊遞迴處理, 狀態變數路徑加上 prefix (例: g_pd[1].)。"""
    ff, comb, gens = [], [], []

    def collect(n):
        if isinstance(n, ast.ProceduralBlockSymbol):
            (ff if n.procedureKind == ast.ProceduralBlockKind.AlwaysFF else comb).append(n)
            return ast.VisitAction.Skip
        if isinstance(n, ast.InstanceSymbol):
            return ast.VisitAction.Skip
        if isinstance(n, ast.GenerateBlockArraySymbol):
            for e in n.entries:
                if not e.isUninstantiated:
                    gens.append((e, '%s%s[%d].' % (prefix, n.name, e.arrayIndex)))
            return ast.VisitAction.Skip
        if isinstance(n, ast.GenerateBlockSymbol) and n is not body:
            if not n.isUninstantiated and n.name:
                gens.append((n, '%s%s.' % (prefix, n.name)))
            return ast.VisitAction.Skip
        return ast.VisitAction.Advance
    body.visit(collect)
    sub = []
    for gb, pfx in gens:
        sub += analyze_scope(gb, mod, pfx)

    # always_ff 內 V <= N (N 為變數) → alias N -> V; 與每個 V 的時脈 / reset
    alias, ffinfo = {}, {}
    for pb in ff:
        clk, rst, low = clock_reset(pb)

        def f(n):
            if isinstance(n, ast.AssignmentExpression):
                v = named_var(n.left)
                if v is not None:
                    ffinfo.setdefault(v.name, (clk, rst, low, v))
                    r = named_var(n.right)
                    if r is not None:
                        alias.setdefault(r.name, set()).add(v.name)
            return ast.VisitAction.Advance
        pb.visit(f)

    fsms = {}
    for pb in ff + comb:
        is_ff = pb in ff
        # case 前的預設值 (comb: N = V 或 N = C)
        defaults = {}
        for s in stmts_of(pb.body.stmt if isinstance(pb.body, ast.TimedStatement) else pb.body):
            if isinstance(s, ast.ExpressionStatement) and isinstance(s.expr, ast.AssignmentExpression):
                v = named_var(s.expr.left)
                if v is not None:
                    vals = rhs_values(s.expr.right)
                    defaults[v.name] = vals if vals is not None else {'self'}

        def visit_case(n):
            if not isinstance(n, ast.CaseStatement):
                return ast.VisitAction.Advance
            sv = named_var(n.expr)
            if sv is None or sv.name not in ffinfo:
                return ast.VisitAction.Advance
            V = sv.name
            # 直接指派 V (always_ff) 或 next-state N (alias[N] 含 V)
            targets = {V} if is_ff else {N for N, vs in alias.items() if V in vs}
            if not targets:
                return ast.VisitAction.Advance
            items, any_assign = [], False
            for ci in n.items:
                it = Item(); must = scan(ci.stmt, targets, it)
                labels = [const_of(e) for e in ci.expressions]
                items.append((labels, it, must)); any_assign |= bool(it.vals) or it.unknown
            dit = None
            if n.defaultCase is not None:
                dit = Item(); dmust = scan(n.defaultCase, targets, dit)
            if not any_assign:
                return ast.VisitAction.Advance
            fs = fsms.get(V)
            if fs is None:
                fs = fsms[V] = Fsm(V, mod)
                fs.path = prefix + V
                fs.clk, fs.rst, fs.rst_active_low, var = ffinfo[V]
                fs.width = var.type.bitWidth
                ct = var.type.canonicalType
                if ct.isEnum:
                    fs.enum = True
                    for m in ct:
                        fs.states[int(m.value.value)] = m.name
            dflt = set()
            for N in targets:
                dflt |= defaults.get(N, {'self'})
            listed = set()
            for labels, it, must in items:
                for L in labels:
                    if L is None:
                        fs.notes.append('non-constant case label in %s' % V); continue
                    listed.add(L)
                    add_arcs(fs, L, it, must, dflt)
            if dit is not None:
                fs.default_item = (dit, dmust, dflt, listed)
            return ast.VisitAction.Skip
        pb.visit(visit_case)

    # case 以外 (非 reset 分支) 對 V 的指派 → 任何狀態 -> C
    for pb in ff:
        _, rst, _ = clock_reset(pb)

        def outer(s, in_rst):
            for x in stmts_of(s):
                if isinstance(x, ast.ConditionalStatement):
                    c = str(x.conditions[0].expr.syntax)
                    r = bool(rst and re.search(r'\b%s\b' % re.escape(rst), c))
                    outer(x.ifTrue, in_rst or r)
                    if x.ifFalse is not None:
                        outer(x.ifFalse, in_rst)
                elif isinstance(x, ast.CaseStatement):
                    sv = named_var(x.expr)
                    if sv is not None and sv.name in fsms:
                        continue              # FSM 本身的 case 已處理
                    for ci in x.items:
                        outer(ci.stmt, in_rst)
                    if x.defaultCase is not None:
                        outer(x.defaultCase, in_rst)
                elif isinstance(x, (ast.BlockStatement, ast.StatementList)):
                    outer(x, in_rst)
                elif isinstance(x, ast.TimedStatement):
                    outer(x.stmt, in_rst)
                elif isinstance(x, ast.ExpressionStatement) and isinstance(x.expr, ast.AssignmentExpression) and not in_rst:
                    v = named_var(x.expr.left)
                    if v is not None and v.name in fsms:
                        vals = rhs_values(x.expr.right)
                        fs = fsms[v.name]
                        if vals is None:
                            r = named_var(x.expr.right)
                            if r is None or v.name not in alias.get(r.name, ()):
                                fs.notes.append('non-constant assignment outside case: %s' % str(x.syntax).strip())
                        else:
                            fs.global_to = getattr(fs, 'global_to', set()) | vals
        outer(pb.body.stmt if isinstance(pb.body, ast.TimedStatement) else pb.body, False)

    out = []
    for fs in fsms.values():
        if hasattr(fs, 'default_item'):
            dit, dmust, dflt, listed = fs.default_item
            for S in list(fs.states) or []:
                if S not in listed:
                    add_arcs(fs, S, dit, dmust, dflt)
        for a, b in list(fs.arcs):
            for s in (a, b):
                fs.states.setdefault(s, 'S%d' % s)
        for C in getattr(fs, 'global_to', ()):
            fs.states.setdefault(C, 'S%d' % C)
            for S in list(fs.states):
                fs.arcs.add((S, C))
        if len(fs.states) < 2:
            continue
        out.append(fs)
    return out + sub


def add_arcs(fs, L, it, must, dflt):
    for C in it.vals:
        fs.arcs.add((L, L if C == 'self' else C))
    if it.unknown:
        fs.notes.append('non-constant next state from %s' % L)
    if not must:
        for d in dflt:
            fs.arcs.add((L, L if d == 'self' else d))


def port_name(path):
    return re.sub(r'\W+', '_', path).strip('_')


def emit(fsms_by_mod, f):
    f.write('// 由 dv/scripts/gen_fsm_cov.py (pyslang) 產生, 勿手動修改。每個 FSM 的每個狀態 / arc 一個 SVA cover。\n')
    for mod, fsms in fsms_by_mod.items():
        ports = {}                         # port 名稱 -> (寬度, 連接的運算式)
        for fs in fsms:
            for n in (fs.clk, fs.rst):
                if n:
                    ports[n] = (1, n)
            ports[port_name(fs.path)] = (fs.width, fs.path)
        decl = ',\n  '.join('input logic %s%s' % ('[%d:0] ' % (w - 1) if w > 1 else '', n) for n, (w, _) in ports.items())
        f.write('\nmodule %s__fsm_cov (\n  %s\n);\n' % (mod, decl))
        for fs in fsms:
            dis = 'disable iff (%s%s) ' % ('!' if fs.rst_active_low else '', fs.rst) if fs.rst else ''
            ck = '@(posedge %s) ' % fs.clk
            W, v_ = fs.width, port_name(fs.path)
            for v, n in sorted(fs.states.items()):
                f.write("  fsmS_%s_%s: cover property (%s%s%s == %d'd%d);\n" % (v_, n, ck, dis, v_, W, v))
            for a, b in sorted(fs.arcs):
                f.write("  fsmA_%s_%s_to_%s: cover property (%s%s%s == %d'd%d ##1 %s == %d'd%d);\n" % (
                    v_, fs.states[a], fs.states[b], ck, dis, v_, W, a, v_, W, b))
        f.write('endmodule\n')
        conn = ', '.join('.%s(%s)' % (n, e) for n, (_, e) in ports.items())
        f.write('bind %s %s__fsm_cov u_fsm_cov (%s);\n' % (mod, mod, conn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('-I', action='append', default=[])
    ap.add_argument('-o', required=True, help='輸出檔 (或 --split 時的輸出目錄)')
    ap.add_argument('--split', action='store_true', help='每個模組一個檔案 fsm_<module>.sv')
    ap.add_argument('--json')
    ap.add_argument('--only', help='只處理這些模組 (逗號分隔)')
    ap.add_argument('--top', help='只處理此頂層模組階層內實際例化的模組')
    ap.add_argument('-G', action='append', default=[], help='頂層參數 (同 Verilator -G<name>=<value>)')
    a = ap.parse_args()
    from pyslang import driver
    d = driver.Driver(); d.addStandardArgs()
    args = ['slang', '--ignore-unknown-modules', '-Wnone'] + ['-I' + i for i in a.I] + ['-G' + g for g in a.G] + \
        (['--top', a.top] if a.top else []) + a.files
    if not d.parseCommandLine(' '.join(args), driver.CommandLineOptions()) or not d.processOptions() or not d.parseAllSources():
        sys.exit('pyslang: cannot parse sources')
    c = d.createCompilation()
    only = set(a.only.split(',')) if a.only else None
    errs = [d for d in c.getAllDiagnostics() if d.isError()]
    if errs:
        print('pyslang: %d errors (first: %s)' % (len(errs), errs[0].code), file=sys.stderr)
    seen, by_mod, skipped = set(), {}, []

    def walk(n):
        if isinstance(n, ast.InstanceSymbol):
            name = n.definition.name
            if name not in seen:
                seen.add(name)
                fl = analyze_body(n.body, only)
                if fl:
                    by_mod[name] = fl
            return ast.VisitAction.Advance
        return ast.VisitAction.Advance
    for ti in c.getRoot().topInstances:
        if a.top is None or ti.name == a.top:
            ti.visit(walk)
    if a.split:                          # 每個模組一個檔案: <dir>/fsm_<module>.sv (只把用到的模組加入各 TB 的建置)
        os.makedirs(a.o, exist_ok=True)
        for m in by_mod:
            with open(os.path.join(a.o, 'fsm_%s.sv' % m), 'w') as f:
                emit({m: by_mod[m]}, f)
    else:
        with open(a.o, 'w') as f:
            emit(by_mod, f)
    rep = {m: [{'var': fs.path, 'clk': fs.clk, 'rst': fs.rst, 'enum': fs.enum,
                'states': {fs.states[k]: k for k in sorted(fs.states)},
                'arcs': sorted('%s->%s' % (fs.states[x], fs.states[y]) for x, y in fs.arcs),
                'notes': fs.notes} for fs in fl] for m, fl in by_mod.items()}
    if a.json:
        json.dump(rep, open(a.json, 'w'), indent=1)
    for m, fl in rep.items():
        for fs in fl:
            print('%-24s %-12s states=%-3d arcs=%-3d%s' % (m, fs['var'], len(fs['states']), len(fs['arcs']),
                                                       ('  NOTE: ' + '; '.join(fs['notes'])) if fs['notes'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
