#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verilator coverage.dat 收斂報告: 依類型 (line / branch / expr / toggle / fsm_state /
fsm_arc / user(SVA cover) / covergroup) 統計 RTL 檔案的覆蓋率, 套用豁免清單, 列出未覆蓋點。

用法: cov_report.py [--rtl REGEX] [--waivers FILE] [--require 100] [--list N] coverage.dat...
  多個 .dat 會先合併 (同一點取最大值以外的加總, 等同 verilator_coverage merge)。
  --require P: 任一類型 (豁免後) 低於 P% 即非零結束。

豁免檔格式 (每行): <類型> <點名稱 regex> # <理由>   — 理由為必填, 沒有理由的豁免會報錯。
"""
import argparse
import collections
import re
import sys

KEYS = {'f': 'file', 'l': 'line', 't': 'type', 'o': 'comment', 'h': 'hier', 'page': 'page'}


def parse_point(raw):
    # 'f<file>\x01l<line>...': 欄位以 \x01 分隔, 鍵與值以 \x02 分隔
    d = {}
    for part in raw.split('\x01'):
        if '\x02' in part:
            k, v = part.split('\x02', 1)
            d[k] = v
    return d


def load(files):
    pts = collections.OrderedDict()
    for fn in files:
        for ln in open(fn, encoding='utf-8', errors='replace'):
            if not ln.startswith('C '):
                continue
            body, cnt = ln[2:].rstrip('\n').rsplit(' ', 1)
            key = body.strip("'")
            pts[key] = pts.get(key, 0) + int(cnt)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dat', nargs='+')
    ap.add_argument('--rtl', default=r'^rtl/')
    ap.add_argument('--waivers')
    ap.add_argument('--require', type=float)
    ap.add_argument('--list', type=int, default=40)
    ap.add_argument('--types', help='只統計這些類型 (逗號分隔)')
    ap.add_argument('--by-module', action='store_true',
                    help='模組層級合併: 同一模組的所有 instance (不同 TB / 不同上層) 的同一點合併計數; '
                         '模組內的 generate 範圍 (名稱 g_*) 仍分開')
    ap.add_argument('--hier-exclude', help='不統計階層符合此 regex 的點 (例: 子模組的 cover 在其自身 TB 量測)')
    ap.add_argument('--expect', action='append', default=[],
                    help='每個符合此 regex 的檔案都必須有被統計的點 (可重複); 防止 --rtl 寫錯時以 0 個點「通過」')
    ap.add_argument('--toggle-rtl', help='toggle 只統計符合此 regex 的檔案 (排除 checker / TB 自身訊號)')
    a = ap.parse_args()
    waivers = []
    if a.waivers:
        for n, ln in enumerate(open(a.waivers, encoding='utf-8'), 1):
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            if '#' not in s:
                sys.exit('%s:%d: waiver without justification' % (a.waivers, n))
            spec, why = s.split('#', 1)
            typ, rx = spec.split(None, 1)
            waivers.append([typ, re.compile(rx.strip()), why.strip(), 0])
    tot = collections.Counter(); hit = collections.Counter(); wv = collections.Counter()
    miss = collections.defaultdict(list)
    pts = load(a.dat)
    seen_files = set()
    if a.by_module:
        merged = collections.OrderedDict()
        for key, cnt in pts.items():
            d = parse_point(key)
            if a.hier_exclude and re.search(a.hier_exclude, d.get('h', '')):
                continue
            if d.get('t') == 'covergroup':      # covergroup 的 h 是 bin 名稱 (不含 instance 路徑), 原樣保留
                merged[key] = merged.get(key, 0) + cnt
                continue
            segs = d.get('h', '').split('.')
            tail = []
            while segs and re.match(r'^g_\w*(\[\d+\])?$', segs[-1]):
                tail.insert(0, segs.pop())
            # 以 instance 名以外的欄位為鍵 (instance 名在不同上層可能不同, 例如 dut / u_cu / u_mma)
            d['h'] = '.'.join(tail)
            nk = '\x01'.join('%s\x02%s' % (k, v) for k, v in sorted(d.items()))
            merged[nk] = merged.get(nk, 0) + cnt
        pts = merged
    for key, cnt in pts.items():
        d = parse_point(key)
        f = d.get('f', '')
        if not re.search(a.rtl, f):
            continue
        t = d.get('t', '?')
        # dv/scripts/gen_fsm_cov.py 產生的 FSM cover (pyslang 抽出的狀態 / arc) 歸類為 fsm_state / fsm_arc
        if t == 'user' and d.get('o', '').startswith(('fsmS_', 'fsmA_')):
            t = 'fsm_state' if d['o'].startswith('fsmS_') else 'fsm_arc'
        if t == 'toggle' and a.toggle_rtl and not re.search(a.toggle_rtl, f):
            continue
        if a.types and t not in a.types.split(','):
            continue
        if a.hier_exclude and re.search(a.hier_exclude, d.get('h', '')):
            continue
        name = '%s:%s %s %s' % (f, d.get('l', '?'), d.get('h', ''), d.get('o', ''))
        seen_files.add(f)
        tot[t] += 1
        if cnt > 0:
            hit[t] += 1
            continue
        for w in waivers:
            if (w[0] == t or w[0] == '*') and w[1].search(name):
                w[3] += 1; wv[t] += 1
                break
        else:
            miss[t].append(name)
    ok = True
    for rx in a.expect:
        if not any(re.search(rx, f) for f in seen_files):
            print('ERROR: no coverage points from files matching %r (wrong --rtl filter?)' % rx)
            ok = False
    if not tot:
        print('ERROR: no coverage points matched --rtl %r' % a.rtl)
        ok = False
    print('%-11s %8s %8s %8s %9s' % ('type', 'hit', 'waived', 'total', 'coverage'))
    for t in sorted(tot):
        pct = 100.0 * (hit[t] + wv[t]) / tot[t]
        print('%-11s %8d %8d %8d %8.2f%%' % (t, hit[t], wv[t], tot[t], pct))
        if a.require is not None and pct + 1e-9 < a.require:
            ok = False
    for w in waivers:
        if w[3] == 0 and (not a.types or w[0] in a.types.split(',')) and w[0] in tot:
            print('NOTE: unused waiver: %s %s  (# %s)' % (w[0], w[1].pattern, w[2]))
    for t in sorted(miss):
        print('--- uncovered %s (%d)' % (t, len(miss[t])))
        for m in miss[t][:a.list]:
            print('   ', m)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
