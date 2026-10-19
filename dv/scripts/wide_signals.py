#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""列出 Verilator --coverage 建置中, 總寬度超過 --coverage-max-width (因此沒有 toggle 覆蓋點) 的訊號。

Verilator 對超過上限的訊號會默默不產生 toggle 點; 本工具從產生的 C++ 標頭
(<Mdir>/*___024root.h) 依儲存型別推算每個訊號的位元數, 讓被排除的訊號明確列入報告。

用法: wide_signals.py <Mdir> <max_width>
"""
import glob
import re
import sys

SCALAR = {'CData': 8, 'SData': 16, 'IData': 32, 'QData': 64}


def bits(t):
    """C++ 儲存型別 → 位元數 (VlWide 以 32-bit word 計, 為上界)。"""
    t = t.strip()
    m = re.fullmatch(r'VlUnpacked<(.*),\s*(\d+)>', t)
    if m:
        return bits(m.group(1)) * int(m.group(2))
    m = re.fullmatch(r'VlWide<(\d+)>(?:/\*(\d+):0\*/)?', t)
    if m:
        return int(m.group(2)) + 1 if m.group(2) else 32 * int(m.group(1))
    m = re.fullmatch(r'(CData|SData|IData|QData)(?:/\*(\d+):0\*/)?', t)
    if m:
        return int(m.group(2)) + 1 if m.group(2) else SCALAR[m.group(1)]
    return 0


def main():
    mdir, cap = sys.argv[1], int(sys.argv[2])
    out = []
    for h in glob.glob(mdir + '/*___024root.h'):
        for ln in open(h, encoding='utf-8', errors='replace'):
            m = re.match(r'\s*(VlUnpacked<.*>|VlWide<\d+>(?:/\*\d+:0\*/)?|[CSIQ]Data(?:/\*\d+:0\*/)?)\s+(\w+);', ln)
            if not m:
                continue
            name = m.group(2)
            if not name.startswith(('orca_lockstep_tb__DOT__dut__DOT__',)) or '__Vdly' in name or '__Vcov' in name or '__Vtogcov' in name:
                continue
            w = bits(m.group(1))
            if w > cap:
                out.append((w, name.replace('orca_lockstep_tb__DOT__', '').replace('__DOT__', '.')))
    print('toggle 未量測 (總寬度 > --coverage-max-width %d): %d 個訊號' % (cap, len(out)))
    try:
        for w, n in sorted(out, reverse=True):
            print('  %8d bit  %s' % (w, n))
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    main()
