#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""以 Yosys (SAT k-induction) 證明 coverage 豁免項目不可達。

每個證明檔 dv/formal_cov/<名稱>.prove (YAML 風格的簡單格式):
    top: orca_irq_aggr
    files: rtl/periph/orca_irq_aggr.sv
    params: NUM_IRQ=8
    reset: rst_n            # active-low reset; 第 0 拍為 reset, 之後自由
    # 每行一個「恆成立」的性質 (SV 運算式, 可引用 DUT 內部訊號); 對應的豁免在 waivers 檔以 [U:<名稱>] 標示
    prove: prdata[31:17] == 0
    prove: prdata[15:8] == 0
    free: npu_mma           # (選用) 這些子模組只保留埠宣告: 其輸出成為每拍自由的變數 (過度近似, 證明仍成立)
做法: 產生包裝模組 (DUT 的所有輸入為自由變數, 性質寫成 always_comb assert), 經 Yosys
`read_verilog -sv -formal; prep; async2sync; sat -tempinduct -prove-asserts` 證明; 失敗時列出反例。
用法: prove_unreachable.py dv/formal_cov/*.prove   (全部成立才回傳 0)
"""
import os
import re
import subprocess
import sys


def parse(fn):
    d = {'prove': [], 'files': [], 'params': {}, 'depth': '20'}
    for ln in open(fn, encoding='utf-8'):
        ln = ln.split('#', 1)[0].strip()
        if not ln:
            continue
        k, v = [x.strip() for x in ln.split(':', 1)]
        if k == 'prove':
            d['prove'].append(v)
        elif k == 'files':
            d['files'] += v.split()
        elif k == 'free':
            d['free'] = v.split()
        elif k == 'params':
            for kv in v.split():
                a, b = kv.split('='); d['params'][a] = b
        else:
            d[k] = v
    return d


def yosys_compat(txt):
    """Yosys 的 SV 剖析器把 `~N'(x)` 讀成 `(~N)'(x)` (size cast 的優先序錯誤): 改寫成等價的 `~(N'(x))`。"""
    out, i = [], 0
    for m in re.finditer(r"([~!-])\s*(\w+)'\(", txt):
        if m.start() < i:
            continue
        j, depth = m.end(), 1
        while depth:
            depth += {'(': 1, ')': -1}.get(txt[j], 0); j += 1
        out.append(txt[i:m.start()] + m.group(1) + '(' + txt[m.start(2):j] + ')')
        i = j
    txt = ''.join(out) + txt[i:]
    # Yosys 不支援 unpacked 陣列埠 `logic [31:0] x [N]`: 改為等價的 packed `logic [N-1:0][31:0] x` (索引方式相同)
    txt = re.sub(r'\b(input|output)(\s+logic\s*)(\[[^\]]+\])\s*(\w+)\s*\[(\w+)\]',
                 lambda m: '%s%s[%s-1:0]%s %s' % (m.group(1), m.group(2), m.group(5), m.group(3), m.group(4)), txt)
    # Yosys 只接受函式中「不在 case 項目內」的 return: 改寫為指定函式名稱 (本專案的函式中 return 都是該路徑的最後一句, 等價)
    def fn_ret(m):
        name = re.search(r'(\w+)\s*\(', m.group(1)).group(1)
        return m.group(1) + re.sub(r'\breturn\s+([^;]+);', lambda r: '%s = %s;' % (name, r.group(1)), m.group(2)) + m.group(3)
    # Yosys 不接受模組標頭或本體內的 `import p::*;`: 改為模組前的檔案層級 import (此專案中 package 名稱不衝突, 等價)
    def mod_imp(m):
        imps = re.findall(r'import\s+\w+::\*\s*;', m.group(0))
        if not imps:
            return m.group(0)
        body = re.sub(r'[ \t]*import\s+\w+::\*\s*;[ \t]*(//[^\n]*)?\n?', '', m.group(0))
        return ' '.join(dict.fromkeys(imps)) + '\n' + body
    txt = re.sub(r'^[ \t]*module\b.*?\bendmodule\b', mod_imp, txt, flags=re.S | re.M)
    # Yosys 不支援具名成員的 assignment pattern `'{f: a, l: b}`: 依同檔 typedef struct packed 的欄位順序改成串接
    structs = {m.group(2): re.findall(r'(\w+)\s*(?=[,;])', m.group(1))
               for m in re.finditer(r'typedef\s+struct\s+packed\s*\{(.*?)\}\s*(\w+)\s*;', txt, flags=re.S)}
    def named_pat(m):
        typ, items = m.group(1), dict(re.findall(r'(\w+)\s*:\s*([^,}]+)', m.group(2)))
        order = structs.get(typ)
        if not order or set(order) != set(items):
            return m.group(0)
        return '%s = {%s}' % (m.group(0).split('=')[0].rstrip(), ', '.join(items[f].strip() for f in order))
    txt = re.sub(r"\b(\w+)\s+\w+\s*=\s*'\{((?:\s*\w+\s*:[^,}]+,?)+)\}", named_pat, txt)
    txt = re.sub(r'(^[ \t]*function\b[^;(]*\w+\s*\([^;]*;)(.*?)(\bendfunction\b)', fn_ret, txt, flags=re.S | re.M)
    return txt


def run(fn):
    d = parse(fn)
    top = d['top']
    chparam = ' '.join('-set %s %s' % (k, v) for k, v in d['params'].items())
    asserts = '\n'.join('  always_comb if (%s) p%d: assert (%s);' % (d.get('reset', "1'b1"), i, p)
                         for i, p in enumerate(d['prove']))
    # 性質直接插入 DUT 頂層模組的 endmodule 之前 (可引用內部訊號); 其他檔案原樣讀入
    files = []
    wd = os.path.join('build', 'formal_cov', os.path.basename(fn).rsplit('.', 1)[0])
    os.makedirs(wd, exist_ok=True)
    for f in d['files']:
        txt = open(f, encoding='utf-8').read()
        txt = yosys_compat(txt)
        for fm in d.get('free', []):
            txt = re.sub(r'(\bmodule\s+%s\b.*?\);).*?(\bendmodule\b)' % re.escape(fm),
                         lambda m: m.group(1) + '\n  // prove_unreachable.py: free (輸出不受驅動)\n' + m.group(2), txt, flags=re.S)
        m = re.search(r'\bmodule\s+%s\b' % re.escape(top), txt)
        if m:
            e = txt.index('endmodule', m.end())
            txt = txt[:e] + '// --- prove_unreachable.py ---\n' + asserts + '\n' + txt[e:]
        f = os.path.join(wd, os.path.basename(f))
        open(f, 'w', encoding='utf-8').write(txt)
        files.append(f)
    ys = ['read_verilog -sv -formal -Irtl/common %s' % ' '.join(files),
          'chparam %s %s' % (chparam, top) if chparam else '',
          'hierarchy -top %s' % top, 'proc', 'flatten', 'opt_clean', 'async2sync', 'dffunmap',
          'sat -tempinduct -prove-asserts -maxsteps %s -set-init-zero -show-ports -verify' % d['depth']]
    r = subprocess.run([os.environ.get('YOSYS', 'yowasp-yosys'), '-q', '-p', '; '.join(x for x in ys if x)],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print('%s %s (%d properties)' % ('PROVED' if ok else 'FAILED', fn, len(d['prove'])))
    if not ok:
        print((r.stdout + r.stderr)[-3000:])
    return ok


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    res = [run(f) for f in sys.argv[1:]]
    sys.exit(0 if all(res) else 1)
