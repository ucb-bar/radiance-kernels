#!/usr/bin/env python3
"""Per-function barrier-in-divergent-region audit.
bar_audit.py tracks vx_split_n/vx_join depth LINEARLY over the whole .s, so an unbalanced pair in
one function (an early return out of a split region is legal) poisons the depth for every later
function and reports false positives.  Depth must be counted per function."""
import re, sys
from collections import Counter
LI = re.compile(r'^\s+li\s+([as][0-9]+|t[0-9]+), (\d+)\s*$')
for path in sys.argv[1:]:
    fn = '?'; depth = 0; regs = {}; hits = []
    for i, l in enumerate(open(path)):
        if 'Begin function' in l:
            fn = l.split('Begin function')[1].strip(); depth = 0; regs = {}
            continue
        m = LI.match(l)
        if m: regs[m.group(1)] = int(m.group(2)); continue
        if 'vx_split_n' in l: depth += 1; continue
        if 'vx_join' in l: depth = max(0, depth - 1); continue
        if 'vx_bar' in l:
            mm = re.search(r'vx_bar\s+([a-z0-9]+),', l)
            hits.append((i + 1, regs.get(mm.group(1)) if mm else None, depth, fn[:46]))
    bad = [h for h in hits if h[2] > 0]
    c = Counter(h[1] for h in hits if h[1] is not None)
    dups = {k: v for k, v in c.items() if v > 1}
    print(f"== {path}: {len(hits)} vx_bar")
    for (ln, rid, d, f) in hits:
        print(f"   line {ln:5d} id={rid} depth={d} in {f}" + ("   <-- DIVERGENT, HANG RISK" if d else ""))
    if dups: print(f"   *** duplicated ids: {dups}  (same id from >1 site = the llvm duplication bug)")
    if not bad and not dups: print("   OK: every barrier at split depth 0 within its own function, no duplicated id")
