#!/usr/bin/env python3
"""Count DISTINCT architectural registers WRITTEN anywhere in the muon .text.
Rename.scala:110-123 keeps one global counter over 1..numPhysRegs-1 that is bumped the first
time ANY warp writes an architectural register and is never reclaimed, so the hardware limit is
   warps_per_core * distinct_arch_regs_written  <=  255.
At 6 warps over 2 cores (3 warps/core) that is 85 registers FOR THE WHOLE KERNEL."""
import sys, re
NOWRITE = {'sw','sb','sh','sw.shared','sw.global','sb.shared','sh.shared','sb.global','sh.global',
           'beq','bne','blt','bge','bltu','bgeu','beqz','bnez','blez','bgez','bltz','bgtz',
           'j','jr','jal','jalr','ret','call','tail','fence','fence.s','ecall','ebreak','nop',
           'vx_bar','vx_split_n','vx_join','vx_tmc','vx_wspawn','vx_pred','vx_fence','unimp'}
REG = re.compile(r'^(x\d+|zero|ra|sp|gp|tp|t[0-6]|s\d+|a\d+|fs\d+|ft\d+|fa\d+)$')
for p in sys.argv[1:]:
    w = {}
    for l in open(p):
        if not l.startswith('\t'): continue
        parts = l.strip().split('#')[0].split(None, 1)
        if not parts: continue
        op = parts[0]
        if op.startswith('.') or op in NOWRITE or op.endswith(':'): continue
        if len(parts) < 2: continue
        d = parts[1].split(',')[0].strip()
        if REG.match(d) and d != 'zero':
            w[d] = w.get(d, 0) + 1
    print(f"{p}: {len(w)} distinct written regs   (limit 85 at 3 warps/core, 127 at 2/core)")
    print("   ", " ".join(sorted(w, key=lambda r: -w[r])[:40]))
