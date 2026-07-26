#!/usr/bin/env python3
"""SUPERSEDED -- USE kernels/fa_mx_hostsf/fa_verify_tiles.py INSTEAD.

*** THIS TOOL IS UNSOUND ON 2-CLUSTER TRACES, WHICH IS EVERY TRACE ON THIS CONFIG. ***
It buckets O stores by the most recent MARK index and never separates clid.  Both clusters write
MARKs and both write the SAME O addresses, interleaved in one trace, so cluster 1's MARK advances
the cursor while cluster 0 is mid-finalize and every bucket mixes clusters and generations.  On the
unmodified baseline it reports 52-85% Frobenius for output that is provably correct.
fa_verify_tiles.py groups by cluster FIRST and then starts a new image on an address repeat, which
is self-validating: it yields exactly 4096 words per image and the images account for exactly the
store words the whole-file verify parses.  That accounting cannot close if the grouping is wrong.
Re-scoring this kernel's verdicts with it changed two things and confirmed the rest:
  * the corruption is confined to CLUSTER 0 (cluster 1 is correct on every tile), not to "the warps
    on core 1" as the mark-bucketed data suggested;
  * its magnitude is 107-111% (garbage), not the 31-61% the mixed buckets showed.
AND IT FAILS IN THE DANGEROUS DIRECTION TOO -- it does not merely add noise, it can report a
FALSE PASS.  On the identical trace (tag zD2, FA_SP_QOVL3 + FA_SP_QKACC, complete and stable) this
tool reports "3.5666% | 3.5666%", i.e. both tiles perfect, while fa_verify_tiles.py reports
cluster 0 tile 1 at 85.1674% WRONG (3 of 4 images).  The mixing hides a broken cluster behind a
correct one.  Never accept a CLEAN verdict from this tool.
Kept only so the earlier numbers in the git history can be traced to their source.

Verify EACH TILE'S O separately, by bucketing the O stores between MARK stamps.

WHY THIS EXISTS.  Every tile of an FA_SP run recomputes the SAME tile from the SAME Q/K/V and
writes O to the SAME GMEM buffer, so a whole-file verify only ever scores the LAST generation that
happened to land -- and a trace read early scores an EARLY generation.  Both directions have
produced misleading "3.5666% OK" and "61% broken" readings on the same binary.  Bucketing the O
stores by the preceding MARK index scores each tile's finalize on its own.
usage: y_pertile.py trace.out [golden.npy]
"""
import re, sys, numpy as np, os
sys.stderr.write("*** fa_pertile.py IS UNSOUND ON 2-CLUSTER TRACES -- use "
                 "kernels/fa_mx_hostsf/fa_verify_tiles.py.  See the docstring. ***\n")
ISSUE = re.compile(r"\[ISSUE\].*?inst=([0-9a-fA-F]+).*?tmask=([0-9a-fA-F]+)"
                   r".*?rs1\.data=\[([0-9a-f ]+)\].*?rs2\.data=\[([0-9a-f ]+)\]")
BASE, NB = 0x40040000, 0x4000
kd = '/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/flash_attention_mx'
gold = np.load(os.path.join(kd, sys.argv[2] if len(sys.argv) > 2 else 'golden_O_u16.npy')
               ).astype(np.uint16).reshape(64, 128)
ef = (gold.astype(np.uint32) << 16).view(np.float32)
cur, buckets = -1, {}
for line in open(sys.argv[1], errors='ignore'):
    if '[ISSUE]' not in line: continue
    i = line.find('rs1.data=[')
    if i < 0: continue
    try: a0 = int(line[i+10:i+18], 16)
    except ValueError: continue
    if 0x40050000 <= a0 < 0x40050200:              # a MARK store: advances the stage cursor
        cur = (a0 - 0x40050000) // 4; continue
    if 'inst=' not in line: continue
    m = ISSUE.search(line)
    if not m or (int(m.group(1), 16) & 0x7F) != 0x23: continue
    A = [int(x, 16) for x in m.group(3).split()]; D = [int(x, 16) for x in m.group(4).split()]
    if any(BASE <= x < BASE+NB for x in A): buckets.setdefault(cur, []).append(list(zip(A, D)))
def score(sub):
    mem = {}
    for lz in sub:
        for a, d in lz:
            if BASE <= a < BASE+NB:
                for b in range(4): mem[a+b] = (d >> (8*b)) & 0xFF
    got = np.zeros(8192, dtype=np.uint16); cov = 0
    for i in range(8192):
        ad = BASE + 2*i
        if ad in mem and ad+1 in mem: got[i] = mem[ad] | (mem[ad+1] << 8); cov += 1
    gf = (got.astype(np.uint32).reshape(64, 128) << 16).view(np.float32)
    return cov, 100*float(np.linalg.norm(gf-ef)/np.linalg.norm(ef))
full = [k for k in sorted(buckets) if score(buckets[k])[0] == 8192]
print(f"  {os.path.basename(sys.argv[1])}: complete O generations at marks {full}")
for k in sorted(buckets):
    c, r = score(buckets[k])
    tag = "  <== tile %d finalize" % ((k-6)//7) if (k-6) % 7 == 0 and c == 8192 else ""
    if c == 8192 or len(buckets[k]) > 100:
        print(f"     after m[{k:3d}]  lines={len(buckets[k]):5d}  covered {c}/8192  Frobenius {r:8.4f}%{tag}")
