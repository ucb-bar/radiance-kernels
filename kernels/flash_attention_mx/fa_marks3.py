#!/usr/bin/env python3
"""fa_marks3.py <out> [--per N] -- PER-CLUSTER tile intervals + per-stage means from MARK stamps.

*** TWO TRAPS, BOTH OF WHICH HAVE PRODUCED WRONG NUMBERS IN THIS PROJECT. ***
(1) DO NOT KEY MARKS BY rs1.data.  fa_marks_cl.py does, i.e. it assumes `mki++` leaves the full
    effective address in the base register.  Whether it does is a CODEGEN COIN FLIP: with a fixed
    number of marks per iteration clang either bumps the base register by 4 per mark (rs1.data IS
    the address -- the FA_SP_QSPLIT builds) or keeps ONE base and puts the index in the store
    IMMEDIATE (the FA_SP_OPV builds, where every mark reports rs1.data=0x40050000 and they all alias
    to index 0).  The project note "any tool deriving an effective address from rs1.data alone is
    WRONG" applies here directly.
(2) THERE IS ONE EXTRA, PRE-LOOP COPY OF THE s0 STAMP, at its own pc.  Include it and every stage
    attribution shifts by one, which makes the empty stage look like it costs 38,000 cycles.
The robust key is TRACE ORDER within a cluster (all marks are stores by the same single thread, so
they are strictly ordered and their values are monotone) with the first one dropped."""
import re, sys
A = re.compile(r'clid=(\d).*?pc=([0-9a-f]{8}).*?inst=([0-9a-f]{16}).*?rs1\.data=\[([0-9a-f]{8}).*?rs2\.data=\[([0-9a-f]{8})')
def main():
    path = sys.argv[1]
    per = int(sys.argv[sys.argv.index('--per')+1]) if '--per' in sys.argv else 7
    # *** MESH-BUSY PER INTERVAL, AND IT IS NOT ALWAYS 16,420. ***  Utilisation is
    # mesh_busy_cycles / interval, and the interval this script measures is ONE ITERATION OF THE
    # MARKED LOOP -- which for FA_SP_SQ32 is a HALF-tile of 32 query rows, i.e. 2 x 4,096 = 8,192
    # mesh cycles, not the 64-row tile's 16,420.  Dividing 16,420 by a half-tile interval reports
    # ~60% for a build whose real figure is ~30%, and that is exactly the kind of number that ships
    # as a headline and then has to be retracted.  Pass --mesh 8192 for any FA_SP_SQ32 trace.
    mesh = int(sys.argv[sys.argv.index('--mesh')+1]) if '--mesh' in sys.argv else 16420
    seq = {0: [], 1: []}
    for line in open(path, errors='replace'):
        if '[ISSUE]' not in line: continue
        g = A.search(line)
        if not g: continue
        if (int(g.group(3)[-2:], 16) & 0x7F) != 0x23: continue           # SW
        if not (0x40050000 <= int(g.group(4), 16) < 0x40050200): continue
        seq[int(g.group(1))].append(int(g.group(5), 16))
    pooled = []
    for c in (0, 1):
        s = seq[c]
        if len(s) < 2: 
            print(f"  cluster {c}: {len(s)} marks -- nothing complete yet"); continue
        s = s[1:]                                                        # drop the pre-loop stamp
        assert all(b >= a for a, b in zip(s, s[1:])), f"cluster {c}: mark values not monotone"
        tops = s[0::per]
        iv = [b - a for a, b in zip(tops, tops[1:])]
        print(f"  cluster {c}: {len(s)} marks, {len(tops)} tile tops, intervals {iv}")
        pooled += iv[1:]
        nfull = len(s) // per
        if nfull >= 3:
            st = []
            for k in range(per):
                d = []
                for it in range(1, nfull):
                    a = s[it*per + k]
                    b = s[it*per + k + 1] if (it*per + k + 1) < len(s) else None
                    if b is not None: d.append(b - a)
                st.append(round(sum(d)/len(d)) if d else 0)
            print(f"    per-stage means (iters 1..{nfull-1}): {st}  sum {sum(st)}")
    if pooled:
        m = sum(pooled)/len(pooled)
        print(f"  POOLED steady intervals (n={len(pooled)}, drops the fill tile): mean {round(m)}  "
              f"min {min(pooled)} max {max(pooled)}   util = {mesh/m*100:.2f}%  "
              f"(mesh-busy {mesh}/interval{'' if mesh == 16420 else '  <-- HALF-TILE'})")
main()
