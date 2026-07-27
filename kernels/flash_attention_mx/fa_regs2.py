#!/usr/bin/env python3
"""fa_regs2.py <TAG.s> [REFERENCE.s] -- renamer budget check that fa_regs.py gets WRONG.

WHY THIS EXISTS.  fa_regs.py counts the union of destination registers over the WHOLE .text and the
header calibrates "53 runs / 55 $finishes" on that number.  That metric gave a FALSE PASS and cost
two 80-minute simulations: the reference FA_SP build and the same build + FA_SP_CBMAX have the
IDENTICAL 53-name union -- not one register appears in one and not the other -- and yet

    reference       whole-file union 53, softmax fn 30 regs   RUNS
    + FA_SP_CBMAX   whole-file union 53, softmax fn 33 regs   $finish at Rename.scala:123,
                                                              196,749,000 ps, mid tile 0

Rename.scala:110-123 bumps ONE per-core counter the first time a given WARP writes a given
architectural register (`assigning = valid && writesToRd && !assigned(wid)(rd)`), never reclaims it,
and asserts when it passes numPhysRegs.  MuonCore.scala:21-25: numWarps 8, numArchRegs 128,
numPhysRegs 256.  So the quantity that must stay under 256 is

    sum over the warps resident on that core of |arch regs THAT WARP writes|

and a warp only claims the registers of the code IT ACTUALLY EXECUTES.  A whole-file union cannot
see that.  FA_SP makes it matter a lot: mu_schedule maps warp w -> core (w & 1), FA_SP_QOVL makes
warp 0 the gemmini agent, so

    core 0 = warps {0, 2, 4} = 1 agent + 2 SIMT       core 1 = warps {1, 3, 5} = 3 SIMT

and CORE 1 -- three warps all running the SIMT path -- binds.  A register added inside a function
every warp runs (the softmax above all) costs 3x there; the same register inside an agent-only
function (`if (tid != 0) return;`, called under a warp-uniform `if (warp == 0)`) costs 1x.

THE RULE THIS SCRIPT CHECKS, therefore:
  (1) whole-file union <= 53   (necessary, not sufficient -- keep it as a coarse screen), AND
  (2) the per-function count of every ALL-WARPS function <= the reference build's.
Pass a reference .s as the second argument and it diffs (2) for you.

HONESTY ABOUT THE LIMITS OF (2).  I could not build a static model that SEPARATES the two known
data points.  The per-ROLE unions are identical too:

    reference and +FA_SP_CBMAX:  SIMT-role union 53, agent-role union 22, 3 x SIMT = 159

so 159 is nowhere near 256 and the difference cannot be a union at all -- it has to come from the
DYNAMIC claim, i.e. from which registers each warp actually writes along the path it takes, summed
over the eight warp SLOTS the runtime also touches.  Modelling that needs basic-block-level
reasoning about warp-uniform branches, which this script does not do.  What survives as a usable
heuristic is the per-function count of the all-warps functions, because that is the one thing that
moved between the two data points (softmax 30 -> 33).  Treat it as a smoke alarm, not a proof.

AND NOTE THE CHEAP HARDWARE GATE, which beats any static analysis: the renamer assert fires inside
the FIRST TILE, ~197M ps in, i.e. about EIGHT MINUTES of wall clock.  Launching the real FA_NT6 run
and grepping its .log for "Assertion failed" after ten minutes is a complete register check.
"""
import re
import sys
import collections

NOWRITE = {
    'sw', 'sb', 'sh', 'sw.shared', 'sw.global', 'sb.shared', 'sh.shared', 'sb.global', 'sh.global',
    'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'blez', 'bgez', 'bltz', 'bgtz',
    'j', 'jr', 'jal', 'jalr', 'ret', 'call', 'tail', 'fence', 'fence.s', 'ecall', 'ebreak', 'nop',
    'vx_bar', 'vx_split_n', 'vx_join', 'vx_tmc', 'vx_wspawn', 'vx_pred', 'vx_fence', 'unimp',
}
REG = re.compile(r'^(x\d+|ra|sp|gp|tp|t[0-6]|s\d+|a\d+)$')
# Functions every warp runs (the SIMT path).  Anything matching AGENT is entered only by warp 0.
SIMT = ('softmax', 'requant', 'finalize', 'rowmax', 'rowsum', 'expreq', 'expitem', 'invl',
        'tree_reduce', 'butterfly', 'prepack', 'dump_', 'copy_smem')
AGENT = ('pack_scales_to_sfmem', 'load_scale_factors', 'fa_mvin', 'fa_mm', 'fa_store_acc', 'fa_cfg',
         'fa_scl', 'fa_gf', 'fa_pack_range', 'copy_scales_to_sfmem', 'fap_', 'configure_mxgemmini',
         'copy_gmem_to_smem', 'copy_P_to_requant')


def parse(path):
    funcs = collections.OrderedDict()
    cur = None
    for line in open(path):
        m = re.match(r'^([A-Za-z_$.][\w.$]*):', line)
        if m and not m.group(1).startswith('.L'):
            cur = m.group(1)
            funcs.setdefault(cur, set())
            continue
        if cur is None or not line.startswith('\t'):
            continue
        parts = line.strip().split('#')[0].split(None, 1)
        if not parts:
            continue
        op = parts[0]
        if op.startswith('.') or op in NOWRITE or len(parts) < 2:
            continue
        d = parts[1].split(',')[0].strip()
        if REG.match(d):
            funcs[cur].add(d)
    return funcs


def role(name):
    if any(p in name for p in AGENT):
        return 'agent'
    if any(p in name for p in SIMT) or 'fa_entry' in name:
        return 'ALL-WARPS'
    return 'other'


def report(path, ref=None):
    f = parse(path)
    union = set().union(*f.values()) if f else set()
    print(f"{path}")
    print(f"  whole-file union: {len(union)} distinct written arch regs "
          f"({'OK' if len(union) <= 53 else 'OVER the empirical 53'})")
    rows = sorted(((len(v), k) for k, v in f.items() if v), reverse=True)
    refc = {}
    if ref:
        refc = {k: len(v) for k, v in parse(ref).items()}
    print(f"  {'function':44s} {'regs':>4s} {'role':>10s}  vs ref")
    for n, k in rows[:14]:
        r = role(k)
        delta = ''
        if refc:
            # match by the demangled-ish prefix, since template mangling can differ
            cand = [c for c in refc if c[:28] == k[:28]]
            if cand:
                d = n - refc[cand[0]]
                delta = f"{d:+d}" + ("   <-- RISK" if d > 0 and r == 'ALL-WARPS' else "")
            else:
                delta = "(new)" + ("   <-- CHECK" if r == 'ALL-WARPS' else "")
        print(f"  {k[:44]:44s} {n:4d} {r:>10s}  {delta}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
