#!/usr/bin/env python3
"""fa_regs3.py <linked GPU .elf> -- THE PER-WARP, PER-CORE RENAMER BUDGET, computed the way
Rename.scala actually counts, instead of the whole-file union both earlier tools report.

WHAT THE HARDWARE DOES (Rename.scala:110-123, MuonCore.scala:21-25):
    assigning = io.rename.valid && writesToRd && !assigned(wid)(rd)
    (globalCounter, overSubscription) = Counter(1 until numPhysRegs, enable = assigning)
    assert(!overSubscription, "total register usage exceeded maximum number of physical registers")
A physical register is handed out the FIRST time a given WARP writes a given ARCHITECTURAL register,
is never reclaimed, and comes from ONE counter per CORE over 1..255.  So the constraint is

    for each core:   sum over its resident warps of |arch regs THAT WARP EVER WRITES|  <=  255

A whole-file union cannot express that.  It OVER-counts (it adds registers that only the warp-0
agent path writes into the budget of warps that never execute it) and it collapses the three-warps-
per-core multiplier, which is the term that actually binds.  fa_regs.py's "53 runs / 55 $finishes"
is therefore a calibration of a proxy, not a budget -- and per the second pass's own note the
proxy already gave a FALSE PASS (FA_SP_CBMAX has the identical 53-name union and aborts).

WHAT THIS SCRIPT DOES: disassembles the LINKED GPU elf (so the runtime is included -- the kernel .s
alone misses _start/init_regs/mu_schedule, which every warp executes), takes each symbol's set of
written destination registers, assigns each symbol a ROLE, and sums per core under FA_SP's warp map
(mu_schedule: warp w -> core w&1; FA_SP_QOVL: warp 0 = gemmini agent, warps 1..5 = SIMT).

fa_entry is executed by ALL warps but CONTAINS the agent regions inline under `if (warp == 0)`, and
registers written only inside those regions are claimed only by warp 0 -- warps 1..5 branch around
them and never issue them.  Charging all of fa_entry to every warp is therefore an UPPER BOUND, and
an over-conservative metric is exactly how the union rule came to veto real wins.  So this tool
computes BOTH:
  * UPPER  -- all of fa_entry charged to every warp (what a naive symbol-level split gives);
  * TIGHT  -- fa_entry's basic blocks split by reachability from the warp-0 guard.  The guard is
    recognisable in the disassembly: FA_SP tests the warp id and branches over the agent region, so
    any block whose only predecessors are the taken side of such a branch, and which contains a call
    to an AGENT symbol, is warp-0-only.  Rather than reconstruct the CFG, TIGHT uses the sound and
    much simpler rule that any register written ONLY in blocks that also call an agent symbol is
    warp-0-only: it walks fa_entry linearly, splits it at calls to agent symbols, and attributes a
    register to the SIMT set only if it is written somewhere that is not inside such a span.
    *** AND THE DATA SAYS THIS CLAIM IS FALSE, SO DO NOT USE TIGHT AS A LIMIT. ***  It computes 210
    for FA_SP_SM1FX and 186 for FA_SP_CBMAX, both of which are OBSERVED to abort, against 156 for the
    build that runs -- so the span heuristic over-attributes to warp 0 and TIGHT is NOT a bound at
    all.  It is kept because it is still MONOTONE in the data (156 runs < 186 aborts < 210 aborts) and
    a second, independent monotone statistic is useful; it is not kept as a legality test.
HOW TO USE THIS TOOL, given both numbers are calibrated rather than absolute.  The observed points are
    UPPER 171 RUNS, 216 RUNS, 246 ABORTS, 270 ABORTS      -> the threshold lies in (216, 246]
    TIGHT 156 RUNS, 186 ABORTS, 210 ABORTS                -> the threshold lies in (156, 186]
so treat UPPER <= 216 (equivalently TIGHT <= 156) as SAFE, UPPER >= 246 as EXPECTED TO ABORT, and
anything between as UNRESOLVED -- to be settled by an eight-minute run, because the renamer assert
fires at ~197,000,000 ps.  What matters is that BOTH statistics order all the data correctly and the
whole-file union does not: it puts an aborting build (FA_SP_CBMAX, 53) and a running build (53) at
the same value, and ranks Sq=32 (55) above the running baseline (53) when its real figure is far
lower.  That is why the union could not have been rescued by moving its threshold.
"""
import re, subprocess, sys, collections

OBJDUMP = "/scratch/yrh/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
# Symbols only the warp-0 gemmini agent ever executes (all are called under `if (warp == 0)` and
# most also early-return on `tid != 0`).
AGENT = ("fa_cfg", "fa_mvin_", "fa_gf", "fa_scl", "fa_mm", "fa_store_acc", "load_scale_factors",
         "pack_scales_to_sfmem", "copy_scales_to_sfmem", "configure_mxgemmini", "fa_cfg_once",
         "copy_gmem_to_smem", "fa_sent", "gemmini")
RUNTIME = ("_start", "skip_nw", "init_regs", "init_regs_all", "vx_wspawn_wait", "mu_schedule",
           "main", "_exit")
# No destination register: stores, branches, and the SIMT control ops.
NO_RD = re.compile(r"^(s[bhwd]|s[bhwd]\.\w+|f?s[wd]|beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|"
                   r"bgez|bltz|bgtz|j|jr|ret|vx_bar|vx_tmc|vx_join|vx_split|vx_wspawn|vx_pred|"
                   r"fence|fence\.\w+|ecall|ebreak|nop|unimp|\.word|<unknown>)$")

def sets_from_elf(path):
    out = subprocess.run([OBJDUMP, "-d", "--triple=riscv32", path],
                         capture_output=True, text=True).stdout
    cur, regs = None, collections.defaultdict(set)
    for ln in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:", ln)
        if m:
            cur = m.group(1); regs[cur]  # touch
            continue
        if cur is None: continue
        m = re.match(r"^\s*[0-9a-f]+:(?:\s[0-9a-f]{2})+\s+([a-z][\w.]*)\s*(.*)$", ln)
        if not m: continue
        op, rest = m.group(1), m.group(2)
        if NO_RD.match(op): continue
        rd = rest.split(",")[0].strip()
        if re.match(r"^[a-z]+[0-9]*$", rd) and rd not in ("zero",):
            regs[cur].add(rd)
    return regs

def role(sym):
    if any(sym.startswith(r) or ("_" + r) in sym for r in RUNTIME): return "runtime"
    if any(a in sym for a in AGENT): return "agent"
    return "simt"          # fa_entry + every all-warps SIMT helper

def entry_split(path):
    """Split fa_entry's written registers into (warp0-only, all-warps) -- see TIGHT above."""
    out = subprocess.run([OBJDUMP, "-d", "--triple=riscv32", path],
                         capture_output=True, text=True).stdout
    inside, agent_only, common, in_span = False, set(), set(), False
    for ln in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:", ln)
        if m:
            inside = m.group(1).startswith("_Z8fa_entry"); continue
        if not inside: continue
        m = re.match(r"^\s*[0-9a-f]+:(?:\s[0-9a-f]{2})+\s+([a-z][\w.]*)\s*(.*)$", ln)
        if not m: continue
        op, rest = m.group(1), m.group(2)
        # a call to an agent symbol opens a warp-0-only span; a barrier closes it (every FA_SP
        # agent region ends at the stage barrier that all warps reach).
        if op in ("jal", "call") and any(a in rest for a in AGENT): in_span = True
        if "vx_bar" in op: in_span = False
        if NO_RD.match(op): continue
        rd = rest.split(",")[0].strip()
        if not re.match(r"^[a-z]+[0-9]*$", rd) or rd == "zero": continue
        (agent_only if in_span else common).add(rd)
    return agent_only - common, common

def main():
    regs = sets_from_elf(sys.argv[1])
    by = collections.defaultdict(set)
    for sym, s in regs.items():
        by[role(sym)] |= s
    rt, ag, si = by["runtime"], by["agent"], by["simt"]
    e_ag, e_common = entry_split(sys.argv[1])
    warp0 = rt | ag | si          # warp 0 runs the agent path AND fa_entry's scaffolding
    warpN = rt | si               # warps 1..5 -- UPPER: all of fa_entry charged to them
    warpT = warpN - e_ag          # TIGHT: fa_entry's warp-0-only spans removed
    core1 = 3 * len(warpN)        # warps {1,3,5}: three SIMT warps -- this is the binding core
    core0 = len(warp0) + 2 * len(warpN)
    c1t, c0t = 3 * len(warpT), len(warp0) + 2 * len(warpT)
    print(f"{sys.argv[1]}")
    print(f"  runtime-only regs {len(rt):3d}   agent-only {len(ag):3d}   all-warps(simt) {len(si):3d}")
    print(f"  per-warp union: warp0 {len(warp0):3d}   warps1-5 {len(warpN):3d}")
    print(f"  CORE 1 = 3 x {len(warpN)} = {core1:4d}   {'OK' if core1 <= 255 else '*** OVER 255 ***'}")
    print(f"  CORE 0 = {len(warp0)} + 2 x {len(warpN)} = {core0:4d}   "
          f"{'OK' if core0 <= 255 else '*** OVER 255 ***'}")
    print(f"  TIGHT (fa_entry's warp-0-only spans removed: {len(e_ag)} regs):")
    print(f"    warps1-5 {len(warpT):3d}   CORE 1 = {c1t:4d}   CORE 0 = {c0t:4d}   "
          f"{'OK' if max(c0t, c1t) <= 255 else '*** OVER 255 ***'}")
    print(f"  headroom on the binding core: UPPER {255 - max(core0, core1):4d} / "
          f"TIGHT {255 - max(c0t, c1t):4d} physical registers "
          f"(= {(255 - max(c0t, c1t)) // 3} more arch regs in an all-warps function)")
main()
