#!/usr/bin/env python3
"""hs_extract.py TAG [MARKS_PER_TILE]

Parse a filtered cyclotron trace (/tmp/hsruns/TAG.out) and print
  * the kernel MARK array (0x40050000, runtime-indexed -> rs1.data is the exact address)
  * per-tile totals and the STEADY-STATE SLOPE
  * host<->GPU handshake spin stamps (0x40051000) and the host diag block (0x40052000)
  * the packed CPROF ring (0x40053000): (id<<20)|(mcycle & 0xFFFFF)

NOTE on trace parsing: the [ISSUE] trace prints rs1.data (the store's BASE REGISTER) but not the
S-type immediate, so only stores whose address is computed at runtime (or whose base pointer was
made opaque) can be located.  mxgemm_core.hpp keeps every diagnostic on its own 4KB page with an
opaque base for exactly this reason.
"""
import re, sys

ISSUE = re.compile(r"inst=([0-9a-fA-F]+).*?rs1\.data=\[([0-9a-f ]+)\].*?rs2\.data=\[([0-9a-f ]+)\]")

CPROF_LBL = {0x00:"QKpf.entry", 0x01:"QKpf.cfg(7 ROCC)", 0x02:"QKpf.mvin(5 ROCC)",
             0x03:"QKpf.scalewait", 0x04:"QKpf.end",
             0x10:"PVpf.entry", 0x11:"PVpf.cfg(7 ROCC)", 0x12:"PVpf.mvin(5 ROCC)",
             0x13:"PVpf.scalewait", 0x14:"PVpf.end",
             0x28:"QKmm.entry", 0x29:"QKmm.leadfence", 0x2a:"QKmm.cfgscale",
             0x2b:"QKmm.issue", 0x2c:"QKmm.trailfence",
             0x20:"PVmm.entry", 0x21:"PVmm.leadfence", 0x22:"PVmm.cfgscale",
             0x23:"PVmm.issue", 0x24:"PVmm.trailfence"}
DIAG = {0:"t_enter", 1:"t_prefill_done", 2:"tiles_served", 3:"t_end", 4:"host_busy_cyc",
        5:"host_wait_cyc", 6:"to_qk", 7:"to_pv", 8:"last_QKDONE", 9:"last_PVDONE",
        10:"probe_256rd_cyc", 11:"probe_64wr_cyc", 12:"probe_128wr_cyc", 13:"probe_sum",
        14:"cfg_capture_timeout", 15:"cfg_ncmds(qk<<8|pv)",
        16:"gpu_spin_qk", 17:"gpu_spin_v", 18:"gpu_tile"}
HSLBL = {0x88:"QKspin_start", 0x8c:"QKspin_end", 0x90:"Vspin_start", 0x94:"Vspin_end"}


def parse(path):
    last = {}     # (clid, addr) -> value  (last write wins)
    order = []    # (clid, addr, value) in trace order
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "[ISSUE]" not in line:
                continue
            m = ISSUE.search(line)
            if not m:
                continue
            if (int(m.group(1), 16) & 0x7F) != 0x23:
                continue
            clid = 1 if "clid=1" in line else 0
            tm = line.find("tmask=")
            mask = int(line[tm+6:line.find(" ", tm)], 16) if tm >= 0 else 0xFFFF
            # tmask is printed as ONE NIBBLE PER LANE (an all-lanes-active store shows
            # tmask=0x1111111111111111), so a bit-per-lane test keeps only lanes 0,4,8,12.
            for lane, (a, d) in enumerate(zip(m.group(2).split(), m.group(3).split())):
                if not (mask >> (4 * lane)) & 0xF:
                    continue
                ea = int(a, 16) & 0xFFFFFFFF
                if 0x40050000 <= ea < 0x40054000:
                    v = int(d, 16)
                    last[(clid, ea)] = v
                    order.append((clid, ea, v))
    return last, order


def main():
    tag = sys.argv[1]
    mpt = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    last, order = parse(f"/tmp/hsruns/{tag}.out")
    for clid in (0, 1):
        marks = sorted(((a - 0x40050000)//4, v) for (c, a), v in last.items()
                       if c == clid and 0x40050000 <= a < 0x40051000)
        if not marks:
            continue
        print(f"=== {tag} cluster {clid}: {len(marks)} marks ===")
        prev = None
        for i, v in marks:
            print(f"  m[{i:3d}] {v:9d}" + (f"  +{v-prev}" if prev is not None else ""))
            prev = v
        md = dict(marks)
        # m[0] is fa_entry's unconditional "0: entry" MARK (flash_attention_mx.cpp:271).  The FA_STEADY
        # loop body emits ELEVEN marks (T + 10: QKpf, QKmm, bar2, softmax, PVpf, requant,
        # pack+bar3, spare, PVmm, finalize) -- flash_attention_mx.cpp:626,643,645,653,662,688,694,
        # 719,730,737,744 -- so tile t's top is m[1 + 11t].  Getting this wrong silently mixes
        # phases across tiles and made the baseline look like a 90k/tile steady state.

        tops = [md[i] for i in range(1, max(md)+2, mpt) if i in md]
        if len(tops) >= 2:
            per = [(tops[k+1]-tops[k]) for k in range(len(tops)-1)]
            print(f"  tile tops m[1+{mpt}k]: {tops}")
            print(f"  per-tile deltas: {per}")
            if len(tops) >= 3:
                slope = (tops[-1]-tops[1]) / (len(tops)-2)
                print(f"  STEADY SLOPE (T[last]-T[1])/{len(tops)-2} = {slope:.0f} cyc/tile"
                      f"   -> mesh util = 16420/{slope:.0f} = {16420*100.0/slope:.2f}%")
            else:
                print(f"  STEADY (only 2 tops) = {per[0]} cyc/tile -> util {16420*100.0/per[0]:.2f}%")
        hs = {a-0x40051000: v for (c, a), v in last.items() if c == clid and 0x40051000 <= a < 0x40052000}
        if hs:
            print("  host-wait stamps: " + ", ".join(f"{HSLBL.get(a,hex(a))}={v}" for a, v in sorted(hs.items())))
        dg = {(a-0x40052000)//4: v for (c, a), v in last.items() if c == clid and 0x40052000 <= a < 0x40053000}
        if dg:
            print("  host diag: " + ", ".join(f"{DIAG.get(i,i)}={v}" for i, v in sorted(dg.items())))
        cp = [(a, v) for (c, a, v) in order if c == clid and 0x40053000 <= a < 0x40054000]
        if cp:
            print("  CPROF (packed id<<20 | cyc&0xFFFFF):")
            p = None
            for a, v in cp:
                i, cy = v >> 20, v & 0xFFFFF
                print(f"    {CPROF_LBL.get(i, hex(i)):20s} {cy:8d}" + (f"  +{cy-p}" if p is not None else ""))
                p = cy


if __name__ == "__main__":
    main()
