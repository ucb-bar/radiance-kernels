#!/usr/bin/env python3
"""Load-phase bandwidth from memstress AGG lines.

usage: memstress_bw.py <bytes_per_core> <log>...

Each AGG line is `AGG core warps sum_store sum_flush sum_load max_store max_load err`. The load
phase of a core lasts max_load cycles (its slowest warp). Aggregate bandwidth is the bytes of all
reporting cores over the longest max_load, which assumes the cores' load phases overlap; the
per-core barriers make that approximately true. One DRAM channel peaks at 8 B/cycle.
"""
import sys

CHANNEL_PEAK = 8.0


def parse(path):
    cores = []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("AGG "):
                v = [int(x) for x in line.split()[1:]]
                cores.append({"core": v[0], "warps": v[1], "max_load": v[6], "err": v[7]})
    return cores


def main():
    per_core = int(sys.argv[1], 0)
    for path in sys.argv[2:]:
        cores = parse(path)
        if not cores:
            print("%-50s no AGG lines" % path)
            continue
        span = max(c["max_load"] for c in cores)
        bw = per_core * len(cores) / span
        print("%-50s cores=%d load_cycles=%d  %.2f B/cyc  %.1f%% of one channel  err=%d" % (
            path, len(cores), span, bw, 100 * bw / CHANNEL_PEAK, sum(c["err"] for c in cores)))


if __name__ == "__main__":
    main()
