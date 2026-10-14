#!/usr/bin/env python3
"""Per-channel DRAM traffic from FireSim FASED profiling (one memory_stats<N>.csv per DRAM channel).

On the Radiance HBM configs there are 5 channels: the host's and one per GPU L2 slice.  Each CSV row is one sample,
taken every `profile_interval` target cycles, of every FASED counter; totalReads / totalWrites are cumulative AXI
transaction counts (L2 misses / writebacks, not L2 hits).

  python3 dram_slices.py <results dir with memory_stats*.csv> [--interval N] [--plot out.png]

Prints the totals per channel, each channel's share of the GPU traffic, and (with --plot) reads and writes per
interval over time.  The host channel is the one with almost no traffic during the kernel.
"""
import argparse, csv, glob, os


def load(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    head = [h.strip() for h in rows[0] if h.strip()]
    data = []
    for r in rows[1:]:
        vals = [v.strip() for v in r if v.strip()]
        if len(vals) >= len(head):
            data.append(dict(zip(head, (int(v) for v in vals[:len(head)]))))
    return head, data


def col(head, *names):
    for n in names:
        for h in head:
            if h.endswith(n):
                return h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--interval", type=int, default=0, help="the profile_interval used (labels only)")
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "**", "memory_stats*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no memory_stats*.csv under {a.dir}")
    series = {}
    print(f"{'channel file':40s} {'samples':>8s} {'reads':>12s} {'writes':>12s}")
    for p in files:
        head, data = load(p)
        rc, wc = col(head, "totalReads"), col(head, "totalWrites")
        if not data or rc is None:
            print(f"{os.path.relpath(p, a.dir):40s} no totalReads column (columns: {head[:6]} ...)")
            continue
        r = [d[rc] for d in data]
        w = [d[wc] for d in data] if wc else [0] * len(r)
        # counters are 32-bit: unwrap
        def unwrap(xs):
            out, off, prev = [], 0, 0
            for x in xs:
                if x < prev:
                    off += 1 << 32
                prev = x
                out.append(x + off)
            return out
        r, w = unwrap(r), unwrap(w)
        series[p] = (r, w)
        print(f"{os.path.relpath(p, a.dir):40s} {len(r):8d} {r[-1]:12d} {w[-1]:12d}")
    tot_r = sum(s[0][-1] for s in series.values())
    tot_w = sum(s[1][-1] for s in series.values())
    print("\nshare of all DRAM transactions (reads / writes):")
    for p, (r, w) in series.items():
        print(f"  {os.path.basename(p):24s} {100 * r[-1] / max(tot_r, 1):6.2f}% / {100 * w[-1] / max(tot_w, 1):6.2f}%")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for p, (r, w) in series.items():
            dr = [b - c for b, c in zip(r[1:], r[:-1])]
            dw = [b - c for b, c in zip(w[1:], w[:-1])]
            x = range(1, len(r))
            ax[0].plot(x, dr, label=os.path.basename(p), lw=1)
            ax[1].plot(x, dw, label=os.path.basename(p), lw=1)
        unit = f"per {a.interval} target cycles" if a.interval else "per sample"
        ax[0].set_ylabel(f"reads {unit}")
        ax[1].set_ylabel(f"writes {unit}")
        ax[1].set_xlabel("sample")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"\nwrote {a.plot}")


if __name__ == "__main__":
    main()
