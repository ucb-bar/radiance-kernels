#!/usr/bin/env python3
"""hs_tiles.py TAG [--golden PATH]

PER-TILE correctness for an FA_STEADY run.

`fa_verify_out.py` merges every store into one image, so it only ever checks the LAST tile of the
LAST cluster to write a cell -- it cannot see "tile 0 right, tiles 1..3 wrong".  Under FA_STEADY
every iteration recomputes the identical result and `finalize_O` writes each of the 4096 O words
exactly once per tile per cluster, so the store stream splits cleanly: walk the O stores of one
cluster in trace order and start a new group whenever an address repeats.

Group k of cluster c is therefore cluster c's tile-k output.  A group with fewer than 4096 words is
INCOMPLETE (the trace was cut) and is reported as such -- NOT as a wrong answer.  That distinction
is the whole point of this script: a truncated final group scores ~sqrt(fraction uncovered), which
is how a perfectly good tile 3 gets reported at 68%.
"""
import argparse
import re
import numpy as np

ISSUE = re.compile(r"inst=([0-9a-fA-F]+).*?tmask=([0-9a-fA-F]+)"
                   r".*?rs1\.data=\[([0-9a-f ]+)\].*?rs2\.data=\[([0-9a-f ]+)\]")
BASE, NW = 0x40040000, 4096          # O_GMEM, 64x128 bf16 = 4096 words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default=None)
    ap.add_argument("--golden",
                    default="/scratch/yrh/radiance-kernels/kernels/fa_mx_hostsf/golden_O_u16.npy")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = args.out or f"/tmp/hsruns/{args.tag}.out"
    golden = np.load(args.golden).astype(np.uint16).reshape(64, 128)
    gf = (golden.astype(np.uint32) << 16).view(np.float32)

    groups = {0: [], 1: []}          # clid -> list of {word_index: value}
    cur = {0: {}, 1: {}}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "[ISSUE]" not in line:
                continue
            m = ISSUE.search(line)
            if not m or (int(m.group(1), 16) & 0x7F) != 0x23:
                continue
            clid = 1 if "clid=1" in line else 0
            # tmask is printed as ONE NIBBLE PER LANE, not one bit: an all-16-lanes-active
            # store shows tmask=0x1111111111111111, and `(mask >> lane) & 1` therefore keeps
            # only lanes 0,4,8,12 -- which made every tile image look 1024/4096 covered.
            mask = int(m.group(2), 16)
            for lane, (a, d) in enumerate(zip(m.group(3).split(), m.group(4).split())):
                if not (mask >> (4 * lane)) & 0xF:
                    continue
                ea = int(a, 16) & 0xFFFFFFFF
                if not (BASE <= ea < BASE + 4 * NW):
                    continue
                w = (ea - BASE) // 4
                if w in cur[clid]:                     # address repeats -> next tile
                    groups[clid].append(cur[clid])
                    cur[clid] = {}
                cur[clid][w] = int(d, 16) & 0xFFFFFFFF
    for c in (0, 1):
        if cur[c]:
            groups[c].append(cur[c])

    print(f"### {args.tag}: per-tile O check (golden {args.golden.split('/')[-1]}, "
          f"3.5666% == CORRECT)")
    verdicts = []
    for c in (0, 1):
        for k, g in enumerate(groups[c]):
            img = np.zeros(NW, dtype=np.uint32)
            for w, v in g.items():
                img[w] = v
            u16 = np.zeros(2 * NW, dtype=np.uint16)
            u16[0::2] = (img & 0xFFFF).astype(np.uint16)
            u16[1::2] = (img >> 16).astype(np.uint16)
            got = u16.reshape(64, 128)
            cov = len(g)
            f32 = (got.astype(np.uint32) << 16).view(np.float32)
            rel = float(np.linalg.norm(f32 - gf) / np.linalg.norm(gf))
            nnan = int(np.isnan(f32).sum())
            if cov < NW:
                verdict = f"INCOMPLETE ({cov}/{NW} words) -- metric meaningless"
            elif abs(rel * 100 - 3.5666) < 0.05:
                verdict = "CORRECT"
            else:
                verdict = "*** WRONG ***"
            verdicts.append((c, k, cov, rel, verdict))
            print(f"  cluster {c} tile {k}: words {cov:5d}/{NW}  Frobenius {100*rel:8.4f}%"
                  + (f"  [{nnan} NaN]" if nnan else "") + f"  {verdict}")
    if not verdicts:
        print("  (no O stores found in the trace)")
    ok = [v for v in verdicts if v[4] == "CORRECT"]
    bad = [v for v in verdicts if v[4].startswith("***")]
    inc = [v for v in verdicts if v[4].startswith("INCOMPLETE")]
    print(f"  SUMMARY: {len(ok)} correct, {len(bad)} wrong, {len(inc)} incomplete "
          f"(of {len(verdicts)} tile-images)")


if __name__ == "__main__":
    main()
