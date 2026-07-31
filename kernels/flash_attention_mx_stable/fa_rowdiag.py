#!/usr/bin/env python3
"""fa_rowdiag.py <trace.out> [<trace.out> ...] [--golden PATH]

PER-ROW diagnosis of a wrong O tile-image, on top of what fa_verify_tiles.py reports.

fa_verify_tiles.py answers "is this tile-image correct".  When it is not, the next question is
always the same -- WHICH STAGE produced the error -- and two cheap row-level statistics separate
the candidates without needing any extra instrumentation in the kernel:

  * NaN ROWS.  finalize_O computes O = O_unnorm / l, so a bf16 NaN (0x7FC0) row means l == 0 for
    that row, i.e. every exp in the row underflowed, i.e. S for that row is badly wrong -- OR that
    finalize read a region the PV move-out never wrote (uninitialised SMEM).  Either way it is
    upstream of finalize's arithmetic.
  * BEST SCALAR MULTIPLE.  If only `l` is wrong, the row is an exact scalar multiple of golden:
    fitting k = <got,gold>/<gold,gold> leaves ~0% residual.  A large residual after that fit means
    O_unnorm ITSELF is wrong, which exonerates l and the softmax denominator and indicts S / P8 /
    the P scales / PV.  This is the statistic that killed "it is just an l bug" for the
    un-overlapped FA_SP path (median residual 78%).

It also prints whether the two clusters agree row-by-row.  Identical fitted (k, residual) in both
clusters means the fault is DETERMINISTIC and data-dependent; rows where they differ are the
timing-dependent part.  That split is what distinguishes a functional defect from a race, and it
is why the un-overlapped path's failure could be separated from the campaign's hazard in one run.

INCOMPLETENESS IS REPORTED, NOT SCORED.  A group with fewer than 4096 words means the trace was
still being written; its Frobenius is ~sqrt(uncovered fraction) and is meaningless.  (A 4032/4096
image reads as 12.6%, which looks exactly like a real error.)
"""
import argparse
import re
import numpy as np

ISSUE = re.compile(r"inst=([0-9a-fA-F]+).*?tmask=([0-9a-fA-F]+)"
                   r".*?rs1\.data=\[([0-9a-f ]+)\].*?rs2\.data=\[([0-9a-f ]+)\]")
BASE, NW = 0x40040000, 4096          # O_GMEM, 64x128 bf16 = 4096 words


def groups_of(path):
    """Split one cluster's O stores into per-tile images, exactly as fa_verify_tiles.py does:
    walk in trace order and start a new group whenever an address repeats."""
    groups = {0: [], 1: []}
    cur = {0: {}, 1: {}}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "[ISSUE]" not in line:
                continue
            m = ISSUE.search(line)
            if not m or (int(m.group(1), 16) & 0x7F) != 0x23:
                continue
            c = 1 if "clid=1" in line else 0
            # tmask is ONE NIBBLE PER LANE, not one bit per lane.
            mask = int(m.group(2), 16)
            for lane, (a, d) in enumerate(zip(m.group(3).split(), m.group(4).split())):
                if not (mask >> (4 * lane)) & 0xF:
                    continue
                ea = int(a, 16) & 0xFFFFFFFF
                if not (BASE <= ea < BASE + 4 * NW):
                    continue
                w = (ea - BASE) // 4
                if w in cur[c]:
                    groups[c].append(cur[c])
                    cur[c] = {}
                cur[c][w] = int(d, 16) & 0xFFFFFFFF
    for c in (0, 1):
        if cur[c]:
            groups[c].append(cur[c])
    return groups


def image(grp):
    img = np.zeros(NW, dtype=np.uint32)
    for w, v in grp.items():
        img[w] = v
    u = np.zeros(2 * NW, dtype=np.uint16)
    u[0::2] = (img & 0xFFFF).astype(np.uint16)
    u[1::2] = (img >> 16).astype(np.uint16)
    return (u.reshape(64, 128).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--rows", action="store_true", help="print the per-row fit for every row")
    args = ap.parse_args()
    gpath = args.golden or __file__.rsplit("/", 1)[0] + "/golden_O_u16.npy"
    g = np.load(gpath).astype(np.uint16).reshape(64, 128)
    gf = (g.astype(np.uint32) << 16).view(np.float32)

    for path in args.traces:
        print(f"=== {path}")
        gr = groups_of(path)
        fits = {}
        for c in (0, 1):
            for k, grp in enumerate(gr[c]):
                f = image(grp)
                nan = np.isnan(f).any(axis=1)
                rel = float(np.linalg.norm(np.nan_to_num(f) - gf) / np.linalg.norm(gf))
                if len(grp) < NW:
                    verdict = f"INCOMPLETE({len(grp)}/{NW}) -- metric meaningless"
                elif abs(rel * 100 - 3.5666) < 0.05 and not nan.any():
                    verdict = "CORRECT"
                else:
                    verdict = "*** WRONG ***"
                ks, rs = {}, []
                for r in range(64):
                    if nan[r]:
                        continue
                    den = float(np.dot(gf[r], gf[r]))
                    kk = float(np.dot(f[r], gf[r])) / den if den else 0.0
                    ks[r] = kk
                    rs.append(float(np.linalg.norm(f[r] - kk * gf[r])
                                    / max(np.linalg.norm(f[r]), 1e-30)))
                    if args.rows:
                        print(f"      cl{c} t{k} row {r:2d}: k={kk:9.5f} resid={100*rs[-1]:7.3f}%")
                fits[(c, k)] = (ks, rs, np.where(nan)[0])
                line = (f"  cl{c} t{k}: words {len(grp):4d}/{NW}  Frobenius {100*rel:8.3f}%  "
                        f"{verdict}")
                if nan.any():
                    line += f"  [NaN rows {list(np.where(nan)[0][:12])}{'...' if nan.sum()>12 else ''}]"
                print(line)
                if verdict.startswith("***") and rs:
                    rsa = np.array(rs)
                    print(f"        scalar-multiple fit over {len(rs)} non-NaN rows: "
                          f"median resid {100*np.median(rsa):.2f}%, max {100*rsa.max():.2f}%  "
                          f"-> {'l-only error' if np.median(rsa) < 0.01 else 'O_unnorm itself is wrong'}")
        # cluster agreement, tile by tile
        for k in range(min(len(gr[0]), len(gr[1]))):
            if (0, k) in fits and (1, k) in fits:
                k0, _, n0 = fits[(0, k)]
                k1, _, n1 = fits[(1, k)]
                both = sorted(set(k0) & set(k1))
                if both:
                    same = sum(1 for r in both if abs(k0[r] - k1[r]) < 1e-6)
                    print(f"        tile {k}: {same}/{len(both)} rows non-NaN in BOTH clusters have "
                          f"IDENTICAL fits (identical => deterministic, not a race); "
                          f"NaN rows cl0={len(n0)} cl1={len(n1)}")


if __name__ == "__main__":
    main()
