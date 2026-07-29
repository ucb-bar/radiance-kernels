#!/usr/bin/env python3
"""Verify the softmax row-denominator l (fp32) from the .out store trace.

l_gmem is written as 32-bit word stores (one fp32 per row) at L_GMEM. Reconstruct
and compare (relative) to golden_l_f32.npy.
"""
import argparse, re, struct, sys
import numpy as np

# tmask is printed as one char per lane (rightmost char = lane 0): "...0001" = lane 0 only.
ISSUE = re.compile(r"\[ISSUE\].*?inst=([0-9a-fA-F]+).*?tmask=([0-9]+).*?rs1\.data=\[([0-9a-f ]+)\]"
                   r".*?rs2\.data=\[([0-9a-f ]+)\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--golden", default="golden_l_f32.npy")
    a = ap.parse_args()
    mem = {}
    for line in open(a.out, errors="ignore"):
        if "[ISSUE]" not in line:
            continue
        m = ISSUE.search(line)
        if not m or (int(m.group(1), 16) & 0x7F) != 0x23:
            continue
        tmask = m.group(2)
        addrs = [int(x, 16) for x in m.group(3).split()]
        data = [int(x, 16) for x in m.group(4).split()]
        for lane, (addr, d) in enumerate(zip(addrs, data)):
            # lane active iff its tmask char (from the right) is '1'
            if lane >= len(tmask) or tmask[len(tmask) - 1 - lane] != '1':
                continue
            if a.base <= addr < a.base + 4 * a.n:
                for b in range(4):
                    mem[addr + b] = (d >> (8 * b)) & 0xFF
    got = np.zeros(a.n, np.float32)
    cov = 0
    for i in range(a.n):
        ad = a.base + 4 * i
        if all((ad + b) in mem for b in range(4)):
            got[i] = struct.unpack("<f", bytes(mem[ad + b] for b in range(4)))[0]
            cov += 1
    g = np.load(a.golden).astype(np.float32)[: a.n]
    rel = float(np.linalg.norm(got - g) / (np.linalg.norm(g) + 1e-30))
    print(f"l covered {cov}/{a.n}; Frobenius rel err = {100*rel:.4f}%")
    print(f"  got[:4]={got[:4].tolist()}")
    print(f"  exp[:4]={g[:4].tolist()}")


if __name__ == "__main__":
    main()
