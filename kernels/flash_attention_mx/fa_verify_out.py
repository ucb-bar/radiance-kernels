#!/usr/bin/env python3
"""Verify a kernel's GMEM output by parsing the cyclotron instruction trace (.out).

The trace logs every SIMT issue as:
  [ISSUE] clid=.. cid=.. wid=.. pc=.. inst=<hex> tmask=<hex> rd=.. rs1=.. \
      rs1.data=[a0 .. a15] rs2=.. rs2.data=[d0 .. d15] rs3=[..]
For a store (sw, opcode 0x23 funct3=2) rs1.data holds the per-lane base address
and rs2.data the per-lane store word; effective addr = base + S-type immediate.

We collect all store words landing in [base, base+nbytes) and reconstruct the
tensor, then compare against a golden .npy/.npz of bf16 codes (uint16).

Usage: fa_verify_out.py <out_file> --base 0x40000000 --nbytes 0x8000 \
           --golden golden_S_u16.npy --rows 128 --cols 128
"""
import argparse
import re
import sys
import numpy as np

ISSUE = re.compile(r"\[ISSUE\].*?inst=([0-9a-fA-F]+).*?tmask=([0-9a-fA-F]+)"
                   r".*?rs1\.data=\[([0-9a-f ]+)\].*?rs2\.data=\[([0-9a-f ]+)\]")


def s_imm(inst):
    inst &= 0xFFFFFFFF
    imm = ((inst >> 25) << 5) | ((inst >> 7) & 0x1F)
    if imm & 0x800:
        imm -= 0x1000
    return imm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--nbytes", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--golden", required=True, help=".npy code array (uint8/uint16)")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--elem-bytes", type=int, default=2, choices=[1, 2],
                    help="element width: 1=fp8/e8m0 codes, 2=bf16 codes")
    args = ap.parse_args()

    eb = args.elem_bytes
    dt = np.uint8 if eb == 1 else np.uint16
    golden = np.load(args.golden).astype(dt).reshape(args.rows, args.cols)
    mem = {}  # byte addr -> byte value (little endian words)
    nstores = 0
    with open(args.out, "r", errors="ignore") as f:
        for line in f:
            if "[ISSUE]" not in line or "inst=" not in line:
                continue
            m = ISSUE.search(line)
            if not m:
                continue
            inst = int(m.group(1), 16)
            if (inst & 0x7F) != 0x23:           # not a STORE
                continue
            # Muon vectorized global store (opcode 0x23): effective address is the
            # per-lane value in rs1.data directly (no S-type immediate), 32-bit word.
            width = 4
            # tmask is printed as ONE NIBBLE PER LANE, not one bit: a store with all 16 lanes
            # active shows tmask=0x1111111111111111 (measured), so `(tmask >> lane) & 1` keeps
            # only lanes 0,4,8,12.  Skipping INACTIVE lanes matters because a masked-off lane's
            # rs1.data still holds a stale address that can fall inside the window.
            tmask = int(m.group(2), 16)
            addrs = [int(x, 16) for x in m.group(3).split()]
            data = [int(x, 16) for x in m.group(4).split()]
            for lane, (a, d) in enumerate(zip(addrs, data)):
                if not (tmask >> (4 * lane)) & 0xF:
                    continue
                ea = a & 0xFFFFFFFF
                if args.base <= ea < args.base + args.nbytes:
                    for b in range(width):
                        mem[ea + b] = (d >> (8 * b)) & 0xFF
                    nstores += 1

    # reconstruct tensor of elem_bytes-wide codes
    got = np.zeros(args.rows * args.cols, dtype=dt)
    covered = 0
    for i in range(args.rows * args.cols):
        a = args.base + eb * i
        if all((a + b) in mem for b in range(eb)):
            v = 0
            for b in range(eb):
                v |= mem[a + b] << (8 * b)
            got[i] = v
            covered += 1
    got = got.reshape(args.rows, args.cols)

    exact = int((got == golden).sum())
    total = args.rows * args.cols
    diff = got.astype(np.int32) - golden.astype(np.int32)
    within1 = int((np.abs(diff) <= 1).sum())
    print(f"store words parsed into range: {nstores}; cells covered: {covered}/{total}")
    print(f"exact match: {exact}/{total} ({100*exact/total:.2f}%)")
    print(f"within 1 code: {within1}/{total} ({100*within1/total:.2f}%)")
    # ---- COVERAGE GATE (2026-07-26) -------------------------------------------------------
    # Uncovered cells stay 0 in `got`, so an INCOMPLETE trace silently inflates the Frobenius
    # error and looks exactly like data corruption.  Measured on a KNOWN-GOOD run (lgfast):
    #   8192/8192 -> 3.5666%   8128 -> 10.61%   8064 -> 13.47%   7904 -> 19.03%   7712 -> 25.56%
    # i.e. Frobenius ~= sqrt(fraction uncovered).  The usual cause is reading the .out while
    # `spike-dasm` is still draining the simulator's stderr pipe: the VCS process exits first, so
    # "the run is done" is NOT enough -- wait until the .out file SIZE STOPS GROWING.
    incomplete = covered < total
    if incomplete:
        print(f"*** INCOMPLETE TRACE: only {covered}/{total} cells covered "
              f"({100.0*(total-covered)/total:.2f}% missing) -- EVERY metric below is MEANINGLESS. "
              f"Wait for the .out size to stop growing (spike-dasm lags the simulator) and re-run. ***")
    if eb == 2:  # bf16: float Frobenius rel err (the meaningful metric)
        gf = (got.astype(np.uint32) << 16).view(np.float32)
        ef = (golden.astype(np.uint32) << 16).view(np.float32)
        rel = float(np.linalg.norm(gf - ef) / (np.linalg.norm(ef) + 1e-30))
        nnan = int(np.isnan(gf).sum())
        print(f"float Frobenius rel err vs golden: {100*rel:.4f}%  "
              f"(max abs diff {np.abs(gf-ef).max():.4f})"
              + (f"  [{nnan} NaN cells]" if nnan else "")
              + ("   <-- INVALID, INCOMPLETE TRACE" if incomplete else ""))
    if covered:
        bad = np.argwhere(got != golden)
        for (r, c) in bad[:8]:
            print(f"  mismatch [{r},{c}]: got=0x{got[r,c]:04x} exp=0x{golden[r,c]:04x}")


if __name__ == "__main__":
    main()
