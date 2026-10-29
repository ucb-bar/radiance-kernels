#!/usr/bin/env python3
"""llama_e2e offline checks: compare the device buffers host.cpp dumped (LL_OFFLINE) with the golden data in the ELF.

    check_offline.py <dump dir> <soc.elf or radiance.elf>

<dump dir> holds lldump.txt (the manifest host.cpp writes) and the lldump_*.bin files it names.  The golden buffers are
read from the kernel's rv32 ELF (the .radiance.elf the build leaves next to the .soc.elf), i.e. as loaded, not as the device left DRAM.  Prints
the same lines host.cpp's on-host checks print and PASS / FAIL with the same gates; exit status 0 = PASS, 1 = FAIL,
2 = the dump is unusable.  The arithmetic copies lib/include/nightly/host.h (nh_compare_bf16) bit for bit, including
its unsigned __int128 wraparound, so the numbers match what an on-host check would have printed.
"""
import math
import os
import struct
import sys

import numpy as np

U128 = 1 << 128


class Elf:
    """PT_LOAD segments of the kernel's rv32 ELF (<name>.radiance.elf next to the .soc.elf), readable by GPU address.
    Not the .soc.elf: its GPU segments and sections are scrambled into the hashed DRAM layout (soc/scramble_gpu_elf.py)
    for FireSim's direct DRAM load; the rv32 ELF from before that step holds the logical layout."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        h = self.f.read(64)
        if h[:4] != b'\x7fELF' or h[4] != 1:
            raise SystemExit(f'{path}: not an ELF32')
        phoff, = struct.unpack_from('<I', h, 0x1C)
        phentsize, phnum = struct.unpack_from('<HH', h, 0x2A)
        self.segs = []
        for i in range(phnum):
            self.f.seek(phoff + i * phentsize)
            p_type, off, _, paddr, filesz, memsz = struct.unpack_from('<IIIIII', self.f.read(24))
            if p_type == 1 and memsz:
                self.segs.append((paddr, filesz, memsz, off))

    def gpu(self, addr, n):
        for paddr, filesz, memsz, off in self.segs:
            if paddr <= addr and addr + n <= paddr + memsz:
                k = max(0, min(n, paddr + filesz - addr))
                self.f.seek(off + addr - paddr)
                return self.f.read(k) + bytes(n - k)
        raise KeyError(f'GPU address {addr:#x} + {n} not in any load segment')


def bf16_to_fix(b, scale_log2):
    """nh_bf16_to_fix, vectorized: bf16 bits -> int64 value * 2^scale_log2 (truncating)."""
    b = b.astype(np.int64)
    sign, e, mant = b >> 15, (b >> 7) & 0xFF, 0x80 | (b & 0x7F)
    sh = e - 127 - 7 + scale_log2
    v = np.where(sh >= 0, np.left_shift(mant, np.clip(sh, 0, 40)), np.right_shift(mant, np.clip(-sh, 0, 40)))
    v = np.where(sh > 40, np.int64(1 << 50), np.where(sh < -40, np.int64(0), v))
    v = np.where(e == 0, np.int64(0), np.where(e == 0xFF, np.int64(1 << 62), v))
    return np.where(sign == 1, -v, v)


def sum_squares(d):
    """exact sum of d^2 over an int64 array, as a Python int."""
    if d.size == 0:
        return 0
    if int(np.abs(d).max()) < (1 << 31):            # split d = hi * 2^16 + lo: every partial sum fits in int64
        hi, lo = d >> 16, d & 0xFFFF
        tot = 0
        for a in range(0, d.size, 1 << 22):
            h, l = hi[a:a + (1 << 22)], lo[a:a + (1 << 22)]
            tot += (int((h * h).sum()) << 32) + (int((h * l).sum()) << 17) + int((l * l).sum())
        return tot
    return sum(int(x) * int(x) for x in d.tolist())


def compare_bf16(got, ref, scale_log2):
    """nh_compare_bf16: (err_bp, exact, n, nonfinite)."""
    n = got.size
    exact = int((got == ref).sum())
    nonfin = int((((got >> 7) & 0xFF) == 0xFF).sum())
    gv, rv = bf16_to_fix(got, scale_log2), bf16_to_fix(ref, scale_log2)
    with np.errstate(over='ignore'):
        d = gv - rv                                  # int64 wraparound, as in the C code
    se, sr = sum_squares(d) % U128, sum_squares(rv) % U128
    if sr == 0:
        return (1000000 if se else 0), exact, n, nonfin
    return math.isqrt(((se * 100000000) % U128) // sr), exact, n, nonfin


def bf_key(x):
    x = x.astype(np.int32)
    return np.where(x & 0x8000, -(x & 0x7FFF), x)


def pct(bp):
    return f'{bp // 100}.{bp % 100:02d}%'


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ep = sys.argv[2]
    if ep.endswith('.soc.elf'):
        ep = ep[:-len('.soc.elf')] + '.radiance.elf'
    ddir, elf = sys.argv[1], Elf(ep)
    man = os.path.join(ddir, 'lldump.txt')
    if not os.path.exists(man):
        print(f'offline_check: no manifest {man}')
        return 2
    lines = open(man).read().splitlines()
    if not lines or not lines[-1].startswith('end ok'):
        print('offline_check: incomplete dump (the host stopped before the manifest end)')
        return 2
    print('== offline checks (check_offline.py): ' + lines[0])
    passed = True
    for ln in lines[1:-1]:
        f = ln.split('|')
        if f[0] == 'check':
            name, fn, dev, gold, n, scale, isbytes, tol = f[1], f[2], *map(int, f[3:])
            got = np.fromfile(os.path.join(ddir, fn), np.uint8 if isbytes else np.uint16)
            ref = np.frombuffer(elf.gpu(gold, n if isbytes else 2 * n), np.uint8 if isbytes else np.uint16)
            if got.size != ref.size:
                print(f'{name}: dump has {got.size} elements, want {ref.size}')
                passed = False
                continue
            if isbytes:
                bad = np.nonzero(got != ref)[0]
                s = f'{name}: mismatched_bytes={bad.size}/{n}'
                if tol:
                    s += f' (allowed {tol})'
                if bad.size:
                    k = int(bad[0])
                    s += f' first_at={k} got={got[k]:#04x} want={ref[k]:#04x}'
                print(s)
                passed &= bad.size <= tol
                continue
            err, exact, n2, nonfin = compare_bf16(got, ref, scale)
            print(f'{name}: frob_rel_err={pct(err)} exact={exact}/{n2} nonfinite={nonfin}')
            # tol 0: bit-exact required; tol > 0: relative Frobenius error <= tol (units of 0.01 %)
            if nonfin or (err > tol if tol else exact != n2):
                passed = False
        elif f[0] == 'logits':
            fn, _, M, V, hf, emu, labels = f[1], *map(int, f[2:])
            lg = np.fromfile(os.path.join(ddir, fn), np.uint16).reshape(M, V)
            nonfin = int((((lg >> 7) & 0xFF) == 0xFF).sum())
            best = bf_key(lg).argmax(1)                  # first maximum, as the host's strict '>'
            s = f'logits: nonfinite={nonfin}'
            bh = None
            if emu:
                be = bf_key(np.frombuffer(elf.gpu(emu, 2 * M * V), np.uint16).reshape(M, V)).argmax(1)
                s += f' top1_vs_emulation={int((best == be).sum())}/{M}'
            if hf:
                bh = bf_key(np.frombuffer(elf.gpu(hf, 2 * M * V), np.uint16).reshape(M, V)).argmax(1)
                agree_hf = int((best == bh).sum())
                s += f' top1_vs_HF={agree_hf}/{M}'
            if labels:
                lab = np.frombuffer(elf.gpu(labels, 4 * M), np.uint32)
                s += f' top1_vs_next_token={int((best == lab).sum())}/{M}'
                s += f' (HF fp32: {int((bh == lab).sum()) if bh is not None else 0})'
            print(s)
            passed &= nonfin == 0
            # real weights: >= 85 % top-1 agreement with the fp32 HF model (host.cpp's gate)
            if hf and agree_hf * 20 < M * 17:
                passed = False
        else:
            print(f'offline_check: unknown manifest line: {ln}')
            return 2
    print('PASS' if passed else 'FAIL')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
