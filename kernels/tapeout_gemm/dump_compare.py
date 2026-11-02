#!/usr/bin/env python3
"""dump_compare.py <run log> M N K TM TN TK [--unit-scales] [--pattern=N]

Compares a TG_DUMP C dump against the golden that gen_data.py computes for the same flags.
The dump may be truncated (the board UART cuts long dumps); only complete rows are compared.
Reports exact matches, the bf16 code-distance histogram, non-finite or huge values, and the
mismatch pattern by row and 16-column block.
"""
import re
import sys
import tempfile

import numpy as np

import gen_data


def main():
    flags = [x for x in sys.argv[1:] if x.startswith("--")]
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    gen_data.UNIT_SCALES = "--unit-scales" in flags
    for x in flags:
        if x.startswith("--pattern="):
            gen_data.PATTERN = int(x.split("=")[1])
    log = open(args[0], errors="replace").read()
    m, n, k, tm, tn, tk = map(int, args[1:7])
    body = log.split("C_DUMP_BEGIN", 1)[1].split("C_DUMP_END", 1)[0]
    vals = np.array([int(x, 16) for x in re.findall(r"\b[0-9a-f]{4}\b", body)], dtype=np.uint16)
    rows = min(vals.size // n, m)
    got = vals[:rows * n].reshape(rows, n)
    a = gen_data.fp8_operand(m * k, gen_data.STREAM_A, cols=k).reshape(m, k)
    b = gen_data.fp8_operand(k * n, gen_data.STREAM_B).reshape(k, n)
    sa = gen_data.scales((k // 32) * m, gen_data.STREAM_SA, cols=m).reshape(k // 32, m)
    sb = gen_data.scales((k // 32) * n, gen_data.STREAM_SB, cols=n).reshape(k // 32, n)
    with tempfile.TemporaryDirectory() as tmp:
        gold = gen_data.golden.mx_matmul(a, b, sa, sb, m, n, k, tmpdir=tmp)[:rows]
    to_f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    o = to_f(got)
    print(f"rows compared {rows}/{m}; exact {(got == gold).sum()}/{rows * n}")
    d = np.abs(got.astype(np.int64) - gold.astype(np.int64))
    print("code distance 0,1,2,3,4-15,>=16:",
          [int((d == 0).sum()), int((d == 1).sum()), int((d == 2).sum()), int((d == 3).sum()),
           int(((d >= 4) & (d < 16)).sum()), int((d >= 16).sum())])
    huge = ~np.isfinite(o) | (np.abs(o) > 1e4)
    print(f"non-finite or |v|>1e4: {int(huge.sum())}")
    bad = d >= 4
    print("bad (dist>=4) per row:", bad.sum(axis=1).tolist())
    print("bad per 16-col block:", bad.reshape(rows, n // 16, 16).sum(axis=(0, 2)).tolist())


if __name__ == "__main__":
    main()
