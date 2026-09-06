#!/usr/bin/env python3
"""Run an MX reference ELF on cyclotron, parse the `OUT <name> R C v0 v1 ...` float line
(gen_selfcontained.py protocol), compare to the capsule golden Y0 with its tolerance."""
import os
import subprocess
import sys
from pathlib import Path

import yaml

CYC = Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron")


def run(elf):
    work = Path(elf).parent
    cfg = work / "config"
    if not cfg.exists():
        cfg.symlink_to(CYC / "config")
    env = dict(os.environ)
    env["CYCLOTRON_MXGEMMINI"] = "1"
    env["RUST_LOG"] = "error"
    p = subprocess.run(
        [str(CYC / "target/release/cyclotron"), str(CYC / "config.toml"),
         "--binary-path", str(elf), "--timing", "--log", "0"],
        capture_output=True, text=True, cwd=str(work), env=env, timeout=900)
    return p.stdout + "\n" + p.stderr


def parse(console):
    for ln in console.splitlines():
        s = ln.strip()
        if s.startswith("OUT "):
            toks = s.split()
            # OUT <name> <rows> <cols> v0 v1 ...
            rows, cols = int(toks[2]), int(toks[3])
            vals = [float(x) for x in toks[4:]]
            return rows, cols, vals
    return None, None, None


def main():
    elf, golden_path = sys.argv[1], sys.argv[2]
    g = yaml.safe_load(Path(golden_path).read_text())
    Y = g["outputs"]["Y0"]
    M, N = len(Y), len(Y[0])
    gp = g["oracle_provenance"]["grade_policy"]
    atol, rtol = float(gp["atol"]), float(gp["rtol"])
    console = run(elf)
    rows, cols, vals = parse(console)
    if vals is None or len(vals) < M * N:
        print("FAIL: no/short OUT line; got", None if vals is None else len(vals))
        print(console[-1500:])
        return 1
    bad, worst = 0, 0.0
    for i in range(M):
        for j in range(N):
            got, exp = vals[i * N + j], float(Y[i][j])
            d = abs(got - exp)
            worst = max(worst, d)
            if d > atol + rtol * abs(exp):
                bad += 1
                if bad <= 12:
                    print(f"  mism [{i}][{j}] got={got} exp={exp} d={d:.5g} tol={atol+rtol*abs(exp):.5g}")
    print(f"atol={atol} rtol={rtol} worst_abs={worst:.6g} mismatches={bad}/{M*N}")
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
