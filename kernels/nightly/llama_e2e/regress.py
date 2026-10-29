#!/usr/bin/env python3
"""llama_e2e regression suite: every model shortened to 1 layer (SmolVLA: 2, one self- and one cross-attention expert
layer) for prefill, batch-16 decode and speculative verify, on 1 / 2 / 4 clusters.  The short runs take minutes on
FireSim instead of hours, and scale_report.py turns their per-op cycles into full-model estimates (within ~1 % of the
full runs it was validated against).

    regress.py build  [--sms "1 2 4"] [--py PY] [--vla-py PY] [--only NAME,...]   make every variant
    regress.py jobs   [--sms "1 2 4"] [--out FILE] [--only NAME,...]              write the FireSim jobs file
    regress.py report [--db PATH] [--since RUN_ID]                                latest result per variant and SM

`jobs` lists only ELFs that exist; run the file with run_all_profiling.py --jobs-file (the chipyard python).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scale_report  # noqa: E402

# (name, make variables): name = the ELF / data name make derives (llama_e2e_<name>.soc.elf); vla = the SmolVLA env.
# --check-l0 (real-weight models only) adds report-only layer-0 stage checks, which match the emulation to the device
# stage by stage; --exact-emu (prefill, real weights) makes the
# goldens in the device's arithmetic (emu_exact.py); timeout = the FireSim job timeout when 1800 s is not enough
VARIANTS = [
    ("tinyllama_l1", dict(MODEL="tinyllama", TAG="l1", ARGS="--layers 1 --check-l0 --exact-emu")),
    ("deepseek_l1", dict(MODEL="deepseek", TAG="l1", ARGS="--layers 1 --check-l0 --exact-emu")),
    ("llama7b_l1", dict(MODEL="llama7b", TAG="l1", ARGS="--layers 1")),
    ("gemma2_l1", dict(MODEL="gemma2", TAG="l1", ARGS="--layers 1")),
    ("smolvla_l2", dict(MODEL="smolvla", TAG="l2", ARGS="--layers 2", vla=True, timeout=3600)),   # 4 SM: > 30 min
    ("dec_tinyllama_l1", dict(MODEL="tinyllama", DECODE="1", TAG="l1", ARGS="--layers 1 --check-l0")),
    ("dec_deepseek_l1", dict(MODEL="deepseek", DECODE="1", TAG="l1", ARGS="--layers 1 --check-l0")),
    ("dec_llama7b_l1", dict(MODEL="llama7b", DECODE="1", TAG="l1", ARGS="--layers 1")),
    ("dec_gemma2_l1", dict(MODEL="gemma2", DECODE="1", TAG="l1", ARGS="--layers 1")),
    ("dec_smolvla_l2", dict(MODEL="smolvla", DECODE="1", TAG="l2", ARGS="--layers 2", vla=True)),
    # speculative verify, T = 4 tokens per step (the dashboard's spec-decode point)
    ("dec_llama7b_v3l1", dict(MODEL="llama7b", DECODE="1", TAG="v3l1", ARGS="--layers 1 --steps 2 --verify 3")),
]
BITSTREAM = {"1": "1SM_DRAM", "2": "2SM_DRAM", "4": "4SM_DRAM"}
FPGAS = '["60:00.0", "61:00.0", "b1:00.0", "b2:00.0", "db:00.0", "da:00.0"]'
DB = "/scratch/nicolas/radiance-runs.db"


def pick(only):
    if not only:
        return VARIANTS
    want = set(only.split(","))
    bad = want - {n for n, _ in VARIANTS}
    if bad:
        raise SystemExit(f"unknown variant(s): {', '.join(sorted(bad))}")
    return [v for v in VARIANTS if v[0] in want]


def elf(name, sm):
    return os.path.join(HERE, "build", f"{sm}sm", f"llama_e2e_{name}.soc.elf")


def build(a):
    for name, v in pick(a.only):
        mk = {k: x for k, x in v.items() if k not in ("vla", "timeout")}
        cmd = ["make", "-C", HERE, f"SMS={a.sms}", f"PY={a.vla_py if v.get('vla') else a.py}"]
        cmd += [f"{k}={x}" for k, x in mk.items()]
        print(f"== {name}: {' '.join(cmd[3:])}", flush=True)
        if subprocess.run(cmd).returncode:
            raise SystemExit(f"{name}: make failed")


def jobs(a):
    lines = ["# llama_e2e regression suite (regress.py jobs): 1-layer variants (SmolVLA 2), checked on the host.",
             "# NOTE: quote every BDF. Unquoted, YAML reads 60:00.0 as a time (=3600.0) and corrupts the card.",
             f"fpgas: {FPGAS}", "defaults: {kind: llm, timeout: 1800}", "jobs:"]
    missing = []
    for name, v in pick(a.only):
        for sm in a.sms.split():
            p = elf(name, sm)
            if os.path.exists(p):
                # the host checks layer 0 stage by stage with --check-l0: such runs took 20 - 30 minutes before
                to = 3600 if "--check-l0" in v["ARGS"] else v.get("timeout")
                to = f", timeout: {to}" if to else ""
                lines.append(f"  - {{name: {name}, bitstream: {BITSTREAM[sm]}, binary: {p}{to}}}")
            else:
                missing.append(p)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"{a.out}: {len(lines) - 5} jobs")
    for p in missing:
        print(f"  not built (skipped): {p}")


def report(a):
    import sqlite3
    c = sqlite3.connect(a.db)
    print(f"{'variant':20s} {'SM':>3s} {'run':>5s} {'status':10s} {'cycles':>14s} {'full model est.':>16s} "
          f"{'per step est.':>14s} {'mesh util':>9s}")
    for name, _ in VARIANTS:
        for sm in ("1", "2", "4"):
            row = c.execute("select id, status, raw_output from runs where workload=? and sm=? and id>=? "
                            "order by id desc limit 1", (name, f"{sm}sm", a.since)).fetchone()
            if not row:
                continue
            rid, st, out = row
            r = scale_report.parse(out or "")
            if r is None:
                print(f"{name:20s} {sm:>3s} {rid:5d} {st:10s} (no per-op table)")
                continue
            e = scale_report.estimate(r)
            ps = f"{e['per_step']:14,.0f}" if e["per_step"] else f"{'':14s}"
            print(f"{name:20s} {sm:>3s} {rid:5d} {st:10s} {r['total']:14,} {e['total']:16,.0f} {ps} {e['util']:8.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sms", default="1 2 4")
    b.add_argument("--py", default="python3", help="python for the LLM generators")
    b.add_argument("--vla-py", default="python3", help="python with lerobot[smolvla] for SmolVLA")
    b.add_argument("--only")
    j = sub.add_parser("jobs")
    j.add_argument("--sms", default="1 2 4")
    j.add_argument("--out", default="/scratch/nicolas/firesim-rundir/regress-jobs.yaml")
    j.add_argument("--only")
    r = sub.add_parser("report")
    r.add_argument("--db", default=DB)
    r.add_argument("--since", type=int, default=0, help="ignore runs before this id")
    a = ap.parse_args()
    {"build": build, "jobs": jobs, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
