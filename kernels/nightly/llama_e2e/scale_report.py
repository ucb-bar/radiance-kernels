#!/usr/bin/env python3
"""Whole-model cycle estimates from short llama_e2e runs (1-2 layers), by scaling each op by its layer count.

    scale_report.py [--db PATH] RUN_ID|LOG ...            estimates for those runs
    scale_report.py [--db PATH] --validate SHORT:FULL ...  estimate SHORT's full model and compare with run FULL

A run's per-op table ("== per op") gives each op's call count and cycles.  Ops that run once per layer scale by
L_full / L_run; the rest (init, final norm, LM head, softcap; SmolVLA's patch embed, connector, Euler steps, ...) are
counted as they are.  SmolVLA has three stacks, scaled separately: vision (v*, 12 layers), prefix (p*, 16) and the
action expert (e*, plus x* = its cross-attention K / V, 16).  Its expert alternates self- and cross-attention layers,
so a SmolVLA short run needs --layers 2 (one of each); with --layers 1 the cross-attention ops are missing.
The gaps between phases (handshake_cycles: host round trips at > 1 cluster) scale with the number of phases.
Decode runs also report cycles per decode step (the step count does not change with the layer count).
"""
import argparse
import re
import sqlite3
import sys

DB = '/scratch/nicolas/radiance-runs.db'
FULL_LAYERS = {'tinyllama': 22, 'deepseek': 28, 'llama7b': 32, 'gemma2': 26}
VLA_STACKS = {'v': 12, 'p': 16, 'e': 16, 'x': 16}       # SmolVLA: op prefix -> the stack's full layer count
VLA_FIXED = {'vpatch', 'vaffine', 'vpost', 'pfinal', 'efinal'}
LLM_FIXED = {'init', 'final', 'lmhead', 'softcap'}


def parse(text):
    """The numbers scale_report needs from one llama_e2e log, or None if it has no per-op table."""
    m = re.search(r'llama_e2e (\S+)(?: expert)?( decode)?: .*? layers=(\d+) .*? clusters=(\d+)', text)
    if not m or '== per op' not in text:
        return None
    r = dict(model=m.group(1), decode=bool(m.group(2)), L=int(m.group(3)), nc=int(m.group(4)), ops={})
    for ln in text.split('== per op')[1].split('== model')[0].splitlines()[1:]:
        p = ln.split()
        if len(p) >= 4 and p[1].isdigit():
            r['ops'][p[0]] = (int(p[1]), int(p[2]), float(p[4].rstrip('%')) if len(p) > 4 else 0.0)
    g = re.search(r'model_cycles=(\d+) phase_cycles=(\d+) handshake_cycles=(\d+)', text)
    r['total'], r['busy'], r['hs'] = map(int, g.groups())
    s = re.search(r'decode: batch=\d+ steps=(\d+)', text)
    r['steps'] = int(s.group(1)) if s else 0
    return r


def factor(model, op, L):
    if model == 'smolvla':
        if op in VLA_FIXED or op[0] not in VLA_STACKS:
            return 1.0
        return VLA_STACKS[op[0]] / min(L, VLA_STACKS[op[0]])   # a run with --layers N has min(stack, N) per stack
    return 1.0 if op in LLM_FIXED else FULL_LAYERS[model] / L


def estimate(r):
    """Full-model cycles, mesh utilization and (decode) cycles per step for a parsed short run."""
    if r['model'] not in FULL_LAYERS and r['model'] != 'smolvla':
        raise SystemExit(f"no full layer count for model {r['model']}")
    if r['model'] == 'smolvla' and r['L'] < 2 and r['L'] < 16:
        print(f"warning: SmolVLA with {r['L']} layer(s) has no cross-attention layer; use --layers 2", file=sys.stderr)
    busy = calls = calls_full = macs = 0.0
    for op, (n, cyc, util) in r['ops'].items():
        f = factor(r['model'], op, r['L'])
        busy += cyc * f
        calls += n
        calls_full += n * f
        macs += util / 100 * 256 * r['nc'] * cyc * f      # MACs back from the op's mesh utilization
    total = busy + r['hs'] * (calls_full / calls if calls else 1)
    return dict(total=total, util=100 * macs / (256 * r['nc'] * total) if total else 0.0,
                per_step=total / r['steps'] if r['steps'] else None)


def load(db, ref):
    """A run id (from the runs DB) or a log path -> (label, parsed)."""
    if ref.isdigit():
        row = sqlite3.connect(db).execute('select workload, sm, raw_output from runs where id=?', (int(ref),)).fetchone()
        if not row:
            raise SystemExit(f'run {ref} not in {db}')
        return f'{ref} {row[0]} {row[1]}', parse(row[2] or '')
    return ref, parse(open(ref).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DB)
    ap.add_argument('--validate', action='store_true', help='arguments are SHORT:FULL pairs')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    if a.validate:
        print(f"{'short':40s} {'full':40s} {'estimate':>14s} {'measured':>14s} {'error':>7s}")
        for pair in a.runs:
            s, f = pair.split(':')
            (ls, rs), (lf, rf) = load(a.db, s), load(a.db, f)
            if rs is None or rf is None:
                print(f'{pair}: no per-op table')
                continue
            e = estimate(rs)['total']
            print(f"{ls:40s} {lf:40s} {e:14,.0f} {rf['total']:14,} {100 * (e / rf['total'] - 1):+6.2f}%")
        return
    print(f"{'run':40s} {'layers':>6s} {'cycles':>14s} {'full model est.':>16s} {'per step est.':>14s} {'mesh util':>9s}")
    for ref in a.runs:
        label, r = load(a.db, ref)
        if r is None:
            print(f'{label}: no per-op table (did the device part finish?)')
            continue
        e = estimate(r)
        ps = f"{e['per_step']:14,.0f}" if e['per_step'] else f"{'':14s}"
        print(f"{label:40s} {r['L']:6d} {r['total']:14,} {e['total']:16,.0f} {ps} {e['util']:8.2f}%")


if __name__ == '__main__':
    main()
