#!/usr/bin/env bash
# fa_final_check.sh [TAG...] -- the correct way to score a steady-state FA_SP run.
# Waits for each run to report DONE and for its trace to stop growing, then reports the steady-state
# tile interval AND the per-CLUSTER per-tile O verdict.  Uses fa_verify_tiles.py, never fa_pertile.py
# (which mixes the two clusters and can report a FALSE PASS -- see fa_pertile.py's docstring).
HS=/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/fa_mx_hostsf
G=/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/flash_attention_mx/golden_O_u16.npy
for t in "$@"; do
  while ! grep -q DONE /tmp/yruns/$t.meta 2>/dev/null; do sleep 60; done
  a=-1; b=$(stat -c %s /tmp/yruns/$t.out); while [ "$a" != "$b" ]; do a=$b; sleep 8; b=$(stat -c %s /tmp/yruns/$t.out); done
  echo "######## $t"
  python3 /tmp/fa_pm.py /tmp/yruns/$t.out --per 7 2>/dev/null | grep -E "tile delta|STEADY"
  ( cd $HS && python3 fa_verify_tiles.py $t --out /tmp/yruns/$t.out --golden $G 2>&1 | tail -12 )
done
