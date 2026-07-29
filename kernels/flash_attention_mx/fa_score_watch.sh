#!/usr/bin/env bash
# fa_score_watch.sh TAG MESH -- the missing half of fa_launch_safe.sh: make a COMPLETED run
# readable without the launching agent's context.
#
# fa_launch_safe.sh guarantees the sim survives the agent; it cannot score it.  This watcher is
# launched in its own setsid session too, waits for the launcher's "DONE" line, waits for the trace
# to STOP GROWING (VCS exits before the trace finishes draining; an incomplete trace scores uncovered
# cells as 0, so the Frobenius comes out as sqrt(fraction uncovered) -- that is how a perfectly good
# tile once got reported at 68%), then writes TAG.score.
#
# MESH is mesh-busy cycles per MARKED LOOP ITERATION: 16420 for a 64-row tile, 8192 for an
# FA_SP_SQ32 HALF-tile.  Passing the wrong one reported 59.97% for a build whose real figure was
# 29.92%, so it is a required argument, not a default.
#
# *** AND ONE TRAP ABOUT VERIFYING DETACHMENT, since I got this wrong in a launcher of my own. ***
# The obvious check -- assert the simv process has PPID 1 -- is WRONG for any chained launch.  simv is
# a GRANDCHILD of the detached process (setsid -> bash -> fa_run.sh -> simv), so its parent is
# legitimately the intermediate shell and the check reports "NOT DETACHED" for four runs that were
# perfectly fine.  The sound test is the SESSION ID: compare `ps -o sid= -p <simv_pid>` against your
# own shell's sid.  Different session => outside the process group the harness tears down when the
# agent dies.  fa_launch_safe.sh checks the launcher pid, which is the other correct place to look.
set -uo pipefail
K=/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/flash_attention_mx
HS=/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/kernels/fa_mx_hostsf
T="${1:?usage: fa_score_watch.sh TAG MESH}"; M="${2:?MESH is required: 16420 per tile, 8192 per SQ32 half-tile}"
R=/tmp/yruns
setsid nohup bash -c '
  T='"$T"'; M='"$M"'; K='"$K"'; HS='"$HS"'; R=/tmp/yruns
  G=$K/golden_O_u16.npy
  for i in $(seq 1 100000); do grep -q "DONE" $R/$T.meta 2>/dev/null && break; sleep 20; done
  a=-1; b=$(stat -c %s $R/$T.out 2>/dev/null||echo 0)
  while [ "$a" != "$b" ]; do a=$b; sleep 10; b=$(stat -c %s $R/$T.out 2>/dev/null||echo 0); done
  {
    echo "== $T   mesh-busy/interval=$M   $(grep -m1 TAG= $R/$T.meta)"
    if grep -qE "exceeded maximum number of physical registers" $R/$T.out 2>/dev/null; then
      echo "RENAMER ABORT -- register budget; cycle numbers meaningless"
    elif grep -q "Assertion failed" $R/$T.out 2>/dev/null; then
      echo "RTL ASSERT -- cycle numbers meaningless:"; grep -m1 "Assertion failed" $R/$T.out
    fi
    python3 $K/fa_marks3.py $R/$T.out --per 7 --mesh $M 2>&1 | grep -E "POOLED|per-stage means"
    ( cd $HS && python3 fa_verify_tiles.py $T --out $R/$T.out --golden $G 2>&1 | tail -14 )
    echo "-- RULE: not 12 of 12 => a corrupt tile => corrupt timing.  The cycle number then only"
    echo "-- identifies the configuration; it is NOT a performance result."
  } > $R/$T.score 2>&1
  echo "SCORED $(date "+%F %T")" >> $R/$T.meta
' </dev/null >>/tmp/yruns/$T.launch 2>&1 &
disown
echo "$T: scorer armed (mesh=$M)"
