#!/usr/bin/env bash
# fa_runtable.sh [rundir]   -- one line per run, read straight off disk.
#
# WHY THIS EXISTS: this campaign has repeatedly lost numbers that only ever existed in a launching
# agent's context.  Every run under $rundir carries its own DEFINES + RV32-segment sha (written by
# the build script) and its own trace, so the whole result set is reconstructible with no agent
# alive.  This just prints it.
#
# Columns:  tag | state | tile-images correct/total | flags
#   state:  DONE (the sim reached $finish) or RUN (still going -- partial images are reported as
#           INCOMPLETE by the scorer and are NOT counted either way).
# A run that is not N-of-N has a corrupt tile and therefore corrupt timing; its cycle number
# identifies the configuration and is NOT a performance result.
set -uo pipefail
DIR=${1:-/tmp/struns}
HERE=$(cd "$(dirname "$0")" && pwd)
printf '%-9s %-5s %-14s %s\n' TAG STATE IMAGES FLAGS
for m in "$DIR"/*.meta; do
  [ -e "$m" ] || continue
  t=$(basename "$m" .meta)
  out="$DIR/$t.out"
  [ -s "$out" ] || continue
  state=RUN; grep -q '^DONE' "$m" 2>/dev/null && state=DONE
  # renamer abort / RTL assert are FAILURES, not "0 correct" -- say so explicitly
  if grep -q "exceeded maximum number of physical registers" "$DIR/$t.log" 2>/dev/null; then
    img="RENAMER-ABORT"
  elif grep -q "Assertion failed" "$DIR/$t.log" 2>/dev/null; then
    img="RTL-ASSERT"
  else
    v=$(python3 "$HERE/fa_rowdiag.py" "$out" 2>/dev/null | grep -cE '  cl[01] t[0-9]+:.*CORRECT')
    w=$(python3 "$HERE/fa_rowdiag.py" "$out" 2>/dev/null | grep -cE '  cl[01] t[0-9]+:.*WRONG')
    img="$v ok / $w wrong"
  fi
  flags=$(head -1 "$DIR/$t.defines" 2>/dev/null | sed 's/FULL_ATTN2 //; s/FA_SP_//g; s/FA_//g')
  printf '%-9s %-5s %-14s %s\n' "$t" "$state" "$img" "${flags:-(no .defines)}"
done
