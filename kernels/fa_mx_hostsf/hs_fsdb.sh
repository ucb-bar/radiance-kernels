#!/usr/bin/env bash
# hs_fsdb.sh TAG MAXCYC DUMPSTART_CYC [SEED]
# Same simulator flags as hs_run.sh (NO +dramsim -- that shifts totals by ~35k cycles and would
# move the failure), plus the debug simv and an FSDB window.  MAXCYC / DUMPSTART are given in CORE
# CYCLES here and doubled internally (+max-cycles and +dump-start count TestDriver trace_count
# units == 2 x core cycles).
set -uo pipefail
TAG=$1; MAXCYC=$2; DUMPST=$3; SEED=${4:-12345}
OUT=/tmp/hsruns; mkdir -p $OUT
F=/tmp/hsfsdb_$TAG.fsdb
rm -f "$F"
cd /scratch/yrh/chipyard/sims/vcs
./simv-chipyard.harness-RadianceTapeoutSimConfig-debug \
  +permissive +max-cycles=$((MAXCYC*2)) +gemmini_timeout=$((MAXCYC*2)) +ntb_random_seed=$SEED \
  +loadmem=/tmp/flash_$TAG.elf +verbose +dump-start=$((DUMPST*2)) +fsdbfile=$F \
  +permissive-off /tmp/flash_$TAG.elf \
  </dev/null 2>&1 > $OUT/${TAG}_fsdb.log \
  | grep -E --line-buffered '4004[0-3][0-9a-f]{3}|4005[0-9a-f]{4}|4006[0-9a-f]{4}' > $OUT/${TAG}_fsdb.out
echo "FSDB_DONE tag=$TAG fsdb=$(stat -c %s $F 2>/dev/null) $(grep -m1 -oE 'at time [0-9]+ ps' $OUT/${TAG}_fsdb.log)" >> $OUT/${TAG}_fsdb.log
