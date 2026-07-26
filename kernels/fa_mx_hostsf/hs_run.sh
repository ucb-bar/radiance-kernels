#!/usr/bin/env bash
# hs_run2.sh TAG [CYCLE_BUDGET] [SEED] -- like hs_run.sh but with a selectable seed, and it
# writes the trace to /tmp/hsruns/<TAG>.out (TAG should already encode the seed if it varies).
# +max-cycles counts HALF cycles (TIMEOUT_CYCLES=N gives N/2 cycles), hence the 2x.
set -uo pipefail
TAG=$1; BUDGET=${2:-600000}; SEED=${3:-12345}
MC=$((BUDGET*2))
OUT=/tmp/hsruns
mkdir -p $OUT
cd /scratch/yrh/chipyard/sims/vcs
./simv-chipyard.harness-RadianceTapeoutSimConfig \
  +permissive +max-cycles=$MC +gemmini_timeout=$MC +ntb_random_seed=$SEED \
  +loadmem=/tmp/flash_$TAG.elf +verbose +permissive-off /tmp/flash_$TAG.elf \
  </dev/null 2>&1 > $OUT/$TAG.log \
  | grep -E --line-buffered '4004[0-3][0-9a-f]{3}|4005[0-9a-f]{4}|4006[0-9a-f]{4}' > $OUT/$TAG.out
echo "SIM_DONE tag=$TAG seed=$SEED $(grep -m1 '^Time:' $OUT/$TAG.log)" >> $OUT/$TAG.log
