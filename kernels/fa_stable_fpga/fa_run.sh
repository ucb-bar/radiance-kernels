#!/usr/bin/env bash
# y_run.sh RUNTAG ELFTAG SEED [BUDGET_CYCLES]
# Same simulation recipe as `make ... run-binary` (+dramsim, +verbose) but:
#   * the ISSUE trace is grep-filtered ON THE FLY to the only lines any tool parses
#     (GMEM stores to 0x40040000-0x40043fff = O, and 0x4005xxxx/0x4006xxxx = MARK/host diag),
#     so the .out is a few MB instead of hundreds and lands on / (319 GB) not /scratch (23 GB);
#   * no spike-dasm, so there is no stderr-drain race between VCS exiting and the trace finishing
#     (a REAL PIPE is used, and bash waits for the whole pipeline -> the .out is complete when
#     this script returns);
#   * the seed is explicit (no +ntb_random_seed_automatic) so multi-seed validation is controlled.
# +max-cycles counts HALF cycles, hence the 2x.
set -uo pipefail
T=$1; E=$2; S=$3; BUDGET=${4:-600000}
MC=$((BUDGET*2))
OUT=/tmp/yruns; mkdir -p $OUT
DS=/scratch/yrh/chipyard/generators/testchipip/src/main/resources/dramsim2_ini
cd /scratch/yrh/chipyard/sims/vcs
echo "START $T elf=$E seed=$S budget=$BUDGET $(date)" > $OUT/$T.meta
./simv-chipyard.harness-RadianceTapeoutSimConfig \
  +permissive +dramsim +dramsim_ini_dir=$DS +max-cycles=$MC +gemmini_timeout=$MC \
  +ntb_random_seed=$S +loadmem=/tmp/flash_$E.elf +verbose +permissive-off /tmp/flash_$E.elf \
  </dev/null 2>&1 > $OUT/$T.log \
  | grep -E --line-buffered '4004[0-3][0-9a-f]{3}|4005[0-9a-f]{4}|4006[0-9a-f]{4}' > $OUT/$T.out
echo "DONE $T $(date) $(grep -m1 '^Time:' $OUT/$T.log) outbytes=$(stat -c %s $OUT/$T.out)" >> $OUT/$T.meta
