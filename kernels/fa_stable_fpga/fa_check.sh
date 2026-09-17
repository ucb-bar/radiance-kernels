#!/usr/bin/env bash
# fa_check.sh Sq Sk d [--seed=N] [--thresh=PCT] [--profile] [--build-only]
#
# One-command MX-FP8 flash-attention correctness check:
#   1) regenerate include/fa_data.h + golden_*.npy for the given sizes
#   2) build flash_attention_mx.soc.elf
#   3) run on VCS (prebuilt debug sim; NO waveform by default -> just the .out trace)
#   4) parse the SIMT store trace and compare O vs golden_O_normfirst; PASS/FAIL + cycles
#
# --profile keeps the FSDB waveform (for radiance-perf-viz); otherwise it is disabled
# (get_waveform_flag=) so the run is trace-only and much lighter on I/O.
set -uo pipefail
KG=/scratch/yrh/ai-workspace/kernel-gen
KDIR=$KG/radiance-kernels/kernels/flash_attention_mx
VCS=$KG/chipyard/sims/vcs
OUTD=$VCS/output/chipyard.harness.TestHarness.RadianceTapeoutSimConfig
ELF=$KDIR/flash_attention_mx.soc.elf
OUT=$OUTD/flash_attention_mx.soc.out
CFG=RadianceTapeoutSimConfig

[ $# -ge 3 ] || { echo "usage: fa_check.sh Sq Sk d [--bk=N] [--seed=N] [--thresh=PCT] [--profile] [--build-only]"; exit 2; }
Sq=$1; Sk=$2; d=$3; shift 3
seed=0; thresh=5.0; profile=0; buildonly=0; bk=0; nodram=0
for a in "$@"; do case $a in
  --bk=*)     bk=${a#*=} ;;
  --seed=*)   seed=${a#*=} ;;
  --thresh=*) thresh=${a#*=} ;;
  --profile)  profile=1 ;;
  --no-dramsim) nodram=1 ;;
  --build-only) buildonly=1 ;;
  *) echo "unknown arg: $a"; exit 2 ;;
esac; done
DRAMFLAG=""; [ $nodram = 1 ] && DRAMFLAG="NO_DRAMSIM=1"   # idealized memory timing
[ $bk = 0 ] && bk=$Sk    # default: single key block (Nblk=1)

source $KG/kernel-build-env.sh 2>/dev/null
cd $KDIR

echo "[1/4] gen data+goldens  Sq=$Sq Sk=$Sk d=$d bk=$bk seed=$seed"
python3 gen_data.py    --Sq $Sq --Sk $Sk --d $d --block_n $bk --seed $seed >/tmp/fa_gen.log 2>&1 || { echo "  GEN(data) FAIL"; tail -20 /tmp/fa_gen.log; exit 1; }
python3 fa_gen_goldens.py --Sq $Sq --Sk $Sk --d $d --block_n $bk --seed $seed >>/tmp/fa_gen.log 2>&1 || { echo "  GEN(golden) FAIL"; tail -20 /tmp/fa_gen.log; exit 1; }

echo "[2/4] build"
rm -f flash_attention_mx.mu.o
make flash_attention_mx.soc.elf >/tmp/fa_build.log 2>&1 || { echo "  BUILD FAIL"; grep -iE 'error' /tmp/fa_build.log | head; exit 1; }
[ $buildonly = 1 ] && { echo "  built (--build-only)"; exit 0; }

rm -f "$OUT" "$OUTD/flash_attention_mx.soc.fsdb"
cd $VCS
# Default: non-debug sim (run-binary) -> fast, emits the +verbose store trace (.out),
# no FSDB. --profile: debug sim (run-binary-debug) -> FSDB for radiance-perf-viz.
# BREAK_SIM_PREREQ=1 skips the (up-to-date) sim rebuild prereqs.
if [ $profile = 1 ]; then TGT=run-binary-debug; else TGT=run-binary; fi
echo "[3/4] run VCS  ($TGT)"
t0=$(date +%s)
make CONFIG=$CFG $TGT LOADMEM=1 BINARY=$ELF BREAK_SIM_PREREQ=1 $DRAMFLAG >/tmp/fa_run.log 2>&1
rc=$?; dt=$(( $(date +%s) - t0 ))
if ! grep -qaE '\$finish at' /tmp/fa_run.log; then
  echo "  RUN did not finish (rc=$rc, ${dt}s)"; tail -8 /tmp/fa_run.log; exit 1
fi
cyc=$(grep -aoE '[0-9]+ cycles' /tmp/fa_run.log "$OUT" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
echo "  finished in ${dt}s, ${cyc:-?} cycles"

echo "[4/4] verify O vs golden_O_normfirst"
nb=$(( Sq * d * 2 ))
rep=$(python3 $KDIR/fa_verify_out.py "$OUT" --base 0x40040000 --nbytes $nb \
        --golden $KDIR/golden_O_flash_u16.npy --rows $Sq --cols $d --elem-bytes 2 2>&1)
echo "$rep" | grep -iE 'cells covered|Frobenius'
err=$(echo "$rep" | grep -oE 'rel err vs golden: [0-9.]+' | grep -oE '[0-9.]+' | head -1)
if [ -z "$err" ]; then echo "RESULT: FAIL (no O parsed)"; exit 1; fi
pass=$(python3 -c "print(1 if float('$err')<=float('$thresh') else 0)")
if [ "$pass" = 1 ]; then echo "RESULT: PASS  (O rel err ${err}% <= ${thresh}%, ${cyc:-?} cyc)";
else echo "RESULT: FAIL  (O rel err ${err}% > ${thresh}%)"; exit 1; fi

if [ $profile = 1 ]; then
  echo "[profile] radiance-perf-viz on the FSDB"
  PV=/nscratch/yrh/claude/agent-mission-control/skills/radiance-perf-viz/scripts
  FSDB="$OUTD/flash_attention_mx.soc.fsdb"
  NPZ=$KDIR/fa_perf_${Sq}x${Sk}x${d}.npz
  bash $PV/radiance_perf.sh --fsdb "$FSDB" --out "$NPZ" --win 300 \
    2>&1 | grep -iE 'cycles|mesh_busy|fp_pipe|smem_gbps|gmem_gbps' | head
  python3 $PV/radiance_perf_plot.py --npz "$NPZ" \
    --out $KDIR/fa_perf_${Sq}x${Sk}x${d}.png --win 200 \
    --title "MX-FP8 FA ${Sq}x${Sk}x${d}" 2>&1 | tail -1
fi
