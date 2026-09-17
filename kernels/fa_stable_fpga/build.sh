#!/usr/bin/env bash
# build.sh TAG "DEF1 DEF2 ..."   -- private copy of flash_attention_mx_stable for FPGA bring-up.
# Sibling of the shared FA dirs so common.mk's relative LLVM_MUON / lib / mxgemmini paths all
# resolve against the SAME consistent on-disk tree the other tracks build against.  Sources copied
# from the on-disk (overlap-sq64) stable dir, which differs from main's 399757f only by comment
# renames plus an extra FA_ST_NOBOUNDS feature that is NOT enabled here.
set -uo pipefail
K=/scratch/yrh/radiance-kernels/kernels/fa_stable_fpga
TAG=$1; DEFS=${2:-}
source /scratch/yrh/ai-workspace/kernel-gen/kernel-build-env.sh 2>/dev/null
cd $K
cp -f kernel.cpp /tmp/_fsf_$TAG.bak
trap 'cp -f /tmp/_fsf_$TAG.bak $K/kernel.cpp' EXIT INT TERM
# NAME=VALUE tokens must become "#define NAME VALUE", not "#define NAME=VALUE 1"
{ for d in $DEFS; do case "$d" in *=*) echo "#define ${d%%=*} ${d#*=}";; *) echo "#define $d 1";; esac; done; cat /tmp/_fsf_$TAG.bak; } > kernel.cpp
export FA_HOST_DEFS="$(for d in $DEFS; do printf ' -D%s' "$d"; done)"
rm -f kernel.mu.o host.host.o kernel.radiance.elf kernel.soc.elf

# TWO-PASS when the C-scale overrun detector is on.  The host must be told the DEVICE address of
# the sentinel, and that address only exists once the rv32 ELF is linked.  Pass 1 builds the GPU
# image; the address is then read out of it with llvm-nm (never assumed); pass 2 builds only the
# host and fuses.  kernel.radiance.elf is already up to date at pass 2, so the GPU image -- and
# therefore the address -- cannot move between the two passes.
NM=/scratch/yrh/radiance-kernels/llvm/llvm-muon/bin/llvm-nm
RE=/scratch/yrh/radiance-kernels/llvm/llvm-muon/bin/llvm-readelf
case " $DEFS " in
*" FA_SCALE_GUARD "*)
  make kernel.radiance.elf > /tmp/fsf_$TAG.log 2>&1 || { echo "BUILD FAIL ($TAG, pass 1)"; grep -iE "error" /tmp/fsf_$TAG.log | head -6; exit 1; }
  LINE=$($NM kernel.radiance.elf | grep C_scale_block | head -1)
  BASE=$(echo "$LINE" | awk '{print $1}')
  TYPE=$(echo "$LINE" | awk '{print $2}')
  if [ -z "$BASE" ]; then echo "GUARD FAIL: C_scale_block not in kernel.radiance.elf"; exit 1; fi
  # GATE 1.  'd'/'D' is .data, which the loader writes.  'b'/'B' is .bss, which on this platform
  # is orphaned outside every LOAD segment and never initialised -- the sentinel would be stale
  # DRAM and every reading of it would be meaningless.  Fail loudly rather than measure noise.
  case "$TYPE" in
    d|D) : ;;
    *) echo "GUARD FAIL: C_scale_block has nm type '$TYPE', want d/D (.data).  A .bss symbol is"
       echo "            never loaded on this platform, so the sentinel would be uninitialised."; exit 1 ;;
  esac
  GADDR=$(printf "0x%08x" $(( 0x$BASE + 2048 )))
  # GATE 2.  The sentinel must fall inside some LOAD segment's FileSiz, not just its MemSiz --
  # bytes past FileSiz are not present in the image and are never written to the device.
  INSIDE=$($RE -l kernel.radiance.elf | awk -v a=$(( 0x$BASE + 2048 )) '
    $1=="LOAD" { v=strtonum($3); f=strtonum($5); if (a>=v && a+256<=v+f) print "yes" }' | head -1)
  if [ "$INSIDE" != "yes" ]; then
    echo "GUARD FAIL: guard at $GADDR is not inside any LOAD segment FileSiz -- it would never"
    echo "            reach the device, so a clean post-run read would prove nothing."
    $RE -l kernel.radiance.elf | grep -E "^ *LOAD"; exit 1
  fi
  echo "GUARD OK: C_scale_block=0x$BASE type=$TYPE  sentinel=$GADDR  inside a LOAD FileSiz"
  export FA_HOST_DEFS="$FA_HOST_DEFS -DFA_GUARD_ADDR=$GADDR"
  rm -f host.host.o kernel.soc.elf
  ;;
esac

# FA_CTX_PERCORE needs the ADDRESS of schedule_context, and that address is produced by the very
# build that consumes it -- adding the context-write code changes .text and can move the symbol.
# So iterate to a fixed point and GATE on it: the address compiled in must equal the address in
# the final image.  Without that gate a moved symbol writes the context to a stale location and
# the experiment silently tests nothing, which is the failure mode this campaign keeps producing.
case " $DEFS " in
*" FA_CTX_PERCORE "*)
  CTXADDR=""
  for pass in 1 2 3; do
    { for d in $DEFS; do case "$d" in *=*) echo "#define ${d%%=*} ${d#*=}";; *) echo "#define $d 1";; esac; done
      [ -n "$CTXADDR" ] && echo "#define FA_CTX_PERCORE $CTXADDR"
      cat /tmp/_fsf_$TAG.bak; } > kernel.cpp
    rm -f kernel.mu.o kernel.radiance.elf
    make kernel.radiance.elf > /tmp/fsf_$TAG.log 2>&1 || { echo "BUILD FAIL ($TAG, ctx pass $pass)"; grep -iE "error" /tmp/fsf_$TAG.log | head -6; exit 1; }
    FOUND=0x$($NM kernel.radiance.elf | grep -w _ZL16schedule_context | awk '{print $1}')
    if [ "$FOUND" = "0x" ]; then echo "CTX FAIL: schedule_context not found in the image"; exit 1; fi
    if [ "$CTXADDR" = "$FOUND" ]; then echo "CTX OK: schedule_context=$FOUND stable after $pass pass(es)"; break; fi
    CTXADDR=$FOUND
  done
  if [ "$CTXADDR" != "$FOUND" ]; then
    echo "CTX FAIL: address never settled (compiled $CTXADDR, image $FOUND) -- the context write"
    echo "          would land at a stale address and the test would measure nothing."; exit 1
  fi
  rm -f host.host.o kernel.soc.elf
  ;;
esac

make kernel.soc.elf > /tmp/fsf_$TAG.log 2>&1; rc=$?
cp -f /tmp/_fsf_$TAG.bak kernel.cpp
if [ $rc -ne 0 ]; then echo "BUILD FAIL ($TAG)"; grep -iE "error" /tmp/fsf_$TAG.log | head -6; exit 1; fi
cp kernel.soc.elf /tmp/fsf_$TAG.soc.elf; cp kernel.radiance.elf /tmp/fsf_$TAG.radiance.elf
echo "BUILT $TAG -> /tmp/fsf_$TAG.soc.elf ($(stat -c%s kernel.soc.elf) bytes)"
