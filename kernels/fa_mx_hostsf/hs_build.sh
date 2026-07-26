#!/usr/bin/env bash
# hs_build.sh TAG "DEFINE1 DEFINE2 ..."  -> /tmp/flash_TAG.elf
#
# Builds in kernels/fa_mx_hostsf, which is a FROZEN snapshot of flash_attention_mx.cpp /
# flash_mx_impl.hpp (owned by other agents) plus LIVE copies of host.cpp / mxgemm_core.hpp
# (owned by me, synced from kernels/flash_attention_mx on every build).  This keeps A/B
# comparisons valid while two other agents are editing the kernel body in the main dir.
set -uo pipefail
KG=/scratch/yrh/ai-workspace/kernel-gen
MAIN=$KG/radiance-kernels/kernels/flash_attention_mx
KDIR=$KG/radiance-kernels/kernels/fa_mx_hostsf
SRC=$KDIR/flash_attention_mx.cpp
ELFSOC=$KDIR/flash_attention_mx.soc.elf
TAG=$1; DEFS=${2:-}
source $KG/kernel-build-env.sh 2>/dev/null

_LOCK=/tmp/hs_build.lock
_t=0
while ! mkdir "$_LOCK" 2>/dev/null; do
  sleep 5; _t=$((_t+5))
  if [ $_t -ge 1200 ]; then echo "hs_build: stealing stale lock after ${_t}s" >&2; rm -rf "$_LOCK"; mkdir "$_LOCK" 2>/dev/null || true; break; fi
done
trap 'cp -f /tmp/_hssrc_$TAG.bak "$SRC" 2>/dev/null; rmdir "$_LOCK" 2>/dev/null' EXIT INT TERM

# sync MY files (single source of truth = the main kernel dir)
cp -f $MAIN/host.cpp        $KDIR/host.cpp
cp -f $MAIN/mxgemm_core.hpp $KDIR/mxgemm_core.hpp

cd $KDIR
cp $SRC /tmp/_hssrc_$TAG.bak
{ for d in $DEFS; do echo "#define $d 1"; done; cat /tmp/_hssrc_$TAG.bak; } > $SRC
rm -f flash_attention_mx.mu.o host.host.o
export FA_HOST_DEFS="$(for d in $DEFS; do printf ' -D%s' "$d"; done)"
make flash_attention_mx.soc.elf >/tmp/hs_build_$TAG.log 2>&1
rc=$?
cp /tmp/_hssrc_$TAG.bak $SRC
if [ $rc -ne 0 ]; then echo "BUILD FAIL ($TAG)"; grep -iE 'error' /tmp/hs_build_$TAG.log | head; exit 1; fi
cp $ELFSOC /tmp/flash_$TAG.elf
echo "BUILT $TAG [$DEFS] -> /tmp/flash_$TAG.elf"
