#!/usr/bin/env bash
# build.sh <tag> "<MU defines>" "<HOST defines>"
# EXISTS BECAUSE: the muon sysroot's include/c++/v1 was WIPED, and the reconstructed bare-metal
# libc++ headers are injected via CPLUS_INCLUDE_PATH by kernel-build-env.sh.  mu_intrinsics.h
# includes <type_traits>, so ANY direct `make` without sourcing that script fails with
#   fatal error: 'type_traits' file not found
# That failure is an UNSOURCED ENV SCRIPT, not a missing header and not broken source.
# fa_stable_fpga/build.sh already did this; fpga_bringup had no build.sh, which is why direct
# `make` here failed while FA built fine at the same moment.
set -uo pipefail
K=/scratch/yrh/radiance-kernels/kernels/fpga_bringup
set +u; source /scratch/yrh/ai-workspace/kernel-gen/kernel-build-env.sh; set -u
cd $K || exit 1
TAG=${1:-default}; MUD=${2:-}; HOSTD=${3:-}
make clean >/dev/null 2>&1
MU_CFLAGS="$MUD" FB_HOST_DEFS="$HOSTD" make -j4 > /tmp/fb_$TAG.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "BUILD FAIL ($TAG)"; grep -iE "error" /tmp/fb_$TAG.log | head -5; exit 1; fi
echo "BUILT $TAG -> $K/kernel.soc.elf ($(stat -c%s kernel.soc.elf) bytes)"
