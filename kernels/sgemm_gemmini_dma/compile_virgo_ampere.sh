#!/bin/sh
#
# This script generates the 8-core-per-cluster version of Virgo GEMM kernels.
# We use the 4-core version for final evaluation; the 8-core kernels should
# behave identically.

if [ ! -f input.a.rand01.fp16.m256n256k256.row.bin ]; then
    echo "input binaries not found, generating operands"
    python3 generate_operands.py
fi

for a in args/*; do
    echo "compiling GEMM kernel for Virgo with dim ${a}"
    cp -f $a args.bin
    aa=$(basename "$a")
    cp -f input.a.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.a.bin
    cp -f input.b.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.b.bin
    touch input.c.bin

    # touch source file to force re-building, as the Makefile does not track
    # binary changes
    touch kernel.cpp

    make CONFIG=gemm.virgo.ampere.dim${aa}
done
