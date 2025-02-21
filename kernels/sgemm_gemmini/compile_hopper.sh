#!/bin/sh

for a in args/*; do
    echo "compiling GEMM kernel for Virgo with dim ${a}"
    cp -f $a args.bin
    aa=$(basename "$a")
    cp ../sgemm_gemmini_dma/input.a.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.a.bin
    cp ../sgemm_gemmini_dma/input.b.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.b.bin
    touch input.c.bin

    # touch source file to force re-building, as the Makefile does not track
    # binary changes
    touch kernel.cpp

    make CONFIG=gemm.virgo.hopper.nodma.dim${aa}
done
