#!/usr/bin/env bash
# y_asm.sh TAG "DEFINES"  -> /tmp/fa_$TAG.s  (assembly only; for bar_audit.py + icache size)
set -u
KG=/scratch/yrh/ai-workspace/kernel-gen
KDIR=$KG/radiance-kernels/kernels/flash_attention_mx
source $KG/kernel-build-env.sh 2>/dev/null
TAG=$1; DEFS=${2:-}
D=""; for d in $DEFS; do D="$D -D$d"; done
cd $KDIR
/scratch/yrh/radiance-kernels/llvm/llvm-muon/bin/clang++ --sysroot=/scratch/yrh/radiance-kernels/llvm/llvm-muon -Xclang -target-feature \
 -Xclang +vortex -march=rv32im_zfinx_zhinx -mabi=ilp32 -O3 -std=c++20 -mcmodel=medany \
 -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections -mllvm -inline-threshold=262144 \
 -I/scratch/yrh/radiance-kernels/lib/include -I/scratch/yrh/radiance-kernels/lib/mxgemmini \
 -DRADIANCE -DRADIANCE_DEVICE -DNDEBUG -DLLVM_VORTEX \
 -include $KG/toolchain-fix/gemmini_host_shim.h $D -S -o /tmp/fa_$TAG.s kernel.cpp \
 2>/tmp/y_asm_$TAG.log || { echo "ASM FAIL $TAG"; tail -5 /tmp/y_asm_$TAG.log; exit 1; }
echo "WROTE /tmp/fa_$TAG.s ($(wc -l < /tmp/fa_$TAG.s) lines)"
