#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <binfile>" >&2
    exit 1
fi

binfile="$1"
if [ ! -f "$binfile" ]; then
    echo "error: binfile not found: $binfile" >&2
    exit 1
fi

objfile="${binfile%.bin}.o"

sanitized_bin="$(printf '%s' "$binfile" | sed 's/[^A-Za-z0-9_]/_/g')"

llvm-objcopy \
    -I binary \
    -O elf32-littleriscv \
    -B riscv \
    --rename-section .data=.rodata.tensor,alloc,load,readonly,data,contents \
    --redefine-sym "_binary_${sanitized_bin}_start=${sanitized_bin}" \
    --redefine-sym "_binary_${sanitized_bin}_end=${sanitized_bin}_end" \
    --redefine-sym "_binary_${sanitized_bin}_size=${sanitized_bin}_size" \
    "$binfile" "$objfile"

echo "generated: ${objfile}; symbol: ${sanitized_bin}"
