#!/bin/sh
#
# Updates symlink to args.bin, input.a.bin, input.b.bin to point to the right
# binary according to the dimension size given as the argument.

if [ "$#" != "2" ]; then
    echo "usage: $0 DIMENSION 1|0"
    echo "second argument indicates using DMA or not."
    exit 1
fi

dim="$1"
dma="$2"
if [ "$2" == "1" ]; then
    layout_a="row.swizzle_fp16"
    layout_b="row"
else
    layout_a="col.swizzle_fp16"
    layout_b="row.swizzle_fp16"
fi

check_exists() {
    if ! [ -f "$1" ]; then
        echo "error: looked for file $1 that does not exist."
        exit 1
    fi
}

args="args.m$1n$1k$1.bin"
input_a="input.a.rand01.fp16.m$1n$1k$1.$layout_a.bin"
input_b="input.b.rand01.fp16.m$1n$1k$1.$layout_b.bin"
check_exists "$args"
check_exists "$input_a"
check_exists "$input_b"

echo "will symlink:"
echo "args.bin -> $args"
echo "input.a.bin -> $input_a"
echo "input.b.bin -> $input_b"
echo "continue? (Y/N)"
read -r -s -n 1 answer
if [ "$answer" != "Y" ] && [ "$answer" != "y" ]; then
  echo "exiting..."
  exit 1
fi

ln -sf -v "$args" "args.bin"
ln -sf -v "$input_a" "input.a.bin"
ln -sf -v "$input_b" "input.b.bin"

echo "done."
