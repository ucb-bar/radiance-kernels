#!/usr/bin/env bash

SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
LLVM_BASE=$(realpath "${SCRIPT_PATH}/../llvm")

cd "${LLVM_BASE}"
git submodule update --init llvm-src

cd llvm-src
rm -rf build
mkdir -p build
cd build

# Check if clang exists
if ! CLANG_PATH=$(command -v clang); then
    echo "Error: clang is not installed or not in your PATH."
    exit 1
fi

# Check if clang++ exists
if ! command -v clang++ >/dev/null 2>&1; then
    echo "Error: clang++ is not installed or not in your PATH."
    exit 1
fi

# Check if ninja exists
if ! command -v ninja >/dev/null 2>&1; then
    echo "Error: ninja is not installed or not in your PATH."
    exit 1
fi

CLANG_PATH="$(dirname ${CLANG_PATH})"

echo "found clang at ${CLANG_PATH}"

MUON_INSTALL_PATH="${LLVM_BASE}/llvm-muon"
../build.sh "${CLANG_PATH}" "${MUON_INSTALL_PATH}"

ninja && ninja install
