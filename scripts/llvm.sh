#!/usr/bin/env bash

set -x

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

# Copy over C++ headers, __config_site, and __assertion_handler
# actually trying to use anything that allocates or asserts will probably lead to incomprehensible linker errors, 
# but header-only functionality should work (tm)
cd "${SCRIPT_PATH}/../llvm"
mkdir -p llvm-muon/include/c++
cp -r llvm-src/libcxx/include llvm-muon/include/c++/v1
cp "${SCRIPT_PATH}/__config_site" llvm-muon/include/c++/v1
cp llvm-src/libcxx/vendor/llvm/default_assertion_handler.in llvm-muon/include/c++/v1/__assertion_handler
