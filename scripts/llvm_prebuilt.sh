#!/usr/bin/env bash

echo "This script is extremely outdated. TODO: bundle a newer llvm-muon" >&2

SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
LLVM_BASE=$(realpath "${SCRIPT_PATH}/../llvm")

cd "${LLVM_BASE}"
tar xJf llvm-muon.tar.xz