#!/usr/bin/env bash

SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
LLVM_BASE=$(realpath "${SCRIPT_PATH}/../llvm")

cd "${LLVM_BASE}"
tar xJf llvm-muon.tar.xz
