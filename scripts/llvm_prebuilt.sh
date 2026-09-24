#!/usr/bin/env bash
# Install the prebuilt Muon LLVM toolchain into llvm/llvm-muon.
#
# Downloads llvm-muon.tar.xz from the ucb-bar/muon-llvm GitHub release that matches the
# llvm/llvm-src submodule commit, checks its md5, and extracts it. Built on Ubuntu 24.04
# (glibc 2.39); on older systems use scripts/llvm.sh to build from source instead.
# Override the release with MUON_LLVM_TAG / MUON_LLVM_MD5, or pass a local tarball as $1.

set -euo pipefail

MUON_LLVM_TAG="${MUON_LLVM_TAG:-muon-2026-09-24}"                    # llvm-src d2f676d
MUON_LLVM_MD5="${MUON_LLVM_MD5:-8b25da8981a6d93df4b8a8ea956cbfd2}"
URL="https://github.com/ucb-bar/muon-llvm/releases/download/${MUON_LLVM_TAG}/llvm-muon.tar.xz"

SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
LLVM_BASE=$(realpath "${SCRIPT_PATH}/../llvm")
cd "${LLVM_BASE}"

if [ -e llvm-muon ]; then
    echo "Error: ${LLVM_BASE}/llvm-muon already exists; move it aside first." >&2
    exit 1
fi

TARBALL="${1:-llvm-muon.tar.xz}"
if [ ! -f "${TARBALL}" ]; then
    echo "Downloading ${URL}"
    curl -fL -o "${TARBALL}" "${URL}"
fi

if [ -n "${MUON_LLVM_MD5}" ]; then
    echo "${MUON_LLVM_MD5}  ${TARBALL}" | md5sum -c -
fi

tar xJf "${TARBALL}"
echo "Installed $("${LLVM_BASE}/llvm-muon/bin/clang" --version | head -1)"
