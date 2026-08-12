#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

g++ \
    -O3 \
    -DNDEBUG \
    -std=c++17 \
    -fPIC \
    -shared \
    "$HERE/cpu_seal_openssl.cpp" \
    -o "$HERE/libcpu_seal.so" \
    $(pkg-config --cflags --libs openssl)

echo "Built:"
ls -lh "$HERE/libcpu_seal.so"

echo
echo "Linked libraries:"
ldd "$HERE/libcpu_seal.so"
