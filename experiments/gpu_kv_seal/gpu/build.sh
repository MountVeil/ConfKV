#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "ERROR: nvcc not found."
    echo "Set NVCC=/path/to/nvcc."
    exit 1
fi

echo "========================================"
echo " ConfKV GPU AES-GCM build"
echo " Target: NVIDIA H100 / Hopper SM90"
echo "========================================"

"$NVCC" \
    -O3 \
    -DNDEBUG \
    -std=c++17 \
    -Xcompiler=-fPIC \
    --shared \
    -lineinfo \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_90,code=compute_90 \
    "$ROOT/gpu_aesgcm.cu" \
    -o "$ROOT/libgpu_aesgcm.so"

echo
echo "Built H100/SM90 library:"
ls -lh "$ROOT/libgpu_aesgcm.so"
