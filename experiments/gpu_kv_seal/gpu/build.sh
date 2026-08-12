#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "nvcc not found."
    echo "Set NVCC=/path/to/nvcc when CUDA toolkit is available."
    exit 1
fi

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
echo "Built:"
ls -lh "$ROOT/libgpu_aesgcm.so"
