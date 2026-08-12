#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$(
    cd "$ROOT/../.."
    pwd
)"

echo "========================================"
echo " GPU AES-GCM Gate"
echo "========================================"

echo
echo "[1/4] Build"
"$ROOT/gpu/build.sh"

echo
echo "[2/4] ABI"
python "$ROOT/scripts/test_gpu_abi.py"

echo
echo "[3/4] Interoperability"
python "$ROOT/scripts/test_gpu_interop.py"

echo
echo "[4/4] Security"
python "$ROOT/scripts/test_gpu_security.py"

echo
echo "========================================"
echo " ALL GPU CORRECTNESS GATES PASSED"
echo "========================================"
echo
echo "Performance benchmark:"
echo "python $ROOT/scripts/bench_gpu_aesgcm.py"
