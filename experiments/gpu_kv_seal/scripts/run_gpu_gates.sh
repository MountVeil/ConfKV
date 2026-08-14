#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

CONFKV_ROOT="$(
    cd "$ROOT/../.."
    pwd
)"

source "$CONFKV_ROOT/env.sh"
cd "$CONFKV_ROOT"

GATE_TMP="$(
    mktemp -d /tmp/confkv-gpu-gates.XXXXXX
)"
trap 'rm -rf -- "$GATE_TMP"' EXIT

echo "========================================"
echo " ConfKV GPU AES-GCM Gates (non-CC)"
echo "========================================"

echo
echo "[1/7] GPU AES-GCM build"
"$ROOT/gpu/build.sh"

echo
echo "[2/7] GPU ABI"
"$PYTHON_BIN" "$ROOT/scripts/test_gpu_abi.py"

run_runtime_gate() {
    local name="$1"
    shift
    local rc

    echo
    echo "$name"

    if "$@"; then
        echo "PASS: $name"
        return 0
    else
        rc=$?
    fi

    if [ "$rc" -eq 77 ]; then
        echo "SKIP: $name (CUDA runtime unavailable)"
        return 0
    fi

    echo "FAIL: $name (exit=$rc)"
    return "$rc"
}

run_runtime_gate \
    "[3/7] GPU/CPU interoperability" \
    "$PYTHON_BIN" \
    "$ROOT/scripts/test_gpu_interop.py"

run_runtime_gate \
    "[4/7] GPU authentication and tamper rejection" \
    "$PYTHON_BIN" \
    "$ROOT/scripts/test_gpu_security.py"

run_runtime_gate \
    "[5/7] ConfKV naive GPU data path" \
    env CONFKV_GPU_MODE=naive \
    "$PYTHON_BIN" \
    "$ROOT/scripts/test_confkv_parallel_gpu_crypto.py" \
    --slots 1 \
    --warmup 1 \
    --iterations 2

run_runtime_gate \
    "[6/7] ConfKV optimized GPU data path" \
    env CONFKV_GPU_MODE=optimized \
    "$PYTHON_BIN" \
    "$ROOT/scripts/test_confkv_parallel_gpu_crypto.py" \
    --slots 1 2 4 8 \
    --warmup 1 \
    --iterations 2

run_runtime_gate \
    "[7/7] Actual LMCache persistent storage path" \
    env CONFKV_GPU_MODE=optimized \
    "$PYTHON_BIN" \
    "$ROOT/scripts/test_actual_lmcache_b3_storage_path.py" \
    --data-dir "$GATE_TMP/l2"

echo
echo "========================================"
echo " NON-CC GPU DATA-PLANE GATES PASSED"
echo "========================================"

echo
echo "Validated:"
echo "  GPU AES-GCM build and ABI"
echo "  GPU/CPU interoperability"
echo "  authentication and tamper rejection"
echo "  ConfKV naive and optimized GPU paths"
echo "  LMCache GPU -> CPU -> L2 -> GPU path"
echo
echo "Not validated:"
echo "  TDX key generation and attestation"
echo "  TDX-to-GPU-CC protected key transfer"
echo "  GPU CC execution"
