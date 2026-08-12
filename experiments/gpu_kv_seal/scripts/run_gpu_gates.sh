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

echo "========================================"
echo " GPU AES-GCM Gates"
echo "========================================"

echo
echo "[1/4] Build"
"$ROOT/gpu/build.sh"

echo
echo "[2/4] ABI"
"$PYTHON_BIN" "$ROOT/scripts/test_gpu_abi.py"

run_runtime_gate() {
    local name="$1"
    local script="$2"
    local rc

    echo
    echo "$name"

    if "$PYTHON_BIN" "$script"; then
        echo "PASS: $name"
        return 0
    else
        rc=$?
    fi

    if [ "$rc" -eq 77 ]; then
        echo "SKIP: $name (CUDA device unavailable)"
        return 0
    fi

    echo "FAIL: $name (exit=$rc)"
    return "$rc"
}

run_runtime_gate \
    "[3/4] GPU/CPU interoperability" \
    "$ROOT/scripts/test_gpu_interop.py"

run_runtime_gate \
    "[4/4] GPU security" \
    "$ROOT/scripts/test_gpu_security.py"

echo
echo "========================================"
echo " GPU GATES COMPLETED"
echo "========================================"

echo
echo "Interpretation:"
echo "  ABI PASS + runtime SKIP => build/ABI ready, not runtime validated"
echo "  all PASS                => runtime correctness/security validated"
echo
echo "Performance benchmark:"
echo "  \"$PYTHON_BIN\" $ROOT/scripts/bench_gpu_aesgcm.py"
