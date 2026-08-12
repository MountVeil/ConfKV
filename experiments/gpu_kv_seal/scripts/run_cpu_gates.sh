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
echo " CPU AES-GCM Gates"
echo "========================================"

echo
echo "[1/4] Build native backend"
"$ROOT/native/build.sh"

echo
echo "[2/4] Security / interoperability"
"$PYTHON_BIN" "$ROOT/scripts/test_native_security.py"

echo
echo "[3/4] Backend switch"
"$PYTHON_BIN" "$ROOT/scripts/test_backend_switch.py"

echo
echo "[4/4] Async serde"
"$PYTHON_BIN" "$ROOT/scripts/test_native_async_serde.py"

echo
echo "========================================"
echo " ALL CPU AES-GCM GATES PASSED"
echo "========================================"
