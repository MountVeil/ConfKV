#!/usr/bin/env bash

CONFKV_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

export CONFKV_ROOT
export LMCACHE_ROOT="$CONFKV_ROOT/LMCache"

# Always import LMCache from the pinned submodule.
export PYTHONPATH="$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ConfKV native CPU AES-GCM backend.
export LMCACHE_AESGCM_NATIVE_LIB="$CONFKV_ROOT/experiments/gpu_kv_seal/native/libcpu_seal.so"

# Do not assume that `python` exists.
export PYTHON_BIN="${PYTHON_BIN:-python3}"
