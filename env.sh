#!/usr/bin/env bash

CONFKV_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

export CONFKV_ROOT

# Use LMCache from the pinned submodule.
export PYTHONPATH="$CONFKV_ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"

# Default optimized CPU AES-GCM library.
export LMCACHE_AESGCM_NATIVE_LIB="$CONFKV_ROOT/experiments/gpu_kv_seal/native/libcpu_seal.so"
