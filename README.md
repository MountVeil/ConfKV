# ConfKV

ConfKV is a research prototype for confidential persistent KV-cache
protection in LLM serving. It is built on top of LMCache and explores
where cryptographic protection should be placed across CPU and GPU
execution paths.

## Motivation

Persistent KV caching introduces a data-lifecycle problem in
confidential LLM serving: KV state may move from confidential GPU
memory into CPU memory and persistent storage.

ConfKV studies cryptographic endpoint placement for this path,
including CPU-side and GPU-side AES-GCM protection.

The current GPU design targets the following persistent-store path:

    GPU plaintext KV
        |
        v
    GPU AES-GCM seal
        |
        v
    GPU persistent ciphertext
        |
        v
    D2H
        |
        v
    CPU / storage relay
        |
        v
    persistent storage

The CPU continues to manage cache metadata, ObjectKeys, lookup,
reservation, and storage decisions. GPU cryptography is only applied
to KV objects selected for persistence.

## Current Status

### CPU

- LMCache stock AES-GCM baseline
- optimized OpenSSL EVP AES-128-GCM backend
- stock/native interoperability
- wrong-key and tamper rejection
- failed-open zeroization
- LMCache serde backend switching
- CPU microbenchmarks

### GPU

Implemented:

- CUDA AES-128-GCM core
- LMCache-compatible persistent frame:
  `[version | 12-byte IV | ciphertext | 16-byte tag]`
- asynchronous seal/open C ABI
- authentication-gated open
- failed-authentication zeroization
- Python ctypes binding
- CUDA runtime test harness
- NVIDIA Hopper (`sm_90`) compilation
- ABI tests

Not yet runtime-validated:

- GPU seal -> reference CPU AES-GCM interoperability
- reference CPU AES-GCM -> GPU open interoperability
- GPU tamper/wrong-key security gates
- H100 performance
- LMCache persistent GPU data-path integration

The GPU backend should therefore currently be considered
**implemented and build/ABI validated, but not yet H100 runtime
validated**.

## Upstream

ConfKV is based on:

- Project: LMCache
- Repository: `LMCache/LMCache`
- Baseline commit:
  `3031f71e66f8872f8c763544e6ad4a654e566629`

The ConfKV-modified LMCache runtime is included as the `LMCache/`
Git submodule. The ConfKV repository pins the submodule to an exact
commit for reproducibility.

## Repository Layout

    LMCache/
        Git submodule containing the ConfKV-modified LMCache runtime

    experiments/gpu_kv_seal/
        gpu/
            CUDA AES-GCM implementation and Python bindings
        native/
            optimized CPU/OpenSSL AES-GCM implementation
        scripts/
            correctness, security, interoperability, and benchmark tools
        configs/
            experiment configurations

## Clone and Environment

Clone ConfKV together with its pinned LMCache runtime:

    git clone --recursive https://github.com/MountVeil/ConfKV.git
    cd ConfKV
    source env.sh

For an existing clone:

    git submodule update --init --recursive
    source env.sh

The ConfKV superproject pins `LMCache/` to an exact commit. The
`branch = confkv` entry in `.gitmodules` indicates the development
branch, but reproducible checkouts always use the commit recorded by
the ConfKV superproject.

## GPU Correctness Gates

Once an H100 is available to CUDA:

    ./experiments/gpu_kv_seal/scripts/run_gpu_gates.sh

After all correctness/security gates pass:

    python3 experiments/gpu_kv_seal/scripts/bench_gpu_aesgcm.py

## License

This repository contains a derivative of LMCache, distributed under
the Apache License 2.0.

See:

- `LICENSE`
- `THIRD_PARTY.md`
- `MODIFICATIONS.md`
