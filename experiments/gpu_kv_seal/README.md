# GPU KV Seal Experiments

This directory contains reproducible experiments for evaluating
persistent KV-cache sealing.

## Experimental classes

### lmcache_total

Measures the real stock LMCache:

    AesGcmSerializer.serialize()

This is the stock system baseline.

### stock_breakdown

Mirrors the current LMCache AES-GCM serializer and separates:

    key/IV setup
    MemoryObj -> Python bytes
    AES-GCM encryption
    ciphertext -> destination MemoryObj

This experiment is explanatory and should not replace the real
`lmcache_total` baseline.

### raw_aes

Measures the cryptography AESGCM API with plaintext already
materialized.

This is a crypto-library lower-level reference, not a system baseline.

## Primary object sizes

Qwen3-8B BF16 KV theoretical sizes:

    128 tokens -> 18 MiB
    256 tokens -> 36 MiB
    512 tokens -> 72 MiB

These are the primary workload-representative points.

## Measurement protocol

Each point uses:

- fixed CPU affinity
- fixed NUMA memory node
- multiple independent Python processes
- warm-up iterations
- raw per-iteration samples
- page-fault counters
- complete software/hardware metadata

The paper-facing result is:

    results/paper/<timestamp>/summary.csv

Raw samples must always be retained.

## Exploratory results

Allocator/page-fault investigations performed before the stable
benchmark protocol are archived under:

    archive/

They may be used to explain implementation behavior but must not be
reported as final performance numbers.

## Future baselines

The experiment framework will later add:

    optimized_cpu
    gpu_seal
    gpu_seal_cc

The runner and result format should remain unchanged so that all
implementations are directly comparable.
