# ConfKV Experiments

This directory contains correctness tests, storage-path tests,
microbenchmarks and the 4-GPU/8-CPU Qwen end-to-end experiment.

These experiments currently validate the non-CC ConfKV data plane.
They do not validate TDX attestation, GPU CC or protected key delivery.

## 1. Environment

The complete Qwen experiment expects:

- four CUDA GPUs;
- eight selected logical CPU cores;
- CUDA toolkit and a working PyTorch CUDA installation;
- `taskset`;
- vLLM with the LMCache connector;
- the pinned LMCache submodule;
- a local Qwen3-8B model;
- sufficient local storage for four independent L2 directories.

The CUDA implementation is currently built for NVIDIA Hopper
`sm_90`.

Initialize the repository and Python environment:

```bash
cd /home/xpf/ConfKV
git submodule update --init --recursive

source .venv/bin/activate
source env.sh
```

For a new environment, install the pinned LMCache source and its
dependencies before sourcing `env.sh`:

```bash
python3 -m pip install -e ./LMCache
```

Install a vLLM version compatible with the pinned LMCache runtime.
The experiment requires both commands to be available:

```bash
command -v lmcache
command -v vllm
```

Check CUDA and the four target GPUs:

```bash
nvidia-smi -L
nvcc --version

python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

for device in range(torch.cuda.device_count()):
    print(device, torch.cuda.get_device_name(device))
PY
```

Build the native backends:

```bash
./experiments/gpu_kv_seal/native/build.sh
./experiments/gpu_kv_seal/gpu/build.sh
```

## 2. Development Keys

The non-CC experiments use test keys stored in protected regular
files:

- `/run/secrets/confkv/master_key`: 32-byte CPU master key;
- `/run/secrets/confkv/k_store`: raw 16-byte GPU AES-128 key.

Create them once:

```bash
sudo install -d -m 700 /run/secrets/confkv

sudo python3 - <<'PY'
import os
from pathlib import Path

keys = {
    Path("/run/secrets/confkv/master_key"): 32,
    Path("/run/secrets/confkv/k_store"): 16,
}

for path, size in keys.items():
    if not path.exists():
        path.write_bytes(os.urandom(size))

    os.chmod(path, 0o600)

    if not path.is_file():
        raise RuntimeError(f"not a regular file: {path}")

    if path.stat().st_size != size:
        raise RuntimeError(
            f"{path} must contain exactly {size} raw bytes"
        )

    print(path, size, "bytes, mode 0600")
PY
```

These are raw binary files, not hexadecimal strings.

The host can observe these development keys, and GPU memory is not
protected without GPU CC. They are suitable only for functional and
performance experiments.

An alternative GPU key path can be selected with:

```bash
export CONFKV_DEV_KEY_PATH=/protected/path/k_store
```

The end-to-end runner also accepts `--cpu-key` and `--gpu-key`.

## 3. Correctness Tests

Run the focused CPU-only/mock unit tests:

```bash
pytest -q \
  LMCache/tests/v1/confkv/test_key_provider.py \
  LMCache/tests/v1/confkv/test_gpu_crypto.py \
  LMCache/tests/v1/confkv/test_store_abort.py
```

Run all non-CC GPU data-plane gates:

```bash
./experiments/gpu_kv_seal/scripts/run_gpu_gates.sh
```

The GPU gates cover:

1. CUDA AES-GCM build;
2. C ABI;
3. GPU/CPU interoperability;
4. authentication and tamper rejection;
5. naive ConfKV GPU execution;
6. optimized ConfKV GPU execution;
7. actual LMCache persistent-storage roundtrip.

For direct debugging of the persistent path:

```bash
python3 \
  experiments/gpu_kv_seal/scripts/test_actual_lmcache_b3_storage_path.py \
  --data-dir /tmp/confkv-storage-smoke
```

This test forces:

```text
paged GPU KV
  -> LMCache gather
  -> GPU encryption
  -> pinned CPU memory
  -> filesystem L2
  -> pinned CPU memory
  -> GPU authentication/decryption
  -> LMCache scatter
  -> paged GPU KV
```

It also tampers with an L2 ciphertext and verifies that unauthenticated
plaintext is not scattered into the KV cache.

## 4. End-to-End Experiment Matrix

The primary runner is:

```text
scripts/run_qwen_e2e_4g8c.py
```

It enforces four GPUs and eight CPU cores and runs:

| Case | Encryption | Execution |
|---|---|---|
| `baseline` | None | Native LMCache |
| `opt_cpu` | CPU AES-128-GCM | Eight CPU workers |
| `confkv_naive` | GPU AES-128-GCM | One slot, synchronous per chunk |
| `confkv_optimized` | GPU AES-128-GCM | Batched, multi-slot, overlapped |

Each case performs:

- `store`: prefill and persist prefixes to L2;
- `load`: use stored prefixes and measure L2 hits;
- `miss`: use never-stored prefixes and measure cache misses.

The runner uses `skip_l1` and different prefixes for each load
concurrency, preventing an L1 hit from being reported as an L2
decrypt result. vLLM prefix caching is disabled.

### Command inspection

Check the generated LMCache and vLLM commands without starting
services:

```bash
python3 experiments/gpu_kv_seal/scripts/run_qwen_e2e_4g8c.py \
  --repo /home/xpf/ConfKV \
  --model /data/models/Qwen3-8B \
  --gpus 0,1,2,3 \
  --cpus 0-7 \
  --dry-run
```

### Small smoke experiment

Run one case first:

```bash
python3 experiments/gpu_kv_seal/scripts/run_qwen_e2e_4g8c.py \
  --repo /home/xpf/ConfKV \
  --model /data/models/Qwen3-8B \
  --gpus 0,1,2,3 \
  --cpus 0-7 \
  --cases confkv_optimized \
  --warmup 1 \
  --requests 2 \
  --miss-requests 2 \
  --load-concurrency 1
```

### Complete experiment

```bash
python3 experiments/gpu_kv_seal/scripts/run_qwen_e2e_4g8c.py \
  --repo /home/xpf/ConfKV \
  --model /data/models/Qwen3-8B \
  --served-model Qwen/Qwen3-8B \
  --gpus 0,1,2,3 \
  --cpus 0-7 \
  --cases baseline opt_cpu confkv_naive confkv_optimized \
  --prefix-tokens 4096 \
  --chunk-size 256 \
  --warmup 2 \
  --requests 16 \
  --miss-requests 8 \
  --store-concurrency 8 \
  --load-concurrency 1 8 \
  --crypto-slots 8 \
  --output-root results/qwen_e2e
```

Relevant parameters:

| Argument | Meaning |
|---|---|
| `--prefix-tokens` | Shared prefix length for each request |
| `--chunk-size` | LMCache chunk size |
| `--requests` | L2-hit requests per concurrency |
| `--miss-requests` | Unseen-prefix requests per concurrency |
| `--load-concurrency` | Requested load concurrency levels |
| `--crypto-slots` | Optimized GPU crypto workspaces/streams |
| `--l1-gb` | LMCache L1 capacity |
| `--odirect` | Enable or disable filesystem direct I/O |
| `--vllm-extra` | Append an additional vLLM argument |

## 5. Result Files

Each run creates a UTC timestamp directory:

```text
results/qwen_e2e/<timestamp>/
    comparison.json
    baseline/
        requests.csv
        summary.json
        lmcache.log
        vllm.log
        l2/
    opt_cpu/
    confkv_naive/
    confkv_optimized/
```

`requests.csv` contains one row per measured request:

- case, phase and concurrency profile;
- success status and error;
- prompt and output token counts;
- TTFT;
- TPOT;
- end-to-end request latency.

`summary.json` contains:

- mean, p50, p90 and p99 TTFT/TPOT/E2E;
- request and output-token throughput;
- L1/L2 hit counters parsed from LMCache;
- L2 file count and stored bytes;
- experiment mode and non-CC security scope.

`comparison.json` combines the summaries from every selected case.

The raw logs are retained so that cache-hit behavior and service
failures can be checked independently.

## 6. Reading Results

Print the primary metrics from the newest run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("results/qwen_e2e")
runs = sorted(path for path in root.iterdir() if path.is_dir())

if not runs:
    raise RuntimeError("no experiment results found")

run = runs[-1]
results = json.loads(
    (run / "comparison.json").read_text()
)

print("run:", run)

for result in results:
    print()
    print(result["case"])

    for profile, metrics in result["metrics"].items():
        print(
            profile,
            "requests=", metrics["successful"],
            "rps=", round(
                metrics["request_throughput_rps"], 3
            ),
            "ttft_p50_ms=", round(
                metrics["ttft_ms"]["p50"], 3
            ),
            "tpot_p50_ms=", round(
                metrics["tpot_ms"]["p50"], 3
            ),
            "e2e_p50_ms=", round(
                metrics["e2e_ms"]["p50"], 3
            ),
        )
PY
```

For paper figures, retain both `requests.csv` and `summary.json`.
Do not report a load result unless `summary.json` records positive L2
hits and the corresponding `lmcache.log` confirms the reload.

## 7. Interpretation

The following results are valid before TDX and GPU CC are available:

- native LMCache baseline;
- CPU AES-GCM overhead;
- naive GPU AES-GCM overhead;
- optimized GPU data-path benefit;
- cache hit/miss behavior;
- storage and PCIe sensitivity;
- non-CC GPU encryption throughput.

The following claims require later TDX and GPU CC experiments:

- protected key establishment;
- remote-attestation correctness;
- GPU CC on/off comparison;
- end-to-end protection against a malicious host.
