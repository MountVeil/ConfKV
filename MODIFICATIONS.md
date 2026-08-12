# ConfKV Modifications

ConfKV is derived from LMCache commit:

`3031f71e66f8872f8c763544e6ad4a654e566629`

## LMCache Runtime Modifications

### `LMCache/lmcache/v1/distributed/serde/aesgcm.py`

Adds selectable AES-GCM backend support while retaining the stock
LMCache AES-GCM implementation.

Current backend modes include:

- `stock`
- `native`

The GPU persistent-data path is developed separately and is not
represented as a CPU serde implementation.

### `LMCache/lmcache/v1/distributed/serde/native_aesgcm.py`

Adds an optimized CPU AES-128-GCM backend using a native OpenSSL EVP
implementation while retaining the persistent frame format and
key-derivation semantics required for interoperability with the stock
LMCache AES-GCM path.

## Research Components

### `experiments/gpu_kv_seal/native/`

Optimized CPU AES-GCM implementation used as a strong CPU control
baseline.

### `experiments/gpu_kv_seal/gpu/`

CUDA AES-128-GCM implementation intended for persistent KV sealing at
the GPU endpoint.

The current implementation:

- compiles for NVIDIA Hopper (`sm_90`);
- exposes a stable C ABI;
- uses the same `P + 29` persistent representation as the CPU path;
- supports asynchronous seal/open operations;
- gates plaintext publication on successful authentication.

GPU runtime correctness and performance have not yet been validated on
an available H100.

### `experiments/gpu_kv_seal/scripts/`

Contains CPU/GPU correctness, security, interoperability, and
performance experiments.
