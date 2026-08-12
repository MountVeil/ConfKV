#!/usr/bin/env python3

import argparse
import csv
import ctypes
import os
import resource
import statistics
import time

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm import AesGcmDeserializer
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider


FRAME_OVERHEAD = 29
MASTER = b"paper-benchmark-master-key-32bytes!!"


class ByteBuf:
    def __init__(self, size: int):
        self._arr = (ctypes.c_ubyte * size)()

    @property
    def byte_array(self):
        return memoryview(self._arr)

    @property
    def address(self):
        return ctypes.addressof(self._arr)


def sched_getcpu():
    try:
        libc = ctypes.CDLL(None)
        libc.sched_getcpu.restype = ctypes.c_int
        return int(libc.sched_getcpu())
    except Exception:
        return -1


def percentile(values, p):
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    default="optimized_cpu",
)

parser.add_argument(
    "--size-mib",
    type=int,
    required=True,
)

parser.add_argument(
    "--warmup",
    type=int,
    default=5,
)

parser.add_argument(
    "--iterations",
    type=int,
    default=30,
)

parser.add_argument(
    "--run-id",
    required=True,
)

parser.add_argument(
    "--output",
    required=True,
)

args = parser.parse_args()

if args.mode != "optimized_cpu":
    raise SystemExit(
        f"bench_optimized_cpu.py only supports optimized_cpu, got {args.mode}"
    )


ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

LIB_PATH = os.path.join(
    ROOT,
    "native",
    "libcpu_seal.so",
)

lib = ctypes.CDLL(LIB_PATH)

lib.lmcache_aes128gcm_seal.argtypes = [
    ctypes.c_void_p,   # key
    ctypes.c_void_p,   # src
    ctypes.c_size_t,   # src_len
    ctypes.c_void_p,   # dst
    ctypes.c_size_t,   # dst_len
]

lib.lmcache_aes128gcm_seal.restype = ctypes.c_int


size = args.size_mib * 1024 * 1024

src = ByteBuf(size)

dst = ByteBuf(
    size + FRAME_OVERHEAD
)

verify = ByteBuf(size)


# ----------------------------------------------------------------------
# Pre-touch buffers.
#
# Important:
# page allocation of src/dst must not contaminate measured sealing time.
# ----------------------------------------------------------------------

ctypes.memset(
    src.address,
    0xA5,
    size,
)

ctypes.memset(
    dst.address,
    0,
    size + FRAME_OVERHEAD,
)

ctypes.memset(
    verify.address,
    0,
    size,
)


# ----------------------------------------------------------------------
# Same LMCache per-cache-salt key derivation.
# ----------------------------------------------------------------------

provider = HkdfKeyProvider(
    MASTER,
    key_len=16,
    info_prefix=b"lmcache-l2-aesgcm-v1",
)

key_desc = ObjectKey(
    chunk_hash=b"\x11" * 32,
    model_name="paper-benchmark",
    kv_rank=0,
    cache_salt="tenant-a",
)

dek = provider.get_key(
    key_desc.cache_salt
)

if len(dek) != 16:
    raise RuntimeError(
        f"Expected AES-128 key, got {len(dek)} bytes"
    )

KeyArray = ctypes.c_ubyte * len(dek)

key_buf = KeyArray.from_buffer_copy(
    dek
)

key_ptr = ctypes.addressof(
    key_buf
)


def seal_once():
    # Include the same cached per-tenant key lookup performed by
    # stock LMCache's serializer.
    current_dek = provider.get_key(
        key_desc.cache_salt
    )

    if current_dek != dek:
        raise RuntimeError("unexpected derived key")

    return lib.lmcache_aes128gcm_seal(
        key_ptr,
        src.address,
        size,
        dst.address,
        size + FRAME_OVERHEAD,
    )


# ----------------------------------------------------------------------
# Warmup
# ----------------------------------------------------------------------

for _ in range(args.warmup):
    rc = seal_once()

    if rc != 0:
        raise RuntimeError(
            f"native seal failed during warmup: rc={rc}"
        )


# ----------------------------------------------------------------------
# Timed samples
# ----------------------------------------------------------------------

rows = []

for iteration in range(args.iterations):

    before = resource.getrusage(
        resource.RUSAGE_SELF
    )

    t0 = time.perf_counter_ns()

    rc = seal_once()

    t1 = time.perf_counter_ns()

    after = resource.getrusage(
        resource.RUSAGE_SELF
    )

    if rc != 0:
        raise RuntimeError(
            f"native seal failed: rc={rc}"
        )

    rows.append({
        "run_id": args.run_id,
        "mode": "optimized_cpu",
        "size_mib": args.size_mib,
        "size_bytes": size,
        "iteration": iteration,
        "cpu": sched_getcpu(),
        "pid": os.getpid(),

        # Native implementation is deliberately treated as one
        # system-stage operation. Do not fake a Python-level breakdown.
        "setup_ns": 0,
        "copy_in_ns": 0,
        "crypto_ns": 0,
        "copy_out_ns": 0,

        "total_ns": t1 - t0,

        "minflt":
            after.ru_minflt -
            before.ru_minflt,

        "majflt":
            after.ru_majflt -
            before.ru_majflt,
    })


# ----------------------------------------------------------------------
# Wire compatibility / correctness.
#
# Native-produced frame must be consumable by stock LMCache.
# ----------------------------------------------------------------------

deserializer = AesGcmDeserializer(
    provider
)

deserializer.deserialize(
    dst,
    verify,
    key_desc,
)

libc = ctypes.CDLL(None)

libc.memcmp.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]

libc.memcmp.restype = ctypes.c_int

if libc.memcmp(
    src.address,
    verify.address,
    size,
) != 0:
    raise RuntimeError(
        "Native seal -> stock LMCache decrypt mismatch"
    )


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

os.makedirs(
    os.path.dirname(args.output),
    exist_ok=True,
)

fieldnames = [
    "run_id",
    "mode",
    "size_mib",
    "size_bytes",
    "iteration",
    "cpu",
    "pid",
    "setup_ns",
    "copy_in_ns",
    "crypto_ns",
    "copy_out_ns",
    "total_ns",
    "minflt",
    "majflt",
]

with open(
    args.output,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


totals = [
    r["total_ns"] / 1e6
    for r in rows
]

median_ms = statistics.median(
    totals
)

p95_ms = percentile(
    totals,
    0.95,
)

throughput = (
    size / (1024 ** 3)
) / (
    median_ms / 1000
)

print(
    f"{'optimized_cpu':16s} "
    f"{args.size_mib:4d} MiB "
    f"median={median_ms:9.3f} ms "
    f"p95={p95_ms:9.3f} ms "
    f"throughput={throughput:7.3f} GiB/s"
)
