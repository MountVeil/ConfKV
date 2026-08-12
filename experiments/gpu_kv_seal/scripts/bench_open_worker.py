#!/usr/bin/env python3

import argparse
import csv
import ctypes
import os
import resource
import statistics
import time

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm import (
    AesGcmDeserializer,
    AesGcmSerializer,
)
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider


FRAME_OVERHEAD = 29
MASTER = b"paper-benchmark-master-key-32bytes!!"


class ByteBuf:
    def __init__(self, size):
        self._arr = (ctypes.c_ubyte * size)()

    @property
    def byte_array(self):
        return memoryview(self._arr)

    @property
    def address(self):
        return ctypes.addressof(self._arr)


def sched_getcpu():
    libc = ctypes.CDLL(None)
    libc.sched_getcpu.restype = ctypes.c_int
    return libc.sched_getcpu()


def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=[
        "lmcache_open",
        "optimized_cpu_open",
    ],
    required=True,
)

parser.add_argument("--size-mib", type=int, required=True)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--run-id", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

size = args.size_mib * 1024 * 1024


# ------------------------------------------------------------
# LMCache-compatible buffers
# ------------------------------------------------------------

plain = ByteBuf(size)
frame = ByteBuf(size + FRAME_OVERHEAD)
dst = ByteBuf(size)

ctypes.memset(
    plain.address,
    0xA5,
    size,
)

ctypes.memset(
    frame.address,
    0,
    size + FRAME_OVERHEAD,
)

ctypes.memset(
    dst.address,
    0,
    size,
)


# ------------------------------------------------------------
# Identical LMCache key semantics
# ------------------------------------------------------------

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

serializer = AesGcmSerializer(provider)
deserializer = AesGcmDeserializer(provider)


# Generate one valid persistent frame outside timing.
n = serializer.serialize(
    plain,
    frame,
    key_desc,
)

assert n == size + FRAME_OVERHEAD


# ------------------------------------------------------------
# Native backend
# ------------------------------------------------------------

lib = None
key_ptr = None
expected_dek = None

if args.mode == "optimized_cpu_open":

    lib = ctypes.CDLL(
        os.path.join(
            ROOT,
            "native",
            "libcpu_seal.so",
        )
    )

    lib.lmcache_aes128gcm_open.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    lib.lmcache_aes128gcm_open.restype = ctypes.c_int

    expected_dek = provider.get_key(
        key_desc.cache_salt
    )

    KeyArray = ctypes.c_ubyte * len(expected_dek)

    key_buf = KeyArray.from_buffer_copy(
        expected_dek
    )

    key_ptr = ctypes.addressof(
        key_buf
    )


def open_once():

    if args.mode == "lmcache_open":

        deserializer.deserialize(
            frame,
            dst,
            key_desc,
        )

        return

    # Keep LMCache's cached per-tenant key lookup inside the
    # measured operation for apples-to-apples semantics.
    dek = provider.get_key(
        key_desc.cache_salt
    )

    if dek != expected_dek:
        raise RuntimeError(
            "unexpected derived key"
        )

    rc = lib.lmcache_aes128gcm_open(
        key_ptr,
        frame.address,
        size + FRAME_OVERHEAD,
        dst.address,
        size,
    )

    if rc != 0:
        raise RuntimeError(
            f"native open failed: rc={rc}"
        )


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

for _ in range(args.warmup):
    open_once()


# ------------------------------------------------------------
# Timed iterations
# ------------------------------------------------------------

rows = []

for iteration in range(args.iterations):

    before = resource.getrusage(
        resource.RUSAGE_SELF
    )

    t0 = time.perf_counter_ns()

    open_once()

    t1 = time.perf_counter_ns()

    after = resource.getrusage(
        resource.RUSAGE_SELF
    )

    rows.append({
        "run_id": args.run_id,
        "mode": args.mode,
        "size_mib": args.size_mib,
        "size_bytes": size,
        "iteration": iteration,
        "cpu": sched_getcpu(),
        "pid": os.getpid(),
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


# ------------------------------------------------------------
# Correctness outside timing
# ------------------------------------------------------------

libc = ctypes.CDLL(None)

libc.memcmp.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]

libc.memcmp.restype = ctypes.c_int

assert libc.memcmp(
    plain.address,
    dst.address,
    size,
) == 0


# ------------------------------------------------------------
# Output raw data
# ------------------------------------------------------------

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
    x["total_ns"] / 1e6
    for x in rows
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
    f"{args.mode:20s} "
    f"{args.size_mib:4d} MiB "
    f"median={median_ms:9.3f} ms "
    f"p95={p95_ms:9.3f} ms "
    f"throughput={throughput:7.3f} GiB/s"
)
