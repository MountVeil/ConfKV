#!/usr/bin/env python3

import argparse
import csv
import ctypes
import os
import resource
import statistics
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm import (
    AesGcmDeserializer,
    AesGcmSerializer,
)
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider


FRAME_OVERHEAD = 1 + 12 + 16
MASTER = b"paper-benchmark-master-key-32bytes!!"


class ByteBuf:
    """CPU byte buffer matching LMCache's MemoryObj.byte_array interface."""

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
    if not values:
        return float("nan")
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def build_crypto():
    provider = HkdfKeyProvider(
        MASTER,
        key_len=16,
        info_prefix=b"lmcache-l2-aesgcm-v1",
    )

    key = ObjectKey(
        chunk_hash=b"\x11" * 32,
        model_name="paper-benchmark",
        kv_rank=0,
        cache_salt="tenant-a",
    )

    return provider, key


def run_lmcache_total(
    serializer,
    src,
    dst,
    key,
):
    before = resource.getrusage(resource.RUSAGE_SELF)

    t0 = time.perf_counter_ns()
    n = serializer.serialize(src, dst, key)
    t1 = time.perf_counter_ns()

    after = resource.getrusage(resource.RUSAGE_SELF)

    assert n == len(src.byte_array) + FRAME_OVERHEAD

    return {
        "setup_ns": 0,
        "copy_in_ns": 0,
        "crypto_ns": 0,
        "copy_out_ns": 0,
        "total_ns": t1 - t0,
        "minflt": after.ru_minflt - before.ru_minflt,
        "majflt": after.ru_majflt - before.ru_majflt,
    }


def run_stock_breakdown(
    provider,
    src,
    dst,
    key,
):
    before = resource.getrusage(resource.RUSAGE_SELF)

    t0 = time.perf_counter_ns()

    dek = provider.get_key(key.cache_salt)
    iv = os.urandom(12)

    t1 = time.perf_counter_ns()

    # Exact stock LMCache plaintext materialization.
    plain = bytes(src.byte_array)

    t2 = time.perf_counter_ns()

    # Exact stock LMCache cryptography API.
    cipher = AESGCM(dek).encrypt(iv, plain, None)

    t3 = time.perf_counter_ns()

    out = memoryview(dst.byte_array).cast("B")
    out[0:1] = b"\x01"
    out[1:13] = iv
    out[13:13 + len(cipher)] = cipher

    t4 = time.perf_counter_ns()

    after = resource.getrusage(resource.RUSAGE_SELF)

    result = {
        "setup_ns": t1 - t0,
        "copy_in_ns": t2 - t1,
        "crypto_ns": t3 - t2,
        "copy_out_ns": t4 - t3,
        "total_ns": t4 - t0,
        "minflt": after.ru_minflt - before.ru_minflt,
        "majflt": after.ru_majflt - before.ru_majflt,
    }

    # Serializer locals disappear after return; emulate that lifecycle.
    del plain
    del cipher

    return result


def run_raw_aes(
    aes,
    plain,
    iv,
):
    before = resource.getrusage(resource.RUSAGE_SELF)

    t0 = time.perf_counter_ns()
    cipher = aes.encrypt(iv, plain, None)
    t1 = time.perf_counter_ns()

    after = resource.getrusage(resource.RUSAGE_SELF)

    result = {
        "setup_ns": 0,
        "copy_in_ns": 0,
        "crypto_ns": t1 - t0,
        "copy_out_ns": 0,
        "total_ns": t1 - t0,
        "minflt": after.ru_minflt - before.ru_minflt,
        "majflt": after.ru_majflt - before.ru_majflt,
    }

    del cipher
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "lmcache_total",
            "stock_breakdown",
            "raw_aes",
        ],
        required=True,
    )

    parser.add_argument("--size-mib", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    size = args.size_mib * 1024 * 1024

    src = ByteBuf(size)
    dst = ByteBuf(size + FRAME_OVERHEAD)

    # Fault in source pages before measurement.
    ctypes.memset(src.address, 0xA5, size)

    provider, key = build_crypto()

    serializer = AesGcmSerializer(provider)
    deserializer = AesGcmDeserializer(provider)

    dek = provider.get_key(key.cache_salt)
    raw_plain = bytes(src.byte_array)
    raw_aes = AESGCM(dek)

    ivs = [
        os.urandom(12)
        for _ in range(args.warmup + args.iterations)
    ]

    def one_iteration(index):
        if args.mode == "lmcache_total":
            return run_lmcache_total(
                serializer,
                src,
                dst,
                key,
            )

        if args.mode == "stock_breakdown":
            return run_stock_breakdown(
                provider,
                src,
                dst,
                key,
            )

        return run_raw_aes(
            raw_aes,
            raw_plain,
            ivs[index],
        )

    # Warmup
    for i in range(args.warmup):
        one_iteration(i)

    rows = []

    for i in range(args.iterations):
        result = one_iteration(args.warmup + i)

        rows.append({
            "run_id": args.run_id,
            "mode": args.mode,
            "size_mib": args.size_mib,
            "size_bytes": size,
            "iteration": i,
            "cpu": sched_getcpu(),
            "pid": os.getpid(),
            **result,
        })

    # Correctness check outside timing.
    if args.mode == "lmcache_total":
        verify = ByteBuf(size)

        n = serializer.serialize(src, dst, key)
        assert n == size + FRAME_OVERHEAD

        deserializer.deserialize(dst, verify, key)

        libc = ctypes.CDLL(None)
        libc.memcmp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        libc.memcmp.restype = ctypes.c_int

        assert libc.memcmp(
            src.address,
            verify.address,
            size,
        ) == 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    totals = [r["total_ns"] / 1e6 for r in rows]

    median_ms = statistics.median(totals)
    p95_ms = percentile(totals, 0.95)

    gib = size / (1024 ** 3)
    throughput = gib / (median_ms / 1000.0)

    print(
        f"{args.mode:16s} "
        f"{args.size_mib:4d} MiB "
        f"median={median_ms:9.3f} ms "
        f"p95={p95_ms:9.3f} ms "
        f"throughput={throughput:7.3f} GiB/s"
    )


if __name__ == "__main__":
    main()
