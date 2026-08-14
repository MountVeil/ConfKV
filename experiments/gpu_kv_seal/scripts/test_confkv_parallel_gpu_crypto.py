#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from lmcache.v1.confkv.gpu_crypto import (
    ConfKVAuthenticationError,
    ConfKVGpuCrypto,
    FRAME_OVERHEAD,
)


class HostMemoryObj:
    """Minimal pinned-host MemoryObj-compatible object."""

    def __init__(self, nbytes: int):
        self.raw_tensor = torch.empty(
            nbytes,
            dtype=torch.uint8,
            pin_memory=True,
        )

    def get_size(self) -> int:
        return int(self.raw_tensor.nbytes)

    def parent(self):
        return None


def throughput(
    size: int,
    chunks: int,
    elapsed_ms: float,
) -> float:
    return (
        size
        * chunks
        / (1024 ** 3)
        / (elapsed_ms / 1000.0)
    )


def measure(fn, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()

    samples = []

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()

        samples.append(
            (t1 - t0) / 1e6
        )

    return (
        statistics.mean(samples),
        statistics.pstdev(samples),
    )


def run_case(
    size: int,
    chunks: int,
    slots: int,
    warmup: int,
    iterations: int,
):
    os.environ[
        "CONFKV_GPU_CRYPTO_SLOTS"
    ] = str(slots)

    crypto = ConfKVGpuCrypto(
        "cuda:0",
        max_plaintext_len=size,
        num_slots=slots,
    )

    if crypto.num_slots != slots:
        raise RuntimeError(
            f"requested {slots} slots, "
            f"got {crypto.num_slots}"
        )

    main_stream = torch.cuda.Stream(
        device="cuda:0"
    )

    plaintexts = [
        torch.empty(
            size,
            dtype=torch.uint8,
            device="cuda:0",
        )
        for _ in range(chunks)
    ]

    outputs = [
        torch.empty_like(x)
        for x in plaintexts
    ]

    frames = [
        HostMemoryObj(
            size + FRAME_OVERHEAD
        )
        for _ in range(chunks)
    ]

    with torch.cuda.stream(main_stream):
        for i, tensor in enumerate(plaintexts):
            tensor.fill_(
                (17 + i) & 0xFF
            )

    main_stream.synchronize()

    def store():
        crypto.seal_batch_and_copy_d2h(
            plaintexts,
            frames,
            stream=main_stream,
        )

        # seal_batch_and_copy_d2h() only performs a GPU-side
        # join. Benchmark completion must be CPU-visible.
        main_stream.synchronize()

    store()

    def load():
        crypto.copy_h2d_and_open_batch(
            frames,
            outputs,
            stream=main_stream,
        )

    load()

    for i in range(chunks):
        if not torch.equal(
            plaintexts[i],
            outputs[i],
        ):
            raise RuntimeError(
                f"roundtrip mismatch: chunk={i}"
            )

    store_ms, store_sigma = measure(
        store,
        warmup,
        iterations,
    )

    # Refresh valid ciphertext before Load timing.
    store()

    load_ms, load_sigma = measure(
        load,
        warmup,
        iterations,
    )

    print(
        f"slots={slots:2d} "
        f"chunks={chunks:2d} "
        f"size={size // (1024**2):3d} MiB  "
        f"store={store_ms:9.3f} ms  "
        f"{throughput(size, chunks, store_ms):8.3f} GiB/s  "
        f"sigma={store_sigma:.3f}"
    )

    print(
        f"slots={slots:2d} "
        f"chunks={chunks:2d} "
        f"size={size // (1024**2):3d} MiB  "
        f"load ={load_ms:9.3f} ms  "
        f"{throughput(size, chunks, load_ms):8.3f} GiB/s  "
        f"sigma={load_sigma:.3f}"
    )

    # --------------------------------------------------------
    # Authentication test
    # --------------------------------------------------------

    store()

    # Frame:
    # [1B version][12B IV][ciphertext][16B tag]
    # Offset 13 is first ciphertext byte.
    frames[0].raw_tensor[13] ^= 1

    outputs[0].fill_(0xA5)

    try:
        crypto.copy_h2d_and_open_batch(
            frames,
            outputs,
            stream=main_stream,
        )
    except ConfKVAuthenticationError:
        pass
    else:
        raise RuntimeError(
            "tampered batch was accepted"
        )

    main_stream.synchronize()

    if torch.count_nonzero(
        outputs[0]
    ).item() != 0:
        raise RuntimeError(
            "failed-auth plaintext "
            "was not zeroized"
        )

    crypto.close()

    print(
        f"slots={slots:2d}: "
        "roundtrip PASS / "
        "tamper PASS / "
        "zeroize PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size-mib",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--chunks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--slots",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    torch.cuda.set_device(0)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Runtime: H100 PRE-CC "
        "(TDX=NO, H100_CC=NO)"
    )

    size = (
        args.size_mib
        * 1024
        * 1024
    )

    for slots in args.slots:
        if slots > args.chunks:
            continue

        run_case(
            size,
            args.chunks,
            slots,
            args.warmup,
            args.iterations,
        )


if __name__ == "__main__":
    main()
