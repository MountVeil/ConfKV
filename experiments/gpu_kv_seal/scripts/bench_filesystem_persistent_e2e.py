#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from lmcache.v1.confkv.gpu_crypto import (
    ConfKVGpuCrypto,
    FRAME_OVERHEAD,
)


KEY = bytes(range(16))
IV_LEN = 12


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    idx = min(
        len(ys) - 1,
        max(0, math.ceil(q * len(ys)) - 1),
    )
    return ys[idx]


def summarize(
    mode: str,
    direction: str,
    size: int,
    chunks: int,
    samples: list[float],
) -> dict:
    mean_ms = statistics.mean(samples)
    total_bytes = size * chunks

    return {
        "mode": mode,
        "direction": direction,
        "size_mib": size // (1024**2),
        "chunks": chunks,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "sigma_ms": statistics.pstdev(samples),
        "logical_gib_s": (
            total_bytes
            / (1024**3)
            / (mean_ms / 1000.0)
        ),
    }


class NativeCpuAes128Gcm:
    """Native OpenSSL AES-GCM baseline used by B2."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)

        self.lib = ctypes.CDLL(str(path))

        self.lib.lmcache_aes128gcm_seal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.lmcache_aes128gcm_seal.restype = ctypes.c_int

        self.lib.lmcache_aes128gcm_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.lmcache_aes128gcm_open.restype = ctypes.c_int

        KeyArray = ctypes.c_ubyte * len(KEY)
        self.key_holder = KeyArray.from_buffer_copy(KEY)
        self.key_ptr = ctypes.addressof(self.key_holder)

    def seal(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        plaintext_len: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_seal(
            self.key_ptr,
            int(src.data_ptr()),
            plaintext_len,
            int(dst.data_ptr()),
            int(dst.nbytes),
        )

        if rc != 0:
            raise RuntimeError(f"native seal rc={rc}")

    def open(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        plaintext_len: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_open(
            self.key_ptr,
            int(src.data_ptr()),
            plaintext_len + FRAME_OVERHEAD,
            int(dst.data_ptr()),
            plaintext_len,
        )

        if rc != 0:
            raise RuntimeError(f"native open rc={rc}")


class HostMemoryObj:
    """Minimal MemoryObj-compatible wrapper around a pinned CPU tensor."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.device.type != "cpu":
            raise ValueError("HostMemoryObj requires a CPU tensor")

        self.raw_tensor = tensor

    def get_size(self) -> int:
        return int(self.raw_tensor.nbytes)

    def parent(self):
        return None


class FilesystemStore:
    """Filesystem persistence layer.

    Store completion:
        write() + fdatasync()

    Load:
        POSIX_FADV_DONTNEED best-effort cache eviction + read()
    """

    def __init__(
        self,
        directory: Path,
        chunks: int,
        io_workers: int,
        durability: str,
    ):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        self.paths = [
            self.directory / f"chunk_{i:04d}.bin"
            for i in range(chunks)
        ]

        self.io_workers = io_workers

        if durability not in {
            "buffered",
            "fdatasync",
        }:
            raise ValueError(
                f"invalid durability mode: {durability}"
            )

        self.durability = durability

    @staticmethod
    def _buffer(tensor: torch.Tensor) -> memoryview:
        if tensor.device.type != "cpu":
            raise ValueError("filesystem buffer must be CPU")

        if not tensor.is_contiguous():
            raise ValueError("filesystem buffer must be contiguous")

        return memoryview(tensor.numpy()).cast("B")

    @classmethod
    def _write_one(
        cls,
        path: Path,
        tensor: torch.Tensor,
        durability: str,
    ) -> None:
        buf = cls._buffer(tensor)

        fd = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC,
            0o600,
        )

        try:
            offset = 0

            while offset < len(buf):
                written = os.writev(
                    fd,
                    [buf[offset:]],
                )

                if written <= 0:
                    raise RuntimeError(
                        f"short write: {path}"
                    )

                offset += written

            if durability == "fdatasync":
                # Durable data completion.
                os.fdatasync(fd)

                # At this point the pages are clean, so DONTNEED
                # can best-effort evict them.
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(
                        fd,
                        0,
                        0,
                        os.POSIX_FADV_DONTNEED,
                    )

        finally:
            os.close(fd)

    @classmethod
    def _read_one(
        cls,
        path: Path,
        tensor: torch.Tensor,
    ) -> None:
        buf = cls._buffer(tensor)

        fd = os.open(path, os.O_RDONLY)

        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    fd,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )

            offset = 0

            while offset < len(buf):
                nread = os.readv(
                    fd,
                    [buf[offset:]],
                )

                if nread == 0:
                    raise RuntimeError(
                        f"unexpected EOF: {path}"
                    )

                offset += nread

        finally:
            os.close(fd)

    def write_all(
        self,
        tensors: list[torch.Tensor],
    ) -> None:
        with ThreadPoolExecutor(
            max_workers=self.io_workers
        ) as pool:
            futures = [
                pool.submit(
                    self._write_one,
                    path,
                    tensor,
                    self.durability,
                )
                for path, tensor in zip(
                    self.paths,
                    tensors,
                    strict=True,
                )
            ]

            for future in futures:
                future.result()

    def read_all(
        self,
        tensors: list[torch.Tensor],
    ) -> None:
        with ThreadPoolExecutor(
            max_workers=self.io_workers
        ) as pool:
            futures = [
                pool.submit(
                    self._read_one,
                    path,
                    tensor,
                )
                for path, tensor in zip(
                    self.paths,
                    tensors,
                    strict=True,
                )
            ]

            for future in futures:
                future.result()

    def prepare_cold_read(self) -> None:
        """Flush buffered writes, then best-effort evict file pages.

        This method is called outside the measured Load interval.
        It prevents buffered Store measurements from turning the
        following Load benchmark into an intentionally warm-cache
        benchmark.
        """
        os.sync()

        if not hasattr(os, "posix_fadvise"):
            return

        for path in self.paths:
            if not path.exists():
                continue

            fd = os.open(
                path,
                os.O_RDONLY,
            )

            try:
                os.posix_fadvise(
                    fd,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )
            finally:
                os.close(fd)

    def cleanup(self) -> None:
        for path in self.paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class PersistentE2E:
    def __init__(
        self,
        *,
        root: Path,
        directory: Path,
        size: int,
        chunks: int,
        cpu_workers: int,
        io_workers: int,
        gpu_slots: int,
        durability: str,
        device: int,
    ):
        self.size = size
        self.frame_size = size + FRAME_OVERHEAD
        self.chunks = chunks
        self.cpu_workers = cpu_workers

        self.device = torch.device(
            f"cuda:{device}"
        )

        torch.cuda.set_device(device)

        self.stream = torch.cuda.Stream(
            device=self.device
        )

        self.fs = FilesystemStore(
            directory,
            chunks,
            io_workers,
            durability,
        )

        native_path = Path(
            os.environ.get(
                "LMCACHE_AESGCM_NATIVE_LIB",
                root
                / "experiments"
                / "gpu_kv_seal"
                / "native"
                / "libcpu_seal.so",
            )
        ).expanduser().resolve()

        self.native = NativeCpuAes128Gcm(
            native_path
        )

        self.gpu_plain = [
            torch.empty(
                size,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(chunks)
        ]

        self.gpu_out = [
            torch.empty_like(t)
            for t in self.gpu_plain
        ]

        self.host_plain = [
            torch.empty(
                size,
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(chunks)
        ]

        self.host_frames = [
            torch.empty(
                self.frame_size,
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(chunks)
        ]

        self.b3_frames = [
            HostMemoryObj(t)
            for t in self.host_frames
        ]

        os.environ[
            "CONFKV_GPU_CRYPTO_SLOTS"
        ] = str(gpu_slots)

        self.gpu_crypto = ConfKVGpuCrypto(
            self.device,
            max_plaintext_len=size,
            num_slots=gpu_slots,
        )

        with torch.cuda.stream(self.stream):
            for i, tensor in enumerate(
                self.gpu_plain
            ):
                tensor.fill_(
                    (31 + i) & 0xFF
                )

        self.stream.synchronize()

    def _parallel(
        self,
        fn,
    ) -> None:
        with ThreadPoolExecutor(
            max_workers=self.cpu_workers
        ) as pool:
            futures = [
                pool.submit(fn, i)
                for i in range(self.chunks)
            ]

            for future in futures:
                future.result()

    # --------------------------------------------------------
    # Plain CPU/GPU transfer
    # --------------------------------------------------------

    def d2h_plain(self) -> None:
        with torch.cuda.stream(self.stream):
            for gpu, host in zip(
                self.gpu_plain,
                self.host_plain,
                strict=True,
            ):
                host.copy_(
                    gpu,
                    non_blocking=True,
                )

        self.stream.synchronize()

    def h2d_plain(self) -> None:
        with torch.cuda.stream(self.stream):
            for host, gpu in zip(
                self.host_plain,
                self.gpu_out,
                strict=True,
            ):
                gpu.copy_(
                    host,
                    non_blocking=True,
                )

        self.stream.synchronize()

    # --------------------------------------------------------
    # B0
    # --------------------------------------------------------

    def b0_store(self) -> None:
        self.d2h_plain()
        self.fs.write_all(
            self.host_plain
        )

    def b0_load(self) -> None:
        self.fs.read_all(
            self.host_plain
        )
        self.h2d_plain()

    # --------------------------------------------------------
    # B1 - stock Python cryptography AESGCM
    # --------------------------------------------------------

    def _stock_seal_one(
        self,
        i: int,
    ) -> None:
        plain = bytes(
            self.host_plain[i].numpy()
        )

        iv = os.urandom(IV_LEN)

        ciphertext_and_tag = AESGCM(
            KEY
        ).encrypt(
            iv,
            plain,
            None,
        )

        dst = self.host_frames[
            i
        ].numpy()

        dst[0] = 1

        dst[1:13] = np.frombuffer(
            iv,
            dtype=np.uint8,
        )

        dst[13:] = np.frombuffer(
            ciphertext_and_tag,
            dtype=np.uint8,
        )

    def _stock_open_one(
        self,
        i: int,
    ) -> None:
        src = bytes(
            self.host_frames[i].numpy()
        )

        plain = AESGCM(
            KEY
        ).decrypt(
            src[1:13],
            src[13:],
            None,
        )

        self.host_plain[
            i
        ].numpy()[:] = np.frombuffer(
            plain,
            dtype=np.uint8,
        )

    def b1_store(self) -> None:
        self.d2h_plain()
        self._parallel(
            self._stock_seal_one
        )
        self.fs.write_all(
            self.host_frames
        )

    def b1_load(self) -> None:
        self.fs.read_all(
            self.host_frames
        )
        self._parallel(
            self._stock_open_one
        )
        self.h2d_plain()

    # --------------------------------------------------------
    # B2 - native OpenSSL CPU AES
    # --------------------------------------------------------

    def _native_seal_one(
        self,
        i: int,
    ) -> None:
        self.native.seal(
            self.host_plain[i],
            self.host_frames[i],
            self.size,
        )

    def _native_open_one(
        self,
        i: int,
    ) -> None:
        self.native.open(
            self.host_frames[i],
            self.host_plain[i],
            self.size,
        )

    def b2_store(self) -> None:
        self.d2h_plain()
        self._parallel(
            self._native_seal_one
        )
        self.fs.write_all(
            self.host_frames
        )

    def b2_load(self) -> None:
        self.fs.read_all(
            self.host_frames
        )
        self._parallel(
            self._native_open_one
        )
        self.h2d_plain()

    # --------------------------------------------------------
    # B3 - production ConfKV GPU crypto
    # --------------------------------------------------------

    def b3_store(self) -> None:
        self.gpu_crypto.seal_batch_and_copy_d2h(
            self.gpu_plain,
            self.b3_frames,
            stream=self.stream,
        )

        # Filesystem workers must not consume host P+29
        # until all D2H copies complete.
        self.stream.synchronize()

        self.fs.write_all(
            self.host_frames
        )

    def b3_load(self) -> None:
        self.fs.read_all(
            self.host_frames
        )

        self.gpu_crypto.copy_h2d_and_open_batch(
            self.b3_frames,
            self.gpu_out,
            stream=self.stream,
        )

        self.stream.synchronize()

    def validate_gpu_output(self) -> None:
        self.stream.synchronize()

        for i in range(self.chunks):
            if not torch.equal(
                self.gpu_plain[i],
                self.gpu_out[i],
            ):
                raise RuntimeError(
                    f"GPU roundtrip mismatch chunk={i}"
                )

    def close(self) -> None:
        self.gpu_crypto.close()


def measure(
    fn,
    warmup: int,
    iterations: int,
) -> list[float]:
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

    return samples


def run_mode(
    bench: PersistentE2E,
    mode: str,
    *,
    warmup: int,
    iterations: int,
):
    store_fn = {
        "B0": bench.b0_store,
        "B1_stock_cpu": bench.b1_store,
        "B2_native_cpu": bench.b2_store,
        "B3_gpu": bench.b3_store,
    }[mode]

    load_fn = {
        "B0": bench.b0_load,
        "B1_stock_cpu": bench.b1_load,
        "B2_native_cpu": bench.b2_load,
        "B3_gpu": bench.b3_load,
    }[mode]

    # First produce a valid persisted representation.
    store_fn()
    load_fn()
    bench.validate_gpu_output()

    store_samples = measure(
        store_fn,
        warmup,
        iterations,
    )

    # Refresh files before measuring Load.
    store_fn()

    # Buffered Store must remain buffered for Store timing.
    # Before Load timing, flush it outside the measured region
    # and best-effort evict clean file pages.
    bench.fs.prepare_cold_read()

    load_samples = measure(
        load_fn,
        warmup,
        iterations,
    )

    bench.validate_gpu_output()

    return (
        store_samples,
        load_samples,
    )


def print_rows(
    rows: list[dict],
) -> None:
    print()
    print(
        "========== FILESYSTEM PERSISTENT E2E =========="
    )

    print(
        "mode              dir   MiB chunks"
        "    mean_ms    p95_ms   logical_GiB/s"
    )

    print(
        "------------------------------------------------"
        "-------------------------"
    )

    for row in rows:
        print(
            f"{row['mode']:17s} "
            f"{row['direction']:5s} "
            f"{row['size_mib']:4d} "
            f"{row['chunks']:6d} "
            f"{row['mean_ms']:10.3f} "
            f"{row['p95_ms']:9.3f} "
            f"{row['logical_gib_s']:15.3f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sizes-mib",
        nargs="+",
        type=int,
        default=[18, 36, 72],
    )

    parser.add_argument(
        "--chunks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--durability",
        choices=[
            "buffered",
            "fdatasync",
        ],
        default="fdatasync",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--keep-files",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(
        os.environ.get(
            "CONFKV_ROOT",
            Path(__file__).resolve().parents[3],
        )
    ).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    torch.cuda.set_device(
        args.device
    )

    print(
        "ConfKV PRE-CC filesystem persistent E2E"
    )
    print("root        :", root)
    print(
        "data dir    :",
        args.data_dir.resolve(),
    )
    print(
        "GPU         :",
        torch.cuda.get_device_name(
            args.device
        ),
    )
    print("torch       :", torch.__version__)
    print("CUDA build  :", torch.version.cuda)
    print("sizes MiB   :", args.sizes_mib)
    print("chunks      :", args.chunks)
    print("CPU workers :", args.cpu_workers)
    print("I/O workers :", args.io_workers)
    print("GPU slots   :", args.gpu_slots)
    print("warmup      :", args.warmup)
    print("iterations  :", args.iterations)
    print(
        "durability  :",
        args.durability,
    )
    print(
        "read cache  : POSIX_FADV_DONTNEED best effort"
    )
    print("TDX         : NO")
    print("H100 CC     : NO")

    rows = []

    for mib in args.sizes_mib:
        size = mib * 1024 * 1024

        print(
            f"\n[SIZE] {mib} MiB"
        )

        for mode in (
            "B0",
            "B1_stock_cpu",
            "B2_native_cpu",
            "B3_gpu",
        ):
            print(
                f"  {mode}",
                flush=True,
            )

            mode_dir = (
                args.data_dir
                / f"{mib}m"
                / mode
            )

            bench = PersistentE2E(
                root=root,
                directory=mode_dir,
                size=size,
                chunks=args.chunks,
                cpu_workers=args.cpu_workers,
                io_workers=args.io_workers,
                gpu_slots=args.gpu_slots,
                durability=args.durability,
                device=args.device,
            )

            try:
                store_samples, load_samples = run_mode(
                    bench,
                    mode,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )

                rows.append(
                    summarize(
                        mode,
                        "store",
                        size,
                        args.chunks,
                        store_samples,
                    )
                )

                rows.append(
                    summarize(
                        mode,
                        "load",
                        size,
                        args.chunks,
                        load_samples,
                    )
                )

            finally:
                bench.close()

                if not args.keep_files:
                    bench.fs.cleanup()

    print_rows(rows)

    args.csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "mode",
        "direction",
        "size_mib",
        "chunks",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "sigma_ms",
        "logical_gib_s",
    ]

    with args.csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV:", args.csv)


if __name__ == "__main__":
    main()
