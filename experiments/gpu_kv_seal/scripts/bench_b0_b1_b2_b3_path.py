#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


FRAME_OVERHEAD = 29
IV_LEN = 12
KEY = bytes(range(16))


def percentile(samples: list[float], q: float) -> float:
    xs = sorted(samples)
    idx = max(
        0,
        min(
            len(xs) - 1,
            math.ceil(q * len(xs)) - 1,
        ),
    )
    return xs[idx]


class NativeCpuAes128Gcm:
    def __init__(
        self,
        lib_path: Path,
        key: bytes,
    ):
        if len(key) != 16:
            raise ValueError(
                "AES-128 key must be 16 bytes"
            )

        if not lib_path.is_file():
            raise FileNotFoundError(
                f"native CPU AES library not found: {lib_path}"
            )

        self.lib = ctypes.CDLL(
            str(lib_path)
        )

        self.lib.lmcache_aes128gcm_seal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.lmcache_aes128gcm_seal.restype = (
            ctypes.c_int
        )

        self.lib.lmcache_aes128gcm_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.lmcache_aes128gcm_open.restype = (
            ctypes.c_int
        )

        KeyArray = ctypes.c_ubyte * 16
        self._key_holder = (
            KeyArray.from_buffer_copy(key)
        )
        self._key_ptr = ctypes.addressof(
            self._key_holder
        )

    def seal(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        plaintext_len: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_seal(
            self._key_ptr,
            int(src.data_ptr()),
            plaintext_len,
            int(dst.data_ptr()),
            int(dst.numel()),
        )

        if rc != 0:
            raise RuntimeError(
                f"native CPU seal failed: rc={rc}"
            )

    def open(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        plaintext_len: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_open(
            self._key_ptr,
            int(src.data_ptr()),
            plaintext_len + FRAME_OVERHEAD,
            int(dst.data_ptr()),
            plaintext_len,
        )

        if rc != 0:
            raise RuntimeError(
                f"native CPU open failed: rc={rc}"
            )


class PathBenchmark:
    def __init__(
        self,
        root: Path,
        device: int,
        warmup: int,
        iterations: int,
    ):
        self.root = root
        self.device_index = device
        self.warmup = warmup
        self.iterations = iterations

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable"
            )

        torch.cuda.set_device(device)

        self.device = torch.device(
            f"cuda:{device}"
        )

        self.stream = torch.cuda.Stream(
            device=self.device
        )

        gpu_dir = (
            root
            / "experiments"
            / "gpu_kv_seal"
            / "gpu"
        )

        sys.path.insert(
            0,
            str(gpu_dir),
        )

        from gpu_aesgcm import (
            GpuAes128GcmKey,
            GpuAesGcmLibrary,
        )

        gpu_lib_path = Path(
            os.environ.get(
                "CONFKV_GPU_AESGCM_LIB",
                str(
                    gpu_dir
                    / "libgpu_aesgcm.so"
                ),
            )
        )

        self.gpu_lib = GpuAesGcmLibrary(
            gpu_lib_path
        )

        self.gpu_key = GpuAes128GcmKey(
            self.gpu_lib,
            KEY,
            device=device,
        )

        cpu_lib_path = Path(
            os.environ.get(
                "LMCACHE_AESGCM_NATIVE_LIB",
                str(
                    root
                    / "experiments"
                    / "gpu_kv_seal"
                    / "native"
                    / "libcpu_seal.so"
                ),
            )
        )

        self.native_cpu = NativeCpuAes128Gcm(
            cpu_lib_path,
            KEY,
        )

        self.stock_cpu = AESGCM(KEY)

    def close(self) -> None:
        self.gpu_key.close()

    def measure(
        self,
        fn,
    ) -> list[float]:
        for _ in range(self.warmup):
            fn()

        samples = []

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()

            fn()

            t1 = time.perf_counter_ns()

            samples.append(
                (t1 - t0) / 1e6
            )

        return samples

    @staticmethod
    def summarize(
        mode: str,
        direction: str,
        size: int,
        samples: list[float],
    ):
        mean_ms = statistics.mean(samples)
        median_ms = statistics.median(samples)
        p95_ms = percentile(
            samples,
            0.95,
        )
        stdev_ms = statistics.pstdev(
            samples
        )

        gib_s = (
            size
            / (1024 ** 3)
            / (mean_ms / 1000.0)
        )

        return {
            "direction": direction,
            "mode": mode,
            "size_mib": (
                size // (1024 * 1024)
            ),
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "stdev_ms": stdev_ms,
            "gib_s": gib_s,
            "iterations": len(samples),
        }

    def run_size(
        self,
        size: int,
    ):
        frame_len = (
            size + FRAME_OVERHEAD
        )

        self.gpu_key.reserve(size)

        gpu_plain = torch.empty(
            size,
            dtype=torch.uint8,
            device=self.device,
        )

        gpu_out = torch.empty(
            size,
            dtype=torch.uint8,
            device=self.device,
        )

        gpu_frame = torch.empty(
            frame_len,
            dtype=torch.uint8,
            device=self.device,
        )

        gpu_auth = torch.empty(
            1,
            dtype=torch.int32,
            device=self.device,
        )

        # LMCache L1 类似的 pinned host staging。
        host_plain = torch.empty(
            size,
            dtype=torch.uint8,
            pin_memory=True,
        )

        host_frame = torch.empty(
            frame_len,
            dtype=torch.uint8,
            pin_memory=True,
        )

        host_plain_np = (
            host_plain.numpy()
        )

        host_frame_np = (
            host_frame.numpy()
        )

        with torch.cuda.stream(
            self.stream
        ):
            gpu_plain.fill_(0xA5)

        self.stream.synchronize()

        # 初始化 host plaintext，不计入实验。
        with torch.cuda.stream(
            self.stream
        ):
            host_plain.copy_(
                gpu_plain,
                non_blocking=True,
            )

        self.stream.synchronize()

        # -------------------------------------------------
        # B0 STORE
        # GPU P -> D2H -> CPU P
        # -------------------------------------------------
        def b0_store():
            with torch.cuda.stream(
                self.stream
            ):
                host_plain.copy_(
                    gpu_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

        # -------------------------------------------------
        # B1 STORE
        # GPU P -> D2H -> stock CPU AES -> CPU P+29
        # -------------------------------------------------
        def b1_store():
            with torch.cuda.stream(
                self.stream
            ):
                host_plain.copy_(
                    gpu_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

            iv = os.urandom(IV_LEN)

            ciphertext = (
                self.stock_cpu.encrypt(
                    iv,
                    bytes(host_plain_np),
                    None,
                )
            )

            host_frame_np[0] = 1

            host_frame_np[1:13] = (
                np.frombuffer(
                    iv,
                    dtype=np.uint8,
                )
            )

            host_frame_np[13:frame_len] = (
                np.frombuffer(
                    ciphertext,
                    dtype=np.uint8,
                )
            )

        # -------------------------------------------------
        # B2 STORE
        # GPU P -> D2H -> native CPU AES -> CPU P+29
        # -------------------------------------------------
        def b2_store():
            with torch.cuda.stream(
                self.stream
            ):
                host_plain.copy_(
                    gpu_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

            self.native_cpu.seal(
                host_plain,
                host_frame,
                size,
            )

        iv_counter = 0

        # -------------------------------------------------
        # B3 STORE
        # GPU P -> GPU AES -> GPU P+29 -> D2H
        # -------------------------------------------------
        def b3_store():
            nonlocal iv_counter

            iv_counter += 1

            iv = iv_counter.to_bytes(
                12,
                "big",
            )

            with torch.cuda.stream(
                self.stream
            ):
                self.gpu_key.seal_async(
                    src_ptr=int(
                        gpu_plain.data_ptr()
                    ),
                    plaintext_len=size,
                    dst_ptr=int(
                        gpu_frame.data_ptr()
                    ),
                    dst_capacity=frame_len,
                    iv=iv,
                    stream_ptr=int(
                        self.stream.cuda_stream
                    ),
                )

                host_frame.copy_(
                    gpu_frame,
                    non_blocking=True,
                )

            self.stream.synchronize()

        results = []

        for mode, fn in (
            ("B0", b0_store),
            ("B1_stock_cpu", b1_store),
            ("B2_native_cpu", b2_store),
            ("B3_gpu", b3_store),
        ):
            results.append(
                self.summarize(
                    mode,
                    "store",
                    size,
                    self.measure(fn),
                )
            )

        # =================================================
        # LOAD
        # =================================================

        # -------------------------------------------------
        # B0 LOAD
        # CPU P -> H2D
        # -------------------------------------------------
        def b0_load():
            with torch.cuda.stream(
                self.stream
            ):
                gpu_out.copy_(
                    host_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

        # -------------------------------------------------
        # Prepare valid B1 frame.
        # -------------------------------------------------
        iv = os.urandom(IV_LEN)

        ciphertext = self.stock_cpu.encrypt(
            iv,
            bytes(host_plain_np),
            None,
        )

        host_frame_np[0] = 1
        host_frame_np[1:13] = (
            np.frombuffer(
                iv,
                dtype=np.uint8,
            )
        )
        host_frame_np[13:frame_len] = (
            np.frombuffer(
                ciphertext,
                dtype=np.uint8,
            )
        )

        # -------------------------------------------------
        # B1 LOAD
        # CPU P+29 -> stock open -> CPU P -> H2D
        # -------------------------------------------------
        def b1_load():
            blob = bytes(
                host_frame_np
            )

            if blob[0] != 1:
                raise RuntimeError(
                    "stock frame version mismatch"
                )

            plaintext = (
                self.stock_cpu.decrypt(
                    blob[1:13],
                    blob[13:frame_len],
                    None,
                )
            )

            host_plain_np[:] = (
                np.frombuffer(
                    plaintext,
                    dtype=np.uint8,
                )
            )

            with torch.cuda.stream(
                self.stream
            ):
                gpu_out.copy_(
                    host_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

        b1_samples = self.measure(
            b1_load
        )

        results.append(
            self.summarize(
                "B1_stock_cpu",
                "load",
                size,
                b1_samples,
            )
        )

        # -------------------------------------------------
        # Prepare valid B2 frame.
        # -------------------------------------------------
        self.native_cpu.seal(
            host_plain,
            host_frame,
            size,
        )

        # -------------------------------------------------
        # B2 LOAD
        # CPU P+29 -> native open -> CPU P -> H2D
        # -------------------------------------------------
        def b2_load():
            self.native_cpu.open(
                host_frame,
                host_plain,
                size,
            )

            with torch.cuda.stream(
                self.stream
            ):
                gpu_out.copy_(
                    host_plain,
                    non_blocking=True,
                )

            self.stream.synchronize()

        b2_samples = self.measure(
            b2_load
        )

        results.append(
            self.summarize(
                "B2_native_cpu",
                "load",
                size,
                b2_samples,
            )
        )

        # -------------------------------------------------
        # Prepare valid B3 frame.
        # -------------------------------------------------
        iv_counter += 1

        iv = iv_counter.to_bytes(
            12,
            "big",
        )

        with torch.cuda.stream(
            self.stream
        ):
            self.gpu_key.seal_async(
                src_ptr=int(
                    gpu_plain.data_ptr()
                ),
                plaintext_len=size,
                dst_ptr=int(
                    gpu_frame.data_ptr()
                ),
                dst_capacity=frame_len,
                iv=iv,
                stream_ptr=int(
                    self.stream.cuda_stream
                ),
            )

            host_frame.copy_(
                gpu_frame,
                non_blocking=True,
            )

        self.stream.synchronize()

        # -------------------------------------------------
        # B3 LOAD
        # CPU P+29 -> H2D -> GPU open/auth
        # -------------------------------------------------
        def b3_load():
            with torch.cuda.stream(
                self.stream
            ):
                gpu_frame.copy_(
                    host_frame,
                    non_blocking=True,
                )

                gpu_auth.zero_()

                self.gpu_key.open_async(
                    src_ptr=int(
                        gpu_frame.data_ptr()
                    ),
                    frame_len=frame_len,
                    dst_ptr=int(
                        gpu_out.data_ptr()
                    ),
                    plaintext_len=size,
                    auth_ok_ptr=int(
                        gpu_auth.data_ptr()
                    ),
                    stream_ptr=int(
                        self.stream.cuda_stream
                    ),
                )

            # 与目前 ConfKV correctness-first
            # auth-before-scatter 路径一致。
            self.stream.synchronize()

            if int(
                gpu_auth.item()
            ) != 1:
                raise RuntimeError(
                    "B3 authentication failed"
                )

        # B0 load 放在这里测，避免改变实验定义。
        b0_samples = self.measure(
            b0_load
        )

        results.append(
            self.summarize(
                "B0",
                "load",
                size,
                b0_samples,
            )
        )

        b3_samples = self.measure(
            b3_load
        )

        results.append(
            self.summarize(
                "B3_gpu",
                "load",
                size,
                b3_samples,
            )
        )

        # correctness check，不计入 timing。
        b3_load()

        if not bool(
            torch.equal(
                gpu_out,
                gpu_plain,
            )
        ):
            raise RuntimeError(
                "B3 plaintext mismatch"
            )

        return results


def print_results(rows):
    print()
    print(
        "================ SINGLE-CHUNK PATH ================"
    )

    print(
        "mode             dir    size      mean"
        "       p50       p95     GiB/s"
    )

    print(
        "----------------------------------------------------"
        "----------------"
    )

    for r in rows:
        print(
            f"{r['mode']:16s} "
            f"{r['direction']:5s} "
            f"{r['size_mib']:4d} MiB "
            f"{r['mean_ms']:9.3f} "
            f"{r['median_ms']:9.3f} "
            f"{r['p95_ms']:9.3f} "
            f"{r['gib_s']:9.3f}"
        )

    print()
    print(
        "================ B3 RELATIVE LATENCY ================"
    )

    by_key = {
        (
            r["direction"],
            r["size_mib"],
            r["mode"],
        ): r
        for r in rows
    }

    sizes = sorted(
        {
            r["size_mib"]
            for r in rows
        }
    )

    for direction in (
        "store",
        "load",
    ):
        for size_mib in sizes:
            b0 = by_key[
                (
                    direction,
                    size_mib,
                    "B0",
                )
            ]["mean_ms"]

            b1 = by_key[
                (
                    direction,
                    size_mib,
                    "B1_stock_cpu",
                )
            ]["mean_ms"]

            b2 = by_key[
                (
                    direction,
                    size_mib,
                    "B2_native_cpu",
                )
            ]["mean_ms"]

            b3 = by_key[
                (
                    direction,
                    size_mib,
                    "B3_gpu",
                )
            ]["mean_ms"]

            print(
                f"{direction:5s} "
                f"{size_mib:3d} MiB: "
                f"B1/B3={b1 / b3:6.2f}x  "
                f"B2/B3={b2 / b3:6.2f}x  "
                f"B0={b0:7.3f} ms  "
                f"B3={b3:7.3f} ms"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes-mib",
        nargs="+",
        type=int,
        default=[
            18,
            36,
            72,
        ],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = Path(
        os.environ.get(
            "CONFKV_ROOT",
            Path(__file__)
            .resolve()
            .parents[3],
        )
    ).resolve()

    print(
        "ConfKV H100 pre-CC B0/B1/B2/B3 path benchmark"
    )

    print(
        "root          :",
        root,
    )

    print(
        "torch         :",
        torch.__version__,
    )

    print(
        "torch CUDA    :",
        torch.version.cuda,
    )

    print(
        "device        :",
        torch.cuda.get_device_name(
            args.device
        ),
    )

    print(
        "warmup        :",
        args.warmup,
    )

    print(
        "iterations    :",
        args.iterations,
    )

    print(
        "sizes MiB     :",
        args.sizes_mib,
    )

    print(
        "environment   : PRE-CC "
        "(no TDX, no H100 CC)"
    )

    bench = PathBenchmark(
        root,
        args.device,
        args.warmup,
        args.iterations,
    )

    rows = []

    try:
        for mib in args.sizes_mib:
            print(
                f"\n[RUN] {mib} MiB"
            )

            rows.extend(
                bench.run_size(
                    mib
                    * 1024
                    * 1024
                )
            )

    finally:
        bench.close()

    print_results(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = [
            "direction",
            "mode",
            "size_mib",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "stdev_ms",
            "gib_s",
            "iterations",
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

        print(
            "\nCSV:",
            args.csv,
        )


if __name__ == "__main__":
    main()
