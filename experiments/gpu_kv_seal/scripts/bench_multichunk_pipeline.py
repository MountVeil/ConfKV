#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    idx = min(
        len(ys) - 1,
        max(0, math.ceil(q * len(ys)) - 1),
    )
    return ys[idx]


class NativeCpuAes128Gcm:
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
        nbytes: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_seal(
            self.key_ptr,
            int(src.data_ptr()),
            nbytes,
            int(dst.data_ptr()),
            int(dst.numel()),
        )
        if rc != 0:
            raise RuntimeError(f"native seal rc={rc}")

    def open(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        nbytes: int,
    ) -> None:
        rc = self.lib.lmcache_aes128gcm_open(
            self.key_ptr,
            int(src.data_ptr()),
            nbytes + FRAME_OVERHEAD,
            int(dst.data_ptr()),
            nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"native open rc={rc}")


class Buffers:
    def __init__(
        self,
        size: int,
        max_chunks: int,
        device: torch.device,
    ):
        self.size = size
        self.frame_len = size + FRAME_OVERHEAD

        # One GPU plaintext source / output is enough because the
        # benchmark measures pipeline throughput rather than unique data.
        self.gpu_plain = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )
        self.gpu_out = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )

        # ConfKV currently owns reusable GPU sealed staging.
        self.gpu_frame = torch.empty(
            self.frame_len,
            dtype=torch.uint8,
            device=device,
        )

        # Separate auth words make batched-auth validation possible.
        self.gpu_auth = [
            torch.empty(
                1,
                dtype=torch.int32,
                device=device,
            )
            for _ in range(max_chunks)
        ]

        # CPU L1-like pinned buffers.
        self.host_plain = [
            torch.empty(
                size,
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(max_chunks)
        ]

        self.host_frame = [
            torch.empty(
                self.frame_len,
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(max_chunks)
        ]


class PipelineBenchmark:
    def __init__(
        self,
        root: Path,
        device_index: int,
        warmup: int,
        iterations: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        torch.cuda.set_device(device_index)

        self.root = root
        self.device_index = device_index
        self.device = torch.device(f"cuda:{device_index}")
        self.stream = torch.cuda.Stream(device=self.device)

        self.warmup = warmup
        self.iterations = iterations

        gpu_dir = (
            root
            / "experiments"
            / "gpu_kv_seal"
            / "gpu"
        )
        sys.path.insert(0, str(gpu_dir))

        from gpu_aesgcm import GpuAes128GcmKey, GpuAesGcmLibrary

        gpu_lib_path = Path(
            os.environ.get(
                "CONFKV_GPU_AESGCM_LIB",
                gpu_dir / "libgpu_aesgcm.so",
            )
        )

        cpu_lib_path = Path(
            os.environ.get(
                "LMCACHE_AESGCM_NATIVE_LIB",
                root
                / "experiments"
                / "gpu_kv_seal"
                / "native"
                / "libcpu_seal.so",
            )
        )

        self.gpu_lib = GpuAesGcmLibrary(gpu_lib_path)
        self.gpu_key = GpuAes128GcmKey(
            self.gpu_lib,
            KEY,
            device=device_index,
        )
        self.native = NativeCpuAes128Gcm(cpu_lib_path)

        self.iv_counter = 1

    def close(self):
        self.gpu_key.close()

    def next_iv(self) -> bytes:
        iv = self.iv_counter.to_bytes(12, "big")
        self.iv_counter += 1
        return iv

    def measure(self, fn):
        for _ in range(self.warmup):
            fn()

        wall_ms: list[float] = []
        cpu_ms: list[float] = []

        for _ in range(self.iterations):
            w0 = time.perf_counter_ns()
            c0 = time.process_time_ns()

            fn()

            c1 = time.process_time_ns()
            w1 = time.perf_counter_ns()

            wall_ms.append((w1 - w0) / 1e6)
            cpu_ms.append((c1 - c0) / 1e6)

        return wall_ms, cpu_ms

    @staticmethod
    def summary(
        mode: str,
        direction: str,
        size: int,
        chunks: int,
        workers: int,
        wall_ms: list[float],
        cpu_ms: list[float],
    ):
        mean_wall = statistics.mean(wall_ms)
        mean_cpu = statistics.mean(cpu_ms)

        total_bytes = size * chunks

        gib_s = (
            total_bytes
            / (1024 ** 3)
            / (mean_wall / 1000.0)
        )

        return {
            "direction": direction,
            "mode": mode,
            "size_mib": size // (1024 * 1024),
            "chunks": chunks,
            "workers": workers,
            "mean_ms": mean_wall,
            "median_ms": statistics.median(wall_ms),
            "p95_ms": percentile(wall_ms, 0.95),
            "aggregate_gib_s": gib_s,
            "mean_cpu_ms": mean_cpu,
            "cpu_cores": (
                mean_cpu / mean_wall
                if mean_wall > 0
                else 0.0
            ),
        }

    def prepare_plain(
        self,
        b: Buffers,
        chunks: int,
    ):
        with torch.cuda.stream(self.stream):
            b.gpu_plain.fill_(0x5A)

            for i in range(chunks):
                b.host_plain[i].copy_(
                    b.gpu_plain,
                    non_blocking=True,
                )

        self.stream.synchronize()

    def prepare_stock_frames(
        self,
        b: Buffers,
        chunks: int,
    ):
        for i in range(chunks):
            plain = bytes(b.host_plain[i].numpy())
            iv = os.urandom(IV_LEN)
            ciphertext = AESGCM(KEY).encrypt(
                iv,
                plain,
                None,
            )

            arr = b.host_frame[i].numpy()

            arr[0] = 1
            arr[1:13] = np.frombuffer(
                iv,
                dtype=np.uint8,
            )
            arr[13:] = np.frombuffer(
                ciphertext,
                dtype=np.uint8,
            )

    def prepare_native_frames(
        self,
        b: Buffers,
        chunks: int,
    ):
        for i in range(chunks):
            self.native.seal(
                b.host_plain[i],
                b.host_frame[i],
                b.size,
            )

    def prepare_gpu_frames(
        self,
        b: Buffers,
        chunks: int,
    ):
        # GPU seal each frame and copy it to its CPU persistent buffer.
        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                self.gpu_key.seal_async(
                    src_ptr=int(b.gpu_plain.data_ptr()),
                    plaintext_len=b.size,
                    dst_ptr=int(b.gpu_frame.data_ptr()),
                    dst_capacity=b.frame_len,
                    iv=self.next_iv(),
                    stream_ptr=int(self.stream.cuda_stream),
                )

                b.host_frame[i].copy_(
                    b.gpu_frame,
                    non_blocking=True,
                )

        self.stream.synchronize()

    # ============================================================
    # STORE
    # ============================================================

    def b0_store(
        self,
        b: Buffers,
        chunks: int,
    ):
        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                b.host_plain[i].copy_(
                    b.gpu_plain,
                    non_blocking=True,
                )

        self.stream.synchronize()

    def cpu_store_pipeline(
        self,
        b: Buffers,
        chunks: int,
        workers: int,
        native: bool,
        executor: ThreadPoolExecutor,
    ):
        futures = []

        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                b.host_plain[i].copy_(
                    b.gpu_plain,
                    non_blocking=True,
                )

                event = torch.cuda.Event(
                    enable_timing=False,
                )
                event.record(self.stream)

                def work(
                    idx=i,
                    ready=event,
                ):
                    ready.synchronize()

                    if native:
                        self.native.seal(
                            b.host_plain[idx],
                            b.host_frame[idx],
                            b.size,
                        )
                    else:
                        plain = bytes(
                            b.host_plain[idx].numpy()
                        )
                        iv = os.urandom(IV_LEN)

                        ciphertext = AESGCM(KEY).encrypt(
                            iv,
                            plain,
                            None,
                        )

                        arr = b.host_frame[idx].numpy()

                        arr[0] = 1
                        arr[1:13] = np.frombuffer(
                            iv,
                            dtype=np.uint8,
                        )
                        arr[13:] = np.frombuffer(
                            ciphertext,
                            dtype=np.uint8,
                        )

                futures.append(
                    executor.submit(work)
                )

        for f in futures:
            f.result()

    def b3_store(
        self,
        b: Buffers,
        chunks: int,
    ):
        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                self.gpu_key.seal_async(
                    src_ptr=int(b.gpu_plain.data_ptr()),
                    plaintext_len=b.size,
                    dst_ptr=int(b.gpu_frame.data_ptr()),
                    dst_capacity=b.frame_len,
                    iv=self.next_iv(),
                    stream_ptr=int(self.stream.cuda_stream),
                )

                b.host_frame[i].copy_(
                    b.gpu_frame,
                    non_blocking=True,
                )

        self.stream.synchronize()

    # ============================================================
    # LOAD
    # ============================================================

    def b0_load(
        self,
        b: Buffers,
        chunks: int,
    ):
        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                b.gpu_out.copy_(
                    b.host_plain[i],
                    non_blocking=True,
                )

        self.stream.synchronize()

    def cpu_load_pipeline(
        self,
        b: Buffers,
        chunks: int,
        workers: int,
        native: bool,
        executor: ThreadPoolExecutor,
    ):
        def decrypt(idx: int):
            if native:
                self.native.open(
                    b.host_frame[idx],
                    b.host_plain[idx],
                    b.size,
                )
            else:
                blob = bytes(
                    b.host_frame[idx].numpy()
                )

                plaintext = AESGCM(KEY).decrypt(
                    blob[1:13],
                    blob[13:b.frame_len],
                    None,
                )

                b.host_plain[idx].numpy()[:] = (
                    np.frombuffer(
                        plaintext,
                        dtype=np.uint8,
                    )
                )

            return idx

        futures = [
            executor.submit(decrypt, i)
            for i in range(chunks)
        ]

        # As each CPU decrypt completes, immediately enqueue H2D.
        with torch.cuda.stream(self.stream):
            for future in as_completed(futures):
                idx = future.result()

                b.gpu_out.copy_(
                    b.host_plain[idx],
                    non_blocking=True,
                )

        self.stream.synchronize()

    def b3_load_current(
        self,
        b: Buffers,
        chunks: int,
    ):
        # Matches current correctness-first ConfKV behavior:
        # each persistent object must authenticate before its caller
        # is permitted to continue toward scatter.
        for i in range(chunks):
            with torch.cuda.stream(self.stream):
                b.gpu_frame.copy_(
                    b.host_frame[i],
                    non_blocking=True,
                )

                b.gpu_auth[i].zero_()

                self.gpu_key.open_async(
                    src_ptr=int(b.gpu_frame.data_ptr()),
                    frame_len=b.frame_len,
                    dst_ptr=int(b.gpu_out.data_ptr()),
                    plaintext_len=b.size,
                    auth_ok_ptr=int(
                        b.gpu_auth[i].data_ptr()
                    ),
                    stream_ptr=int(
                        self.stream.cuda_stream
                    ),
                )

            self.stream.synchronize()

            if int(b.gpu_auth[i].item()) != 1:
                raise RuntimeError(
                    f"B3 auth failed for chunk {i}"
                )

    def b3_load_batched_auth(
        self,
        b: Buffers,
        chunks: int,
    ):
        # Diagnostic optimization ceiling only:
        # same stream preserves H2D -> open ordering, but we defer
        # CPU observation of auth results until the whole batch finishes.
        with torch.cuda.stream(self.stream):
            for i in range(chunks):
                b.gpu_frame.copy_(
                    b.host_frame[i],
                    non_blocking=True,
                )

                b.gpu_auth[i].zero_()

                self.gpu_key.open_async(
                    src_ptr=int(b.gpu_frame.data_ptr()),
                    frame_len=b.frame_len,
                    dst_ptr=int(b.gpu_out.data_ptr()),
                    plaintext_len=b.size,
                    auth_ok_ptr=int(
                        b.gpu_auth[i].data_ptr()
                    ),
                    stream_ptr=int(
                        self.stream.cuda_stream
                    ),
                )

        self.stream.synchronize()

        for i in range(chunks):
            if int(b.gpu_auth[i].item()) != 1:
                raise RuntimeError(
                    f"B3 batched auth failed for chunk {i}"
                )

    def run_size(
        self,
        size: int,
        chunk_counts: list[int],
        worker_counts: list[int],
    ):
        max_chunks = max(chunk_counts)

        self.gpu_key.reserve(size)

        b = Buffers(
            size,
            max_chunks,
            self.device,
        )

        self.prepare_plain(
            b,
            max_chunks,
        )

        rows = []

        for chunks in chunk_counts:
            print(
                f"\n  chunks={chunks}"
            )

            # ---------------- B0 store ----------------
            wall, cpu = self.measure(
                lambda: self.b0_store(
                    b,
                    chunks,
                )
            )
            rows.append(
                self.summary(
                    "B0",
                    "store",
                    size,
                    chunks,
                    0,
                    wall,
                    cpu,
                )
            )

            # ---------------- B1/B2 store ----------------
            for workers in worker_counts:
                if workers > chunks:
                    continue

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    wall, cpu = self.measure(
                        lambda executor=executor: (
                            self.cpu_store_pipeline(
                                b,
                                chunks,
                                workers,
                                False,
                                executor,
                            )
                        )
                    )

                rows.append(
                    self.summary(
                        "B1_stock_cpu",
                        "store",
                        size,
                        chunks,
                        workers,
                        wall,
                        cpu,
                    )
                )

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    wall, cpu = self.measure(
                        lambda executor=executor: (
                            self.cpu_store_pipeline(
                                b,
                                chunks,
                                workers,
                                True,
                                executor,
                            )
                        )
                    )

                rows.append(
                    self.summary(
                        "B2_native_cpu",
                        "store",
                        size,
                        chunks,
                        workers,
                        wall,
                        cpu,
                    )
                )

            # ---------------- B3 store ----------------
            wall, cpu = self.measure(
                lambda: self.b3_store(
                    b,
                    chunks,
                )
            )

            rows.append(
                self.summary(
                    "B3_gpu",
                    "store",
                    size,
                    chunks,
                    0,
                    wall,
                    cpu,
                )
            )

            # ==================================================
            # LOAD
            # ==================================================

            wall, cpu = self.measure(
                lambda: self.b0_load(
                    b,
                    chunks,
                )
            )

            rows.append(
                self.summary(
                    "B0",
                    "load",
                    size,
                    chunks,
                    0,
                    wall,
                    cpu,
                )
            )

            # ---------------- stock CPU load ----------------
            self.prepare_stock_frames(
                b,
                chunks,
            )

            for workers in worker_counts:
                if workers > chunks:
                    continue

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    wall, cpu = self.measure(
                        lambda executor=executor: (
                            self.cpu_load_pipeline(
                                b,
                                chunks,
                                workers,
                                False,
                                executor,
                            )
                        )
                    )

                rows.append(
                    self.summary(
                        "B1_stock_cpu",
                        "load",
                        size,
                        chunks,
                        workers,
                        wall,
                        cpu,
                    )
                )

            # ---------------- native CPU load ----------------
            self.prepare_native_frames(
                b,
                chunks,
            )

            for workers in worker_counts:
                if workers > chunks:
                    continue

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    wall, cpu = self.measure(
                        lambda executor=executor: (
                            self.cpu_load_pipeline(
                                b,
                                chunks,
                                workers,
                                True,
                                executor,
                            )
                        )
                    )

                rows.append(
                    self.summary(
                        "B2_native_cpu",
                        "load",
                        size,
                        chunks,
                        workers,
                        wall,
                        cpu,
                    )
                )

            # ---------------- B3 load ----------------
            self.prepare_gpu_frames(
                b,
                chunks,
            )

            wall, cpu = self.measure(
                lambda: self.b3_load_current(
                    b,
                    chunks,
                )
            )

            rows.append(
                self.summary(
                    "B3_gpu_current",
                    "load",
                    size,
                    chunks,
                    0,
                    wall,
                    cpu,
                )
            )

            wall, cpu = self.measure(
                lambda: self.b3_load_batched_auth(
                    b,
                    chunks,
                )
            )

            rows.append(
                self.summary(
                    "B3_gpu_batched_auth",
                    "load",
                    size,
                    chunks,
                    0,
                    wall,
                    cpu,
                )
            )

        return rows


def print_rows(rows):
    print()
    print(
        "================ MULTI-CHUNK PIPELINE ================"
    )
    print(
        "mode                  dir   MiB chunks workers"
        "    mean_ms    p95_ms   agg_GiB/s CPUcores"
    )
    print(
        "------------------------------------------------------"
        "-------------------------------"
    )

    for r in rows:
        print(
            f"{r['mode']:21s} "
            f"{r['direction']:5s} "
            f"{r['size_mib']:4d} "
            f"{r['chunks']:6d} "
            f"{r['workers']:7d} "
            f"{r['mean_ms']:10.3f} "
            f"{r['p95_ms']:9.3f} "
            f"{r['aggregate_gib_s']:11.3f} "
            f"{r['cpu_cores']:8.2f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes-mib",
        nargs="+",
        type=int,
        default=[18, 36, 72],
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
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

    args = parser.parse_args()

    root = Path(
        os.environ.get(
            "CONFKV_ROOT",
            Path(__file__).resolve().parents[3],
        )
    ).resolve()

    print("ConfKV PRE-CC multi-chunk pipeline")
    print("root       :", root)
    print("torch      :", torch.__version__)
    print("CUDA       :", torch.version.cuda)
    print(
        "GPU        :",
        torch.cuda.get_device_name(args.device),
    )
    print("sizes MiB  :", args.sizes_mib)
    print("chunks     :", args.chunks)
    print("workers    :", args.workers)
    print("warmup     :", args.warmup)
    print("iterations :", args.iterations)
    print("TDX        : NO")
    print("H100 CC    : NO")

    bench = PipelineBenchmark(
        root,
        args.device,
        args.warmup,
        args.iterations,
    )

    rows = []

    try:
        for mib in args.sizes_mib:
            print(f"\n[SIZE] {mib} MiB")

            rows.extend(
                bench.run_size(
                    mib * 1024 * 1024,
                    args.chunks,
                    args.workers,
                )
            )
    finally:
        bench.close()

    print_rows(rows)

    args.csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "direction",
        "mode",
        "size_mib",
        "chunks",
        "workers",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "aggregate_gib_s",
        "mean_cpu_ms",
        "cpu_cores",
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

    print("\nCSV:", args.csv)


if __name__ == "__main__":
    main()
