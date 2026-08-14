#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
from pathlib import Path
import statistics
import sys

import torch


FRAME_OVERHEAD = 29
KEY = bytes(range(16))


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    idx = min(
        len(ys) - 1,
        max(0, math.ceil(q * len(ys)) - 1),
    )
    return ys[idx]


class CryptoSlot:
    """One independent native GPU AES-GCM execution context."""

    def __init__(
        self,
        *,
        key,
        stream: torch.cuda.Stream,
    ):
        self.key = key
        self.stream = stream


class ParallelGpuAesOnly:
    """Parallel GPU AES-GCM benchmark without PCIe transfers."""

    def __init__(
        self,
        root: Path,
        device_index: int,
        size: int,
        chunks: int,
        slots: int,
    ):
        if slots <= 0:
            raise ValueError("slots must be positive")

        if slots > chunks:
            raise ValueError(
                "slots cannot exceed chunks"
            )

        self.root = root
        self.device_index = device_index
        self.device = torch.device(
            f"cuda:{device_index}"
        )

        self.size = size
        self.frame_size = (
            size + FRAME_OVERHEAD
        )
        self.chunks = chunks
        self.num_slots = slots

        torch.cuda.set_device(
            device_index
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

        lib_path = Path(
            os.environ.get(
                "CONFKV_GPU_AESGCM_LIB",
                gpu_dir / "libgpu_aesgcm.so",
            )
        ).expanduser().resolve()

        if not lib_path.is_file():
            raise FileNotFoundError(
                lib_path
            )

        self.library = (
            GpuAesGcmLibrary(
                lib_path
            )
        )

        self.control_stream = (
            torch.cuda.Stream(
                device=self.device
            )
        )

        self.slots: list[
            CryptoSlot
        ] = []

        for _ in range(
            self.num_slots
        ):
            key = GpuAes128GcmKey(
                self.library,
                KEY,
                device=device_index,
            )

            # Each native key handle owns an
            # independent GHASH workspace.
            key.reserve(
                size
            )

            self.slots.append(
                CryptoSlot(
                    key=key,
                    stream=torch.cuda.Stream(
                        device=self.device
                    ),
                )
            )

        # Each logical chunk has independent
        # input/output/frame/auth storage.
        self.plaintexts = [
            torch.empty(
                size,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(chunks)
        ]

        self.frames = [
            torch.empty(
                self.frame_size,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(chunks)
        ]

        self.outputs = [
            torch.empty(
                size,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(chunks)
        ]

        self.auth = [
            torch.empty(
                1,
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(chunks)
        ]

        self._iv_counter = 1

        with torch.cuda.stream(
            self.control_stream
        ):
            for i, tensor in enumerate(
                self.plaintexts
            ):
                tensor.fill_(
                    (17 + i) & 0xFF
                )

        self.control_stream.synchronize()

    def next_iv(self) -> bytes:
        """Return a fresh unique 96-bit IV.

        IV generation is deliberately outside the GPU event timing.
        The benchmark measures GPU AES-GCM execution rather than
        CPU random-number-generation overhead.
        """
        iv = self._iv_counter.to_bytes(
            12,
            "big",
        )

        self._iv_counter += 1

        return iv

    def _begin_gpu_measurement(
        self,
    ) -> torch.cuda.Event:
        start = torch.cuda.Event(
            enable_timing=True
        )

        start.record(
            self.control_stream
        )

        for slot in self.slots:
            slot.stream.wait_event(
                start
            )

        return start

    def _finish_gpu_measurement(
        self,
        start: torch.cuda.Event,
    ) -> float:
        # Join all crypto streams into one
        # control stream without serializing
        # the workers with each other.
        for slot in self.slots:
            done = torch.cuda.Event(
                enable_timing=False
            )

            done.record(
                slot.stream
            )

            self.control_stream.wait_event(
                done
            )

        end = torch.cuda.Event(
            enable_timing=True
        )

        end.record(
            self.control_stream
        )

        end.synchronize()

        return float(
            start.elapsed_time(end)
        )

    def seal_once(self) -> float:
        """Seal all logical chunks and return GPU makespan in ms."""

        ivs = [
            self.next_iv()
            for _ in range(
                self.chunks
            )
        ]

        start = (
            self._begin_gpu_measurement()
        )

        for chunk_idx in range(
            self.chunks
        ):
            slot = self.slots[
                chunk_idx
                % self.num_slots
            ]

            with torch.cuda.stream(
                slot.stream
            ):
                slot.key.seal_async(
                    src_ptr=int(
                        self.plaintexts[
                            chunk_idx
                        ].data_ptr()
                    ),
                    plaintext_len=self.size,
                    dst_ptr=int(
                        self.frames[
                            chunk_idx
                        ].data_ptr()
                    ),
                    dst_capacity=(
                        self.frame_size
                    ),
                    iv=ivs[chunk_idx],
                    stream_ptr=int(
                        slot.stream.cuda_stream
                    ),
                )

        return (
            self._finish_gpu_measurement(
                start
            )
        )

    def open_once(self) -> float:
        """Open all logical chunks and return GPU makespan in ms."""

        start = (
            self._begin_gpu_measurement()
        )

        for chunk_idx in range(
            self.chunks
        ):
            slot = self.slots[
                chunk_idx
                % self.num_slots
            ]

            with torch.cuda.stream(
                slot.stream
            ):
                slot.key.open_async(
                    src_ptr=int(
                        self.frames[
                            chunk_idx
                        ].data_ptr()
                    ),
                    frame_len=(
                        self.frame_size
                    ),
                    dst_ptr=int(
                        self.outputs[
                            chunk_idx
                        ].data_ptr()
                    ),
                    plaintext_len=self.size,
                    auth_ok_ptr=int(
                        self.auth[
                            chunk_idx
                        ].data_ptr()
                    ),
                    stream_ptr=int(
                        slot.stream.cuda_stream
                    ),
                )

        return (
            self._finish_gpu_measurement(
                start
            )
        )

    def prepare_open_frames(
        self,
    ) -> None:
        # One valid seal pass before measuring open.
        self.seal_once()

        for auth in self.auth:
            auth.zero_()

        torch.cuda.synchronize(
            self.device
        )

    def validate_open(
        self,
    ) -> None:
        torch.cuda.synchronize(
            self.device
        )

        for i in range(
            self.chunks
        ):
            if int(
                self.auth[i].item()
            ) != 1:
                raise RuntimeError(
                    "AES-GCM authentication "
                    f"failed for chunk {i}"
                )

            if not torch.equal(
                self.plaintexts[i],
                self.outputs[i],
            ):
                raise RuntimeError(
                    "AES-GCM plaintext mismatch "
                    f"for chunk {i}"
                )

    def close(self) -> None:
        for slot in self.slots:
            slot.stream.synchronize()

        for slot in self.slots:
            slot.key.close()

        self.slots.clear()


def summarize(
    *,
    mode: str,
    size: int,
    chunks: int,
    slots: int,
    samples: list[float],
):
    mean_ms = statistics.mean(
        samples
    )

    total_bytes = (
        size * chunks
    )

    gib_s = (
        total_bytes
        / (1024 ** 3)
        / (mean_ms / 1000.0)
    )

    return {
        "mode": mode,
        "size_mib": (
            size // (1024 ** 2)
        ),
        "chunks": chunks,
        "slots": slots,
        "mean_ms": mean_ms,
        "median_ms": (
            statistics.median(
                samples
            )
        ),
        "p95_ms": percentile(
            samples,
            0.95,
        ),
        "sigma_ms": (
            statistics.pstdev(
                samples
            )
        ),
        "aggregate_gib_s": gib_s,
    }


def benchmark_case(
    *,
    root: Path,
    device: int,
    size: int,
    chunks: int,
    slots: int,
    warmup: int,
    iterations: int,
):
    bench = ParallelGpuAesOnly(
        root,
        device,
        size,
        chunks,
        slots,
    )

    try:
        # ---------------- Seal ----------------

        for _ in range(warmup):
            bench.seal_once()

        seal_samples = [
            bench.seal_once()
            for _ in range(
                iterations
            )
        ]

        seal = summarize(
            mode="seal",
            size=size,
            chunks=chunks,
            slots=slots,
            samples=seal_samples,
        )

        # ---------------- Open ----------------

        bench.prepare_open_frames()

        for _ in range(warmup):
            bench.open_once()

        open_samples = [
            bench.open_once()
            for _ in range(
                iterations
            )
        ]

        opened = summarize(
            mode="open",
            size=size,
            chunks=chunks,
            slots=slots,
            samples=open_samples,
        )

        bench.validate_open()

        return [
            seal,
            opened,
        ]

    finally:
        bench.close()


def print_rows(rows):
    print()
    print(
        "========== PARALLEL GPU AES-ONLY =========="
    )

    print(
        "mode  MiB chunks slots"
        "    mean_ms    p95_ms"
        "    sigma   agg_GiB/s"
    )

    print(
        "-------------------------------------------"
        "---------------------"
    )

    for row in rows:
        print(
            f"{row['mode']:5s} "
            f"{row['size_mib']:4d} "
            f"{row['chunks']:6d} "
            f"{row['slots']:5d} "
            f"{row['mean_ms']:10.3f} "
            f"{row['p95_ms']:9.3f} "
            f"{row['sigma_ms']:8.3f} "
            f"{row['aggregate_gib_s']:11.3f}"
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
        type=int,
        default=8,
    )

    parser.add_argument(
        "--slots",
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
        required=True,
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
        "ConfKV parallel GPU AES-only benchmark"
    )

    print(
        "GPU        :",
        torch.cuda.get_device_name(
            args.device
        ),
    )

    print(
        "torch      :",
        torch.__version__,
    )

    print(
        "CUDA build :",
        torch.version.cuda,
    )

    print(
        "sizes MiB  :",
        args.sizes_mib,
    )

    print(
        "chunks     :",
        args.chunks,
    )

    print(
        "slots      :",
        args.slots,
    )

    print(
        "warmup     :",
        args.warmup,
    )

    print(
        "iterations :",
        args.iterations,
    )

    print(
        "TDX        : NO"
    )

    print(
        "H100 CC    : NO"
    )

    print(
        "PCIe copy  : EXCLUDED"
    )

    rows = []

    for mib in args.sizes_mib:
        size = (
            mib
            * 1024
            * 1024
        )

        print(
            f"\n[SIZE] {mib} MiB"
        )

        for slots in args.slots:
            if slots > args.chunks:
                continue

            print(
                f"  slots={slots}"
            )

            case_rows = benchmark_case(
                root=root,
                device=args.device,
                size=size,
                chunks=args.chunks,
                slots=slots,
                warmup=args.warmup,
                iterations=args.iterations,
            )

            rows.extend(
                case_rows
            )

            # Release per-case GPU buffers/workspaces
            # before moving to the next slot count.
            gc.collect()
            torch.cuda.empty_cache()

    print_rows(rows)

    args.csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "mode",
        "size_mib",
        "chunks",
        "slots",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "sigma_ms",
        "aggregate_gib_s",
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
    print(
        "CSV:",
        args.csv,
    )


if __name__ == "__main__":
    main()
