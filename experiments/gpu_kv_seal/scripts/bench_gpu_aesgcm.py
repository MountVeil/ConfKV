#!/usr/bin/env python3

import ctypes
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "gpu"

sys.path.insert(
    0,
    str(GPU_DIR),
)

from cuda_runtime import (  # noqa: E402
    CudaRuntime,
    CudaRuntimeError,
)
from gpu_aesgcm import (  # noqa: E402
    GpuAes128GcmKey,
    GpuAesGcmLibrary,
)


SIZES_MIB = (
    18,
    36,
    72,
)

WARMUP = 5
ITERATIONS = 30

KEY = bytes(range(16))


rt = CudaRuntime()

try:
    count = rt.device_count()
except CudaRuntimeError as e:
    print(
        "SKIP: CUDA device unavailable:",
        e,
    )
    raise SystemExit(77)

if count == 0:
    print("SKIP: no CUDA devices")
    raise SystemExit(77)

rt.set_device(0)

gpu = GpuAesGcmLibrary()

stream = rt.stream_create()
start = rt.event_create()
end = rt.event_create()


def measure(
    enqueue,
):
    samples = []

    for _ in range(WARMUP):
        enqueue()
        rt.stream_sync(
            stream
        )

    for _ in range(ITERATIONS):
        rt.event_record(
            start,
            stream,
        )

        enqueue()

        rt.event_record(
            end,
            stream,
        )

        rt.event_sync(
            end
        )

        samples.append(
            rt.event_elapsed_ms(
                start,
                end,
            )
        )

    return samples


try:
    with GpuAes128GcmKey(
        gpu,
        KEY,
        device=0,
    ) as key:

        key.reserve(
            max(SIZES_MIB)
            * 1024
            * 1024
        )

        for mib in SIZES_MIB:
            size = mib * 1024 * 1024

            frame_len = (
                gpu.frame_size(size)
            )

            d_plain = rt.malloc(size)
            d_frame = rt.malloc(
                frame_len
            )
            d_out = rt.malloc(size)
            d_auth = rt.malloc(
                ctypes.sizeof(
                    ctypes.c_int
                )
            )

            try:
                rt.memset(
                    d_plain,
                    0xA5,
                    size,
                )

                iv_counter = [1]

                def seal_once():
                    value = iv_counter[0]
                    iv_counter[0] += 1

                    iv = value.to_bytes(
                        12,
                        "big",
                    )

                    key.seal_async(
                        src_ptr=d_plain,
                        plaintext_len=size,
                        dst_ptr=d_frame,
                        dst_capacity=frame_len,
                        iv=iv,
                        stream_ptr=stream,
                    )

                # Build one valid frame
                seal_once()
                rt.stream_sync(
                    stream
                )

                def open_once():
                    rt.memset_async(
                        d_auth,
                        0,
                        ctypes.sizeof(
                            ctypes.c_int
                        ),
                        stream,
                    )

                    key.open_async(
                        src_ptr=d_frame,
                        frame_len=frame_len,
                        dst_ptr=d_out,
                        plaintext_len=size,
                        auth_ok_ptr=d_auth,
                        stream_ptr=stream,
                    )

                seal_samples = measure(
                    seal_once
                )

                # Rebuild one final valid frame.
                seal_once()
                rt.stream_sync(
                    stream
                )

                open_samples = measure(
                    open_once
                )

                for name, samples in (
                    (
                        "gpu_seal",
                        seal_samples,
                    ),
                    (
                        "gpu_open",
                        open_samples,
                    ),
                ):
                    mean_ms = (
                        statistics.mean(
                            samples
                        )
                    )

                    stdev_ms = (
                        statistics.pstdev(
                            samples
                        )
                    )

                    gib_s = (
                        size
                        / (1024 ** 3)
                        / (
                            mean_ms
                            / 1000
                        )
                    )

                    print(
                        f"{name:10s} "
                        f"{mib:3d} MiB "
                        f"{mean_ms:10.3f} ms "
                        f"{gib_s:9.3f} GiB/s "
                        f"sigma={stdev_ms:.3f} ms"
                    )

            finally:
                rt.free(d_auth)
                rt.free(d_out)
                rt.free(d_frame)
                rt.free(d_plain)

finally:
    rt.event_destroy(end)
    rt.event_destroy(start)
    rt.stream_destroy(stream)
