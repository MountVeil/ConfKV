#!/usr/bin/env python3

import ctypes
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


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


KEY = bytes(range(16))


def plaintext(size: int) -> bytes:
    pattern = bytes(range(256))

    return (
        pattern
        * ((size + 255) // 256)
    )[:size]


def get_device_or_skip(rt):
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


rt = CudaRuntime()
get_device_or_skip(rt)

gpu = GpuAesGcmLibrary()

stream = rt.stream_create()

try:
    with GpuAes128GcmKey(
        gpu,
        KEY,
        device=0,
    ) as key:

        key.reserve(
            4 * 1024 * 1024
        )

        for size in (
            1,
            15,
            16,
            17,
            31,
            32,
            33,
            4096,
            4 * 1024 * 1024,
        ):
            plain = plaintext(size)

            iv = (
                size.to_bytes(
                    12,
                    "big",
                )
            )

            cpu_ct_tag = (
                AESGCM(KEY).encrypt(
                    iv,
                    plain,
                    None,
                )
            )

            expected_frame = (
                b"\x01"
                + iv
                + cpu_ct_tag
            )

            frame_len = len(
                expected_frame
            )

            d_plain = rt.malloc(
                max(size, 1)
            )
            d_frame = rt.malloc(
                frame_len
            )
            d_out = rt.malloc(
                max(size, 1)
            )
            d_auth = rt.malloc(
                ctypes.sizeof(
                    ctypes.c_int
                )
            )

            try:
                # ---------------------------------
                # GPU seal -> CPU AESGCM open
                # ---------------------------------

                rt.h2d(
                    d_plain,
                    plain,
                )

                key.seal_async(
                    src_ptr=d_plain,
                    plaintext_len=size,
                    dst_ptr=d_frame,
                    dst_capacity=frame_len,
                    iv=iv,
                    stream_ptr=stream,
                )

                rt.stream_sync(
                    stream
                )

                gpu_frame = rt.d2h(
                    d_frame,
                    frame_len,
                )

                assert (
                    gpu_frame
                    == expected_frame
                ), (
                    f"GPU frame mismatch "
                    f"at size={size}"
                )

                cpu_open = (
                    AESGCM(KEY).decrypt(
                        gpu_frame[1:13],
                        gpu_frame[13:],
                        None,
                    )
                )

                assert cpu_open == plain

                # ---------------------------------
                # CPU AESGCM seal -> GPU open
                # ---------------------------------

                rt.h2d(
                    d_frame,
                    expected_frame,
                )

                rt.memset(
                    d_out,
                    0xCC,
                    max(size, 1),
                )

                rt.memset(
                    d_auth,
                    0,
                    ctypes.sizeof(
                        ctypes.c_int
                    ),
                )

                key.open_async(
                    src_ptr=d_frame,
                    frame_len=frame_len,
                    dst_ptr=d_out,
                    plaintext_len=size,
                    auth_ok_ptr=d_auth,
                    stream_ptr=stream,
                )

                rt.stream_sync(
                    stream
                )

                auth_raw = rt.d2h(
                    d_auth,
                    ctypes.sizeof(
                        ctypes.c_int
                    ),
                )

                auth_ok = int.from_bytes(
                    auth_raw,
                    byteorder=sys.byteorder,
                    signed=True,
                )

                assert auth_ok == 1

                gpu_open = rt.d2h(
                    d_out,
                    size,
                )

                assert gpu_open == plain

                print(
                    f"PASS size={size}: "
                    "GPU seal <-> CPU AESGCM"
                )

            finally:
                rt.free(d_auth)
                rt.free(d_out)
                rt.free(d_frame)
                rt.free(d_plain)

finally:
    rt.stream_destroy(
        stream
    )


print()
print("==============================")
print("GPU AES-GCM INTEROP GATE PASSED")
print("==============================")
