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
WRONG_KEY = bytes(
    reversed(
        range(16)
    )
)

SIZE = 4 * 1024 * 1024


def auth_value(rt, ptr):
    raw = rt.d2h(
        ptr,
        ctypes.sizeof(
            ctypes.c_int
        ),
    )

    return int.from_bytes(
        raw,
        byteorder=sys.byteorder,
        signed=True,
    )


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

plain = (
    bytes(range(256))
    * (SIZE // 256)
)

iv = bytes.fromhex(
    "00112233445566778899aabb"
)

ct_tag = AESGCM(KEY).encrypt(
    iv,
    plain,
    None,
)

good_frame = (
    b"\x01"
    + iv
    + ct_tag
)


def run_open_test(
    key_obj,
    frame,
    expected_auth,
    name,
):
    d_frame = rt.malloc(
        len(frame)
    )
    d_out = rt.malloc(
        SIZE
    )
    d_auth = rt.malloc(
        ctypes.sizeof(
            ctypes.c_int
        )
    )

    try:
        rt.h2d(
            d_frame,
            frame,
        )

        rt.memset(
            d_out,
            0xCC,
            SIZE,
        )

        rt.memset(
            d_auth,
            0,
            ctypes.sizeof(
                ctypes.c_int
            ),
        )

        key_obj.open_async(
            src_ptr=d_frame,
            frame_len=len(frame),
            dst_ptr=d_out,
            plaintext_len=SIZE,
            auth_ok_ptr=d_auth,
            stream_ptr=stream,
        )

        rt.stream_sync(
            stream
        )

        auth = auth_value(
            rt,
            d_auth,
        )

        assert auth == expected_auth

        result = rt.d2h(
            d_out,
            SIZE,
        )

        if expected_auth:
            assert result == plain
        else:
            assert result == (
                b"\x00" * SIZE
            )

        print(
            f"PASS {name}"
        )

    finally:
        rt.free(d_auth)
        rt.free(d_out)
        rt.free(d_frame)


try:
    with GpuAes128GcmKey(
        gpu,
        KEY,
        device=0,
    ) as key:

        key.reserve(SIZE)

        run_open_test(
            key,
            good_frame,
            1,
            "valid frame",
        )

        # Ciphertext
        x = bytearray(good_frame)
        x[13 + 100] ^= 0x01

        run_open_test(
            key,
            bytes(x),
            0,
            "ciphertext tamper rejected + zeroized",
        )

        # IV
        x = bytearray(good_frame)
        x[1] ^= 0x01

        run_open_test(
            key,
            bytes(x),
            0,
            "IV tamper rejected + zeroized",
        )

        # Tag
        x = bytearray(good_frame)
        x[-1] ^= 0x01

        run_open_test(
            key,
            bytes(x),
            0,
            "tag tamper rejected + zeroized",
        )

        # Version
        x = bytearray(good_frame)
        x[0] = 2

        run_open_test(
            key,
            bytes(x),
            0,
            "version tamper rejected + zeroized",
        )

    with GpuAes128GcmKey(
        gpu,
        WRONG_KEY,
        device=0,
    ) as wrong:

        wrong.reserve(SIZE)

        run_open_test(
            wrong,
            good_frame,
            0,
            "wrong key rejected + zeroized",
        )

finally:
    rt.stream_destroy(
        stream
    )


print()
print("==============================")
print("GPU AES-GCM SECURITY GATE PASSED")
print("==============================")
