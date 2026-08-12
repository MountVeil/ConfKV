#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "gpu"

sys.path.insert(
    0,
    str(GPU_DIR),
)

from gpu_aesgcm import (  # noqa: E402
    FRAME_OVERHEAD,
    GpuAesGcmLibrary,
)


lib = GpuAesGcmLibrary()


print(
    "Library:",
    lib.path,
)


for size in (
    1,
    15,
    16,
    17,
    1024,
    4 * 1024 * 1024,
    18 * 1024 * 1024,
    36 * 1024 * 1024,
    72 * 1024 * 1024,
):
    frame = lib.frame_size(
        size
    )

    assert (
        frame
        == size + FRAME_OVERHEAD
    )

    print(
        f"PASS frame_size "
        f"{size} -> {frame}"
    )


assert (
    lib.error_string(-1)
    == "invalid argument"
)

assert (
    lib.error_string(-2)
    == "GPU allocation failure"
)

assert (
    lib.error_string(-3)
    == "CUDA failure"
)

print(
    "PASS error-string ABI"
)

print()
print(
    "=============================="
)
print(
    "GPU AES-GCM ABI GATE PASSED"
)
print(
    "=============================="
)
