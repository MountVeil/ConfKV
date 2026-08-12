#!/usr/bin/env python3

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
from pathlib import Path


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


class CudaRuntimeError(RuntimeError):
    pass


def _find_cudart() -> Path | str:
    found = ctypes.util.find_library("cudart")
    if found:
        return found

    candidates: list[Path] = [
        Path("/usr/local/cuda/lib64/libcudart.so"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so"),
        Path("/usr/local/cuda-12.0/lib64/libcudart.so"),
        Path("/usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudart.so"),
    ]

    nvcc = shutil.which("nvcc")

    if nvcc:
        cuda_root = Path(nvcc).resolve().parent.parent

        candidates.extend(
            [
                cuda_root / "lib64" / "libcudart.so",
                cuda_root
                / "targets"
                / "x86_64-linux"
                / "lib"
                / "libcudart.so",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot locate libcudart.so"
    )


class CudaRuntime:
    def __init__(self):
        self.path = _find_cudart()
        self.lib = ctypes.CDLL(str(self.path))
        self._configure()

    def _configure(self):
        l = self.lib

        l.cudaGetErrorString.argtypes = [
            ctypes.c_int
        ]
        l.cudaGetErrorString.restype = (
            ctypes.c_char_p
        )

        l.cudaGetDeviceCount.argtypes = [
            ctypes.POINTER(ctypes.c_int)
        ]
        l.cudaGetDeviceCount.restype = (
            ctypes.c_int
        )

        l.cudaSetDevice.argtypes = [
            ctypes.c_int
        ]
        l.cudaSetDevice.restype = (
            ctypes.c_int
        )

        l.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        l.cudaMalloc.restype = ctypes.c_int

        l.cudaFree.argtypes = [
            ctypes.c_void_p
        ]
        l.cudaFree.restype = ctypes.c_int

        l.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        l.cudaMemcpy.restype = ctypes.c_int

        l.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        l.cudaMemset.restype = ctypes.c_int

        l.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        l.cudaMemsetAsync.restype = ctypes.c_int

        l.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        l.cudaStreamCreate.restype = (
            ctypes.c_int
        )

        l.cudaStreamDestroy.argtypes = [
            ctypes.c_void_p
        ]
        l.cudaStreamDestroy.restype = (
            ctypes.c_int
        )

        l.cudaStreamSynchronize.argtypes = [
            ctypes.c_void_p
        ]
        l.cudaStreamSynchronize.restype = (
            ctypes.c_int
        )

        l.cudaEventCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        l.cudaEventCreate.restype = (
            ctypes.c_int
        )

        l.cudaEventRecord.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        l.cudaEventRecord.restype = (
            ctypes.c_int
        )

        l.cudaEventSynchronize.argtypes = [
            ctypes.c_void_p
        ]
        l.cudaEventSynchronize.restype = (
            ctypes.c_int
        )

        l.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        l.cudaEventElapsedTime.restype = (
            ctypes.c_int
        )

        l.cudaEventDestroy.argtypes = [
            ctypes.c_void_p
        ]
        l.cudaEventDestroy.restype = (
            ctypes.c_int
        )

    def error_string(self, rc: int) -> str:
        s = self.lib.cudaGetErrorString(rc)

        if not s:
            return f"CUDA error {rc}"

        return s.decode()

    def check(self, rc: int, op: str):
        if rc == 0:
            return

        raise CudaRuntimeError(
            f"{op}: rc={rc}: "
            f"{self.error_string(rc)}"
        )

    def device_count(self) -> int:
        n = ctypes.c_int()

        self.check(
            self.lib.cudaGetDeviceCount(
                ctypes.byref(n)
            ),
            "cudaGetDeviceCount",
        )

        return n.value

    def set_device(self, device: int):
        self.check(
            self.lib.cudaSetDevice(device),
            "cudaSetDevice",
        )

    def malloc(self, size: int) -> int:
        ptr = ctypes.c_void_p()

        self.check(
            self.lib.cudaMalloc(
                ctypes.byref(ptr),
                size,
            ),
            "cudaMalloc",
        )

        assert ptr.value is not None
        return int(ptr.value)

    def free(self, ptr: int):
        if ptr:
            self.check(
                self.lib.cudaFree(
                    ctypes.c_void_p(ptr)
                ),
                "cudaFree",
            )

    def memset(
        self,
        ptr: int,
        value: int,
        size: int,
    ):
        self.check(
            self.lib.cudaMemset(
                ctypes.c_void_p(ptr),
                value,
                size,
            ),
            "cudaMemset",
        )

    def memset_async(
        self,
        ptr: int,
        value: int,
        size: int,
        stream: int,
    ):
        self.check(
            self.lib.cudaMemsetAsync(
                ctypes.c_void_p(ptr),
                value,
                size,
                ctypes.c_void_p(stream),
            ),
            "cudaMemsetAsync",
        )

    def h2d(self, dst: int, data: bytes):
        if not data:
            return

        buf = (
            ctypes.c_ubyte * len(data)
        ).from_buffer_copy(data)

        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst),
                ctypes.cast(
                    buf,
                    ctypes.c_void_p,
                ),
                len(data),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D",
        )

    def d2h(self, src: int, size: int) -> bytes:
        buf = (
            ctypes.c_ubyte * size
        )()

        self.check(
            self.lib.cudaMemcpy(
                ctypes.cast(
                    buf,
                    ctypes.c_void_p,
                ),
                ctypes.c_void_p(src),
                size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy D2H",
        )

        return bytes(buf)

    def stream_create(self) -> int:
        s = ctypes.c_void_p()

        self.check(
            self.lib.cudaStreamCreate(
                ctypes.byref(s)
            ),
            "cudaStreamCreate",
        )

        assert s.value is not None
        return int(s.value)

    def stream_sync(self, stream: int):
        self.check(
            self.lib.cudaStreamSynchronize(
                ctypes.c_void_p(stream)
            ),
            "cudaStreamSynchronize",
        )

    def stream_destroy(self, stream: int):
        self.check(
            self.lib.cudaStreamDestroy(
                ctypes.c_void_p(stream)
            ),
            "cudaStreamDestroy",
        )

    def event_create(self) -> int:
        e = ctypes.c_void_p()

        self.check(
            self.lib.cudaEventCreate(
                ctypes.byref(e)
            ),
            "cudaEventCreate",
        )

        assert e.value is not None
        return int(e.value)

    def event_record(
        self,
        event: int,
        stream: int,
    ):
        self.check(
            self.lib.cudaEventRecord(
                ctypes.c_void_p(event),
                ctypes.c_void_p(stream),
            ),
            "cudaEventRecord",
        )

    def event_sync(self, event: int):
        self.check(
            self.lib.cudaEventSynchronize(
                ctypes.c_void_p(event)
            ),
            "cudaEventSynchronize",
        )

    def event_elapsed_ms(
        self,
        start: int,
        end: int,
    ) -> float:
        ms = ctypes.c_float()

        self.check(
            self.lib.cudaEventElapsedTime(
                ctypes.byref(ms),
                ctypes.c_void_p(start),
                ctypes.c_void_p(end),
            ),
            "cudaEventElapsedTime",
        )

        return float(ms.value)

    def event_destroy(self, event: int):
        self.check(
            self.lib.cudaEventDestroy(
                ctypes.c_void_p(event)
            ),
            "cudaEventDestroy",
        )
