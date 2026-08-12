#!/usr/bin/env python3

from __future__ import annotations

import ctypes
from pathlib import Path


FRAME_OVERHEAD = 29
IV_LEN = 12
KEY_LEN = 16


class GpuAesGcmError(RuntimeError):
    pass


class GpuAesGcmLibrary:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).with_name(
                "libgpu_aesgcm.so"
            )

        self.path = Path(path).resolve()

        if not self.path.exists():
            raise FileNotFoundError(
                f"GPU AES-GCM library not found: {self.path}"
            )

        self.lib = ctypes.CDLL(
            str(self.path)
        )

        self._configure_abi()

    def _configure_abi(self):
        lib = self.lib

        lib.lmcache_gpu_aesgcm_frame_size.argtypes = [
            ctypes.c_size_t,
        ]
        lib.lmcache_gpu_aesgcm_frame_size.restype = (
            ctypes.c_size_t
        )

        lib.lmcache_gpu_aes128gcm_key_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.lmcache_gpu_aes128gcm_key_create.restype = (
            ctypes.c_int
        )

        lib.lmcache_gpu_aes128gcm_key_destroy.argtypes = [
            ctypes.c_void_p,
        ]
        lib.lmcache_gpu_aes128gcm_key_destroy.restype = (
            ctypes.c_int
        )

        lib.lmcache_gpu_aes128gcm_reserve.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.lmcache_gpu_aes128gcm_reserve.restype = (
            ctypes.c_int
        )

        lib.lmcache_gpu_aes128gcm_seal_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_void_p,
        ]
        lib.lmcache_gpu_aes128gcm_seal_async.restype = (
            ctypes.c_int
        )

        lib.lmcache_gpu_aes128gcm_open_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.lmcache_gpu_aes128gcm_open_async.restype = (
            ctypes.c_int
        )

        lib.lmcache_gpu_aesgcm_strerror.argtypes = [
            ctypes.c_int,
        ]
        lib.lmcache_gpu_aesgcm_strerror.restype = (
            ctypes.c_char_p
        )

    def frame_size(self, plaintext_len: int) -> int:
        if plaintext_len < 0:
            raise ValueError(
                "plaintext_len must be non-negative"
            )

        return int(
            self.lib.lmcache_gpu_aesgcm_frame_size(
                plaintext_len
            )
        )

    def error_string(self, rc: int) -> str:
        p = self.lib.lmcache_gpu_aesgcm_strerror(
            rc
        )

        if not p:
            return f"error {rc}"

        return p.decode(
            "utf-8",
            errors="replace",
        )

    def check(self, rc: int, operation: str):
        if rc == 0:
            return

        raise GpuAesGcmError(
            f"{operation} failed: "
            f"rc={rc} "
            f"({self.error_string(rc)})"
        )


class GpuAes128GcmKey:
    """
    Thin owner of the native GPU AES-128-GCM key handle.

    Device pointers and CUDA stream handles are passed as integer
    addresses. This deliberately keeps the native binding independent
    of PyTorch; torch integration lives in the test/LMCache layers.
    """

    def __init__(
        self,
        library: GpuAesGcmLibrary,
        key: bytes,
        device: int = 0,
    ):
        if len(key) != KEY_LEN:
            raise ValueError(
                "AES-128 key must be exactly 16 bytes"
            )

        self.library = library
        self.device = int(device)
        self._handle = ctypes.c_void_p()

        KeyArray = ctypes.c_uint8 * KEY_LEN
        key_buf = KeyArray.from_buffer_copy(
            key
        )

        rc = (
            library.lib
            .lmcache_gpu_aes128gcm_key_create(
                key_buf,
                KEY_LEN,
                self.device,
                ctypes.byref(self._handle),
            )
        )

        library.check(
            rc,
            "key_create",
        )

        if not self._handle.value:
            raise GpuAesGcmError(
                "key_create succeeded but returned NULL handle"
            )

    @property
    def handle(self) -> int:
        if not self._handle.value:
            raise GpuAesGcmError(
                "GPU AES-GCM key has been closed"
            )

        return int(self._handle.value)

    def reserve(
        self,
        max_plaintext_len: int,
    ):
        rc = (
            self.library.lib
            .lmcache_gpu_aes128gcm_reserve(
                self._handle,
                max_plaintext_len,
            )
        )

        self.library.check(
            rc,
            "reserve",
        )

    def seal_async(
        self,
        *,
        src_ptr: int,
        plaintext_len: int,
        dst_ptr: int,
        dst_capacity: int,
        iv: bytes,
        stream_ptr: int,
    ):
        if len(iv) != IV_LEN:
            raise ValueError(
                "AES-GCM IV must be exactly 12 bytes"
            )

        IvArray = ctypes.c_uint8 * IV_LEN
        iv_buf = IvArray.from_buffer_copy(
            iv
        )

        rc = (
            self.library.lib
            .lmcache_gpu_aes128gcm_seal_async(
                self._handle,
                ctypes.c_void_p(src_ptr),
                plaintext_len,
                ctypes.c_void_p(dst_ptr),
                dst_capacity,
                iv_buf,
                ctypes.c_void_p(stream_ptr),
            )
        )

        self.library.check(
            rc,
            "seal_async",
        )

    def open_async(
        self,
        *,
        src_ptr: int,
        frame_len: int,
        dst_ptr: int,
        plaintext_len: int,
        auth_ok_ptr: int,
        stream_ptr: int,
    ):
        rc = (
            self.library.lib
            .lmcache_gpu_aes128gcm_open_async(
                self._handle,
                ctypes.c_void_p(src_ptr),
                frame_len,
                ctypes.c_void_p(dst_ptr),
                plaintext_len,
                ctypes.c_void_p(auth_ok_ptr),
                ctypes.c_void_p(stream_ptr),
            )
        )

        self.library.check(
            rc,
            "open_async",
        )

    def close(self):
        if not self._handle.value:
            return

        rc = (
            self.library.lib
            .lmcache_gpu_aes128gcm_key_destroy(
                self._handle
            )
        )

        self._handle = ctypes.c_void_p()

        self.library.check(
            rc,
            "key_destroy",
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.close()

    def __del__(self):
        # Do not raise from __del__.
        try:
            if getattr(
                self,
                "_handle",
                None,
            ) is not None:
                if self._handle.value:
                    self.library.lib.lmcache_gpu_aes128gcm_key_destroy(
                        self._handle
                    )
                    self._handle = ctypes.c_void_p()
        except Exception:
            pass
