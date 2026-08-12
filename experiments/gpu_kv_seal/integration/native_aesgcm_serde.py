import ctypes
import os
from pathlib import Path

import torch
from cryptography.exceptions import InvalidTag

from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    Serializer,
)
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.key_provider import (
    HkdfKeyProvider,
    KeyProvider,
)


_VERSION = 1
_IV_LEN = 12
_TAG_LEN = 16
_HDR_LEN = 1 + _IV_LEN
_FRAME_OVERHEAD = _HDR_LEN + _TAG_LEN

_HKDF_INFO_PREFIX = b"lmcache-l2-aesgcm-v1"


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "native" / "libcpu_seal.so"

_lib = ctypes.CDLL(str(LIB_PATH))

_lib.lmcache_aes128gcm_seal.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.lmcache_aes128gcm_seal.restype = ctypes.c_int

_lib.lmcache_aes128gcm_open.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.lmcache_aes128gcm_open.restype = ctypes.c_int


def _plaintext_bytes(layout_desc: MemoryLayoutDesc) -> int:
    total = 0

    for shape, dtype in zip(
        layout_desc.shapes,
        layout_desc.dtypes,
        strict=True,
    ):
        n = 1

        for dim in shape:
            n *= int(dim)

        total += (
            n
            * torch.empty(
                (),
                dtype=dtype,
            ).element_size()
        )

    return total


def _byte_view(obj):
    return memoryview(
        obj.byte_array
    ).cast("B")


def _address(view) -> int:
    if len(view) == 0:
        raise ValueError("zero-length buffer unsupported")

    return ctypes.addressof(
        ctypes.c_ubyte.from_buffer(view)
    )


def _key_ptr(raw_key: bytes):
    KeyArray = ctypes.c_ubyte * len(raw_key)

    holder = KeyArray.from_buffer_copy(
        raw_key
    )

    return holder, ctypes.addressof(holder)


class NativeAesGcmSerializer(Serializer):

    def __init__(
        self,
        key_provider: KeyProvider,
    ):
        self._keys = key_provider

    def serialize(
        self,
        src,
        dst,
        key: ObjectKey,
    ) -> int:

        src_view = _byte_view(src)
        dst_view = _byte_view(dst)

        plaintext_len = len(src_view)
        frame_len = (
            plaintext_len
            + _FRAME_OVERHEAD
        )

        if len(dst_view) < frame_len:
            raise ValueError(
                "native aesgcm: destination too small"
            )

        # Same steady-state key lookup semantics as stock LMCache.
        dek = self._keys.get_key(
            key.cache_salt
        )

        if len(dek) != 16:
            raise ValueError(
                "native aesgcm currently supports AES-128 only"
            )

        key_holder, key_ptr = _key_ptr(dek)

        rc = _lib.lmcache_aes128gcm_seal(
            key_ptr,
            _address(src_view),
            plaintext_len,
            _address(dst_view),
            len(dst_view),
        )

        # Keep holder alive through the native call.
        _ = key_holder

        if rc != 0:
            raise RuntimeError(
                f"native aesgcm seal failed: rc={rc}"
            )

        return frame_len

    def estimate_serialized_size(
        self,
        layout_desc: MemoryLayoutDesc,
    ) -> int:

        return (
            _plaintext_bytes(layout_desc)
            + _FRAME_OVERHEAD
        )


class NativeAesGcmDeserializer(Deserializer):

    def __init__(
        self,
        key_provider: KeyProvider,
    ):
        self._keys = key_provider

    def deserialize(
        self,
        src,
        dst,
        key: ObjectKey,
    ) -> None:

        src_view = _byte_view(src)
        dst_view = _byte_view(dst)

        plaintext_len = len(dst_view)

        frame_len = (
            plaintext_len
            + _FRAME_OVERHEAD
        )

        if len(src_view) < frame_len:
            raise ValueError(
                "native aesgcm: malformed frame"
            )

        dek = self._keys.get_key(
            key.cache_salt
        )

        if len(dek) != 16:
            raise ValueError(
                "native aesgcm currently supports AES-128 only"
            )

        key_holder, key_ptr = _key_ptr(dek)

        rc = _lib.lmcache_aes128gcm_open(
            key_ptr,
            _address(src_view),
            frame_len,
            _address(dst_view),
            plaintext_len,
        )

        _ = key_holder

        if rc == -11:
            raise InvalidTag(
                "native aesgcm authentication failed"
            )

        if rc != 0:
            raise ValueError(
                f"native aesgcm open failed: rc={rc}"
            )


def create_native_aesgcm_serde(
    kwargs: dict[str, object],
) -> SerdeProcessor:

    aes_bits = int(
        kwargs.get(
            "aes_bits",
            128,
        )
    )

    if aes_bits != 128:
        raise ValueError(
            "experimental native backend currently "
            "supports aes_bits=128 only"
        )

    master_key_path = str(
        kwargs.get(
            "master_key_path",
            "",
        )
    )

    if not master_key_path:
        raise ValueError(
            "native_aesgcm requires master_key_path"
        )

    with open(
        master_key_path,
        "rb",
    ) as f:
        master_key = f.read()

    provider = HkdfKeyProvider(
        master_key,
        key_len=16,
        info_prefix=_HKDF_INFO_PREFIX,
    )

    max_workers = int(
        kwargs.get(
            "max_workers",
            1,
        )
    )

    return AsyncSerdeProcessor(
        NativeAesGcmSerializer(
            provider
        ),
        NativeAesGcmDeserializer(
            provider
        ),
        max_workers=max_workers,
    )


register_serde_factory(
    "native_aesgcm_exp",
    create_native_aesgcm_serde,
)
