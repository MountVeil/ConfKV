#!/usr/bin/env python3

import ctypes
import os

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm import (
    AesGcmDeserializer,
    AesGcmSerializer,
)
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider


ROOT = "experiments/gpu_kv_seal"
LIB = f"{ROOT}/native/libcpu_seal.so"

FRAME_OVERHEAD = 29
SIZE = 4 * 1024 * 1024

MASTER = b"paper-benchmark-master-key-32bytes!!"


class ByteBuf:
    def __init__(self, size):
        self._arr = (ctypes.c_ubyte * size)()

    @property
    def byte_array(self):
        return memoryview(self._arr)

    @property
    def address(self):
        return ctypes.addressof(self._arr)

    def bytes(self):
        return bytes(self._arr)


lib = ctypes.CDLL(LIB)

lib.lmcache_aes128gcm_seal.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
lib.lmcache_aes128gcm_seal.restype = ctypes.c_int

lib.lmcache_aes128gcm_open.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
lib.lmcache_aes128gcm_open.restype = ctypes.c_int


provider = HkdfKeyProvider(
    MASTER,
    key_len=16,
    info_prefix=b"lmcache-l2-aesgcm-v1",
)

key = ObjectKey(
    chunk_hash=b"\x11" * 32,
    model_name="paper-benchmark",
    kv_rank=0,
    cache_salt="tenant-a",
)

wrong_key_desc = ObjectKey(
    chunk_hash=b"\x11" * 32,
    model_name="paper-benchmark",
    kv_rank=0,
    cache_salt="tenant-b",
)

serializer = AesGcmSerializer(provider)
deserializer = AesGcmDeserializer(provider)


def key_buffer(desc):
    raw = provider.get_key(desc.cache_salt)

    arr = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)

    return arr


def all_zero(buf):
    return not any(buf.bytes())


src = ByteBuf(SIZE)
ctypes.memset(src.address, 0xA5, SIZE)


# ------------------------------------------------------------------
# 1. Native seal -> stock LMCache open
# ------------------------------------------------------------------

frame = ByteBuf(SIZE + FRAME_OVERHEAD)

kb = key_buffer(key)

rc = lib.lmcache_aes128gcm_seal(
    ctypes.addressof(kb),
    src.address,
    SIZE,
    frame.address,
    SIZE + FRAME_OVERHEAD,
)

assert rc == 0

dst = ByteBuf(SIZE)

deserializer.deserialize(
    frame,
    dst,
    key,
)

assert dst.bytes() == src.bytes()

print("PASS native seal -> stock LMCache open")


# ------------------------------------------------------------------
# 2. Stock LMCache seal -> native open
# ------------------------------------------------------------------

stock_frame = ByteBuf(SIZE + FRAME_OVERHEAD)

n = serializer.serialize(
    src,
    stock_frame,
    key,
)

assert n == SIZE + FRAME_OVERHEAD

dst2 = ByteBuf(SIZE)

rc = lib.lmcache_aes128gcm_open(
    ctypes.addressof(kb),
    stock_frame.address,
    n,
    dst2.address,
    SIZE,
)

assert rc == 0
assert dst2.bytes() == src.bytes()

print("PASS stock LMCache seal -> native open")


# ------------------------------------------------------------------
# Helpers for adversarial tests
# ------------------------------------------------------------------

def native_open_modified(frame_bytes, key_desc):
    fb = ByteBuf(len(frame_bytes))
    frame_copy = (
        ctypes.c_ubyte * len(frame_bytes)
    ).from_buffer_copy(frame_bytes)

    ctypes.memmove(
        fb.address,
        ctypes.addressof(frame_copy),
        len(frame_bytes),
    )

    out = ByteBuf(SIZE)

    # Make zeroization observable.
    ctypes.memset(
        out.address,
        0xCC,
        SIZE,
    )

    kb2 = key_buffer(key_desc)

    rc = lib.lmcache_aes128gcm_open(
        ctypes.addressof(kb2),
        fb.address,
        len(frame_bytes),
        out.address,
        SIZE,
    )

    return rc, out


original_frame = bytearray(frame.bytes())


# ------------------------------------------------------------------
# 3. Wrong tenant/key
# ------------------------------------------------------------------

rc, out = native_open_modified(
    original_frame,
    wrong_key_desc,
)

assert rc != 0
assert all_zero(out)

print("PASS wrong key rejected + provisional plaintext wiped")


# ------------------------------------------------------------------
# 4. Ciphertext tamper
# ------------------------------------------------------------------

tampered = bytearray(original_frame)

tampered[13 + SIZE // 2] ^= 0x01

rc, out = native_open_modified(
    tampered,
    key,
)

assert rc != 0
assert all_zero(out)

print("PASS ciphertext tamper rejected + plaintext wiped")


# ------------------------------------------------------------------
# 5. Tag tamper
# ------------------------------------------------------------------

tampered = bytearray(original_frame)

tampered[-1] ^= 0x01

rc, out = native_open_modified(
    tampered,
    key,
)

assert rc != 0
assert all_zero(out)

print("PASS tag tamper rejected + plaintext wiped")


# ------------------------------------------------------------------
# 6. IV tamper
# ------------------------------------------------------------------

tampered = bytearray(original_frame)

tampered[1] ^= 0x01

rc, out = native_open_modified(
    tampered,
    key,
)

assert rc != 0
assert all_zero(out)

print("PASS IV tamper rejected + plaintext wiped")


print()
print("====================================")
print("NATIVE AES-GCM SECURITY GATE PASSED")
print("====================================")
