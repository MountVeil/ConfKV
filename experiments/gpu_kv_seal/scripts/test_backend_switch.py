#!/usr/bin/env python3

import ctypes
import os
import tempfile

import torch

from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.serde.base import SerdeConfig
from lmcache.v1.distributed.serde.factory import create_serde_processor


SIZE = 4 * 1024 * 1024


class Buf:
    def __init__(self, capacity):
        self._arr = (ctypes.c_ubyte * capacity)()
        self._capacity = capacity
        self._used = capacity

    @property
    def byte_array(self):
        return memoryview(self._arr)[:self._used]

    @property
    def address(self):
        return ctypes.addressof(self._arr)

    def set_used_size(self, size):
        self._used = size

    def bytes(self):
        return bytes(memoryview(self._arr)[:self._used])


def make_processor(backend, key_path):
    os.environ["LMCACHE_AESGCM_BACKEND"] = backend

    return create_serde_processor(
        SerdeConfig(
            type="aesgcm",
            kwargs={
                "master_key_path": key_path,
                "aes_bits": 128,
                "max_workers": 1,
            },
        )
    )


def wait_result(query, task):
    import time

    deadline = time.monotonic() + 5

    while time.monotonic() < deadline:
        result = query(task)

        if result is not None:
            return result

        time.sleep(0.001)

    raise TimeoutError


def serialize(processor, src, frame, key):
    task = processor.submit_serialize(
        [src],
        [frame],
        [key],
    )

    assert wait_result(
        processor.query_serialize_result,
        task,
    ) is True


def deserialize(processor, frame, dst, key):
    task = processor.submit_deserialize(
        [frame],
        [dst],
        [key],
    )

    assert wait_result(
        processor.query_deserialize_result,
        task,
    ) is True


with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b"paper-benchmark-master-key-32bytes!!")
    key_path = f.name


try:
    key = ObjectKey(
        chunk_hash=b"\x11" * 32,
        model_name="paper-benchmark",
        kv_rank=0,
        cache_salt="tenant-a",
    )

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([SIZE // 2])],
        dtypes=[torch.bfloat16],
    )

    src = Buf(SIZE)
    ctypes.memset(src.address, 0xA5, SIZE)

    # --------------------------------------------------
    # stock seal -> native open
    # --------------------------------------------------

    stock = make_processor("stock", key_path)

    frame1 = Buf(
        stock.estimate_serialized_size(layout)
    )

    serialize(
        stock,
        src,
        frame1,
        key,
    )

    stock.close()

    native = make_processor("native", key_path)

    out1 = Buf(SIZE)

    deserialize(
        native,
        frame1,
        out1,
        key,
    )

    assert out1.bytes() == src.bytes()

    print("PASS stock seal -> native open")

    native.close()

    # --------------------------------------------------
    # native seal -> stock open
    # --------------------------------------------------

    native = make_processor("native", key_path)

    frame2 = Buf(
        native.estimate_serialized_size(layout)
    )

    serialize(
        native,
        src,
        frame2,
        key,
    )

    native.close()

    stock = make_processor("stock", key_path)

    out2 = Buf(SIZE)

    deserialize(
        stock,
        frame2,
        out2,
        key,
    )

    assert out2.bytes() == src.bytes()

    print("PASS native seal -> stock open")

    stock.close()

    print()
    print("==============================")
    print("BACKEND SWITCH GATE PASSED")
    print("==============================")

finally:
    os.unlink(key_path)
