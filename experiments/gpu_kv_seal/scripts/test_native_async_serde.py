#!/usr/bin/env python3

import ctypes
import os
import tempfile
import time

# Import performs experimental factory registration.
import sys

sys.path.insert(
    0,
    "experiments/gpu_kv_seal/integration",
)

import native_aesgcm_serde  # noqa: F401

from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.serde.base import (
    SerdeConfig,
)
from lmcache.v1.distributed.serde.factory import (
    create_serde_processor,
    get_registered_serde_types,
)

import torch


SIZE = 4 * 1024 * 1024


class Buf:

    def __init__(self, capacity):
        self._arr = (
            ctypes.c_ubyte
            * capacity
        )()

        self._capacity = capacity
        self._used = capacity

    @property
    def byte_array(self):
        return memoryview(
            self._arr
        )[:self._used]

    @property
    def address(self):
        return ctypes.addressof(
            self._arr
        )

    def set_used_size(self, size):
        assert (
            0 <= size <= self._capacity
        )

        self._used = size

    def reset_used_size(self):
        self._used = self._capacity

    def bytes(self):
        return bytes(
            memoryview(
                self._arr
            )[:self._used]
        )


def wait_result(query, task_id):

    deadline = (
        time.monotonic()
        + 5.0
    )

    while (
        time.monotonic()
        < deadline
    ):
        result = query(
            task_id
        )

        if result is not None:
            return result

        time.sleep(0.001)

    raise TimeoutError(
        "serde task timeout"
    )


with tempfile.NamedTemporaryFile(
    delete=False
) as f:

    f.write(
        b"paper-benchmark-master-key-32bytes!!"
    )

    key_path = f.name


try:

    assert (
        "native_aesgcm_exp"
        in get_registered_serde_types()
    )

    config = SerdeConfig(
        type="native_aesgcm_exp",
        kwargs={
            "master_key_path":
                key_path,

            "aes_bits":
                128,

            "max_workers":
                1,
        },
    )

    processor = (
        create_serde_processor(
            config
        )
    )

    key = ObjectKey(
        chunk_hash=(
            b"\x11"
            * 32
        ),
        model_name="paper-benchmark",
        kv_rank=0,
        cache_salt="tenant-a",
    )

    layout = MemoryLayoutDesc(
        shapes=[
            torch.Size(
                [SIZE // 2]
            )
        ],
        dtypes=[
            torch.bfloat16
        ],
    )

    estimated = (
        processor
        .estimate_serialized_size(
            layout
        )
    )

    assert (
        estimated
        == SIZE + 29
    )

    src = Buf(SIZE)

    ctypes.memset(
        src.address,
        0xA5,
        SIZE,
    )

    frame = Buf(
        estimated
    )

    # -------------------------------
    # Async serialize
    # -------------------------------

    task = (
        processor.submit_serialize(
            [src],
            [frame],
            [key],
        )
    )

    ok = wait_result(
        processor
        .query_serialize_result,
        task,
    )

    assert ok is True

    assert (
        len(frame.byte_array)
        == SIZE + 29
    )

    print(
        "PASS async native serialize"
    )

    # -------------------------------
    # Async deserialize
    # -------------------------------

    out = Buf(SIZE)

    task = (
        processor.submit_deserialize(
            [frame],
            [out],
            [key],
        )
    )

    ok = wait_result(
        processor
        .query_deserialize_result,
        task,
    )

    assert ok is True

    assert (
        out.bytes()
        == src.bytes()
    )

    print(
        "PASS async native deserialize"
    )

    # -------------------------------
    # Authentication failure
    # -------------------------------

    frame_view = (
        memoryview(
            frame._arr
        ).cast("B")
    )

    frame_view[
        13 + SIZE // 2
    ] ^= 0x01

    ctypes.memset(
        out.address,
        0xCC,
        SIZE,
    )

    task = (
        processor.submit_deserialize(
            [frame],
            [out],
            [key],
        )
    )

    ok = wait_result(
        processor
        .query_deserialize_result,
        task,
    )

    assert ok is False

    assert all(
        b == 0
        for b in out.bytes()
    )

    print(
        "PASS async tamper rejection "
        "+ zeroization"
    )

    processor.close()

    print()
    print(
        "================================"
    )
    print(
        "NATIVE ASYNC SERDE GATE PASSED"
    )
    print(
        "================================"
    )

finally:

    os.unlink(
        key_path
    )
