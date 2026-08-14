#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import torch

from lmcache import torch_dev
import lmcache.c_ops as lmc_ops

from lmcache.utils import EngineType

from lmcache.v1.confkv import gpu_crypto as confkv_gpu_crypto

from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)

from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)

from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
)

from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2AdapterConfig,
)

from lmcache.v1.distributed.storage_manager import (
    StorageManager,
)

from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    downsample_and_stage_block_ids,
    get_layout_desc,
    transfer_kv_per_object_group,
)

from lmcache.v1.platform.cache_context import (
    create_cache_context,
)


FRAME_OVERHEAD = 29


class LocalTensorWrapper:
    """Minimal same-process wrapper for GPUCacheContext construction.

    This smoke intentionally excludes CUDA IPC serialization.  The real
    GPUCacheContext still unwraps the object through to_tensor(), exactly
    as it does for the multiprocess DeviceIPCWrapper.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
    ) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor


def layout_nbytes(
    layout: MemoryLayoutDesc,
) -> int:
    total = 0

    for shape, dtype in zip(
        layout.shapes,
        layout.dtypes,
        strict=True,
    ):
        element_size = torch.empty(
            (),
            dtype=dtype,
        ).element_size()

        total += (
            shape.numel()
            * element_size
        )

    return total


def make_sealed_layout(
    plaintext_nbytes: int,
) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[
            torch.Size(
                [
                    plaintext_nbytes
                    + FRAME_OVERHEAD
                ]
            )
        ],
        dtypes=[
            torch.uint8
        ],
    )


def make_qwen3_like_kv(
    *,
    chunks: int,
    chunk_size: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size must be divisible by block_size"
        )

    blocks_per_chunk = (
        chunk_size // block_size
    )

    num_blocks = (
        chunks * blocks_per_chunk
    )

    result: list[torch.Tensor] = []

    for layer_idx in range(
        num_layers
    ):
        tensor = torch.empty(
            (
                2,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

        # Each paged block receives a distinct deterministic value.
        # This makes block-mapping/scatter corruption observable.
        block_values = (
            torch.arange(
                num_blocks,
                device=device,
                dtype=torch.float32,
            )
            + layer_idx * 257
        ).to(
            torch.bfloat16
        ).view(
            1,
            num_blocks,
            1,
            1,
            1,
        )

        tensor.copy_(
            block_values.expand_as(
                tensor
            )
        )

        result.append(
            tensor
        )

    return result


def wait_for_l2_files(
    *,
    base_path: Path,
    expected_count: int,
    expected_size: int,
    timeout_s: float,
) -> list[Path]:
    deadline = (
        time.monotonic()
        + timeout_s
    )

    while True:
        files = sorted(
            base_path.glob(
                "*.data"
            )
        )

        if (
            len(files)
            == expected_count
            and all(
                p.stat().st_size
                == expected_size
                for p in files
            )
        ):
            return files

        if (
            time.monotonic()
            >= deadline
        ):
            state = [
                (
                    p.name,
                    p.stat().st_size,
                )
                for p in files
            ]

            raise TimeoutError(
                "timed out waiting for "
                f"{expected_count} L2 files "
                f"of size {expected_size}; "
                f"current={state}"
            )

        time.sleep(
            0.05
        )


def delete_all_from_l1(
    storage_manager: StorageManager,
    keys: list[ObjectKey],
    timeout_s: float,
) -> None:
    """Ensure that every requested key is absent from L1.

    LMCache's default prefetch policy makes L2-loaded objects temporary.
    finish_read_prefetched() therefore removes them automatically when
    their final read lock is released.

    delete_l1_keys() reports only the number successfully deleted and
    the number skipped because they are locked.  Missing keys are an
    idempotent no-op, so deleted=0/skipped=0 can legitimately mean that
    every requested key is already absent.

    Therefore, the only state that requires retrying is skipped > 0.
    """
    deadline = (
        time.monotonic()
        + timeout_s
    )

    while True:
        deleted, skipped = (
            storage_manager.delete_l1_keys(
                keys,
                force=False,
            )
        )

        if skipped == 0:
            print(
                "L1 absence    :",
                f"deleted_now={deleted}, "
                "locked_remaining=0",
            )
            return

        if (
            time.monotonic()
            >= deadline
        ):
            raise TimeoutError(
                "could not make all keys "
                "absent from L1; "
                f"deleted_now={deleted}, "
                f"locked_remaining={skipped}"
            )

        time.sleep(
            0.05
        )


def prefetch_from_l2(
    *,
    storage_manager: StorageManager,
    keys: list[ObjectKey],
    sealed_layout: MemoryLayoutDesc,
    timeout_s: float,
) -> None:
    spec = PrefetchRequestSpec(
        keys=keys,
        group_layout_descs={
            0: sealed_layout,
        },
        extra_count=0,
        policy=TrimPolicy.PREFIX,
        attn_desc=(
            DEFAULT_ATTN_WINDOW_DESC
        ),
        mode=PrefetchMode.LOOKUP,
    )

    handle = (
        storage_manager.submit_prefetch_task(
            spec,
            external_request_id=(
                "confkv-b3-"
                "actual-storage-smoke"
            ),
        )
    )

    print(
        "prefetch L1 hits:",
        len(
            handle.l1_found_indices
        ),
    )

    if handle.l1_found_indices:
        raise RuntimeError(
            "prefetch unexpectedly hit L1; "
            "this would not validate L2 reload"
        )

    ready = (
        storage_manager.wait_prefetch_status(
            handle,
            timeout=timeout_s,
        )
    )

    if not ready:
        raise TimeoutError(
            "L2 prefetch timed out"
        )

    found = (
        storage_manager.query_prefetch_status(
            handle
        )
    )

    if found is None:
        raise RuntimeError(
            "prefetch completed but "
            "returned no result bitmap"
        )

    hits = found.popcount()

    print(
        "prefetch L2 hits:",
        hits,
        "/",
        len(keys),
    )

    if hits != len(keys):
        raise RuntimeError(
            "not all objects were "
            "loaded from L2"
        )


def zero_paged_kv(
    kv_caches: list[torch.Tensor],
) -> None:
    for tensor in kv_caches:
        tensor.zero_()

    torch.cuda.synchronize()


def assert_paged_kv_equal(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
) -> None:
    for layer_idx, (
        lhs,
        rhs,
    ) in enumerate(
        zip(
            actual,
            expected,
            strict=True,
        )
    ):
        if not torch.equal(
            lhs,
            rhs,
        ):
            raise RuntimeError(
                "paged KV mismatch "
                f"at layer {layer_idx}"
            )


def assert_paged_kv_zero(
    kv_caches: list[torch.Tensor],
) -> None:
    for layer_idx, tensor in enumerate(
        kv_caches
    ):
        if torch.count_nonzero(
            tensor
        ).item() != 0:
            raise RuntimeError(
                "authentication failure "
                "published plaintext/scatter "
                f"to layer {layer_idx}"
            )


def tamper_ciphertext(
    path: Path,
) -> None:
    # Frame:
    # [1-byte version]
    # [12-byte IV]
    # [ciphertext]
    # [16-byte tag]
    #
    # Offset 13 is the first ciphertext byte.
    with path.open(
        "r+b",
        buffering=0,
    ) as f:
        f.seek(
            13
        )

        original = f.read(
            1
        )

        if len(original) != 1:
            raise RuntimeError(
                "cannot read ciphertext byte"
            )

        f.seek(
            13
        )

        f.write(
            bytes(
                [
                    original[0]
                    ^ 0x01
                ]
            )
        )

        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--chunks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    if not hasattr(
        lmc_ops,
        "multi_layer_block_kv_transfer",
    ):
        raise RuntimeError(
            "LMCache native CUDA transfer "
            "kernel is unavailable"
        )

    device = torch.device(
        f"cuda:{args.device}"
    )

    torch.cuda.set_device(
        device
    )

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    if data_dir.exists():
        shutil.rmtree(
            data_dir
        )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========== ACTUAL LMCACHE B3 "
        "STORAGE-PATH SMOKE =========="
    )

    print(
        "GPU          :",
        torch.cuda.get_device_name(
            device
        ),
    )

    print(
        "data dir     :",
        data_dir,
    )

    print(
        "chunks       :",
        args.chunks,
    )

    print(
        "chunk size   :",
        args.chunk_size,
    )

    print(
        "block size   :",
        args.block_size,
    )

    print(
        "GPU slots    :",
        args.gpu_slots,
    )

    print(
        "TDX          : NO"
    )

    print(
        "H100 CC      : NO"
    )

    # Qwen3-8B-like KV geometry:
    #
    # 36 layers
    # 8 KV heads
    # head_dim = 128
    # BF16
    #
    # 256 tokens => 36 MiB persistent plaintext.
    kv_caches = make_qwen3_like_kv(
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        device=device,
    )

    expected = [
        tensor.clone()
        for tensor in kv_caches
    ]

    wrappers = [
        LocalTensorWrapper(
            tensor
        )
        for tensor in kv_caches
    ]

    cache_context = None
    storage_manager = None
    gpu_crypto = None

    try:
        cache_context = (
            create_cache_context(
                wrappers,
                lmcache_tokens_per_chunk=(
                    args.chunk_size
                ),
                layout_hints={
                    "kv_layout": "NHD",
                },
                engine_group_infos=[],
                engine_type=(
                    EngineType.VLLM
                ),
                separate_object_groups=False,
                full_sw_kv=False,
            )
        )

        manager = (
            cache_context
            .kv_layer_groups_manager
        )

        print(
            "kernel groups:",
            manager.num_kernel_groups,
        )

        print(
            "object groups:",
            manager.num_object_groups,
        )

        if (
            manager.num_kernel_groups
            != 1
            or manager.num_object_groups
            != 1
        ):
            raise RuntimeError(
                "expected one non-hybrid "
                "Qwen-like KV group"
            )

        plain_layout = (
            get_layout_desc(
                cache_context,
                args.chunk_size,
                object_group_id=0,
            )
        )

        plaintext_nbytes = (
            layout_nbytes(
                plain_layout
            )
        )

        sealed_layout = (
            make_sealed_layout(
                plaintext_nbytes
            )
        )

        sealed_nbytes = (
            plaintext_nbytes
            + FRAME_OVERHEAD
        )

        print(
            "plaintext/chunk:",
            plaintext_nbytes,
            "bytes",
            f"({plaintext_nbytes / (1 << 20):.3f} MiB)",
        )

        print(
            "sealed/chunk   :",
            sealed_nbytes,
            "bytes",
        )

        expected_plaintext = (
            36
            * 2
            * args.chunk_size
            * 8
            * 128
            * 2
        )

        if (
            plaintext_nbytes
            != expected_plaintext
        ):
            raise RuntimeError(
                "unexpected Qwen3-like "
                "chunk geometry: "
                f"{plaintext_nbytes} != "
                f"{expected_plaintext}"
            )

        fs_config = (
            FSL2AdapterConfig(
                base_path=str(
                    data_dir
                ),
                use_odirect=False,
            )
        )

        storage_config = (
            StorageManagerConfig(
                l1_manager_config=(
                    L1ManagerConfig(
                        memory_config=(
                            L1MemoryManagerConfig(
                                size_in_bytes=(
                                    512 << 20
                                ),
                                use_lazy=True,
                                init_size_in_bytes=(
                                    128 << 20
                                ),
                                shm_name="",
                            )
                        )
                    )
                ),
                eviction_config=(
                    EvictionConfig(
                        eviction_policy=(
                            "noop"
                        )
                    )
                ),
                l2_adapter_config=(
                    L2AdaptersConfig(
                        adapters=[
                            fs_config
                        ]
                    )
                ),
                store_policy="default",
                prefetch_policy="default",
                prefetch_max_in_flight=8,
            )
        )

        storage_manager = (
            StorageManager(
                storage_config
            )
        )

        gpu_crypto = (
            confkv_gpu_crypto
            .ConfKVGpuCrypto(
                device,
                max_plaintext_len=(
                    plaintext_nbytes
                ),
                num_slots=(
                    args.gpu_slots
                ),
            )
        )

        kv_rank = (
            ObjectKey.ComputeKVRank(
                world_size=1,
                global_rank=0,
                local_world_size=1,
                local_rank=0,
            )
        )

        keys = [
            ObjectKey(
                chunk_hash=(
                    (i + 1).to_bytes(
                        32,
                        byteorder="big",
                    )
                ),
                model_name=(
                    "confkv-actual-b3-smoke"
                ),
                kv_rank=kv_rank,
                object_group_id=0,
            )
            for i in range(
                args.chunks
            )
        ]

        blocks_per_chunk = (
            args.chunk_size
            // args.block_size
        )

        num_blocks = (
            args.chunks
            * blocks_per_chunk
        )

        raw_block_ids = [
            list(
                range(
                    num_blocks
                )
            )
        ]

        # ====================================================
        # STORE:
        # paged GPU KV
        # -> real LMCache gather
        # -> GPU AES-GCM
        # -> L1 sealed MemoryObj
        # -> real StoreController
        # -> real FSL2Adapter
        # ====================================================

        print()
        print(
            "========== STORE =========="
        )

        reserved = (
            storage_manager.reserve_write(
                keys,
                sealed_layout,
                "new",
            )
        )

        if (
            len(reserved)
            != len(keys)
        ):
            raise RuntimeError(
                "failed to reserve every "
                "B3 sealed object"
            )

        memory_objs = [
            reserved[key]
            for key in keys
        ]

        block_ids_gpu = (
            downsample_and_stage_block_ids(
                cache_context,
                [
                    list(
                        raw_block_ids[0]
                    )
                ],
            )
        )

        def seal_batch(
            plaintext_gpus,
            dsts,
        ):
            gpu_crypto.seal_batch_and_copy_d2h(
                plaintext_gpus,
                dsts,
                stream=(
                    cache_context.stream
                ),
            )

        with (
            torch_dev.device(
                cache_context.device
            ),
            torch_dev.stream(
                cache_context.stream
            ),
        ):
            transfer_kv_per_object_group(
                cache_context,
                block_ids_gpu,
                memory_objs,
                object_group_id=0,
                batch_size=(
                    cache_context
                    .max_batch_size
                ),
                skip_first_n_tokens=0,
                direction=(
                    lmc_ops
                    .TransferDirection
                    .D2H
                ),
                d2h_batch_transform=(
                    seal_batch
                ),
            )

        cache_context.stream.synchronize()

        storage_manager.finish_write(
            keys
        )

        files = wait_for_l2_files(
            base_path=data_dir,
            expected_count=(
                len(keys)
            ),
            expected_size=(
                sealed_nbytes
            ),
            timeout_s=args.timeout,
        )

        print(
            "L2 files     :",
            len(files),
        )

        for path in files:
            with path.open(
                "rb"
            ) as f:
                header = f.read(
                    13
                )

            if (
                len(header) != 13
                or header[0] != 1
            ):
                raise RuntimeError(
                    "invalid B3 AES-GCM "
                    f"frame on disk: {path}"
                )

        print(
            "PASS: real FSL2 contains "
            "P+29 B3 frames"
        )

        # StoreController may briefly retain an L1 read lock
        # while the async L2 store finishes. Wait until it is
        # safe to remove every object from L1.
        delete_all_from_l1(
            storage_manager,
            keys,
            timeout_s=args.timeout,
        )

        print(
            "PASS: all stored keys "
            "removed from L1"
        )

        # ====================================================
        # LOAD:
        # FSL2
        # -> real PrefetchController
        # -> sealed L1 MemoryObj
        # -> H2D
        # -> GPU auth/open
        # -> real LMCache scatter
        # -> paged GPU KV
        # ====================================================

        print()
        print(
            "========== LOAD =========="
        )

        prefetch_from_l2(
            storage_manager=(
                storage_manager
            ),
            keys=keys,
            sealed_layout=(
                sealed_layout
            ),
            timeout_s=args.timeout,
        )

        zero_paged_kv(
            kv_caches
        )

        block_ids_gpu = (
            downsample_and_stage_block_ids(
                cache_context,
                [
                    list(
                        raw_block_ids[0]
                    )
                ],
            )
        )

        def open_batch(
            srcs,
            plaintext_gpus,
        ):
            gpu_crypto.copy_h2d_and_open_batch(
                srcs,
                plaintext_gpus,
                stream=(
                    cache_context.stream
                ),
            )

        with (
            storage_manager
            .read_prefetched_results(
                keys
            )
        ) as loaded_objs:
            if (
                loaded_objs is None
                or len(loaded_objs)
                != len(keys)
            ):
                raise RuntimeError(
                    "prefetched B3 objects "
                    "not resident in L1"
                )

            with (
                torch_dev.device(
                    cache_context.device
                ),
                torch_dev.stream(
                    cache_context.stream
                ),
            ):
                transfer_kv_per_object_group(
                    cache_context,
                    block_ids_gpu,
                    loaded_objs,
                    object_group_id=0,
                    batch_size=(
                        cache_context
                        .max_batch_size
                    ),
                    skip_first_n_tokens=0,
                    direction=(
                        lmc_ops
                        .TransferDirection
                        .H2D
                    ),
                    h2d_batch_transform=(
                        open_batch
                    ),
                )

            cache_context.stream.synchronize()

        storage_manager.finish_read_prefetched(
            keys
        )

        assert_paged_kv_equal(
            kv_caches,
            expected,
        )

        print(
            "PASS: FSL2 -> L1 -> "
            "GPU open -> paged-KV "
            "roundtrip"
        )

        # ====================================================
        # TAMPER:
        # corrupt one actual FSL2 ciphertext byte,
        # force another L2 reload,
        # verify auth failure prevents scatter.
        # ====================================================

        print()
        print(
            "========== TAMPER =========="
        )

        delete_all_from_l1(
            storage_manager,
            keys,
            timeout_s=args.timeout,
        )

        tamper_ciphertext(
            files[0]
        )

        print(
            "tampered     :",
            files[0].name,
        )

        prefetch_from_l2(
            storage_manager=(
                storage_manager
            ),
            keys=keys,
            sealed_layout=(
                sealed_layout
            ),
            timeout_s=args.timeout,
        )

        zero_paged_kv(
            kv_caches
        )

        block_ids_gpu = (
            downsample_and_stage_block_ids(
                cache_context,
                [
                    list(
                        raw_block_ids[0]
                    )
                ],
            )
        )

        auth_failed = False

        try:
            with (
                storage_manager
                .read_prefetched_results(
                    keys
                )
            ) as loaded_objs:
                if (
                    loaded_objs is None
                ):
                    raise RuntimeError(
                        "tampered objects "
                        "not loaded into L1"
                    )

                with (
                    torch_dev.device(
                        cache_context.device
                    ),
                    torch_dev.stream(
                        cache_context.stream
                    ),
                ):
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_gpu,
                        loaded_objs,
                        object_group_id=0,
                        batch_size=(
                            cache_context
                            .max_batch_size
                        ),
                        skip_first_n_tokens=0,
                        direction=(
                            lmc_ops
                            .TransferDirection
                            .H2D
                        ),
                        h2d_batch_transform=(
                            open_batch
                        ),
                    )

                cache_context.stream.synchronize()

        except Exception as exc:
            auth_failed = True

            print(
                "auth failure  :",
                type(exc).__name__,
                str(exc),
            )

        if not auth_failed:
            raise RuntimeError(
                "tampered FSL2 ciphertext "
                "was accepted"
            )

        assert_paged_kv_zero(
            kv_caches
        )

        print(
            "PASS: tampered FSL2 frame "
            "rejected"
        )

        print(
            "PASS: auth failure caused "
            "no paged-KV scatter"
        )

        print()
        print(
            "===================================="
        )

        print(
            "ACTUAL_LMCACHE_B3_STORAGE_PATH: PASS"
        )

        print(
            "real GPUCacheContext       : PASS"
        )

        print(
            "native LMCache KV transfer : PASS"
        )

        print(
            "GPU B3 seal/open           : PASS"
        )

        print(
            "real StorageManager        : PASS"
        )

        print(
            "real StoreController       : PASS"
        )

        print(
            "real PrefetchController    : PASS"
        )

        print(
            "real FSL2Adapter/NVMe      : PASS"
        )

        print(
            "forced L2 reload           : PASS"
        )

        print(
            "roundtrip                  : PASS"
        )

        print(
            "tamper rejection           : PASS"
        )

        print(
            "fail-closed scatter        : PASS"
        )

        print(
            "TDX                        : NOT VALIDATED"
        )

        print(
            "H100 CC                    : NOT VALIDATED"
        )

        print(
            "MP RPC / vLLM              : NOT YET VALIDATED"
        )

        print(
            "===================================="
        )

    finally:
        if gpu_crypto is not None:
            gpu_crypto.close()

        if cache_context is not None:
            cache_context.close()

        if storage_manager is not None:
            storage_manager.close()


if __name__ == "__main__":
    main()
