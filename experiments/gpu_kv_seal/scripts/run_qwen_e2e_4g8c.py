#!/usr/bin/env python3
"""4-GPU/8-CPU Qwen + LMCache/ConfKV non-CC E2E suite."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import time
from typing import Any
import urllib.request

CASES = (
    "baseline",
    "opt_cpu",
    "confkv_naive",
    "confkv_optimized",
)
CONFKV_CASES = frozenset({
    "confkv_naive",
    "confkv_optimized",
})


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen
    log: Any
    log_path: Path


@dataclass
class Result:
    case: str
    phase: str
    profile: str
    item: int
    concurrency: int
    ok: bool
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    error: str = ""


def indices(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad range {part!r}")
            out.update(range(lo, hi + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def pct(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else math.nan,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
    }


def start(
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> Proc:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", buffering=1)
    handle.write(f"COMMAND: {shlex.join(cmd)}\n\n")
    popen = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return Proc(name, popen, handle, log)


def stop(proc: Proc | None) -> None:
    if proc is None:
        return

    try:
        if proc.popen.poll() is None:
            os.killpg(proc.popen.pid, signal.SIGINT)
            try:
                proc.popen.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.popen.pid, signal.SIGTERM)
                try:
                    proc.popen.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.popen.pid, signal.SIGKILL)
                    proc.popen.wait(10)
    finally:
        proc.log.close()


def wait_tcp(proc: Proc, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.popen.poll() is not None:
            raise RuntimeError(
                f"{proc.name} exited; inspect {proc.log_path}"
            )
        try:
            with socket.create_connection(
                ("127.0.0.1", port),
                timeout=1,
            ):
                return
        except OSError:
            time.sleep(0.5)

    raise TimeoutError(
        f"timeout waiting for {proc.name} port {port}"
    )


def wait_http(proc: Proc, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.popen.poll() is not None:
            raise RuntimeError(
                f"{proc.name} exited; inspect {proc.log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(1)

    raise TimeoutError(f"timeout waiting for {url}")


def wait_l2(
    path: Path,
    timeout: float,
    quiet: float,
) -> tuple[int, int, float]:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last = None
    unchanged = time.monotonic()

    while time.monotonic() < deadline:
        files = [
            p for p in path.rglob("*.data")
            if p.is_file()
        ]
        current = (
            len(files),
            sum(p.stat().st_size for p in files),
        )

        if current != last:
            last = current
            unchanged = time.monotonic()
        elif current[0] and time.monotonic() - unchanged >= quiet:
            return (
                current[0],
                current[1],
                time.monotonic() - started,
            )

        time.sleep(0.5)

    raise TimeoutError(
        f"L2 did not quiesce; last state={last}"
    )


def check_secret(
    path: Path,
    exact: int | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_mode & 0o077:
        raise PermissionError(
            f"secret must be mode 0600/0400: {path}"
        )

    if exact is not None and path.stat().st_size != exact:
        raise ValueError(
            f"{path} must contain exactly {exact} raw bytes"
        )


def env_for(
    args: argparse.Namespace,
    repo: Path,
    case: str,
) -> dict[str, str]:
    env = dict(os.environ)

    # Prevent parent-shell settings from contaminating another case.
    for key in (
        "CONFKV_GPU_SEAL_ENABLE",
        "CONFKV_KEY_PROVIDER",
        "CONFKV_DEV_KEY_PATH",
        "CONFKV_STORE_KEY_PATH",
        "CONFKV_GPU_MODE",
        "CONFKV_GPU_CRYPTO_SLOTS",
        "LMCACHE_AESGCM_BACKEND",
        "LMCACHE_AESGCM_NATIVE_LIB",
    ):
        env.pop(key, None)

    env.update(
        CONFKV_ROOT=str(repo),
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        LMCACHE_DISABLE_BANNER="1",
    )

    env["PYTHONPATH"] = str(repo / "LMCache") + (
        os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else ""
    )

    if case == "baseline":
        env["CONFKV_GPU_SEAL_ENABLE"] = "0"
        env["LMCACHE_AESGCM_BACKEND"] = "stock"

    elif case == "opt_cpu":
        env["CONFKV_GPU_SEAL_ENABLE"] = "0"
        env["LMCACHE_AESGCM_BACKEND"] = "native"
        env["LMCACHE_AESGCM_NATIVE_LIB"] = str(
            repo
            / "experiments/gpu_kv_seal/native/libcpu_seal.so"
        )

    elif case in CONFKV_CASES:
        env["CONFKV_GPU_SEAL_ENABLE"] = "1"
        env["LMCACHE_AESGCM_BACKEND"] = "stock"
        env["CONFKV_KEY_PROVIDER"] = "dev"
        env["CONFKV_DEV_KEY_PATH"] = str(
            Path(args.gpu_key).resolve()
        )
        env["CONFKV_GPU_MODE"] = (
            "naive"
            if case == "confkv_naive"
            else "optimized"
        )
        env["CONFKV_GPU_CRYPTO_SLOTS"] = str(
            1
            if case == "confkv_naive"
            else args.crypto_slots
        )
        env["CONFKV_GPU_AESGCM_LIB"] = str(
            repo
            / "experiments/gpu_kv_seal/gpu/libgpu_aesgcm.so"
        )
        env["CONFKV_GPU_AESGCM_BINDING"] = str(
            repo
            / "experiments/gpu_kv_seal/gpu/gpu_aesgcm.py"
        )

    return env


def l2_json(
    args: argparse.Namespace,
    l2: Path,
    case: str,
) -> str:
    spec: dict[str, Any] = {
        "type": "fs",
        "base_path": str(l2),
        "use_odirect": args.odirect,
        "persist_enabled": True,
    }

    if case == "opt_cpu":
        spec["serde"] = {
            "type": "aesgcm",
            "key_provider": "hkdf",
            "aes_bits": 128,
            "master_key_path": str(
                Path(args.cpu_key).resolve()
            ),
            "max_workers": 8,
        }

    return json.dumps(spec, separators=(",", ":"))


def lmcache_cmd(
    args: argparse.Namespace,
    l2: Path,
    case: str,
) -> list[str]:
    binary = args.lmcache_bin or shutil.which("lmcache")
    if not binary:
        raise RuntimeError(
            "lmcache not found; source env.sh first"
        )

    return [
        "taskset",
        "-c",
        args.cpus,
        binary,
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.lmcache_port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(args.lmcache_http_port),
        "--chunk-size",
        str(args.chunk_size),
        "--max-gpu-workers",
        "4",
        "--max-cpu-workers",
        "8",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--l1-size-gb",
        str(args.l1_gb),
        "--l1-init-size-gb",
        str(args.l1_init_gb),
        "--eviction-policy",
        "LRU",

        # 必须使用 skip_l1：
        # L2 store 完成后移除 L1 副本，保证 load 测到 L2 解密。
        "--l2-store-policy",
        "skip_l1",
        "--l2-prefetch-policy",
        "default",
        "--l2-prefetch-max-in-flight",
        "8",
        "--l2-adapter",
        l2_json(args, l2, case),
    ]


def vllm_cmd(args: argparse.Namespace) -> list[str]:
    binary = args.vllm_bin or shutil.which("vllm")
    if not binary:
        raise RuntimeError(
            "vllm not found; source env.sh first"
        )

    connector = {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path":
            "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "127.0.0.1",
            "lmcache.mp.port": args.lmcache_port,
            "lmcache.mp.mp_transfer_mode": "lmcache_driven",
        },
    }

    cmd = [
        "taskset",
        "-c",
        args.cpus,
        binary,
        "serve",
        args.model,
        "--served-model-name",
        args.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.vllm_port),
        "--tensor-parallel-size",
        "4",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--kv-transfer-config",
        json.dumps(connector, separators=(",", ":")),
    ]

    for extra in args.vllm_extra:
        cmd.extend(shlex.split(extra))

    return cmd


def prompts(
    args: argparse.Namespace,
    count: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
    )

    body = tokenizer.encode(
        "ConfKV persistent KV benchmark protected prefix. ",
        add_special_tokens=False,
    )

    prefixes = []
    for item in range(count):
        ids = tokenizer.encode(
            f"Unique document {item:08d}. ",
            add_special_tokens=False,
        )

        while len(ids) < args.prefix_tokens:
            ids.extend(
                body[:args.prefix_tokens - len(ids)]
            )

        prefixes.append(ids[:args.prefix_tokens])

    store_suffix = tokenizer.encode(
        "\nStore pass. Answer:",
        add_special_tokens=False,
    )
    load_suffix = tokenizer.encode(
        "\nLoad pass. Answer:",
        add_special_tokens=False,
    )

    return prefixes, store_suffix, load_suffix


async def request(
    session: Any,
    args: argparse.Namespace,
    *,
    case: str,
    phase: str,
    profile: str,
    item: int,
    concurrency: int,
    prompt: list[int],
    max_tokens: int,
) -> Result:
    payload = {
        "model": args.served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    started = time.perf_counter()
    first = None
    done = False
    usage: dict[str, Any] = {}

    try:
        async with session.post(
            (
                f"http://127.0.0.1:"
                f"{args.vllm_port}/v1/completions"
            ),
            json=payload,
        ) as response:
            if response.status >= 300:
                body = (await response.text())[:500]
                raise RuntimeError(
                    f"HTTP {response.status}: {body}"
                )

            while True:
                raw = await response.content.readline()
                if not raw:
                    break

                line = raw.decode(
                    errors="replace"
                ).strip()

                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break

                obj = json.loads(data)

                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if (
                    choices
                    and choices[0].get("text")
                    and first is None
                ):
                    first = time.perf_counter()

        if not done:
            raise RuntimeError(
                "stream ended before [DONE]"
            )
        if first is None:
            raise RuntimeError(
                "stream completed without generated text"
            )

        ended = time.perf_counter()
        output_tokens = int(
            usage.get("completion_tokens", max_tokens)
        )

        return Result(
            case=case,
            phase=phase,
            profile=profile,
            item=item,
            concurrency=concurrency,
            ok=True,
            prompt_tokens=int(
                usage.get("prompt_tokens", len(prompt))
            ),
            output_tokens=output_tokens,
            ttft_ms=(first - started) * 1000,
            tpot_ms=(
                (ended - first)
                * 1000
                / (output_tokens - 1)
                if output_tokens > 1
                else 0.0
            ),
            e2e_ms=(ended - started) * 1000,
        )

    except Exception as exc:
        return Result(
            case=case,
            phase=phase,
            profile=profile,
            item=item,
            concurrency=concurrency,
            ok=False,
            prompt_tokens=len(prompt),
            output_tokens=0,
            ttft_ms=math.nan,
            tpot_ms=math.nan,
            e2e_ms=(
                time.perf_counter() - started
            ) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


async def group(
    args: argparse.Namespace,
    *,
    case: str,
    phase: str,
    profile: str,
    concurrency: int,
    items: list[tuple[int, list[int]]],
    max_tokens: int,
) -> tuple[list[Result], float]:
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout
    )
    connector = aiohttp.TCPConnector(
        limit=concurrency
    )
    started = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:

        async def one(
            item: int,
            prompt: list[int],
        ) -> Result:
            async with semaphore:
                return await request(
                    session,
                    args,
                    case=case,
                    phase=phase,
                    profile=profile,
                    item=item,
                    concurrency=concurrency,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

        rows = await asyncio.gather(
            *(
                one(item, prompt)
                for item, prompt in items
            )
        )

    return (
        list(rows),
        time.perf_counter() - started,
    )


def log_metrics(path: Path) -> dict[str, int]:
    text = path.read_text(errors="replace")
    hits = re.findall(
        (
            r"(\d+)/(\d+) retained keys "
            r"\((\d+) L1, (\d+) L2\)"
        ),
        text,
    )

    return {
        "prefetch_events": len(hits),
        "l1_hit_keys": sum(
            int(hit[2]) for hit in hits
        ),
        "l2_hit_keys": sum(
            int(hit[3]) for hit in hits
        ),
    }


def summarize(
    rows: list[Result],
    durations: dict[tuple[str, str], float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    groups = sorted({
        (row.phase, row.profile)
        for row in rows
    })

    for phase, profile in groups:
        selected = [
            row for row in rows
            if (
                row.phase,
                row.profile,
            ) == (
                phase,
                profile,
            )
        ]
        good = [
            row for row in selected
            if row.ok
        ]
        duration = durations[(phase, profile)]

        output[f"{phase}/{profile}"] = {
            "requests": len(selected),
            "successful": len(good),
            "request_throughput_rps":
                len(good) / duration,
            "output_throughput_tps":
                sum(
                    row.output_tokens
                    for row in good
                ) / duration,
            "ttft_ms": stats([
                row.ttft_ms for row in good
            ]),
            "tpot_ms": stats([
                row.tpot_ms for row in good
            ]),
            "e2e_ms": stats([
                row.e2e_ms for row in good
            ]),
            "errors": [
                row.error
                for row in selected
                if not row.ok
            ],
        }

    return output


def case_metadata(case: str) -> dict[str, Any]:
    common = {
        "experiment_scope":
            "non-cc-data-plane-performance-only",
        "tdx_enabled": False,
        "gpu_cc_enabled": False,
        "remote_attestation": False,
    }

    variants = {
        "baseline": {
            "encryption": "none",
            "encryption_location": "none",
            "key_provider": "none",
            "execution_mode": "native-lmcache",
            "security_mode": "unencrypted",
        },
        "opt_cpu": {
            "encryption": "aes-128-gcm",
            "encryption_location": "cpu",
            "key_provider": "hkdf-file",
            "execution_mode": "cpu-thread-pool",
            "security_mode": "dev-no-cc",
        },
        "confkv_naive": {
            "encryption": "aes-128-gcm",
            "encryption_location": "gpu",
            "key_provider": "dev",
            "execution_mode": "naive",
            "security_mode": "dev-no-cc",
        },
        "confkv_optimized": {
            "encryption": "aes-128-gcm",
            "encryption_location": "gpu",
            "key_provider": "dev",
            "execution_mode": "optimized",
            "security_mode": "dev-no-cc",
        },
    }

    if case not in variants:
        raise ValueError(f"unknown case: {case}")

    return {
        **common,
        **variants[case],
    }


def run_case(
    args: argparse.Namespace,
    repo: Path,
    run_dir: Path,
    case: str,
    prefixes: list[list[int]],
    miss_prefixes: list[list[int]],
    store_suffix: list[int],
    load_suffix: list[int],
) -> dict[str, Any]:
    case_dir = run_dir / case
    l2 = case_dir / "l2"
    l2.mkdir(parents=True)

    env = env_for(args, repo, case)
    server = None
    engine = None
    rows: list[Result] = []
    durations: dict[tuple[str, str], float] = {}
    drain = math.nan

    try:
        server = start(
            "LMCache",
            lmcache_cmd(args, l2, case),
            repo,
            env,
            case_dir / "lmcache.log",
        )
        wait_tcp(
            server,
            args.lmcache_port,
            args.startup_timeout,
        )

        engine = start(
            "vLLM",
            vllm_cmd(args),
            repo,
            env,
            case_dir / "vllm.log",
        )
        wait_http(
            engine,
            (
                f"http://127.0.0.1:"
                f"{args.vllm_port}/health"
            ),
            args.startup_timeout,
        )

        time.sleep(args.stabilize)

        store_items = [
            (
                item,
                [
                    *prefix,
                    *store_suffix,
                ],
            )
            for item, prefix in enumerate(prefixes)
        ]

        stored, elapsed = asyncio.run(
            group(
                args,
                case=case,
                phase="store",
                profile=f"c{args.store_concurrency}",
                concurrency=args.store_concurrency,
                items=store_items,
                max_tokens=args.store_output_tokens,
            )
        )
        rows.extend(stored)
        durations[
            (
                "store",
                f"c{args.store_concurrency}",
            )
        ] = elapsed

        _, _, drain = wait_l2(
            l2,
            args.storage_timeout,
            args.storage_quiet,
        )

        cursor = args.warmup

        if args.warmup:
            warm_items = [
                (
                    item,
                    [
                        *prefixes[item],
                        *load_suffix,
                    ],
                )
                for item in range(args.warmup)
            ]

            warm, _ = asyncio.run(
                group(
                    args,
                    case=case,
                    phase="warmup",
                    profile="excluded",
                    concurrency=min(
                        args.warmup,
                        args.store_concurrency,
                    ),
                    items=warm_items,
                    max_tokens=8,
                )
            )

            if not all(row.ok for row in warm):
                raise RuntimeError([
                    row.error
                    for row in warm
                    if not row.ok
                ])

        # 每个并发档使用不同的 prefix，避免第二次命中 L1。
        for concurrency in args.load_concurrency:
            current = range(
                cursor,
                cursor + args.requests,
            )
            items = [
                (
                    item,
                    [
                        *prefixes[item],
                        *load_suffix,
                    ],
                )
                for item in current
            ]
            profile = f"c{concurrency}"

            loaded, elapsed = asyncio.run(
                group(
                    args,
                    case=case,
                    phase="load",
                    profile=profile,
                    concurrency=concurrency,
                    items=items,
                    max_tokens=args.load_output_tokens,
                )
            )
            rows.extend(loaded)
            durations[("load", profile)] = elapsed
            cursor += args.requests

        miss_cursor = 0
        for concurrency in args.load_concurrency:
            current = range(
                miss_cursor,
                miss_cursor + args.miss_requests,
            )
            items = [
                (
                    item,
                    [
                        *miss_prefixes[item],
                        *load_suffix,
                    ],
                )
                for item in current
            ]
            profile = f"c{concurrency}"

            missed, elapsed = asyncio.run(
                group(
                    args,
                    case=case,
                    phase="miss",
                    profile=profile,
                    concurrency=concurrency,
                    items=items,
                    max_tokens=args.load_output_tokens,
                )
            )
            rows.extend(missed)
            durations[("miss", profile)] = elapsed
            miss_cursor += args.miss_requests

    finally:
        stop(engine)
        stop(server)
        time.sleep(2)

    if not rows:
        raise RuntimeError(f"{case}: no results")

    with (
        case_dir / "requests.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0])),
        )
        writer.writeheader()
        writer.writerows(
            asdict(row) for row in rows
        )

    failed = [
        row for row in rows
        if not row.ok
    ]
    if failed:
        first_failure = failed[0]
        raise RuntimeError(
            f"{case}: {len(failed)} request(s) failed; "
            f"first={first_failure.phase}/"
            f"{first_failure.profile}/"
            f"item-{first_failure.item}: "
            f"{first_failure.error}"
        )

    lmcache_metrics = log_metrics(
        case_dir / "lmcache.log"
    )

    # 防止把 L1 hit 或 cache miss 误认为解密性能。
    if lmcache_metrics["l2_hit_keys"] <= 0:
        raise RuntimeError(
            f"{case}: no L2 hits; "
            "refusing false decrypt result"
        )

    result = {
        "case": case,
        "experiment": case_metadata(case),
        "store_drain_after_responses_s": drain,
        "l2_files": len(
            list(l2.rglob("*.data"))
        ),
        "l2_bytes": sum(
            path.stat().st_size
            for path in l2.rglob("*.data")
        ),
        "lmcache": lmcache_metrics,
        "metrics": summarize(
            rows,
            durations,
        ),
    }

    (case_dir / "summary.json").write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=True,
        )
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        default="/home/xpf/ConfKV",
    )
    parser.add_argument(
        "--model",
        default="/data/models/Qwen3-8B",
    )
    parser.add_argument(
        "--served-model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
    )
    parser.add_argument(
        "--cpus",
        default="0-7",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
    )
    parser.add_argument(
        "--cpu-key",
        default="/run/secrets/confkv/master_key",
    )
    parser.add_argument(
        "--gpu-key",
        default="/run/secrets/confkv/k_store",
    )
    parser.add_argument(
        "--crypto-slots",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lmcache-bin",
        default="",
    )
    parser.add_argument(
        "--vllm-bin",
        default="",
    )
    parser.add_argument(
        "--lmcache-port",
        type=int,
        default=5555,
    )
    parser.add_argument(
        "--lmcache-http-port",
        type=int,
        default=8080,
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--l1-gb",
        type=float,
        default=16,
    )
    parser.add_argument(
        "--l1-init-gb",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--odirect",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--vllm-extra",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--store-output-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--load-output-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--miss-requests",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--store-concurrency",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--load-concurrency",
        nargs="+",
        type=int,
        default=[1, 8],
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=600,
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600,
    )
    parser.add_argument(
        "--storage-timeout",
        type=float,
        default=300,
    )
    parser.add_argument(
        "--storage-quiet",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--stabilize",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--output-root",
        default="results/qwen_e2e",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(args.repo).resolve()

    cpus = indices(args.cpus)
    gpus = indices(args.gpus)

    if len(cpus) != 8 or len(gpus) != 4:
        raise ValueError(
            "need exactly 8 CPUs and 4 GPUs; "
            f"cpus={cpus}, gpus={gpus}"
        )

    if args.prefix_tokens % args.chunk_size:
        raise ValueError(
            "prefix tokens must be a "
            "chunk-size multiple"
        )

    if not shutil.which("taskset"):
        raise RuntimeError("taskset not found")

    # dry-run 可在密钥和 CC 尚未就绪时执行。
    if (
        not args.dry_run
        and "opt_cpu" in args.cases
    ):
        check_secret(
            Path(args.cpu_key).resolve()
        )

    if (
        not args.dry_run
        and any(case in CONFKV_CASES for case in args.cases)
    ):
        # DevKeyProvider consumes one raw 16-byte AES-128 test key.
        # This validates the data path but provides no host/CC security.
        check_secret(
            Path(args.gpu_key).resolve(),
            exact=16,
        )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_dir = (
        repo
        / args.output_root
        / stamp
    ).resolve()
    run_dir.mkdir(parents=True)

    if args.dry_run:
        for case in args.cases:
            print(
                case,
                "LMCache:",
                shlex.join(
                    lmcache_cmd(
                        args,
                        run_dir / case / "l2",
                        case,
                    )
                ),
            )
            print(
                case,
                "vLLM:",
                shlex.join(vllm_cmd(args)),
            )
        return

    stored_count = (
        args.warmup
        + args.requests
        * len(args.load_concurrency)
    )
    miss_count = (
        args.miss_requests
        * len(args.load_concurrency)
    )

    (
        all_prefixes,
        store_suffix,
        load_suffix,
    ) = prompts(
        args,
        stored_count + miss_count,
    )
    prefixes = all_prefixes[:stored_count]
    miss_prefixes = all_prefixes[stored_count:]

    summaries = []

    for case in args.cases:
        print(
            f"\n===== {case} =====",
            flush=True,
        )
        summaries.append(
            run_case(
                args,
                repo,
                run_dir,
                case,
                prefixes,
                miss_prefixes,
                store_suffix,
                load_suffix,
            )
        )

    (run_dir / "comparison.json").write_text(
        json.dumps(
            summaries,
            indent=2,
            allow_nan=True,
        )
    )

    print(f"\nRESULTS: {run_dir}")


if __name__ == "__main__":
    main()
