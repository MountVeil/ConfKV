#!/usr/bin/env bash

set -euo pipefail

CONFKV_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
    pwd
)"

source "$CONFKV_ROOT/env.sh"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"

source "$ROOT/configs/cpu_paper.env"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/paper/$STAMP"

mkdir -p \
    "$OUT/raw" \
    "$OUT/meta"

echo "Experiment output:"
echo "  $OUT"
echo

# ----------------------------------------------------------------------
# Freeze execution environment
# ----------------------------------------------------------------------

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "git_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO" branch --show-current)"
    echo "cpu_core=$CPU_CORE"
    echo "numa_node=$NUMA_NODE"
    echo "sizes_mib=$SIZES_MIB"
    echo "process_repeats=$PROCESS_REPEATS"
    echo "warmup=$WARMUP"
    echo "iterations=$ITERATIONS"
    echo "modes=$MODES"
    echo "python=$(python --version 2>&1)"
    echo "openssl=$(openssl version)"
    echo "glibc=$(ldd --version | head -1)"
} > "$OUT/meta/experiment.txt"

git -C "$REPO" status --porcelain \
    > "$OUT/meta/git_status.txt"

uname -a \
    > "$OUT/meta/uname.txt"

LC_ALL=C lscpu \
    > "$OUT/meta/lscpu.txt"

numactl --hardware \
    > "$OUT/meta/numa.txt"

free -h \
    > "$OUT/meta/memory.txt"

uptime \
    > "$OUT/meta/load.txt"

cat /sys/kernel/mm/transparent_hugepage/enabled \
    > "$OUT/meta/thp.txt" 2>/dev/null || true

"$PYTHON_BIN" -m pip freeze \
    > "$OUT/meta/pip_freeze.txt"

"$PYTHON_BIN" - <<'PY' > "$OUT/meta/python_packages.txt"
import cryptography
import torch

print("torch =", torch.__version__)
print("cryptography =", cryptography.__version__)

try:
    import lmcache
    print("lmcache =", lmcache.__file__)
except Exception as e:
    print("lmcache import error:", e)
PY


# Prevent unrelated numerical libraries from creating worker threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

for mode in $MODES; do
    for size in $SIZES_MIB; do
        for rep in $(seq 1 "$PROCESS_REPEATS"); do

            RUN_ID="${mode}_${size}MiB_r${rep}"

            echo "========================================"
            echo "$RUN_ID"
            echo "========================================"

            BENCH="$ROOT/scripts/bench_seal_worker.py"

            case "$mode" in
                optimized_cpu)
                    BENCH="$ROOT/scripts/bench_optimized_cpu.py"
                    ;;
                lmcache_open|optimized_cpu_open)
                    BENCH="$ROOT/scripts/bench_open_worker.py"
                    ;;
            esac

            numactl \
                --physcpubind="$CPU_CORE" \
                --membind="$NUMA_NODE" \
                "$PYTHON_BIN" "$BENCH" \
                    --mode "$mode" \
                    --size-mib "$size" \
                    --warmup "$WARMUP" \
                    --iterations "$ITERATIONS" \
                    --run-id "$RUN_ID" \
                    --output "$OUT/raw/${RUN_ID}.csv"

        done
    done
done


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

"$PYTHON_BIN" "$ROOT/scripts/summarize_results.py" \
    --raw-dir "$OUT/raw" \
    --output-dir "$OUT"

echo
echo "DONE"
echo "$OUT"
