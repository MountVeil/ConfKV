#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict


def median(values):
    return statistics.median(values) if values else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("--raw-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))

if not files:
    raise SystemExit(f"No CSV files under {args.raw_dir}")

rows = []

for path in files:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for field in [
                "size_mib",
                "size_bytes",
                "iteration",
                "cpu",
                "pid",
                "setup_ns",
                "copy_in_ns",
                "crypto_ns",
                "copy_out_ns",
                "total_ns",
                "minflt",
                "majflt",
            ]:
                row[field] = int(row[field])

            rows.append(row)


# ----------------------------------------------------------------------
# Per-process statistics
# ----------------------------------------------------------------------

by_process = defaultdict(list)

for row in rows:
    key = (
        row["mode"],
        row["size_mib"],
        row["run_id"],
    )
    by_process[key].append(row)


process_rows = []

for (mode, size_mib, run_id), samples in sorted(by_process.items()):
    total_ms = [x["total_ns"] / 1e6 for x in samples]
    setup_ms = [x["setup_ns"] / 1e6 for x in samples]
    copy_in_ms = [x["copy_in_ns"] / 1e6 for x in samples]
    crypto_ms = [x["crypto_ns"] / 1e6 for x in samples]
    copy_out_ms = [x["copy_out_ns"] / 1e6 for x in samples]
    minflt = [x["minflt"] for x in samples]

    total_sorted = sorted(total_ms)
    p95_index = round((len(total_sorted) - 1) * 0.95)

    med_total = median(total_ms)

    gib = (size_mib * 1024 * 1024) / (1024 ** 3)
    throughput = gib / (med_total / 1000.0)

    process_rows.append({
        "mode": mode,
        "size_mib": size_mib,
        "run_id": run_id,
        "samples": len(samples),
        "median_total_ms": med_total,
        "p95_total_ms": total_sorted[p95_index],
        "median_setup_ms": median(setup_ms),
        "median_copy_in_ms": median(copy_in_ms),
        "median_crypto_ms": median(crypto_ms),
        "median_copy_out_ms": median(copy_out_ms),
        "median_minflt": median(minflt),
        "throughput_gib_s": throughput,
    })


os.makedirs(args.output_dir, exist_ok=True)

per_process_path = os.path.join(
    args.output_dir,
    "per_process.csv",
)

with open(per_process_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(process_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(process_rows)


# ----------------------------------------------------------------------
# Aggregate independent-process medians
# ----------------------------------------------------------------------

groups = defaultdict(list)

for row in process_rows:
    groups[(row["mode"], row["size_mib"])].append(row)


summary_rows = []

for (mode, size_mib), procs in sorted(groups.items()):
    total = [x["median_total_ms"] for x in procs]
    throughput = [x["throughput_gib_s"] for x in procs]

    summary_rows.append({
        "mode": mode,
        "size_mib": size_mib,
        "processes": len(procs),

        "median_latency_ms":
            median(total),

        "min_process_median_ms":
            min(total),

        "max_process_median_ms":
            max(total),

        "stdev_process_median_ms":
            statistics.stdev(total)
            if len(total) > 1 else 0.0,

        "median_throughput_gib_s":
            median(throughput),

        "median_setup_ms":
            median([x["median_setup_ms"] for x in procs]),

        "median_copy_in_ms":
            median([x["median_copy_in_ms"] for x in procs]),

        "median_crypto_ms":
            median([x["median_crypto_ms"] for x in procs]),

        "median_copy_out_ms":
            median([x["median_copy_out_ms"] for x in procs]),

        "median_minflt":
            median([x["median_minflt"] for x in procs]),
    })


summary_path = os.path.join(
    args.output_dir,
    "summary.csv",
)

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(summary_rows)


print()
print("========== PAPER SUMMARY ==========")

for row in summary_rows:
    print(
        f"{row['mode']:16s} "
        f"{row['size_mib']:4d} MiB  "
        f"{row['median_latency_ms']:9.3f} ms  "
        f"{row['median_throughput_gib_s']:7.3f} GiB/s  "
        f"σ={row['stdev_process_median_ms']:.3f} ms"
    )

print()
print(f"per-process: {per_process_path}")
print(f"summary    : {summary_path}")
