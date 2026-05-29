#!/usr/bin/env python3
"""Compare a DiLoCo run against the synchronous data-parallel baseline.

Reads the two metrics CSVs written by the worker (rank 0) and the baseline
binary -- both share the schema
`round,total_samples,wall_clock_s,comm_bytes,val_loss,train_loss` -- and prints
a quantitative summary.

The credible claim is that DiLoCo reaches a similar final validation loss while
communicating far less, so the summary centers on the final validation loss and
the communication-volume reduction at matched compute. (Wall-clock is reported
too, but on localhost it reflects compute, not network transfer, so it does not
capture DiLoCo's real-world advantage.)

Uses only the standard library::

    python3 scripts/analyze.py \
        --diloco runs/metrics_diloco.csv \
        --sync runs/metrics_sync.csv
"""

import argparse
import csv
import sys


def load(path):
    """Load a metrics CSV into a dict of column name -> list of floats."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"error: {path} has no data rows")
    cols = {key: [float(r[key]) for r in rows] for key in rows[0]}
    return cols


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def summarize(diloco, sync):
    d_final = diloco["val_loss"][-1]
    s_final = sync["val_loss"][-1]
    d_comm = diloco["comm_bytes"][-1]
    s_comm = sync["comm_bytes"][-1]
    d_wall = diloco["wall_clock_s"][-1]
    s_wall = sync["wall_clock_s"][-1]
    reduction = (s_comm / d_comm) if d_comm else float("nan")

    print("\n=== DiLoCo vs synchronous baseline ===")
    print(f"{'metric':<26}{'DiLoCo':>16}{'Synchronous':>16}")
    print("-" * 58)
    print(f"{'final val loss':<26}{d_final:>16.4f}{s_final:>16.4f}")
    print(f"{'total communication':<26}{human_bytes(d_comm):>16}{human_bytes(s_comm):>16}")
    print(f"{'wall-clock (s)':<26}{d_wall:>16.1f}{s_wall:>16.1f}")
    print("-" * 58)
    print(f"communication reduction: {reduction:.1f}x  (DiLoCo communicates this many times less)")
    print(f"val-loss gap (DiLoCo - sync): {d_final - s_final:+.4f}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diloco", required=True, help="DiLoCo metrics CSV")
    p.add_argument("--sync", required=True, help="synchronous baseline metrics CSV")
    args = p.parse_args()

    summarize(load(args.diloco), load(args.sync))


if __name__ == "__main__":
    main()
