#!/usr/bin/env python3
"""Aggregate a K-sweep produced by scripts/run_sweep.sh and plot the
communication-quality frontier.

Walks <sweep>/<config>/seed<s>/{manifest.json,metrics_diloco.csv,metrics_sync.csv},
groups runs by (sharding mode, K) across seeds, and reports the final held-out
quality as **bits-per-character** (BPC = val_loss / ln 2, the canonical char-LM
metric) against cumulative communication.

The headline figure is BPC vs. communication volume (log x): the synchronous
baseline sits at the right (most communication); DiLoCo marches left as K grows
(communication falls ~1/K) while quality stays flat until the knee. That knee is
the empirical communication-quality tradeoff.

A text + CSV summary always prints. The plot needs matplotlib::

    pip install matplotlib
    python3 scripts/plot_sweep.py --sweep runs/k_sweep
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict

LN2 = math.log(2.0)


def read_final_row(path):
    """Last data row of a metrics CSV, as a dict of floats (None if absent)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[-1]
    return {
        "total_samples": float(r["total_samples"]),
        "comm_bytes": float(r["comm_bytes"]),
        "val_loss": float(r["val_loss"]),
        "bpc": float(r["val_loss"]) / LN2,
    }


def read_curve(path):
    """Full (total_samples, bpc) curve from a metrics CSV."""
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            xs.append(float(r["total_samples"]))
            ys.append(float(r["val_loss"]) / LN2)
    return xs, ys


def load_sweep(sweep_dir):
    """Return (runs, curves). `runs` is a list of per-run dicts; `curves` maps
    (mode, k, seed) -> (diloco_curve, sync_curve) for the convergence panel."""
    runs = []
    curves = {}
    for manifest_path in sorted(glob.glob(os.path.join(sweep_dir, "**", "manifest.json"), recursive=True)):
        run_dir = os.path.dirname(manifest_path)
        with open(manifest_path) as f:
            m = json.load(f)
        diloco = read_final_row(os.path.join(run_dir, "metrics_diloco.csv"))
        sync = read_final_row(os.path.join(run_dir, "metrics_sync.csv"))
        if diloco is None or sync is None:
            print(f"warning: skipping incomplete run {run_dir}")
            continue
        key = (m["data_sharding"], int(m["inner_steps"]), int(m["seed"]))
        runs.append({"mode": m["data_sharding"], "k": int(m["inner_steps"]),
                     "seed": int(m["seed"]), "diloco": diloco, "sync": sync})
        curves[key] = (
            read_curve(os.path.join(run_dir, "metrics_diloco.csv")),
            read_curve(os.path.join(run_dir, "metrics_sync.csv")),
        )
    return runs, curves


def mean_std(xs):
    xs = list(xs)
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def aggregate(runs):
    """Per (mode, k) DiLoCo stats, and per-mode pooled baseline stats."""
    by_mode_k = defaultdict(list)
    sync_by_mode = defaultdict(list)
    for r in runs:
        by_mode_k[(r["mode"], r["k"])].append(r)
        sync_by_mode[r["mode"]].append(r["sync"])

    diloco = {}
    for (mode, k), group in by_mode_k.items():
        bpc_m, bpc_s = mean_std(g["diloco"]["bpc"] for g in group)
        comm = statistics.mean(g["diloco"]["comm_bytes"] for g in group)
        diloco[(mode, k)] = {"bpc": bpc_m, "bpc_std": bpc_s, "comm": comm,
                             "n_seeds": len(group)}

    sync = {}
    for mode, rows in sync_by_mode.items():
        bpc_m, bpc_s = mean_std(s["bpc"] for s in rows)
        sync[mode] = {"bpc": bpc_m, "bpc_std": bpc_s,
                      "comm": statistics.mean(s["comm_bytes"] for s in rows)}
    return diloco, sync


def summarize(diloco, sync, out_csv):
    modes = sorted({m for (m, _) in diloco})
    print("\n=== K-sweep summary (final held-out bits-per-character) ===")
    rows = []
    for mode in modes:
        base = sync[mode]
        print(f"\nmode = {mode}   (synchronous baseline: BPC {base['bpc']:.4f} "
              f"± {base['bpc_std']:.4f}, comm {base['comm'] / 1e6:.1f} MB)")
        print(f"  {'K':>5}  {'BPC':>14}  {'comm (MB)':>10}  {'comm reduction':>14}")
        for (m, k) in sorted(diloco, key=lambda mk: (mk[0], mk[1])):
            if m != mode:
                continue
            d = diloco[(m, k)]
            reduction = base["comm"] / d["comm"] if d["comm"] else float("nan")
            print(f"  {k:>5}  {d['bpc']:>7.4f} ± {d['bpc_std']:.4f}  "
                  f"{d['comm'] / 1e6:>10.1f}  {reduction:>12.1f}x")
            rows.append({"mode": m, "k": k, "bpc": f"{d['bpc']:.6f}",
                         "bpc_std": f"{d['bpc_std']:.6f}",
                         "comm_bytes": int(d["comm"]),
                         "comm_reduction_vs_sync": f"{reduction:.3f}",
                         "n_seeds": d["n_seeds"],
                         "sync_bpc": f"{base['bpc']:.6f}",
                         "sync_comm_bytes": int(base["comm"])})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


def plot(diloco, sync, curves, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({m for (m, _) in diloco})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Headline: BPC vs communication (log x), the Pareto frontier ----------
    for i, mode in enumerate(modes):
        pts = sorted(((k, diloco[(m, k)]) for (m, k) in diloco if m == mode),
                     key=lambda kv: kv[0])
        ks = [k for k, _ in pts]
        comm = [d["comm"] / 1e6 for _, d in pts]
        bpc = [d["bpc"] for _, d in pts]
        err = [d["bpc_std"] for _, d in pts]
        ax1.errorbar(comm, bpc, yerr=err, marker="o", capsize=3,
                     label=f"DiLoCo ({mode})", color=f"C{i}")
        for k, c, b in zip(ks, comm, bpc):
            ax1.annotate(f"K={k}", (c, b), textcoords="offset points",
                         xytext=(6, 6), fontsize=8, color=f"C{i}")
        base = sync[mode]
        ax1.scatter([base["comm"] / 1e6], [base["bpc"]], marker="*", s=180,
                    color=f"C{i}", edgecolor="black", zorder=5,
                    label=f"Synchronous ({mode})")
    ax1.set_xscale("log")
    ax1.set_xlabel("cumulative communication (MB, log scale)")
    ax1.set_ylabel("validation bits-per-character")
    ax1.set_title("Communication-quality frontier\n(same compute; left = cheaper)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    # --- Convergence: BPC vs compute for one mode, seed 0 --------------------
    conv_mode = modes[0]
    ks_present = sorted({k for (m, k) in diloco if m == conv_mode})
    plotted_sync = False
    for j, k in enumerate(ks_present):
        key = (conv_mode, k, 0)
        if key not in curves:
            continue
        (dx, dy), (sx, sy) = curves[key]
        ax2.plot(dx, dy, marker=".", color=f"C{j}", label=f"DiLoCo K={k}")
        if not plotted_sync:
            ax2.plot(sx, sy, "--", color="black", alpha=0.7, label="Synchronous")
            plotted_sync = True
    ax2.set_xlabel("total sequences processed (compute)")
    ax2.set_ylabel("validation bits-per-character")
    ax2.set_title(f"Convergence per unit compute ({conv_mode}, seed 0)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("DiLoCo: equal quality at K× lower communication", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="sweep directory (e.g. runs/k_sweep)")
    args = p.parse_args()

    runs, curves = load_sweep(args.sweep)
    if not runs:
        raise SystemExit(f"no complete runs found under {args.sweep}")
    diloco, sync = aggregate(runs)
    summarize(diloco, sync, os.path.join(args.sweep, "summary.csv"))
    try:
        plot(diloco, sync, curves, os.path.join(args.sweep, "sweep_plot.png"))
    except ImportError:
        print("\nmatplotlib not installed; skipping plot (pip install matplotlib). "
              "Summary above and summary.csv still reflect the sweep.")


if __name__ == "__main__":
    main()
