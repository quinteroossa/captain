#!/usr/bin/env python
"""Compare protection solutions across two or more ES policy runs.

Requires eval_policy.py to have been run first for each model
(it saves eval_results.json and eval_protection_grid.npy).

Computes:
  - Jaccard similarity between protection maps (spatial overlap)
  - Side-by-side protection map plots
  - Metric comparison table (threat counts, cost, recovery rates)
  - Cost efficiency: conservation benefit per unit cost

Usage:
    uv run python experiments/compare_policies.py \\
        --runs es_baseline_full es_stochastic_v3 \\
        --out-dir results/comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["LC", "NT", "VU", "EN", "CR"]
THREAT_WEIGHTS = np.array([1, 0, -8, -16, -32])  # mirrors ES reward weights


# =============================================================================
# Args
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare ES policy evaluation results")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run names to compare (must have run eval_policy.py first)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root results directory (default: results/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for comparison plots/table (default: results/comparison)")
    return parser.parse_args()


# =============================================================================
# Jaccard
# =============================================================================

def jaccard(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Jaccard similarity between two binary protection grids (NaN = outside study area)."""
    valid = ~(np.isnan(grid_a) | np.isnan(grid_b))
    a = (grid_a[valid] > 0.5).astype(bool)
    b = (grid_b[valid] > 0.5).astype(bool)
    intersection = (a & b).sum()
    union        = (a | b).sum()
    return float(intersection / union) if union > 0 else 0.0


# =============================================================================
# Plots
# =============================================================================

def plot_side_by_side(grids: dict, out_path: Path):
    n     = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (run_name, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(run_name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Protected")
    plt.suptitle("Protection maps — spatial comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_difference(grid_a: np.ndarray, grid_b: np.ndarray,
                    label_a: str, label_b: str, out_path: Path):
    """Cells unique to A (blue), unique to B (red), shared (grey)."""
    valid = ~(np.isnan(grid_a) | np.isnan(grid_b))
    diff  = np.full(grid_a.shape, np.nan)
    diff[valid & (grid_a > 0.5) & (grid_b > 0.5)] = 0.5   # shared
    diff[valid & (grid_a > 0.5) & (grid_b <= 0.5)] = 1.0  # unique to A
    diff[valid & (grid_a <= 0.5) & (grid_b > 0.5)] = 0.0  # unique to B

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.colors.ListedColormap(["#e74c3c", "#bdc3c7", "#3498db"])
    bounds = [-0.1, 0.25, 0.75, 1.1]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(diff, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"Spatial overlap: {label_a} vs {label_b}", fontsize=11)
    ax.axis("off")
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#3498db", label=f"Unique to {label_b}"),
        Patch(color="#bdc3c7", label="Shared"),
        Patch(color="#e74c3c", label=f"Unique to {label_a}"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metric_comparison(results: dict, metric: str, title: str, out_path: Path):
    run_names = list(results.keys())
    values = []
    for run in run_names:
        vals = results[run]["protected"]["mean_threat_counts"]
        values.append([vals.get(cls, 0) for cls in CLASS_NAMES])
    values = np.array(values)

    x     = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(run_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (run, vals) in enumerate(zip(run_names, values)):
        offset = (i - len(run_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=run)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Number of species")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cost_breakdown(results: dict, out_path: Path):
    run_names = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(run_names)
    for i, run in enumerate(run_names):
        vals = results[run]["protected"].get("mean_cost_by_category", {})
        bar_vals = [vals.get(cls, 0) for cls in CLASS_NAMES]
        offset = (i - len(run_names) / 2 + 0.5) * width
        ax.bar(x + offset, bar_vals, width, label=run)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Budget spent on habitat (rescaled)")
    ax.set_title("Cost breakdown by initial threat category")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Table
# =============================================================================

def print_comparison_table(results: dict, grids: dict):
    run_names = list(results.keys())
    col_w = max(len(r) for r in run_names) + 2

    print("\n" + "=" * 70)
    print("METRIC COMPARISON TABLE")
    print("=" * 70)

    # Threat counts
    header = f"{'Metric':<30}" + "".join(f"{r:>{col_w}}" for r in run_names)
    print(f"\n{header}")
    print("-" * len(header))

    for cls in CLASS_NAMES:
        row = f"  Species {cls} (final)          "[:30]
        for run in run_names:
            val = results[run]["protected"]["mean_threat_counts"].get(cls, "-")
            row += f"{val:>{col_w}.1f}"
        print(row)

    row = f"  Total cost                    "[:30]
    for run in run_names:
        val = results[run]["protected"].get("mean_cost", "-")
        row += f"{val:>{col_w}.4f}"
    print(row)

    # Recovery rates
    print()
    for cls in CLASS_NAMES:
        rates = {}
        for run in run_names:
            r = results[run]["protected"].get("mean_recovery_rate", {}).get(cls)
            rates[run] = r
        if any(v is not None for v in rates.values()):
            row = f"  Recovery rate {cls}             "[:30]
            for run in run_names:
                v = rates[run]
                row += f"{v:>{col_w}.3f}" if v is not None else f"{'—':>{col_w}}"
            print(row)

    # Decline rates
    print()
    for cls in CLASS_NAMES:
        rates = {}
        for run in run_names:
            r = results[run]["protected"].get("mean_decline_rate", {}).get(cls)
            rates[run] = r
        if any(v is not None for v in rates.values()):
            row = f"  Decline rate {cls}              "[:30]
            for run in run_names:
                v = rates[run]
                row += f"{v:>{col_w}.3f}" if v is not None else f"{'—':>{col_w}}"
            print(row)

    # Jaccard between each pair
    run_list = list(grids.keys())
    if len(run_list) >= 2:
        print("\n  Jaccard similarity (spatial overlap between protection maps):")
        for i in range(len(run_list)):
            for j in range(i + 1, len(run_list)):
                j_val = jaccard(grids[run_list[i]], grids[run_list[j]])
                print(f"    {run_list[i]} vs {run_list[j]}: {j_val:.4f}")

    # Transition matrices
    for run in run_names:
        tm = np.array(results[run]["protected"]["mean_transition_matrix"])
        print(f"\n  Transition matrix — {run} (rows=initial, cols=final):")
        header_tm = "         " + "  ".join(f"{c:>5}" for c in CLASS_NAMES)
        print(header_tm)
        for i, row_name in enumerate(CLASS_NAMES):
            row_str = "  ".join(f"{tm[i][j]:5.1f}" for j in range(len(CLASS_NAMES)))
            print(f"    {row_name:2s}   [ {row_str} ]")


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    out_dir = args.out_dir or (args.results_dir / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    grids   = {}

    for run in args.runs:
        results_file = args.results_dir / run / "eval_results.json"
        grid_file    = args.results_dir / run / "eval_protection_grid.npy"

        if not results_file.exists():
            raise FileNotFoundError(
                f"No eval results for '{run}'. Run eval_policy.py first:\n"
                f"  uv run python experiments/eval_policy.py --run-name {run} --data-dir <data-dir>"
            )
        if not grid_file.exists():
            raise FileNotFoundError(f"No protection grid for '{run}' at {grid_file}")

        with open(results_file) as f:
            results[run] = json.load(f)
        grids[run] = np.load(grid_file)
        print(f"  Loaded: {run}")

    # Plots
    plot_side_by_side(grids, out_dir / "protection_maps.png")

    run_list = list(grids.keys())
    if len(run_list) == 2:
        plot_difference(
            grids[run_list[0]], grids[run_list[1]],
            run_list[0], run_list[1],
            out_dir / "protection_difference.png",
        )

    plot_metric_comparison(
        results, metric="threat_counts",
        title="Final species counts by threat category",
        out_path=out_dir / "threat_counts_comparison.png",
    )
    plot_cost_breakdown(results, out_dir / "cost_breakdown.png")

    # Console table
    print_comparison_table(results, grids)

    # Save comparison JSON
    comparison = {
        "runs": args.runs,
        "jaccard": {},
        "metrics": {run: results[run]["protected"] for run in args.runs},
    }
    for i in range(len(run_list)):
        for j in range(i + 1, len(run_list)):
            key = f"{run_list[i]}_vs_{run_list[j]}"
            comparison["jaccard"][key] = jaccard(grids[run_list[i]], grids[run_list[j]])

    out_json = out_dir / "comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Summary → {out_json}")


if __name__ == "__main__":
    main()
