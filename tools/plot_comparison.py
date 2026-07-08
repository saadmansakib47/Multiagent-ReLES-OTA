"""
tools/plot_comparison.py — Automated Side-by-Side Algorithm Comparison Charts
==============================================================================
Reads results/leaderboard.csv and generates two side-by-side bar charts
comparing any two algorithms.  The previous chart is automatically deleted
so the charts/ folder always contains exactly the latest comparison.

Usage (CLI)
-----------
    # Compare two algorithms (positional, or via flags)
    python tools/plot_comparison.py --algo1 FP3O_Safety_True --algo2 IPPO_Safety_False

    # Auto-compare all entries in the leaderboard (generates one chart per pair)
    python tools/plot_comparison.py --all

    # List available experiment names
    python tools/plot_comparison.py --list

The charts are saved to results/charts/compare_<A>_vs_<B>.png
All previous charts in results/charts/ are wiped before writing new ones
(unless --keep-old is passed).
"""

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import PATHS_CFG, BENCHMARK_CFG

CHARTS_DIR = Path(PATHS_CFG["charts_dir"])
LEADERBOARD = Path(PATHS_CFG["leaderboard_csv"])

# ── Design tokens ────────────────────────────────────────────────────────────
PALETTE = {
    "FP3O":  "#4C72B0",
    "MAPPO": "#DD8452",
    "IPPO":  "#55A868",
    "OTHER": "#C44E52",
}
BACKGROUND  = "#F8F9FA"
GRID_ALPHA  = 0.3
BAR_ALPHA   = 0.88
ERROR_COLOR = "#2d2d2d"
TARGET_COLOR= "#E83030"


def _algo_color(name: str) -> str:
    for key in PALETTE:
        if key in name.upper():
            return PALETTE[key]
    return PALETTE["OTHER"]


def _load_leaderboard() -> pd.DataFrame:
    if not LEADERBOARD.exists():
        raise FileNotFoundError(f"Leaderboard not found: {LEADERBOARD}\nRun training first.")
    return pd.read_csv(LEADERBOARD)


def _clear_old_charts() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in CHARTS_DIR.glob("compare_*.png"):
        f.unlink()


def _safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def plot_pair(
    df: pd.DataFrame,
    algo1: str,
    algo2: str,
    keep_old: bool = False,
) -> Path:
    """
    Generate a side-by-side comparison chart for two experiments.
    Returns the path to the saved PNG.
    """
    rows = {n: df[df["Experiment"] == n] for n in [algo1, algo2]}
    for name, row in rows.items():
        if row.empty:
            raise ValueError(f"Experiment '{name}' not found in leaderboard.\n"
                             f"Available: {list(df['Experiment'])}")

    if not keep_old:
        _clear_old_charts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BACKGROUND)
    fig.suptitle(
        f"Algorithm Comparison\n{algo1}  vs  {algo2}",
        fontsize=14, fontweight="bold", y=1.01
    )

    names    = [algo1, algo2]
    colors   = [_algo_color(n) for n in names]
    means    = [float(rows[n]["Mean_Return"].iloc[0]) for n in names]
    cis      = [float(rows[n]["CI_95"].iloc[0]) for n in names]

    # ── Left panel: Mean Return ───────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(BACKGROUND)
    bars0 = ax0.bar(names, means, color=colors, alpha=BAR_ALPHA,
                    width=0.45, edgecolor="white", linewidth=1.2)
    ax0.errorbar(names, means, yerr=cis, fmt="none",
                 ecolor=ERROR_COLOR, elinewidth=2, capsize=6, capthick=2)

    # Target line
    target = BENCHMARK_CFG["target_return_bd"]
    ax0.axhline(target, color=TARGET_COLOR, linestyle="--", linewidth=1.4,
                label=f"Target ≥ {target}")
    ax0.legend(fontsize=9)

    ax0.set_ylabel("Mean Episode Return", fontsize=11)
    ax0.set_title("Mean Return  (↑ better)", fontsize=11, pad=8)
    ax0.yaxis.grid(True, alpha=GRID_ALPHA)
    ax0.set_axisbelow(True)
    ax0.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar, m, ci in zip(bars0, means, cis):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + ci + abs(max(means) - min(means)) * 0.015,
                 f"{m:.2f} ± {ci:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Right panel: p-value + seeds metadata ────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(BACKGROUND)
    ax1.axis("off")

    # Build info table
    table_data = []
    for name in names:
        row = rows[name].iloc[0]
        seeds     = int(row.get("Seeds", "?"))
        ts        = int(row.get("Timesteps", 0))
        p_val     = str(row.get("p_value_vs_IPPO", "N/A"))
        n_agents  = int(row.get("N_Agents", 0))
        n_blocks  = int(row.get("N_Blocks", 0))
        table_data.append([name, f"{seeds}", f"{ts:,}", p_val, f"{n_agents}", f"{n_blocks}"])

    col_labels = ["Experiment", "Seeds", "Steps/Seed", "p-value vs IPPO", "Agents", "Blocks"]
    tbl = ax1.table(
        cellText  = table_data,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0.35, 1, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(BACKGROUND)

    # p-value interpretation
    sig_lines = []
    for name in names:
        row   = rows[name].iloc[0]
        p_raw = row.get("p_value_vs_IPPO", "N/A")
        try:
            p = float(p_raw)
            sig = "✓ significant" if p < BENCHMARK_CFG["p_value_threshold"] else "✗ not significant"
            sig_lines.append(f"{name}: p={p:.4f}  ({sig})")
        except (ValueError, TypeError):
            sig_lines.append(f"{name}: p={p_raw}")

    # Verdict
    better = names[0] if means[0] > means[1] else names[1]
    verdict_color = _algo_color(better)
    ax1.text(0.5, 0.25, "\n".join(sig_lines),
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=9, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef2ff", edgecolor="#aab4e8"))
    ax1.text(0.5, 0.10,
             f"▶  Better algorithm:  {better}",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=11, fontweight="bold", color=verdict_color)
    ax1.set_title("Experiment Metadata & Statistics", fontsize=11, pad=8)

    plt.tight_layout()

    fname = f"compare_{_safe_filename(algo1)}_vs_{_safe_filename(algo2)}.png"
    out   = CHARTS_DIR / fname
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"\n  Chart saved -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side algorithm comparison bar charts."
    )
    parser.add_argument("--algo1",    type=str, default=None,
                        help="First experiment name (exact match from leaderboard)")
    parser.add_argument("--algo2",    type=str, default=None,
                        help="Second experiment name")
    parser.add_argument("--all",      action="store_true",
                        help="Generate charts for all consecutive pairs in leaderboard")
    parser.add_argument("--list",     action="store_true",
                        help="List available experiment names")
    parser.add_argument("--keep-old", action="store_true",
                        help="Do not delete previous charts before writing new ones")
    args = parser.parse_args()

    df = _load_leaderboard()

    if args.list:
        print("\nAvailable experiments in leaderboard:")
        for name in df["Experiment"]:
            print(f"  • {name}")
        return

    if args.all:
        exps = list(df["Experiment"])
        for i in range(0, len(exps) - 1, 2):
            plot_pair(df, exps[i], exps[i + 1], keep_old=args.keep_old)
        if len(exps) % 2 == 1:
            print(f"  (odd number of experiments; '{exps[-1]}' has no pair)")
        return

    if args.algo1 and args.algo2:
        plot_pair(df, args.algo1, args.algo2, keep_old=args.keep_old)
        return

    # Default: compare the first two in the leaderboard
    if len(df) >= 2:
        exps = list(df["Experiment"])
        print(f"  No --algo1/--algo2 specified. Comparing: '{exps[0]}' vs '{exps[1]}'")
        plot_pair(df, exps[0], exps[1], keep_old=args.keep_old)
    else:
        print("Not enough experiments in the leaderboard to compare. Train at least two algorithms.")


if __name__ == "__main__":
    main()
