"""
tools/training_registry.py — Training Run Tracker
===================================================
Records every training run with full metadata (algorithm, timestamps,
seeds, hyperparameters, leaderboard score) to results/training_registry.json.

This solves the "how many times has the model been trained?" question.
Every call to `log_run()` appends one record; the registry is never
overwritten — only appended.

Usage
-----
    from tools.training_registry import log_run, query_runs, print_summary

    # After training completes:
    log_run(
        algorithm = "fp3o",
        safety    = True,
        n_seeds   = 10,
        timesteps = 500_000,
        mean_return = -18.4,
        ci_95       = 0.42,
        extra = {"n_agents": 4, "n_blocks": 16}
    )
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from config import PATHS_CFG

REGISTRY_PATH = Path(PATHS_CFG["training_registry"])


def _load() -> list:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return []


def _save(records: list) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def log_run(
    algorithm: str,
    safety: bool,
    n_seeds: int,
    timesteps: int,
    mean_return: float,
    ci_95: float,
    p_value: str = "N/A",
    extra: dict | None = None,
) -> int:
    """
    Append one training run to the registry.

    Returns
    -------
    run_id : int
        Auto-incremented integer ID for this run.
    """
    records = _load()
    run_id  = len(records) + 1

    record = {
        "run_id":       run_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "algorithm":    algorithm.upper(),
        "safety":       safety,
        "n_seeds":      n_seeds,
        "timesteps_per_seed": timesteps,
        "total_env_steps":    n_seeds * timesteps,
        "mean_return":  round(mean_return, 4),
        "ci_95":        round(ci_95, 4),
        "p_value_vs_ippo": p_value,
        **(extra or {}),
    }
    records.append(record)
    _save(records)
    return run_id


def query_runs(algorithm: str | None = None, safety: bool | None = None) -> list:
    """Return all runs, optionally filtered by algorithm and/or safety flag."""
    records = _load()
    if algorithm is not None:
        records = [r for r in records if r.get("algorithm") == algorithm.upper()]
    if safety is not None:
        records = [r for r in records if r.get("safety") == safety]
    return records


def print_summary():
    """Pretty-print the training registry to stdout."""
    records = _load()
    if not records:
        print("No training runs recorded yet.")
        return

    print(f"\n{'='*72}")
    print(f"  Training Registry  ({len(records)} total runs)")
    print(f"{'='*72}")
    header = f"  {'ID':>3}  {'Algorithm':<10}  {'Safety':<6}  {'Seeds':>5}  "
    header += f"{'Steps/seed':>10}  {'Mean Return':>12}  {'CI±':>6}  {'Timestamp'}"
    print(header)
    print("  " + "-" * 68)
    for r in records:
        ts = r["timestamp"][:10]
        print(
            f"  {r['run_id']:>3}  {r['algorithm']:<10}  {str(r['safety']):<6}  "
            f"{r['n_seeds']:>5}  {r['timesteps_per_seed']:>10,}  "
            f"{r['mean_return']:>12.4f}  {r['ci_95']:>6.4f}  {ts}"
        )
    print()


if __name__ == "__main__":
    print_summary()
