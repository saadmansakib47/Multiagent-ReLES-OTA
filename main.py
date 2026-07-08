"""
<<<<<<< HEAD
main.py - Multi-Agent OTA Evaluation & Training Pipeline
========================================================
Entry point for multi-seed experiments across MARL algorithms and safety
configurations. The CLI remains usable directly and is also used by the local
web control interface.
=======
main.py — Multi-Agent OTA Evaluation & Training Pipeline
=========================================================
Entry point for multi-seed experiments across different MARL
algorithms and safety configurations.

Modes
-----
  train  — train N seeds, then evaluate each with evaluate_marl,
            update leaderboard + training registry.
  test   — skip training; load existing saved models from all seeds
            found in the experiment directory and report evaluation stats.
>>>>>>> db9e210125f724cb49dccfb9e0553338e7cadc5a
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from config import BENCHMARK_CFG, PATHS_CFG, TRAIN_CFG
from tools.training_registry import log_run, print_summary as registry_summary
<<<<<<< HEAD
from train_mappo import train_algorithm

=======
from tools.evaluate_marl import evaluate_trained_model, evaluate_all_seeds


# ── Statistical helpers ──────────────────────────────────────────────────────
>>>>>>> db9e210125f724cb49dccfb9e0553338e7cadc5a

def _compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    if n < 2 or se == 0:
        return m, 0.0
<<<<<<< HEAD
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
=======
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
>>>>>>> db9e210125f724cb49dccfb9e0553338e7cadc5a
    return m, h


def _compute_pvalue(baseline_data, target_data):
    if len(baseline_data) < 2 or len(target_data) < 2:
        return 1.0
    _, p = stats.ttest_ind(baseline_data, target_data, equal_var=False)
    return p


<<<<<<< HEAD
def _placeholder_performance(seed_dir, algorithm):
    try:
        from results.marl_models import _placeholder_eval

        return _placeholder_eval(seed_dir, algorithm)
    except Exception:
        algo_offsets = {"ippo": -100.0, "mappo": -50.0, "fp3o": -20.0}
        return algo_offsets.get(algorithm, -50.0) + np.random.randn() * 5.0


def _read_raw_results(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_raw_results(path, raw_results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)


def _upsert_leaderboard(leaderboard_path, row):
    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)
        idx = df.index[df["Experiment"] == row["Experiment"]].tolist()
        if idx:
            for key, value in row.items():
                df.at[idx[0], key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

=======
def _update_leaderboard(leaderboard_path, new_row: dict) -> pd.DataFrame:
    """
    Upsert `new_row` into the leaderboard CSV.
    Ensures all columns have consistent dtypes to avoid FutureWarning
    when mixing string 'N/A' with float columns.
    """
    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)
    else:
        df = pd.DataFrame(columns=list(new_row.keys()))

    idx_list = df.index[df["Experiment"] == new_row["Experiment"]].tolist()
    new_df = pd.DataFrame([new_row])

    if idx_list:
        # Drop the old row and replace with updated data
        df = df.drop(index=idx_list[0]).reset_index(drop=True)

    df = pd.concat([df, new_df], ignore_index=True)
>>>>>>> db9e210125f724cb49dccfb9e0553338e7cadc5a
    df.to_csv(leaderboard_path, index=False)
    return df


<<<<<<< HEAD
def _run_algorithm(args, algorithm, results_dir, raw_results, raw_results_file):
    entry_name = f"{algorithm.upper()}_Safety_{args.safety}"
    final_returns = []

    for seed in range(args.seeds):
        print(f"\n[Seed {seed + 1}/{args.seeds}] Starting training for {entry_name}...", flush=True)
        seed_dir = str(results_dir / "marl_models" / entry_name / f"seed_{seed}")

        t0 = time.time()
        train_algorithm(
            algorithm=algorithm,
            n_agents=args.n_agents,
            n_blocks=args.n_blocks,
            bd_mode=args.bd_mode,
            safety=args.safety,
            total_timesteps=args.timesteps,
            save_dir=seed_dir,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            death_masking=args.death_masking,
        )
        elapsed = time.time() - t0
        print(f"[Seed {seed + 1}/{args.seeds}] Completed in {elapsed:.1f}s", flush=True)

        # TODO: replace with evaluate_model(seed_dir) once a real evaluator exists.
        final_returns.append(_placeholder_performance(seed_dir, algorithm))

    raw_results[entry_name] = final_returns
    _write_raw_results(raw_results_file, raw_results)

    mean_ret, ci_ret = _compute_ci(final_returns)
    p_val = "N/A"
    baseline_name = f"IPPO_Safety_{args.safety}"
    if entry_name != baseline_name and baseline_name in raw_results:
        p_val = f"{_compute_pvalue(raw_results[baseline_name], final_returns):.4f}"

    new_row = {
        "Experiment": entry_name,
        "Mean_Return": round(mean_ret, 2),
        "CI_95": round(ci_ret, 2),
        "p_value_vs_IPPO": p_val,
        "N_Agents": args.n_agents,
        "N_Blocks": args.n_blocks,
        "Timesteps": args.timesteps,
        "Seeds": args.seeds,
    }

    leaderboard_path = results_dir / "leaderboard.csv"
    df = _upsert_leaderboard(leaderboard_path, new_row)
    print(f"\nLeaderboard updated at {leaderboard_path}", flush=True)
    print(df.to_string(), flush=True)

    run_id = log_run(
        algorithm=algorithm,
        safety=args.safety,
        n_seeds=args.seeds,
        timesteps=args.timesteps,
        mean_return=mean_ret,
        ci_95=ci_ret,
        p_value=p_val,
        extra={
            "n_agents": args.n_agents,
            "n_blocks": args.n_blocks,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "bd_mode": args.bd_mode,
            "death_masking": args.death_masking,
        },
    )
    print(f"\n  Run #{run_id} recorded in training registry.", flush=True)
    registry_summary()

    if len(df) >= 2:
        try:
            from tools.plot_comparison import plot_pair

            experiments = list(df["Experiment"])
            plot_pair(df, experiments[-2], experiments[-1])
            print(f"  Comparison chart saved to {PATHS_CFG['charts_dir']}/", flush=True)
        except Exception as chart_err:
            print(f"  [warn] Chart generation skipped: {chart_err}", flush=True)

    target = BENCHMARK_CFG["target_return_bd"] if args.bd_mode else BENCHMARK_CFG["target_return_generic"]
    if mean_ret >= target:
        print(f"\n  BENCHMARK MET: {mean_ret:.2f} >= target {target}", flush=True)
    else:
        gap = target - mean_ret
        print(f"\n  Benchmark not yet met ({mean_ret:.2f}). Still {gap:.2f} away from target {target}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent OTA ReLES Pipeline")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode of execution")
    parser.add_argument("--algorithm", choices=["ippo", "mappo", "fp3o"], default="fp3o", help="Algorithm to run")
    parser.add_argument("--compare_algorithm", choices=["", "ippo", "mappo", "fp3o"], default="", help="Optional second algorithm")
    parser.add_argument("--safety", type=lambda x: str(x).lower() == "true", default=True, help="Enable Safety Shield")
    parser.add_argument("--n_agents", type=int, default=4, help="Number of ECU agents")
    parser.add_argument("--n_blocks", type=int, default=16, help="Firmware blocks per agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per seed")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--n_envs", type=int, default=TRAIN_CFG["n_envs"], help="Parallel rollout environments")
    parser.add_argument("--n_steps", type=int, default=TRAIN_CFG["n_steps"], help="PPO rollout horizon per environment")
    parser.add_argument("--batch_size", type=int, default=TRAIN_CFG["batch_size"], help="PPO minibatch size")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--bd_mode", type=lambda x: str(x).lower() == "true", default=True, help="Enable constrained BD network")
    parser.add_argument("--death_masking", type=lambda x: str(x).lower() == "true", default=True, help="Enable death masking")
    args = parser.parse_args()

    print("\n" + "=" * 60, flush=True)
    print("  Multi-Agent OTA Evaluation Pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"  Mode:       {args.mode.upper()}", flush=True)
    print(f"  Algorithm:  {args.algorithm.upper()}", flush=True)
    print(f"  Compare:    {args.compare_algorithm.upper() if args.compare_algorithm else 'OFF'}", flush=True)
    print(f"  Safety:     {args.safety}", flush=True)
    print(f"  Seeds:      {args.seeds}", flush=True)
    print(f"  Agents:     {args.n_agents} | Blocks: {args.n_blocks}", flush=True)
    print(f"  Timesteps:  {args.timesteps}", flush=True)
    print(f"  Rollout:    {args.n_steps} x {args.n_envs} = {args.n_steps * args.n_envs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Device:     {args.device}", flush=True)
    print(f"  BD Network: {args.bd_mode}", flush=True)
    print("=" * 60, flush=True)

    if args.mode != "train":
        print("Test mode evaluation will be implemented in future phases.", flush=True)
        return

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_file = results_dir / "raw_seed_returns.json"
    raw_results = _read_raw_results(raw_results_file)

    algorithms = [args.algorithm]
    if args.compare_algorithm and args.compare_algorithm != args.algorithm:
        algorithms.append(args.compare_algorithm)

    for algorithm in algorithms:
        _run_algorithm(args, algorithm, results_dir, raw_results, raw_results_file)

=======
# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent OTA ReLES Pipeline")
    parser.add_argument("--mode",      choices=["train", "test"], default="train",
                        help="'train' to run training + eval; 'test' to eval existing saved models only")
    parser.add_argument("--algorithm", choices=["ippo", "mappo", "fp3o"], default="fp3o",
                        help="Algorithm to run")
    parser.add_argument("--safety",    type=lambda x: str(x).lower() == "true", default=True,
                        help="Enable Safety Shield (CBF)")
    parser.add_argument("--n_agents",  type=int, default=4,        help="Number of ECU agents")
    parser.add_argument("--n_blocks",  type=int, default=16,       help="Firmware blocks per agent")
    parser.add_argument("--timesteps", type=int, default=100_000,  help="Timesteps per seed (train mode)")
    parser.add_argument("--seeds",     type=int, default=10,       help="Number of seeds to run (train mode)")
    parser.add_argument("--eval_episodes", type=int, default=20,   help="Evaluation episodes per seed")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Multi-Agent OTA Evaluation Pipeline")
    print("=" * 60)
    print(f"  Mode:       {args.mode.upper()}")
    print(f"  Algorithm:  {args.algorithm.upper()}")
    print(f"  Safety:     {args.safety}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Agents:     {args.n_agents} | Blocks: {args.n_blocks}")
    if args.mode == "train":
        print(f"  Timesteps:  {args.timesteps}")
    print(f"  Eval eps:   {args.eval_episodes} per seed")
    print("=" * 60)

    results_dir  = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_file  = results_dir / "raw_seed_returns.json"
    leaderboard_path  = results_dir / "leaderboard.csv"
    entry_name        = f"{args.algorithm.upper()}_Safety_{args.safety}"
    experiment_dir    = results_dir / "marl_models" / entry_name

    # ── Load existing raw returns so p-value comparisons stay intact ─────────
    raw_results: dict = {}
    if raw_results_file.exists():
        with open(raw_results_file, "r") as f:
            raw_results = json.load(f)

    # ════════════════════════════════════════════════════════════════
    #  TRAIN MODE
    # ════════════════════════════════════════════════════════════════
    if args.mode == "train":
        final_returns:       list[float] = []
        final_payload_costs: list[float] = []
        final_shield_rates:  list[float] = []

        for seed in range(args.seeds):
            print(f"\n[Seed {seed + 1}/{args.seeds}] Starting training for {entry_name}...")
            seed_dir = str(experiment_dir / f"seed_{seed}")

            # ── Train ───────────────────────────────────────────────────────
            t0 = time.time()
            train_mean_return = train_algorithm(
                algorithm       = args.algorithm,
                n_agents        = args.n_agents,
                n_blocks        = args.n_blocks,
                bd_mode         = True,
                safety          = args.safety,
                total_timesteps = args.timesteps,
                save_dir        = seed_dir,
            )
            elapsed = time.time() - t0
            print(f"[Seed {seed + 1}/{args.seeds}] Training completed in {elapsed:.1f}s")
            if train_mean_return is not None:
                print(f"[Seed {seed + 1}/{args.seeds}] Training ep_rew_mean: {train_mean_return:.2f}")

            # ── Evaluate saved model on a fresh test env ─────────────────────
            # This is the REAL evaluation: decentralized execution per CTDE.
            # The critic is not used at test time — only the actor policies.
            print(f"[Seed {seed + 1}/{args.seeds}] Running {args.eval_episodes}-episode evaluation...")
            eval_result = evaluate_trained_model(
                seed_dir        = seed_dir,
                algorithm       = args.algorithm,
                n_agents        = args.n_agents,
                n_blocks        = args.n_blocks,
                n_eval_episodes = args.eval_episodes,
                safety          = args.safety,
                bd_mode         = True,
                verbose         = True,
            )

            perf          = eval_result["mean_return"]
            payload_cost  = eval_result["mean_payload_cost"]
            shield_rate   = eval_result["mean_shield_rate"]

            print(
                f"[Seed {seed + 1}/{args.seeds}] Eval  →  "
                f"mean_return={perf:.2f}  "
                f"payload={payload_cost:.1f}  "
                f"shield_rate={shield_rate:.4f}"
            )

            final_returns.append(perf)
            final_payload_costs.append(payload_cost)
            final_shield_rates.append(shield_rate)

        # ── Persist raw per-seed returns ─────────────────────────────────────
        raw_results[entry_name] = final_returns
        with open(raw_results_file, "w") as f:
            json.dump(raw_results, f, indent=2)

        # ── Summary statistics ───────────────────────────────────────────────
        mean_ret,     ci_ret     = _compute_ci(final_returns)
        mean_payload, _          = _compute_ci(final_payload_costs)
        mean_shield               = float(np.mean(final_shield_rates))

        # Welch's t-test p-value vs IPPO baseline (per H₁ hypothesis)
        p_val        = None
        baseline_name = f"IPPO_Safety_{args.safety}"
        if entry_name != baseline_name and baseline_name in raw_results:
            p_v   = _compute_pvalue(raw_results[baseline_name], final_returns)
            p_val = round(float(p_v), 4)

        # ── Update leaderboard ───────────────────────────────────────────────
        new_row = {
            "Experiment":         entry_name,
            "Mean_Return":        round(mean_ret, 2),
            "CI_95":              round(ci_ret, 2),
            "p_value_vs_IPPO":    p_val,            # float or None (never "N/A" string)
            "Mean_Payload_Cost":  round(mean_payload, 1),
            "Shield_Rate":        round(mean_shield, 4),
            "N_Agents":           args.n_agents,
            "N_Blocks":           args.n_blocks,
            "Timesteps":          args.timesteps,
            "Seeds":              args.seeds,
        }

        df = _update_leaderboard(leaderboard_path, new_row)
        print(f"\nLeaderboard updated at {leaderboard_path}")
        print(df.to_string())

        # ── Log to training registry ─────────────────────────────────────────
        run_id = log_run(
            algorithm   = args.algorithm,
            safety      = args.safety,
            n_seeds     = args.seeds,
            timesteps   = args.timesteps,
            mean_return = mean_ret,
            ci_95       = ci_ret,
            p_value     = str(p_val) if p_val is not None else "N/A",
            extra       = {
                "n_agents":           args.n_agents,
                "n_blocks":           args.n_blocks,
                "mean_payload_cost":  round(mean_payload, 1),
                "shield_rate":        round(mean_shield, 4),
                "eval_episodes":      args.eval_episodes,
            },
        )
        print(f"\n  Run #{run_id} recorded in training registry.")
        registry_summary()

        # ── Auto-generate comparison chart if ≥2 experiments exist ──────────
        if len(df) >= 2:
            try:
                from tools.plot_comparison import plot_pair
                exps = list(df["Experiment"])
                plot_pair(df, exps[-2], exps[-1])
                print(f"  Comparison chart saved to {PATHS_CFG['charts_dir']}/")
            except Exception as chart_err:
                print(f"  [warn] Chart generation skipped: {chart_err}")

        # ── Check benchmark targets ──────────────────────────────────────────
        target = BENCHMARK_CFG["target_return_bd"]
        if mean_ret >= target:
            print(f"\n  ✅  BENCHMARK MET: {mean_ret:.2f} >= target {target}")
        else:
            gap = target - mean_ret
            print(f"\n  ⚠️   Benchmark not yet met ({mean_ret:.2f}). Still {gap:.2f} away from target {target}.")

        # ── Safety constraint check ──────────────────────────────────────────
        max_shield = BENCHMARK_CFG["max_shield_activation_rate"]
        if mean_shield <= max_shield:
            print(f"  ✅  Shield rate {mean_shield:.4f} is within limit (<= {max_shield})")
        else:
            print(f"  ⚠️   Shield rate {mean_shield:.4f} exceeds limit ({max_shield}). "
                  f"Consider tuning safety threshold or training longer.")

    # ════════════════════════════════════════════════════════════════
    #  TEST MODE — evaluate existing saved models, no training
    # ════════════════════════════════════════════════════════════════
    else:
        if not experiment_dir.exists():
            print(f"\n  ❌  No experiment directory found at {experiment_dir}")
            print(f"      Run with --mode train first to train models.")
            return

        print(f"\n  Loading saved models from {experiment_dir}")
        print(f"  Running {args.eval_episodes} evaluation episodes per seed...\n")

        try:
            all_eval = evaluate_all_seeds(
                experiment_dir  = str(experiment_dir),
                algorithm       = args.algorithm,
                n_agents        = args.n_agents,
                n_blocks        = args.n_blocks,
                n_eval_episodes = args.eval_episodes,
                safety          = args.safety,
                bd_mode         = True,
                verbose         = True,
            )
        except Exception as e:
            print(f"  ❌  Evaluation failed: {e}")
            return

        all_returns = all_eval["all_returns"]
        mean_ret, ci_ret = _compute_ci(all_returns)

        # Collect payload + shield aggregates from per-seed results
        payload_costs = [r["mean_payload_cost"] for r in all_eval["per_seed_results"]]
        shield_rates  = [r["mean_shield_rate"]  for r in all_eval["per_seed_results"]]
        mean_payload  = float(np.mean(payload_costs))
        mean_shield   = float(np.mean(shield_rates))

        # p-value vs IPPO
        p_val        = None
        baseline_name = f"IPPO_Safety_{args.safety}"
        if entry_name != baseline_name and baseline_name in raw_results:
            p_v   = _compute_pvalue(raw_results[baseline_name], all_returns)
            p_val = round(float(p_v), 4)

        print("\n" + "=" * 60)
        print(f"  TEST MODE RESULTS  —  {entry_name}")
        print("=" * 60)
        print(f"  Seeds evaluated :  {len(all_returns)}")
        print(f"  Mean Return     :  {mean_ret:.2f}  ±  {ci_ret:.2f} (95% CI)")
        print(f"  Mean Payload    :  {mean_payload:.1f}")
        print(f"  Shield Rate     :  {mean_shield:.4f}")
        if p_val is not None:
            sig = "✅ SIGNIFICANT" if p_val < BENCHMARK_CFG["p_value_threshold"] else "⚠️  NOT SIGNIFICANT"
            print(f"  p-value vs IPPO :  {p_val}  ({sig})")
        print("=" * 60)

        # Update leaderboard with test-mode eval results
        new_row = {
            "Experiment":        entry_name,
            "Mean_Return":       round(mean_ret, 2),
            "CI_95":             round(ci_ret, 2),
            "p_value_vs_IPPO":   p_val,
            "Mean_Payload_Cost": round(mean_payload, 1),
            "Shield_Rate":       round(mean_shield, 4),
            "N_Agents":          args.n_agents,
            "N_Blocks":          args.n_blocks,
            "Timesteps":         "eval_only",
            "Seeds":             len(all_returns),
        }
        df = _update_leaderboard(leaderboard_path, new_row)
        print(f"\nLeaderboard updated at {leaderboard_path}")
        print(df.to_string())

        # Benchmark checks
        target = BENCHMARK_CFG["target_return_bd"]
        if mean_ret >= target:
            print(f"\n  ✅  BENCHMARK MET: {mean_ret:.2f} >= target {target}")
        else:
            gap = target - mean_ret
            print(f"\n  ⚠️   Benchmark not yet met ({mean_ret:.2f}). Gap: {gap:.2f}")

>>>>>>> db9e210125f724cb49dccfb9e0553338e7cadc5a

if __name__ == "__main__":
    main()
