"""
tools/benchmark_runner.py — Fault-Tolerant Multi-Seed Benchmark Runner (Tier 1)
================================================================================
Automates execution of the 10-seed non-bijective fleet benchmark with
atomic seed-level persistence.

Features:
---------
1. Tier 1 Persistence: Automatically checks `results/benchmark_tracker.json`.
   Completed seeds are NEVER re-executed.
2. Crash Resilient: Power outages, thermal cut-offs, or manual interruptions
   will only interrupt the current seed. All prior seeds remain intact.
3. Factory Matrix: Supports running individual algorithms or the full 2x3 matrix:
   {IPPO, MAPPO, FP3O} x {Blind, Type-Conditioned}.
4. Automated Evaluation: Evaluates final policies over 20 episodes and logs:
   - Mean Episode Return (+/- std)
   - Shield Activation Rate (%)
   - Mean Fleet Payload Cost (B)

Usage:
------
    # Run 10 seeds for FP3O Type-Conditioned on N=8:
    python tools/benchmark_runner.py --algo fp3o --condition type_conditioned --seeds 10 --n_agents 8

    # Run the full 2x3 benchmark matrix (10 seeds each):
    python tools/benchmark_runner.py --matrix --seeds 10 --n_agents 8

    # Inspect current benchmark progress:
    python tools/benchmark_runner.py --status
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.seed_tracker import SeedTracker
from train_mappo import train_algorithm
from marl_ota_env import MultiAgentOTAEnv


def evaluate_policy(model, env: MultiAgentOTAEnv, n_eval_episodes: int = 20) -> Dict[str, float]:
    """Evaluate a trained model over deterministic episodes to measure returns and shield rate."""
    returns = []
    payloads = []
    shield_interventions = 0
    total_steps = 0

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_ret = 0.0

        while env.agents:
            actions = {}
            for a in env.agents:
                action, _ = model.predict(obs[a], deterministic=True)
                actions[a] = action

            obs, rews, terms, truncs, infos = env.step(actions)
            ep_ret += sum(rews.values())
            total_steps += len(env.agents)

            for a in env.agents:
                info = infos.get(a, {})
                if info.get("shield_active", False) or info.get("safety_override", False):
                    shield_interventions += 1

        total_payload = sum(env.cum_enc_cost[a] + env.cum_tx_cost[a] for a in env.possible_agents)
        returns.append(ep_ret)
        payloads.append(total_payload)

    shield_rate = (shield_interventions / max(total_steps, 1)) if total_steps > 0 else 0.0

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_payload": float(np.mean(payloads)),
        "std_payload": float(np.std(payloads)),
        "shield_rate": float(shield_rate),
        "eval_episodes": n_eval_episodes
    }


def run_seed_benchmark(
    algo: str,
    condition: str,
    n_seeds: int = 10,
    n_agents: int = 8,
    fleet_preset: str = "balanced_8",
    timesteps: int = 100_000,
    coupled_channel: bool = True,
    gateway_bw_mbps: float = 25.0,
    save_models_dir: str = "results/models",
    tracker: SeedTracker = None,
    checkpoint_freq: int = 10_000,
    checkpoint_dir: str = "results/checkpoints",
) -> None:
    if tracker is None:
        tracker = SeedTracker()

    type_conditioning = (condition.lower() == "type_conditioned")
    models_dir = Path(save_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {algo.upper()} | Condition: {condition} | Fleet: {fleet_preset} (N={n_agents})")
    print(f"Target: {n_seeds} seeds x {timesteps:,} steps | Coupled: {coupled_channel} ({gateway_bw_mbps} Mbps)")
    print(f"{'='*70}")

    for seed in range(n_seeds):
        key = tracker.make_key(algo, condition, seed, n_agents, fleet_preset)

        # ── Tier 1 Check: Skip completed seeds ──
        if tracker.is_completed(algo, condition, seed, n_agents, fleet_preset):
            metrics = tracker._data["seeds"][key].get("metrics", {})
            print(f"[{seed+1}/{n_seeds}] [SKIP] Seed {seed} already completed. Return: {metrics.get('mean_return', 'N/A'):.2f}, Shield: {metrics.get('shield_rate', 0)*100:.1f}%")
            continue

        print(f"\n[{seed+1}/{n_seeds}] [EXEC] Starting Seed {seed} for {algo.upper()} ({condition})...")
        tracker.mark_started(algo, condition, seed, n_agents, fleet_preset)

        model_filename = f"{algo.lower()}_{condition.lower()}_{fleet_preset}_s{seed}.zip"
        model_path = models_dir / model_filename

        try:
            # -- Tier-2: Find latest step checkpoint for this seed -------
            _ckpt_dir = Path(checkpoint_dir)
            _existing = sorted(
                _ckpt_dir.glob(f"seed_{seed}_step_*.zip"),
                key=lambda p: int(p.stem.split("_step_")[1])
            ) if _ckpt_dir.exists() else []
            _resume_path = str(_existing[-1]) if _existing else None
            if _resume_path:
                _rs = int(Path(_resume_path).stem.split("_step_")[1])
                print(f"  [tier2] Checkpoint found at step {_rs:,} -> will resume")

            # Execute training with this specific seed
            model, _ = train_algorithm(
                algorithm=algo.lower(),
                n_agents=n_agents,
                n_blocks=16,
                safety=True,
                total_timesteps=timesteps,
                type_conditioning=type_conditioning,
                coupled_channel=coupled_channel,
                gateway_bw_mbps=gateway_bw_mbps,
                fleet_preset=fleet_preset,
                seed=seed,
                save_path=str(model_path),
                return_model=True,
                device="auto",
                checkpoint_freq=checkpoint_freq,
                checkpoint_dir=str(_ckpt_dir),
                resume_from_checkpoint=_resume_path,
            )

            # Evaluate policy
            eval_env = MultiAgentOTAEnv(
                n_agents=n_agents,
                n_blocks=16,
                safety_shield=True,
                type_conditioning=type_conditioning,
                coupled_channel=coupled_channel,
                gateway_bw_mbps=gateway_bw_mbps,
                fleet_preset=fleet_preset,
            )
            metrics = evaluate_policy(model, eval_env, n_eval_episodes=20)

            # Atomically mark completed
            tracker.mark_completed(
                algo=algo,
                condition=condition,
                seed=seed,
                model_path=str(model_path),
                metrics=metrics,
                n_agents=n_agents,
                fleet_preset=fleet_preset,
            )

            print(f"[{seed+1}/{n_seeds}] [DONE] Seed {seed} completed & saved to {model_path}!")
            print(f"       Mean Return: {metrics['mean_return']:.2f} | Shield Rate: {metrics['shield_rate']*100:.1f}% | Payload: {metrics['mean_payload']:.1f} B")

        except KeyboardInterrupt:
            # Tier-3: user pressed Ctrl+C, emergency checkpoint already saved.
            print(f"\n[PAUSED] Seed {seed} paused. Re-run to resume from last checkpoint.")
            return  # Stop run; Tier-1 keeps this seed as "started"
        except Exception as e:
            print(f"\n[ERROR] Seed {seed} encountered an error: {e}")
            raise e


def print_status():
    tracker = SeedTracker()
    summary = tracker.get_progress_summary()
    print("\n" + "="*50)
    print("BENCHMARK TRACKER STATUS (Tier 1)")
    print("="*50)
    print(f"Total Registered Seeds: {summary['total_registered']}")
    print(f"Completed Seeds:        {summary['completed']}")
    print(f"In-Progress / Stalled:  {summary['in_progress']}")
    print("="*50)

    for k, v in tracker._data.get("seeds", {}).items():
        status = v.get("status")
        ret = v.get("metrics", {}).get("mean_return", "N/A")
        sr = v.get("metrics", {}).get("shield_rate", "N/A")
        sr_str = f"{sr*100:.1f}%" if isinstance(sr, (int, float)) else sr
        ret_str = f"{ret:.2f}" if isinstance(ret, (int, float)) else str(ret)
        print(f"  {k:45s} | {status:11s} | Return: {ret_str:8s} | Shield: {sr_str}")


def main():
    parser = argparse.ArgumentParser(description="Fault-Tolerant Benchmark Runner (Tier 1)")
    parser.add_argument("--algo", type=str, default="fp3o", choices=["fp3o", "ippo", "mappo"])
    parser.add_argument("--condition", type=str, default="type_conditioned", choices=["type_conditioned", "blind"])
    parser.add_argument("--n_agents", type=int, default=8)
    parser.add_argument("--fleet_preset", type=str, default="balanced_8")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--coupled", action="store_true", default=True)
    parser.add_argument("--matrix", action="store_true", help="Run the full 2x3 factorial benchmark")
    parser.add_argument("--status", action="store_true", help="Print benchmark progress status")
    parser.add_argument("--checkpoint_freq", type=int, default=10_000,
                        help="Save step checkpoint every N steps (Tier 2)")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints",
                        help="Directory for Tier 2 step checkpoints")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    tracker = SeedTracker()

    if args.matrix:
        matrix_configs = [
            ("fp3o", "type_conditioned"),
            ("fp3o", "blind"),
            ("ippo", "type_conditioned"),
            ("ippo", "blind"),
            ("mappo", "type_conditioned"),
            ("mappo", "blind"),
        ]
        for algo, condition in matrix_configs:
            run_seed_benchmark(
                algo=algo,
                condition=condition,
                n_seeds=args.seeds,
                n_agents=args.n_agents,
                fleet_preset=args.fleet_preset,
                timesteps=args.timesteps,
                coupled_channel=args.coupled,
                tracker=tracker,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_dir=args.checkpoint_dir,
            )
    else:
        run_seed_benchmark(
            algo=args.algo,
            condition=args.condition,
            n_seeds=args.seeds,
            n_agents=args.n_agents,
            fleet_preset=args.fleet_preset,
            timesteps=args.timesteps,
            coupled_channel=args.coupled,
            tracker=tracker,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_dir=args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()
