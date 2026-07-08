"""
tools/evaluate_marl.py — Decentralized Execution Evaluation for MARL Models
=============================================================================
Loads a saved SB3 model checkpoint and evaluates it on a fresh
MultiAgentOTAEnv test environment.

Design follows the CTDE (Centralized Training, Decentralized Execution)
paradigm described in:
  - MAPPO-PIS  (arXiv:2408.06656)
  - FP3O       (arXiv:2310.05053)
  - RSR-RSMARL (arXiv:2506.00982)

At test time, ONLY the decentralized actors are used — the centralized critic
is dropped. Each agent observes its own local obs dict and calls
model.predict(obs, deterministic=True) to select its action.

Usage
-----
    from tools.evaluate_marl import evaluate_trained_model

    result = evaluate_trained_model(
        seed_dir   = "results/marl_models/FP3O_Safety_True/seed_0",
        algorithm  = "fp3o",
        n_agents   = 4,
        n_blocks   = 16,
        n_eval_episodes = 20,
    )
    print(result["mean_return"])   # actual test-time mean episode return
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── lazy imports keep this module lightweight when only the signature is needed ──
def _load_sb3_model(seed_dir: str):
    """
    Load the SB3 PPO model saved by train_algorithm().

    train_algorithm() saves to:
        {seed_dir}/{algorithm}_{tag}_final.zip   (SB3 model)

    We look for any .zip file under seed_dir and load the most recent one.
    Falls back to checking for the standard naming pattern.
    """
    from stable_baselines3 import PPO  # pyrefly: ignore [missing-import]

    seed_path = Path(seed_dir)

    # Try standard name patterns first (most specific → most general)
    candidates = []
    for tag in ["bd", "generic"]:
        for alg in ["fp3o", "mappo", "ippo"]:
            p = seed_path / f"{alg}_{tag}_final.zip"
            if p.exists():
                candidates.append(str(p))

    # Fallback: any .zip in the directory
    if not candidates:
        candidates = sorted(glob.glob(str(seed_path / "*.zip")))

    if not candidates:
        raise FileNotFoundError(
            f"No saved model (.zip) found in {seed_dir}. "
            "Make sure train_algorithm() completed successfully."
        )

    model_path = candidates[0]
    model = PPO.load(model_path)
    return model, model_path


def evaluate_trained_model(
    seed_dir: str,
    algorithm: str        = "fp3o",
    n_agents: int         = 4,
    n_blocks: int         = 16,
    n_eval_episodes: int  = 20,
    safety: bool          = True,
    bd_mode: bool         = True,
    verbose: bool         = False,
) -> dict:
    """
    Run decentralized-execution evaluation of a saved MARL model.

    For each episode:
      1. Reset the MultiAgentOTAEnv (fresh test environment, bd_mode=True by default).
      2. For each active agent, extract its local obs from the dict obs.
      3. Call model.predict(obs, deterministic=True) to get the action.
         NOTE: SB3 PPO operates on a SINGLE flattened/dict obs at a time during
         predict(). We adapt this by running agents sequentially within each step
         (deterministic decentralized execution — standard CTDE eval).
      4. Accumulate episode return, payload cost, and shield activations.

    Parameters
    ----------
    seed_dir        : directory where the model .zip was saved by train_algorithm()
    algorithm       : "ippo", "mappo", or "fp3o" (used for logging only)
    n_agents        : must match the training configuration
    n_blocks        : must match the training configuration
    n_eval_episodes : number of independent test episodes (default: 20)
    safety          : whether the CBF safety shield was enabled during training
    bd_mode         : use Bangladesh network conditions for test env (default: True)
    verbose         : print per-episode stats

    Returns
    -------
    dict with keys:
        mean_return           : float — mean episode return across eval episodes
        std_return            : float — std of episode returns
        mean_payload_cost     : float — mean total fleet payload cost per episode
        std_payload_cost      : float — std of fleet payload costs
        mean_shield_rate      : float — fraction of steps where shield activated
        n_eval_episodes       : int   — number of episodes evaluated
        model_path            : str   — path of the loaded model
        algorithm             : str
    """
    from marl_ota_env import MultiAgentOTAEnv  # pyrefly: ignore [missing-import]

    # ── Load model ──────────────────────────────────────────────────────────
    model, model_path = _load_sb3_model(seed_dir)
    if verbose:
        print(f"  [eval] Loaded model: {model_path}")

    # ── Create fresh evaluation environment ─────────────────────────────────
    # Separate from the training env — ensures no data leakage.
    eval_env = MultiAgentOTAEnv(
        n_agents          = n_agents,
        n_blocks          = n_blocks,
        bd_mode           = bd_mode,
        stochastic_latency= True,   # keep realistic stochastic conditions
        safety_shield     = safety,
    )

    episode_returns:       list[float] = []
    episode_payload_costs: list[float] = []
    episode_shield_rates:  list[float] = []

    for ep in range(n_eval_episodes):
        obs_dict, _ = eval_env.reset(seed=1000 + ep)  # deterministic eval seeds
        ep_return    = 0.0
        ep_payload   = 0.0
        total_steps  = 0
        shield_steps = 0

        while eval_env.agents:
            # ── Build action dict for all active agents ──────────────────────
            # Each agent gets its own local obs slice passed to model.predict.
            # SB3's predict() works on a single observation (no vectorised env
            # at eval time) — we process each agent independently.
            actions: dict[str, np.ndarray] = {}
            for agent in eval_env.agents:
                agent_obs = obs_dict[agent]
                # model.predict expects a batch dimension; we add it via {k: v[None]}
                batched_obs = {k: v[np.newaxis, ...] for k, v in agent_obs.items()}
                action, _ = model.predict(batched_obs, deterministic=True)
                # action shape is (1, 2) for MultiDiscrete([n_blocks, 3])
                actions[agent] = action[0] if action.ndim > 1 else action

            obs_dict, rewards, terminations, truncations, infos = eval_env.step(actions)

            ep_return  += sum(rewards.values())
            total_steps += 1

            # Track shield activations
            for agent, info in infos.items():
                if info.get("shielded", False):
                    shield_steps += 1

        # Accumulate fleet payload cost from final agent states
        fleet_payload = sum(
            eval_env.cum_enc_cost[a] + eval_env.cum_tx_cost[a]
            for a in eval_env.possible_agents
        )

        shield_rate = shield_steps / max(total_steps * n_agents, 1)

        episode_returns.append(ep_return)
        episode_payload_costs.append(fleet_payload)
        episode_shield_rates.append(shield_rate)

        if verbose:
            print(
                f"  [eval] ep={ep+1:3d}/{n_eval_episodes}  "
                f"return={ep_return:8.2f}  "
                f"payload={fleet_payload:8.1f}  "
                f"shield_rate={shield_rate:.3f}"
            )

    eval_env.close()

    result = {
        "mean_return":       float(np.mean(episode_returns)),
        "std_return":        float(np.std(episode_returns)),
        "mean_payload_cost": float(np.mean(episode_payload_costs)),
        "std_payload_cost":  float(np.std(episode_payload_costs)),
        "mean_shield_rate":  float(np.mean(episode_shield_rates)),
        "n_eval_episodes":   n_eval_episodes,
        "model_path":        model_path,
        "algorithm":         algorithm,
    }

    if verbose:
        print(
            f"\n  [eval] Summary  |  "
            f"mean_return={result['mean_return']:.2f} ± {result['std_return']:.2f}  |  "
            f"mean_payload={result['mean_payload_cost']:.1f}  |  "
            f"shield_rate={result['mean_shield_rate']:.4f}"
        )

    return result


def evaluate_all_seeds(
    experiment_dir: str,
    algorithm: str        = "fp3o",
    n_agents: int         = 4,
    n_blocks: int         = 16,
    n_eval_episodes: int  = 20,
    safety: bool          = True,
    bd_mode: bool         = True,
    verbose: bool         = True,
) -> dict:
    """
    Evaluate all seeds under an experiment directory.

    Expects structure:
        experiment_dir/
          seed_0/
          seed_1/
          ...

    Returns
    -------
    dict with keys:
        per_seed_results  : list of per-seed result dicts
        all_returns       : list of mean returns (one per seed)
        mean_return       : float — grand mean across seeds
        std_return        : float — std of seed-level mean returns
    """
    exp_path = Path(experiment_dir)
    seed_dirs = sorted(exp_path.glob("seed_*"))

    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed_* directories found under {experiment_dir}. "
            "Have you run training first?"
        )

    if verbose:
        print(f"\n  [eval] Evaluating {len(seed_dirs)} seeds in {experiment_dir}")

    per_seed_results = []
    all_returns      = []

    for seed_dir in seed_dirs:
        try:
            res = evaluate_trained_model(
                seed_dir        = str(seed_dir),
                algorithm       = algorithm,
                n_agents        = n_agents,
                n_blocks        = n_blocks,
                n_eval_episodes = n_eval_episodes,
                safety          = safety,
                bd_mode         = bd_mode,
                verbose         = verbose,
            )
            per_seed_results.append(res)
            all_returns.append(res["mean_return"])
            if verbose:
                print(f"    seed {seed_dir.name}: mean_return={res['mean_return']:.2f}")
        except Exception as e:
            print(f"    [warn] Skipping {seed_dir.name}: {e}")

    if not all_returns:
        raise RuntimeError(f"All seeds failed evaluation under {experiment_dir}.")

    return {
        "per_seed_results": per_seed_results,
        "all_returns":      all_returns,
        "mean_return":      float(np.mean(all_returns)),
        "std_return":       float(np.std(all_returns)),
    }


# ── CLI quick-eval ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a saved MARL model")
    parser.add_argument("--seed_dir",    required=True, help="Path to seed directory")
    parser.add_argument("--algorithm",   default="fp3o", choices=["ippo", "mappo", "fp3o"])
    parser.add_argument("--n_agents",    type=int, default=4)
    parser.add_argument("--n_blocks",    type=int, default=16)
    parser.add_argument("--episodes",    type=int, default=20)
    parser.add_argument("--safety",      type=lambda x: x.lower() == "true", default=True)
    args = parser.parse_args()

    result = evaluate_trained_model(
        seed_dir        = args.seed_dir,
        algorithm       = args.algorithm,
        n_agents        = args.n_agents,
        n_blocks        = args.n_blocks,
        n_eval_episodes = args.episodes,
        safety          = args.safety,
        verbose         = True,
    )
    print("\n── Evaluation Result ──")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")
