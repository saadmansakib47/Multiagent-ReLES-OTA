"""
train_mappo.py — MAPPO/IPPO Training Scaffold for Phase 2
==========================================================
Training script for multi-agent OTA update learning.

Phase 2 Step 1 (current):
  - IPPO (Independent PPO): each agent has its own PPO policy,
    no parameter sharing, simplest MARL baseline.
  - Uses SuperSuit to wrap PettingZoo env into SB3-compatible VecEnv.

Phase 2 Steps 2+:
  - MAPPO: centralized critic, decentralized actors.
  - Parameter sharing across agents.

Usage
-----
    # IPPO training (default, recommended for Step 1)
    python train_mappo.py --mode ippo --n_agents 4 --timesteps 100000

    # Quick sanity run
    python train_mappo.py --mode random --n_agents 4

References
----------
- MAPPO-PIS  (arXiv:2408.06656)
- FP3O       (arXiv:2310.05053)
- MARL-CC    (arXiv:2511.17653)
"""

import argparse
import json
# pyrefly: ignore [missing-import]
import numpy as np
import time
import sys
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── Env imports ──
from marl_ota_env import MultiAgentOTAEnv


# ══════════════════════════════════════════════════════════════
#  Random-agent baseline (no learning — sanity/benchmark only)
# ══════════════════════════════════════════════════════════════

def run_random_marl(n_agents: int = 4, n_blocks: int = 16, n_episodes: int = 20,
                    bd_mode: bool = False) -> dict:
    """
    Run N episodes with purely random agents.
    Used as the MARL baseline (equivalent to Phase 1's random baseline).
    """
    print(f"\n Running Random Multi-Agent Baseline")
    print(f"   n_agents={n_agents}, n_blocks={n_blocks}, n_episodes={n_episodes}, bd_mode={bd_mode}")

    env = MultiAgentOTAEnv(
        n_agents=n_agents, n_blocks=n_blocks,
        bd_mode=bd_mode, stochastic_latency=bd_mode
    )

    episode_payloads     = []
    episode_memories     = []
    episode_steps        = []
    per_agent_payloads   = defaultdict(list)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        step   = 0
        ep_payload = 0.0

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, _, terms, truncs, infos = env.step(actions)
            step += 1

        # Aggregate after episode
        total_payload = sum(env.cum_enc_cost[a] + env.cum_tx_cost[a] for a in env.possible_agents)
        total_memory  = sum(env.cum_memory[a] for a in env.possible_agents)
        episode_payloads.append(total_payload)
        episode_memories.append(total_memory)
        episode_steps.append(step)

        for agent in env.possible_agents:
            per_agent_payloads[agent].append(env.cum_enc_cost[agent] + env.cum_tx_cost[agent])

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"fleet_payload={total_payload:.1f}  "
                  f"steps={step}")

    results = {
        "method":          "random_marl",
        "n_agents":        n_agents,
        "n_episodes":      n_episodes,
        "bd_mode":         bd_mode,
        "mean_fleet_payload": float(np.mean(episode_payloads)),
        "std_fleet_payload":  float(np.std(episode_payloads)),
        "mean_fleet_memory":  float(np.mean(episode_memories)),
        "mean_episode_steps": float(np.mean(episode_steps)),
        "per_agent_mean_payload": {
            a: float(np.mean(per_agent_payloads[a])) for a in env.possible_agents
        },
    }

    print(f"\n  Mean fleet payload : {results['mean_fleet_payload']:.2f} ± {results['std_fleet_payload']:.2f}")
    print(f"  Mean fleet memory  : {results['mean_fleet_memory']:.2f}")
    print(f"  Mean episode steps : {results['mean_episode_steps']:.1f}")
    return results


# ══════════════════════════════════════════════════════════════
#  IPPO training (Independent PPO via SuperSuit + SB3)
# ══════════════════════════════════════════════════════════════

def train_algorithm(
    algorithm: str      = "fp3o",
    n_agents: int       = 4,
    n_blocks: int       = 16,
    bd_mode: bool       = False,
    safety: bool        = True,
    total_timesteps: int = 500_000,
    save_dir: str       = "results/marl_models",
    n_envs: int          = 4,
    n_steps: int         = 256,
    batch_size: int      = 128,
    n_epochs: int        = 10,
    ent_coef: float      = 0.01,
    device: str          = "auto",
    death_masking: bool  = True,
    seed: int            = 42,
) -> None:
    """
    Train Independent PPO (IPPO) on the multi-agent OTA env.

    IPPO treats each agent as an independent learner with its own
    PPO policy. No parameter sharing, no centralized critic.
    Simple but effective baseline for MARL — often competitive with MAPPO.

    SuperSuit is used to:
      1. Convert PettingZoo Parallel → AEC (parallel_to_aec)
      2. Wrap into a Gymnasium-compatible VecEnv for SB3
    """
    try:
        # pyrefly: ignore [import, missing-import]
        import supersuit as ss
        # pyrefly: ignore [import, missing-import]
        from stable_baselines3 import PPO
        # pyrefly: ignore [import, missing-import]
        from stable_baselines3.common.vec_env import VecMonitor
        # pyrefly: ignore [import, missing-import]
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        # pyrefly: ignore [import, missing-import]
        from fp3o_policy import FP3OPolicy, ValueNormalizationCallback, make_fp3o_policy_kwargs
    except ImportError as e:
        print(f"  ❌  Missing dependency: {e}")
        print("  Install with:  pip install supersuit stable-baselines3 sb3-contrib")
        return

    # ── Invalid-action rate tracker ──
    class InvalidActionRateCallback(BaseCallback):
        """
        Counts steps where info["invalid_action"] == True across the rollout
        and logs ``rollout/invalid_action_rate`` via SB3's logger.

        With action masking active in FP3OPolicy this rate should be exactly 0%
        (invalid block indices are never sampled).  Any non-zero value indicates
        a masking path that was not reached (e.g. the obs-based mask was None).
        """
        def __init__(self):
            super().__init__(verbose=0)
            self._invalid_steps = 0
            self._total_steps   = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                self._total_steps += 1
                if info.get("invalid_action", False):
                    self._invalid_steps += 1
            return True

        def _on_rollout_end(self) -> None:
            rate = (
                self._invalid_steps / self._total_steps
                if self._total_steps > 0 else 0.0
            )
            self.logger.record("rollout/invalid_action_rate", rate)
            self.logger.record("rollout/invalid_action_steps", self._invalid_steps)
            if self._invalid_steps > 0:
                print(
                    f"  [warn] invalid_action_rate={rate:.4%}  "
                    f"({self._invalid_steps}/{self._total_steps} steps) — "
                    "masking may not be covering all paths"
                )
            # Reset accumulators for next rollout
            self._invalid_steps = 0
            self._total_steps   = 0


    print(f"\n Training {algorithm.upper()} on MultiAgentOTAEnv")
    print(f"   n_agents={n_agents}, n_blocks={n_blocks}, bd_mode={bd_mode}, safety={safety}")
    print(f"   total_timesteps={total_timesteps:,}, n_envs={n_envs}, n_steps={n_steps}, batch_size={batch_size}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Build env ──
    def make_env():
        return MultiAgentOTAEnv(
            n_agents=n_agents, n_blocks=n_blocks,
            bd_mode=bd_mode, stochastic_latency=bd_mode,
            safety_shield=safety
        )

    raw_env = make_env()

    # SuperSuit: PettingZoo Parallel → SB3-compatible VecEnv
    # MarkovVectorEnv handles variable-length episodes via black_death masking.
    
    # pyrefly: ignore [missing-import]
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv
    env = MarkovVectorEnv(raw_env, black_death=death_masking)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=n_envs, num_cpus=1, base_class="stable_baselines3")

    # ── Patch missing VecEnv interface methods onto ConcatVecEnv ──────────────
    # SB3's VecEnvWrapper.__init__ calls get_attr("render_mode") on the inner
    # env; ConcatVecEnv does not implement get_attr / set_attr / env_method,
    # which causes AttributeError inside VecMonitor.
    # Setting render_mode explicitly silences the SB3 UserWarning.
    env.render_mode = None
    if not hasattr(env, "get_attr"):
        def _get_attr(attr_name, indices=None):
            val  = getattr(env, attr_name, None)
            idxs = list(range(env.num_envs)) if indices is None else (
                [indices] if isinstance(indices, int) else list(indices)
            )
            return [val for _ in idxs]
        def _set_attr(attr_name, value, indices=None): pass
        def _env_method(method_name, *method_args, indices=None, **method_kwargs): return []
        env.get_attr   = _get_attr
        env.set_attr   = _set_attr
        env.env_method = _env_method

    # Dummy seed stub required by older SB3 internals.
    env.seed = lambda seed=None: None
    env = VecMonitor(env)

    # ── PPO model ──
    policy_kwargs = make_fp3o_policy_kwargs(
        n_blocks=n_blocks,
        ecu_type="generic"
    )
    policy_kwargs["algorithm"] = algorithm
    policy_kwargs["share_features_extractor"] = False

    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA was requested but is not available. Falling back to CPU.")
        selected_device = "cpu"
    else:
        selected_device = device

    from typing import Callable
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """Linear learning rate schedule."""
        def func(progress_remaining: float) -> float:
            # SB3's num_timesteps can slightly overshoot total_timesteps at the end of a rollout,
            # causing progress_remaining to be slightly negative. Clamp it to 0 to prevent
            # a negative learning rate (which causes instant gradient explosion and NaNs).
            return max(0.0, progress_remaining * initial_value)
        return func

    model = PPO(
        policy          = FP3OPolicy,
        env             = env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        device          = selected_device,
        tensorboard_log = f"{save_dir}/logs/{algorithm}_{'bd' if bd_mode else 'generic'}",
        learning_rate   = linear_schedule(3e-4),
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = n_epochs,
        gae_lambda      = 0.95,
        gamma           = 0.99,
        clip_range      = 0.2,
        ent_coef        = ent_coef,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        seed            = seed,
    )
    # Seed RNGs with the per-seed value so each seed genuinely differs
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n  Starting {algorithm.upper()} training on {selected_device}...")
    t0 = time.time()
    callback = CallbackList([
        ValueNormalizationCallback(),
        InvalidActionRateCallback(),
    ])
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    elapsed = time.time() - t0

    tag = "bd" if bd_mode else "generic"
    model.save(f"{save_dir}/{algorithm}_{tag}_final")
    print(f"\n  Training done in {elapsed:.1f}s  →  saved to {save_dir}/{algorithm}_{tag}_final")

    # ── Extract final training-time ep_rew_mean from VecMonitor buffer ──────
    # ep_info_buffer is a deque populated by VecMonitor with dicts {r, l, t}.
    # This gives the rolling mean episode return over the last N episodes
    # seen during training — useful as a sanity check scalar.
    train_mean_return = None
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
        train_mean_return = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
        print(f"  Training-time ep_rew_mean (last {len(model.ep_info_buffer)} eps): {train_mean_return:.2f}")

    env.close()
    return train_mean_return


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 MARL Training — OTA Update Environment")
    parser.add_argument("--mode",       choices=["random", "train"], default="train",
                        help="Training mode: 'random' for baseline, 'train' for learning")
    parser.add_argument("--algorithm",  choices=["ippo", "mappo", "fp3o"], default="fp3o",
                        help="MARL algorithm to train")
    parser.add_argument("--n_agents",   type=int, default=4,       help="Number of ECU agents")
    parser.add_argument("--n_blocks",   type=int, default=16,      help="Firmware blocks per agent")
    parser.add_argument("--bd_mode",    action="store_true",       help="Enable BD network parameters")
    parser.add_argument("--safety",     type=lambda x: str(x).lower() == 'true', default=True, help="Enable Safety Shield")
    parser.add_argument("--timesteps",  type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--n_envs",     type=int, default=10,      help="Number of parallel rollout environments")
    parser.add_argument("--n_steps",    type=int, default=2048,    help="PPO rollout horizon per environment")
    parser.add_argument("--batch_size", type=int, default=64,      help="PPO minibatch size")
    parser.add_argument("--device",     choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--death_masking", type=lambda x: str(x).lower() == 'true', default=True,
                        help="Keep finished agents masked with zero observations")
    parser.add_argument("--episodes",   type=int, default=20,      help="Episodes for random baseline")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 55 + "╗")
    print("║   Phase 2 — Multi-Agent OTA Training                  ║")
    print("╚" + "═" * 55 + "╝")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print(f"  Agents: {args.n_agents} | Blocks/agent: {args.n_blocks} | BD mode: {args.bd_mode} | Safety: {args.safety}")

    if args.mode == "random":
        results = run_random_marl(
            n_agents   = args.n_agents,
            n_blocks   = args.n_blocks,
            n_episodes = args.episodes,
            bd_mode    = args.bd_mode,
        )
        out_path = Path("results/marl_random_baseline.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    elif args.mode == "train":
        train_algorithm(
            algorithm       = args.algorithm,
            n_agents        = args.n_agents,
            n_blocks        = args.n_blocks,
            bd_mode         = args.bd_mode,
            safety          = args.safety,
            total_timesteps = args.timesteps,
            n_envs          = args.n_envs,
            n_steps         = args.n_steps,
            batch_size      = args.batch_size,
            device          = args.device,
            death_masking   = args.death_masking,
        )


if __name__ == "__main__":
    main()
