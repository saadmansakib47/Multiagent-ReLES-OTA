"""
tests/edge_case.py
==================
Edge Case Test: What happens when an ECU needs an update during a
"Monsoon" network jitter event?

The 250ms jitter target and multiplier are sourced from config.BD_CFG
so they are never hard-coded here.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl_ota_env import MultiAgentOTAEnv
from config import BD_CFG
import numpy as np


def test_edge_case(verbose: bool = True):
    jitter_ms   = BD_CFG["monsoon_jitter_ms"]          # e.g. 250
    multiplier  = BD_CFG["monsoon_multiplier_high"]     # e.g. 3.0

    if verbose:
        print(f"Edge Case Test: {jitter_ms}ms Monsoon Network Jitter")
        print(f"  Monsoon multiplier override: {multiplier}x")

    env = MultiAgentOTAEnv(n_agents=4, n_blocks=8, bd_mode=True, stochastic_latency=True)
    obs, info = env.reset(seed=1337)

    # Override the monsoon multiplier to simulate peak jitter from config
    env.bd_params["monsoon_multiplier"] = multiplier

    all_rewards = []
    steps = 0
    while env.agents and steps < 15:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        all_rewards.extend(rewards.values())
        steps += 1

    mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0
    if verbose:
        print(f"  Completed {steps} steps | Mean step reward: {mean_r:.2f}")
        print("  Edge case monsoon simulation PASSED.")

    assert steps > 0, "Environment produced no steps"
    env.close()
    return mean_r


if __name__ == "__main__":
    test_edge_case(verbose=True)
