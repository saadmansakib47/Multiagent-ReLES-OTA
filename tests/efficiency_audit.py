"""
tests/efficiency_audit.py
=========================
Audit: compare inference throughput for the FP3O policy across
different firmware block sizes (8 / 16 / 24 blocks).

Metric: forward-pass throughput (samples/sec) with batch_size=32.
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl_ota.envs import MultiAgentOTAEnv
from marl_ota.policies import FP3OPolicy, make_fp3o_policy_kwargs


BATCH = 32
WARMUP_ITERS = 10
MEASURE_ITERS = 100


def _obs_to_tensor(obs_sample, batch: int, device: torch.device) -> dict:
    """Convert a gym obs-dict sample to a batched tensor dict."""
    out = {}
    for k, v in obs_sample.items():
        arr = np.array(v, dtype=np.float32)
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        out[k] = t.unsqueeze(0).expand(batch, *arr.shape).clone()
    return out


def audit_depth(n_blocks: int, device: torch.device) -> float:
    """Return forward-pass throughput in samples/sec for a given n_blocks config."""
    env = MultiAgentOTAEnv(n_agents=4, n_blocks=n_blocks)
    obs_space = env.observation_space("agent_0")
    act_space = env.action_space("agent_0")

    policy_kwargs = make_fp3o_policy_kwargs(n_blocks=n_blocks, ecu_type="generic")
    policy = FP3OPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        **policy_kwargs,
    ).to(device)
    policy.eval()

    raw_obs = obs_space.sample()
    obs_tensor = _obs_to_tensor(raw_obs, BATCH, device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            policy.forward(obs_tensor)

    # Measure
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            policy.forward(obs_tensor)
    elapsed = time.perf_counter() - t0

    throughput = (MEASURE_ITERS * BATCH) / elapsed
    env.close()
    return throughput, elapsed


def test_efficiency_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEfficiency Audit — FP3O Policy Throughput  (device={device})")
    print(f"  batch={BATCH}, warmup={WARMUP_ITERS} iters, measure={MEASURE_ITERS} iters\n")
    print(f"  {'n_blocks':>10}  {'samples/sec':>14}  {'total_time (s)':>16}")
    print("  " + "-" * 46)

    depths = [8, 16, 24]
    for depth in depths:
        throughput, elapsed = audit_depth(depth, device)
        print(f"  {depth:>10}  {throughput:>14,.0f}  {elapsed:>16.4f}")

    print("\nEfficiency audit completed.")


if __name__ == "__main__":
    test_efficiency_audit()
