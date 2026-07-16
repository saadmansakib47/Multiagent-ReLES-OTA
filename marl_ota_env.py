"""
marl_ota_env.py — Multi-Agent OTA Update Environment
======================================================
Phase 2 implementation of the ReLES-OTA framework as a Multi-Agent
Reinforcement Learning environment following the PettingZoo Parallel API.

Architecture
------------
- N concurrent ECU agents, each managing its own firmware block list
- Heterogeneous fleet: each agent has independent old/new firmware blocks
- Death masking: finished agents return zero-vector obs (agent_id only)
  → prevents distribution shift in the centralized critic (MAPPO)
- Stochastic latency: Truncated Gaussian [50–200ms] per step per agent
  → forces latency-robust policies (RSR-RSMARL approach)

References
----------
- Original ReLES-OTA paper (Bhattacherjee et al., 2025)
- RSR-RSMARL (arXiv:2506.00982) — robust MARL + stochastic state reps
- MAPPO-PIS (arXiv:2408.06656) — death masking for variable-length episodes
- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
"""

import numpy as np
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
from pettingzoo import ParallelEnv
# pyrefly: ignore [missing-import]
from pettingzoo.utils import parallel_to_aec

import hashlib
import time
from typing import Dict, List, Optional, Tuple

from ota_core import (
    estimate_delta_size,
    calculate_tx_cost,
    load_bd_params,
)

# Import config (provides all tunable constants in one place)
try:
    from config import SAFETY_CFG, BD_CFG
except ImportError:
    # Fallback defaults if run outside the project root
    SAFETY_CFG = {"enabled": True, "memory_budget_frac": 0.85}
    BD_CFG = {"monsoon_jitter_ms": 250, "monsoon_multiplier_low": 1.5,
              "monsoon_multiplier_high": 3.0}


# ══════════════════════════════════════════════════════════════
#  Multi-Agent OTA Environment
# ══════════════════════════════════════════════════════════════

class MultiAgentOTAEnv(ParallelEnv):
    """
    Multi-Agent OTA Firmware Update Environment (PettingZoo Parallel API).

    Each agent represents one ECU in a vehicle fleet simultaneously
    receiving an over-the-air firmware update.

    Parameters
    ----------
    n_agents   : number of concurrent ECU agents (vehicles in fleet)
    n_blocks   : firmware blocks per agent (each agent manages this many)
    block_size : bytes per block
    bd_mode    : enable Bangladesh-specific network parameters
    bd_params_path : path to bd_params.json
    stochastic_latency : sample latency from Truncated Gaussian each step
                         (True for Phase 2 MARL; False for Phase 1 compat)
    max_steps  : hard episode limit per agent (safety cap)
    ecu_types  : optional dict mapping agent_id → ECU type string
                 (e.g. {"ecu_0": "engine", "ecu_1": "braking"})
                 Used by FP3O specialized heads to select the correct
                 action head for each heterogeneous ECU.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "marl_ota_v0",
    }

    # ──────────────────────────────────────────────────────────
    #  Init
    # ──────────────────────────────────────────────────────────

    def __init__(
        self,
        n_agents: int = 4,
        n_blocks: int = 24,
        block_size: int = 4096,
        bd_mode: bool = False,
        bd_params_path: str = "bd_params.json",
        stochastic_latency: bool = True,
        max_steps: int = None,
        render_mode: Optional[str] = None,
        ecu_types: Optional[Dict[str, str]] = None,
        safety_shield: bool = True,
    ):
        super().__init__()

        self.safety_shield      = safety_shield
        self.render_mode        = render_mode
        self.n_agents_total     = n_agents
        self.n_blocks           = n_blocks
        self.block_size         = block_size
        self.bd_mode            = bd_mode
        self.stochastic_latency = stochastic_latency
        self.max_steps          = max_steps if max_steps is not None else n_blocks * 3

        # ECU type labels for FP3O heterogeneous action heads.
        # Valid types: "engine", "braking", "infotainment", "generic"
        _default_types = ["engine", "braking", "infotainment", "generic"]
        self.ecu_types: Dict[str, str] = {
            f"ecu_{i}": (
                ecu_types.get(f"ecu_{i}", _default_types[i % len(_default_types)])
                if ecu_types else _default_types[i % len(_default_types)]
            )
            for i in range(n_agents)
        }

        # Load network parameters
        # load_bd_params() always merges with safe defaults — safe to call regardless of bd_mode
        self.bd_params = load_bd_params(bd_params_path if bd_mode else "nonexistent_path_to_use_defaults")

        # Agent IDs (stable — never mutated after __init__)
        self.possible_agents: List[str] = [f"ecu_{i}" for i in range(n_agents)]

        # Spaces defined here — actual per-episode reset in reset()
        self._obs_space  = self._build_obs_space()
        self._act_space  = spaces.MultiDiscrete([n_blocks, 3])

        # ── Per-agent firmware state (regenerated on reset) ──
        self._old_blocks:       Dict[str, List[bytes]]  = {}
        self._new_blocks:       Dict[str, List[bytes]]  = {}
        self._similarity_bias:  Dict[str, np.ndarray]   = {}

        # ── Per-agent episode state ──
        self.masks:         Dict[str, np.ndarray] = {}
        self.cum_enc_cost:  Dict[str, float]      = {}
        self.cum_tx_cost:   Dict[str, float]      = {}
        self.cum_memory:    Dict[str, float]      = {}
        self.current_step:  Dict[str, int]        = {}

        # PettingZoo required: live agent list (shrinks as agents die)
        self.agents: List[str] = []

        # Death-masking: track which agents have finished
        self.terminations:  Dict[str, bool] = {}
        self.truncations:   Dict[str, bool] = {}

        self._episode_start = 0.0

    # ──────────────────────────────────────────────────────────
    #  Observation / Action spaces (PettingZoo API)
    # ──────────────────────────────────────────────────────────

    def _build_obs_space(self) -> spaces.Dict:
        """
        Observation per agent:
          mask             : MultiBinary(n_blocks)  — which blocks remain
          cum_encoding_cost: Box(1,)  — accumulated encoding cost so far
          cum_tx_cost      : Box(1,)  — accumulated transmission cost
          memory_used      : Box(1,)  — fraction of memory budget used [0,1]
          step             : Box(1,)  — current step count
          agent_id         : Box(n_agents,)  — one-hot agent identifier
                             (preserved even in zero-vector death-mask obs)
          state            : Box(state_dim,) — global state representation
        """
        state_dim = self.n_agents_total * (self.n_blocks + 4)
        return spaces.Dict({
            "mask":             spaces.MultiBinary(self.n_blocks),
            "cum_encoding_cost": spaces.Box(0, np.inf, (1,), dtype=np.float32),
            "cum_tx_cost":       spaces.Box(0, np.inf, (1,), dtype=np.float32),
            "memory_used":       spaces.Box(0, 1.0,   (1,), dtype=np.float32),
            "step":              spaces.Box(0, self.n_blocks + 10, (1,), dtype=np.int32),
            "agent_id":          spaces.Box(0, 1.0, (self.n_agents_total,), dtype=np.float32),
            "state":             spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        })

    def _get_global_state(self) -> np.ndarray:
        state_parts = []
        mem_budget = self.bd_params.get("memory_budget_fraction", 1.0)
        mem_cap    = self.n_blocks * self.block_size * 2.0 * mem_budget

        for a in self.possible_agents:
            if self.terminations[a] or self.truncations[a]:
                state_parts.append(np.zeros(self.n_blocks + 4, dtype=np.float32))
            else:
                m = self.masks[a].astype(np.float32)
                e = np.array([self.cum_enc_cost[a]], dtype=np.float32)
                t = np.array([self.cum_tx_cost[a]], dtype=np.float32)
                mem = np.array([min(self.cum_memory[a] / max(mem_cap, 1.0), 1.0)], dtype=np.float32)
                s = np.array([self.current_step[a]], dtype=np.float32)
                state_parts.extend([m, e, t, mem, s])
        return np.concatenate(state_parts)

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        return self._act_space

    # ──────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset all agents for a new episode."""

        if seed is not None:
            np.random.seed(seed)

        # Restore full agent list
        self.agents = list(self.possible_agents)

        # Generate fresh firmware blocks for each agent
        for agent in self.possible_agents:
            self._old_blocks[agent]      = [np.random.bytes(self.block_size) for _ in range(self.n_blocks)]
            self._new_blocks[agent]      = [np.random.bytes(self.block_size) for _ in range(self.n_blocks)]
            self._similarity_bias[agent] = np.random.uniform(0.4, 0.9, self.n_blocks)

        # Reset per-agent episode state
        for agent in self.possible_agents:
            self.masks[agent]        = np.ones(self.n_blocks, dtype=np.int32)
            self.cum_enc_cost[agent] = 0.0
            self.cum_tx_cost[agent]  = 0.0
            self.cum_memory[agent]   = 0.0
            self.current_step[agent] = 0
            self.terminations[agent] = False
            self.truncations[agent]  = False

        self._episode_start = time.time()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos        = {agent: {} for agent in self.agents}

        return observations, infos

    # ──────────────────────────────────────────────────────────
    #  Observation builder (with death masking)
    # ──────────────────────────────────────────────────────────

    def _get_obs(self, agent: str) -> dict:
        """
        Build the observation dict for `agent`.

        Death masking (RSR-RSMARL / MAPPO-PIS):
        ----------------------------------------
        When an agent has terminated (all its blocks processed), we do NOT
        remove it from the environment. Instead, we return a ZERO-VECTOR for
        all fields EXCEPT `agent_id`.

        This provides two benefits:
        1. Prevents distribution shift in the centralized critic — the critic
           always sees a fixed-size observation tensor.
        2. The critic can still identify which agents are "dead" via the
           zero-pattern, and learn to predict the post-death average reward.
        """
        # One-hot agent identity vector (always present)
        agent_idx  = self.possible_agents.index(agent)
        agent_id_vec = np.zeros(self.n_agents_total, dtype=np.float32)
        agent_id_vec[agent_idx] = 1.0
        
        state = self._get_global_state()

        if self.terminations[agent] or self.truncations[agent]:
            # ── DEATH MASK: zero everything except agent_id ──
            return {
                "mask":              np.zeros(self.n_blocks, dtype=np.int8),
                "cum_encoding_cost": np.zeros(1, dtype=np.float32),
                "cum_tx_cost":       np.zeros(1, dtype=np.float32),
                "memory_used":       np.zeros(1, dtype=np.float32),
                "step":              np.zeros(1, dtype=np.int32),
                "agent_id":          agent_id_vec,
                "state":             state,
            }

        # ── Normal observation ──
        mem_budget = self.bd_params.get("memory_budget_fraction", 1.0)
        mem_cap    = self.n_blocks * self.block_size * 2.0 * mem_budget  # approx max memory

        return {
            "mask":              self.masks[agent].copy(),
            "cum_encoding_cost": np.array([self.cum_enc_cost[agent]], dtype=np.float32),
            "cum_tx_cost":       np.array([self.cum_tx_cost[agent]],  dtype=np.float32),
            "memory_used":       np.array(
                                     [min(self.cum_memory[agent] / max(mem_cap, 1.0), 1.0)],
                                     dtype=np.float32
                                 ),
            "step":              np.array([self.current_step[agent]], dtype=np.int32),
            "agent_id":          agent_id_vec,
            "state":             state,
        }

    # ──────────────────────────────────────────────────────────
    #  Step
    # ──────────────────────────────────────────────────────────

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step for ALL active agents simultaneously.

        Parameters
        ----------
        actions : dict mapping agent_id → action array [block_idx, operation]

        Returns
        -------
        observations  : {agent: obs_dict}
        rewards       : {agent: float}
        terminations  : {agent: bool}  — True if agent finished ALL blocks
        truncations   : {agent: bool}  — True if hard step limit hit
        infos         : {agent: dict}
        """

        rewards:     Dict[str, float] = {}
        infos:       Dict[str, dict]  = {}

        # ── Safety: global episode step-budget cap (avoid infinite loops) ──
        if all(self.terminations[a] or self.truncations[a] for a in self.possible_agents):
            for agent in self.possible_agents:
                if agent not in rewards:
                    rewards[agent] = 0.0
                    infos[agent] = {"done": True}
            self.agents = []
            obs = {a: self._get_obs(a) for a in self.possible_agents}
            return obs, rewards, dict(self.terminations), dict(self.truncations), infos

        # ── Compute valid actions and proposed overhead ──
        valid_actions = {} 
        proposed_overhead = {}
        for agent in self.possible_agents:
            if self.terminations[agent] or self.truncations[agent]:
                continue
                
            action = actions.get(agent)
            if action is None:
                continue
                
            block_idx, operation = int(action[0]), int(action[1])
            if block_idx >= self.n_blocks or self.masks[agent][block_idx] == 0:
                continue
                
            delta_size = estimate_delta_size(block_idx, operation, self._similarity_bias[agent], self.block_size)
            encoding_cost = delta_size * (1.25 if operation == 2 else 1.0)
            tx_cost = calculate_tx_cost(delta_size, self.bd_params, stochastic=self.stochastic_latency)
            if self.bd_mode:
                tx_cost *= self.bd_params.get("monsoon_multiplier", 1.0)
            overhead = delta_size * (2.6 if operation == 2 else 1.9)
            
            valid_actions[agent] = (block_idx, operation, delta_size, overhead, encoding_cost, tx_cost)
            proposed_overhead[agent] = overhead

        current_M = sum(self.cum_memory[a] for a in self.possible_agents)
        mem_budget = self.bd_params.get("memory_budget_fraction", 1.0)
        # Use safety threshold from config; alpha=0.5 scales the safety margin
        safety_frac = SAFETY_CFG["memory_budget_frac"]  # e.g. 0.85
        alpha = 0.5
        M_limit = self.n_agents_total * self.n_blocks * self.block_size * 2.0 * mem_budget * safety_frac

        def _eval_coalition(S):
            S_overhead = {a: proposed_overhead[a] for a in S}
            S_total_delta = sum(S_overhead.values())
            crashed = False
            shielded_in_S = set()
            if S_total_delta > alpha * (M_limit - current_M):
                if self.safety_shield:
                    sorted_agents = sorted(S_overhead.keys(), key=lambda a: S_overhead[a], reverse=True)
                    rem_delta = S_total_delta
                    for a in sorted_agents:
                        if rem_delta <= max(0, alpha * (M_limit - current_M)):
                            break
                        rem_delta -= S_overhead[a]
                        shielded_in_S.add(a)
                else:
                    if current_M + S_total_delta > M_limit:
                        crashed = True
            
            if crashed:
                # Cap the maximum crash penalty to -100.0 to prevent gradient explosions
                return -100.0
                
            val = 0.0
            for a in S:
                if a not in shielded_in_S:
                    _, _, delta_size, overhead, enc_cost, tx_c = valid_actions[a]
                    # Scale costs down to avoid reward magnitudes of thousands
                    val -= (enc_cost + tx_c) * 0.001 + overhead * 0.0003
                    if np.sum(self.masks[a]) == 1:
                        # Completion bonus: small fixed reward for finishing all blocks
                        val += 10.0
            return val

        active_agents = [a for a in self.possible_agents if a in valid_actions]
        shapley_rewards = {a: 0.0 for a in self.possible_agents}
        
        if len(active_agents) > 0:
            import itertools
            import random
            
            if len(active_agents) <= 4:
                perms = list(itertools.permutations(active_agents))
            else:
                perms = [tuple(random.sample(active_agents, len(active_agents))) for _ in range(10)]
                
            mc_sums = {a: 0.0 for a in active_agents}
            for p in perms:
                S = set()
                v_prev = 0.0
                for a in p:
                    S.add(a)
                    v_curr = _eval_coalition(S)
                    mc_sums[a] += (v_curr - v_prev)
                    v_prev = v_curr
                    
            for a in active_agents:
                shapley_rewards[a] = mc_sums[a] / len(perms)

        # ── Apply Grand Coalition ──
        S_all = set(active_agents)
        S_total_delta = sum(proposed_overhead[a] for a in S_all)
        crashed = False
        shielded = set()
        if S_total_delta > alpha * (M_limit - current_M):
            if self.safety_shield:
                sorted_agents = sorted(active_agents, key=lambda a: proposed_overhead[a], reverse=True)
                rem_delta = S_total_delta
                for a in sorted_agents:
                    if rem_delta <= max(0, alpha * (M_limit - current_M)):
                        break
                    rem_delta -= proposed_overhead[a]
                    shielded.add(a)
            else:
                if current_M + S_total_delta > M_limit:
                    crashed = True
                    
        if crashed:
            for agent in self.possible_agents:
                self.truncations[agent] = True
                # Scaled-down crash penalty to prevent statistical explosions
                rewards[agent] = -20.0
                infos[agent] = {"crash": True}
            self.agents = []
            obs = {a: self._get_obs(a) for a in self.possible_agents}
            return obs, rewards, dict(self.terminations), dict(self.truncations), infos

        for agent in self.possible_agents:
            if self.terminations[agent] or self.truncations[agent]:
                rewards[agent] = 0.0
                infos[agent] = {"dead": True}
                continue
                
            action = actions.get(agent)
            if action is None:
                rewards[agent] = -5.0
                infos[agent] = {"missing_action": True}
                continue
                
            block_idx, operation = int(action[0]), int(action[1])

            # Per-agent step-budget truncation: if an agent has exhausted its
            # step budget without finishing, truncate it with a penalty.
            if self.current_step[agent] >= self.max_steps:
                rewards[agent] = -15.0
                infos[agent] = {"step_limit": True}
                self.truncations[agent] = True
                self.current_step[agent] += 1
                continue

            if block_idx >= self.n_blocks or self.masks[agent][block_idx] == 0:
                # Invalid action penalty — must be clearly worse than taking
                # any valid action (cheapest valid step ~-0.66, most expensive ~-6.54), so use -15.0
                rewards[agent] = -15.0
                infos[agent] = {"invalid_action": True, "block_idx": block_idx}
                self.current_step[agent] += 1
                continue
                
            if agent in shielded:
                rewards[agent] = 0.0
                infos[agent] = {"shielded": True}
                self.current_step[agent] += 1
                continue
                
            # Execute action
            _, _, delta_size, overhead, enc_cost, tx_cost = valid_actions[agent]
            
            self.cum_enc_cost[agent] += enc_cost
            self.cum_tx_cost[agent] += tx_cost
            self.cum_memory[agent] += overhead
            self.masks[agent][block_idx] = 0
            self.current_step[agent] += 1
            
            done = bool(np.all(self.masks[agent] == 0))
            if done:
                self.terminations[agent] = True
                
            # Small progress bonus: valid action taken this step.
            # Encourages agents to take valid actions rather than explore invalid ones.
            rewards[agent] = float(shapley_rewards[agent]) + 0.5
            infos[agent] = {
                "payload_bytes": self.cum_enc_cost[agent] + self.cum_tx_cost[agent],
                "memory_used": self.cum_memory[agent],
                "blocks_processed": int(np.sum(self.masks[agent] == 0)),
                "done": done,
            }

        # ── Observations for ALL agents (including dead → zero-vector) ──
        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}

        # ── Update live agent list (PettingZoo convention) ──
        # Remove agents that are done OR truncated
        self.agents = [
            a for a in self.possible_agents
            if not self.terminations[a] and not self.truncations[a]
        ]

        return (
            observations,
            rewards,
            dict(self.terminations),
            dict(self.truncations),
            infos,
        )

    # ──────────────────────────────────────────────────────────
    #  Render
    # ──────────────────────────────────────────────────────────

    def render(self):
        """Human-readable per-agent status."""
        print("\n" + "─" * 65)
        print(f"  MultiAgentOTAEnv  |  Step summary  |  {len(self.agents)} agents alive")
        print("─" * 65)
        for agent in self.possible_agents:
            status = "✓ DONE" if self.terminations[agent] else \
                     "✗ TRUNC" if self.truncations[agent] else "● ALIVE"
            remaining = int(np.sum(self.masks.get(agent, np.zeros(self.n_blocks)) == 1))
            payload   = self.cum_enc_cost.get(agent, 0) + self.cum_tx_cost.get(agent, 0)
            mem       = self.cum_memory.get(agent, 0)
            step      = self.current_step.get(agent, 0)
            print(
                f"  {agent:8s}  [{status:7s}]  "
                f"step={step:3d}/{self.n_blocks}  "
                f"remaining={remaining:3d}  "
                f"payload={payload:8.1f}  mem={mem:8.1f}"
            )
        print("─" * 65)

    def action_masks(self, agent: str) -> np.ndarray:
        """
        Return a boolean mask of length ``n_blocks + 3`` for masking-aware
        policy distributions.

        Layout
        ------
        - Indices  0 … n_blocks-1 : block dimension (1=valid, 0=already done)
        - Indices  n_blocks … n_blocks+2 : operation dimension (always valid)

        Dead-agent note
        ---------------
        Terminated / truncated agents return **all-True** (dummy valid mask).
        ``step()`` discards their sampled action and returns ``rewards=0.0``
        anyway, so the distribution content is irrelevant.  An all-False mask
        would cause NaN / errors in MaskableMultiCategoricalDistribution.
        """
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            # Dummy all-valid mask — action is discarded by step() regardless
            return np.ones(self.n_blocks + 3, dtype=bool)
        block_mask = self.masks[agent].astype(bool)     # shape (n_blocks,)
        op_mask    = np.ones(3, dtype=bool)             # operations always valid
        return np.concatenate([block_mask, op_mask])

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════
#  AEC wrapper (for algorithms that require turn-based API)
# ══════════════════════════════════════════════════════════════

def make_marl_ota_aec(**kwargs):
    """Wrap MultiAgentOTAEnv in PettingZoo AEC API for compatibility."""
    env = MultiAgentOTAEnv(**kwargs)
    env = parallel_to_aec(env)
    return env


# ══════════════════════════════════════════════════════════════
#  Quick smoke-test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke-testing MultiAgentOTAEnv ...")

    env = MultiAgentOTAEnv(n_agents=4, n_blocks=8, bd_mode=True, stochastic_latency=True)
    obs, infos = env.reset(seed=42)
    print(f"Reset OK — {len(env.agents)} agents, obs keys: {list(obs['ecu_0'].keys())}")

    step_count = 0
    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        step_count += 1
        if step_count % 4 == 0:
            env.render()

    print(f"\nEpisode finished in {step_count} steps.")
    print("All agents terminated:", all(env.terminations.values()))
