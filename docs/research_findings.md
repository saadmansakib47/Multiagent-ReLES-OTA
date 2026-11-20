# Research Findings — ReLES-OTA Multiagent MARL Experiment

*This document captures key experimental findings in detail for use when writing the thesis.*

---

## Finding 1: Homogeneous Environment Makes FP3O's Specialization Unnecessary

**Phase**: Phase 4, Step 1  
**Training Runs**: #17 (FP3O), #18 (IPPO), #19 (MAPPO) — all 2M timesteps, 2 seeds  
**Git Checkpoint**: Prior to any heterogeneity changes

### Results
| Algorithm | Mean Return | CI_95 | Mean Payload Cost | Shield Rate |
| :--- | :---: | :---: | :---: | :---: |
| IPPO | **12.85** | 0.01 | 38,335 | 0.0 |
| MAPPO | **12.85** | ~0.01 | ~38,500 | 0.0 |
| FP3O | 11.44 | 1.21 | 39,250 | 0.0 |

### Analysis
All three algorithms converged to near-identical mean returns (~12–13), with IPPO and MAPPO slightly edging out FP3O. This is a direct consequence of the **environment being functionally homogeneous**: the ECU type labels (`engine`, `braking`, `infotainment`, `generic`) were stored in `self.ecu_types` and used to route FP3O's policy heads, but had **zero effect on the environment's reward function, physics model, or constraints**. Every ECU agent faced the identical optimization problem: clear blocks using the cheapest operation (Copy).

In such a homogeneous setting, the "Surprising Effectiveness of IPPO" phenomenon (Yu et al., 2022) applies directly:
- IPPO's single shared head receives **4× more gradient updates per step** because all 4 agents' experiences pool into one head.
- FP3O's 4 specialized heads each receive only **25% of the data**, learning 4× slower, with no performance benefit from the specialization.

**Thesis Implication**: This finding should be reported as a baseline negative result — FP3O's overhead is unjustified in a homogeneous fleet. The thesis then motivates heterogeneity as the natural next step.

**Reference**: Yu, C. et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *arXiv:2103.01955*.

---

## Finding 2: Uniform Cost Multipliers Fail to Create Qualitative Policy Differences

**Phase**: Phase 4, Steps 2–3  
**Training Runs**: #20 (FP3O, uniform multipliers), #21 (IPPO, uniform multipliers) — 2M timesteps, 2 seeds

### Configuration
```python
cost_multipliers = {"engine": 1.5, "braking": 1.2, "infotainment": 0.7, "generic": 1.0}
```

### Results
| Algorithm | Mean Return |
| :--- | :---: |
| FP3O | -12.00 |
| IPPO | 9.15 |

### Analysis
Applying a uniform cost multiplier (e.g. 1.5× for all operations for engine, 0.7× for infotainment) failed because it did not change the **ordinal ranking** of operations. For every ECU type, the cost ordering remained `Copy < Modify < MB`. The qualitatively optimal policy — always prefer `Copy` — was identical for all ECU types.

Additionally, the large gradient magnitude disparity (1.5× for engine vs 0.7× for infotainment) created **conflicting gradient magnitudes flowing through the shared backbone**, destabilizing the common feature representation.

**Thesis Implication**: Reward-scale heterogeneity alone is insufficient to justify specialized architectures. Heterogeneity must be **structural** — it must change which action is optimal, not just how costly any action is.

---

## Finding 3: `agent_id` Observation Enables IPPO to Silently Replicate Heterogeneous Policies

**Phase**: Phase 4, Step 4  
**Training Runs**: #22 (FP3O, per-op multipliers), #23 (IPPO, per-op multipliers) — 2M timesteps, 2 seeds  
**Git Checkpoint**: `38ac072` — per-op multipliers active, `agent_id` still in observation

### Configuration
```python
per_op_multipliers = {
    "engine":       [1.0, 1.8, 0.6],   # prefer MB (op 2); penalise Modify
    "braking":      [1.0, 1.2, 0.8],
    "infotainment": [0.6, 1.0, 1.5],   # prefer Copy (op 0); penalise MB
    "generic":      [1.0, 1.0, 1.0],
}
```

### Results
| Algorithm | Mean Return | CI_95 |
| :--- | :---: | :---: |
| IPPO | **16.71** | 0.006 |
| FP3O | 8.68 | 31.5 |

### Analysis — The `agent_id` Confounder
Despite per-operation multipliers creating genuinely contradictory optimal policies (engine → prefer MB; infotainment → prefer Copy), IPPO *still* dominated. The root cause is the **`agent_id` field in the observation**.

Every agent's observation included `agent_id`, a one-hot vector of shape `(n_agents,)` indicating the specific agent index (e.g., `[1, 0, 0, 0]` for `ecu_0`). This means:

1. **IPPO's shared head has full access to agent identity information** and can learn completely different policy logits per agent ID via standard neural conditioning. The "shared" head effectively becomes a heterogeneous lookup table with 4 entries.
2. **IPPO still benefits from 4× more training data per gradient step** because all agents pool experience into one set of shared weights.
3. IPPO therefore has **the best of both worlds**: heterogeneous behavior via `agent_id` conditioning AND sample efficiency via shared weights with 4× data.

FP3O's specialized heads, by contrast, only get 1/4 of the data and their shared backbone still receives mixed gradients from all ECU types simultaneously.

### Key Thesis Insight
This is a subtle but important methodological confound in MARL evaluation. Including a **per-agent numeric identifier** in observations effectively converts any shared-policy baseline into a fully agent-conditioned policy, collapsing the distinction between "shared" and "specialized" architectures. This is related to the observation aliasing problem in partially observable MARL.

The experiment demonstrates that in a heterogeneous MARL benchmark, the observation space design is as important as the algorithm design. If a baseline algorithm can observe *which specific agent it is* (not just *what type of agent it is*), it can memorize agent-specific strategies and systematically outperform architectures with genuine type-based specialization.

**Methodological Contribution**: This finding is well worth discussing explicitly in the experimental section of the thesis as a lesson in fair MARL benchmark design.

---

## Planned Fix: `ecu_type` One-Hot Replacing `agent_id` (Phase 4, Step 5)

Replace `agent_id` (per-agent numeric index, size = n_agents) with `ecu_type` (per-type semantic identifier, size = 4):

```python
# Old observation field  
"agent_id": spaces.Box(0, 1.0, (n_agents_total,), dtype=np.float32)

# New observation field
"ecu_type": spaces.Box(0, 1.0, (4,), dtype=np.float32)  # [engine, braking, infotainment, generic]
```

Under this regime:
- **IPPO's head** can still condition on ECU category (4 semantic values), but *cannot* distinguish between two different agents of the same type. It can no longer memorize per-agent strategies.
- **FP3O's specialized heads** still have dedicated weights per ECU category — their structural inductive bias is preserved and now provides a genuine advantage.
- **FP3O's head routing** in `fp3o_policy.py` reads `ecu_type` instead of `agent_id` for the ECU head index lookup.

This makes the comparison architecturally fair and is the correct experimental setup to validate the FP3O paper's claims in a heterogeneous vehicle firmware OTA context.

### Files to Change
- `marl_ota_env.py`: Replace `agent_id` with `ecu_type` in `_build_obs_space()` and `_get_obs()`
- `fp3o_policy.py`: Update `_build_action_mask()` and `_get_action_dist_from_latent()` to read `ecu_type` instead of `agent_id`
