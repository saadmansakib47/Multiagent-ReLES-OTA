# MA-ReLES-OTA

### Multi-Agent Reinforcement Learning for Coordinated Automotive OTA Updates

> A high-performance research extension of the [ReLES-OTA (2025)](https://github.com/) framework — scaling single-ECU firmware optimization into a full **Multi-Agent RL** coordination system for resource-constrained vehicular networks.

---

## Overview

Modern vehicles are **"Data Centers on Wheels"** — containing dozens of heterogeneous ECUs (Engine, Braking, Infotainment) that must often be updated simultaneously over constrained mobile networks.

**The Combinatorial Optimization Gap:**

> How can multiple agents coordinate block-update sequences when competing for a shared, constrained vehicle memory pool under adverse network jitter?

This project addresses that gap using state-of-the-art MARL coordination algorithms under the **Centralized Training, Decentralized Execution (CTDE)** paradigm.

---

## 🔬 Research Phases

### Phase 1 — Bangladesh-Specific Benchmarking _(Completed)_

Replicated and localized the single-agent PPO framework for the Bangladeshi network environment.

| Parameter        | Value     |
| ---------------- | --------- |
| Injected Latency | 120–250ms |
| Packet Loss      | 3–8%      |
| Memory Budget    | 60%       |

**Results vs. sequential baseline:**

- 📉 **21% reduction** in payload cost
- 📉 **22% lower** memory overhead

---

### Phase 2 — Multi-Agent Scaling & Coordination _(In Progress)_

Refactoring the environment to support the CTDE paradigm for **N concurrent ECU agents**, with full Shapley-based credit assignment and CBF safety shields.

---

## Architectural Pillars

| Pillar          | Paper      | Implementation                                                                                            |
| --------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| **Stability**   | MAPPO      | Centralized Critic with Death Masking (`0_a`) to stabilize training as ECUs finish at different times     |
| **Versatility** | FP3O       | Partial Parameter Sharing — shared backbone for firmware encoding + unique action heads per hardware type |
| **Fairness**    | MARL-CC    | Monte Carlo Shapley Values for credit assignment based on marginal contribution to team efficiency        |
| **Safety**      | RSR-RSMARL | CBF-based Safety Shield preventing RL actions from exceeding hardware register or RAM limits              |

---

## Formal Hypothesis

**H₀ (Null):** Coordinated MARL provides no significant improvement over independent agents under stochastic network jitter.

**H₁ (Alternative):** A coordinated framework using Shapley-based credit assignment and stochastic latency modeling significantly reduces update completion time variance and prevents system-wide memory overflows.

_Target validation: Phase 2 with 10-seed statistical testing via W&B._

---

## Tech Stack

**Hardware**

- CPU: AMD Ryzen 5 5600 (6-Core, 12T) @ 3.50 GHz
- GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
- RAM: 12GB

**Software**

```
Python 3.10+
PyTorch
Gymnasium
PettingZoo
Stable Baselines 3        # Extended for MAPPO / FP3O
Weights & Biases (W&B)    # 10-seed statistical validation [Phase 2]
```

---

## Project Structure

```
MA-ReLES-OTA/
│
├── ota_env.py          # Multi-agent env — N-agent dicts + Stochastic Latency Stubs (50–200ms)
├── train_ppo.py        # Training pipeline — Value Normalization + parallel rollouts (n_envs=10)
│
└── results/            # [Phase 2] Shaded learning curves + ablation studies
                        #           (Shapley / Safety module removal)
```

---

## 📊 Results _(Phase 2 — Upcoming)_

Planned outputs:

- Shaded learning curves across 10 seeds
- Ablation study: effect of removing Shapley credit assignment
- Ablation study: effect of removing CBF Safety Shield
- Completion time variance under stochastic jitter

---

## Contributors

| Name               | Role               | Focus                                  |
| ------------------ | ------------------ | -------------------------------------- |
| **Saadman Sakib**  | Architect          | Shared Backbone & Pipeline Integration |
| **Mohtasim Dipto** | Environment Dev    | N-Agent Scaling & Death Masking        |
| **Mahin Islam**    | Documentation Lead | Statistical Rigor & Hypothesis Testing |

---

## 📎 Citation

If you build on this work, please cite the original ReLES-OTA framework and this extension. BibTeX entries will be added upon publication.

---

<p align="center">
  <i>Undergraduate Thesis Research — Software & Systems Lab</i>
</p>
