# MA-ReLES-OTA

### Multi-Agent Reinforcement Learning for Coordinated Automotive OTA Updates

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Parallel_API-orange.svg)](https://pettingzoo.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A research extension of the **ReLES-OTA** framework that models automotive firmware update scheduling as a heterogeneous **Multi-Agent Reinforcement Learning (MARL)** problem under shared wireless gateway contention and physical memory constraints.

---

## Quick Start — 4 Standard Commands

Setting up and running the project is streamlined into 4 primary commands:

### 1. Run Automated Test Suite
Verify non-bijective fleet scaling, FP3O neural policy routing, and PettingZoo API compliance:
```bash
python tests/test_bijective_confound.py
```
*(You can also run `python tests/test_fp3o.py` and `python tests/test_marl_env.py` to test individual components).*

### 2. Train MARL Algorithms (CLI)
Train an algorithm (`fp3o`, `ippo`, or `mappo`) on a non-bijective scalable fleet preset:
```bash
python scripts/train.py --algorithm fp3o --fleet_preset balanced_8 --timesteps 100000
```

### 3. Launch Web Diagnostic Interface
Start the local interactive Web UI dashboard to configure experiments, monitor real-time training curves, and inspect leaderboards:
```bash
python scripts/web_ui.py
```
*Open your browser at `http://localhost:8000`.*

### 4. Run Multi-Seed Benchmark Pipeline
Execute full factorial multi-seed training and automated evaluation across independent seeds:
```bash
python scripts/run_benchmark.py --algorithm fp3o --seeds 10 --timesteps 100000
```

---

## Repository Architecture

The codebase follows the packaging standards of premier AI and cyber-physical research laboratories:

```text
MA-ReLES-OTA/
├── docs/                       # Research papers, empirical findings, and modular logs
│   ├── project_logs/
│   │   ├── phase_10_iccit_audit.md   # Active peer review audit & remediation roadmap
│   │   └── phases_01_to_09.md        # Archived historical engineering logs
│   ├── research_explanation.md       # Comprehensive domain & systems overview
│   ├── research_findings.md          # Detailed empirical analysis
│   └── interpretation.md             # Practical engineering implications
├── marl_ota/                   # Core Python package (pip install -e .)
│   ├── __init__.py                   # Top-level package exports
│   ├── config.py                     # Central hyperparameters & fleet presets
│   ├── core/
│   │   ├── ota_core.py               # Bit diffing, payload models, channel cost functions
│   │   └── network_params.json       # Vehicular CPS network simulation parameters
│   ├── envs/
│   │   ├── marl_ota_env.py           # PettingZoo Parallel multi-agent vehicular CPS environment
│   │   └── ota_env.py                # Single-agent Gymnasium baseline
│   ├── policies/
│   │   └── fp3o_policy.py            # FP3O shared-backbone & specialized actor heads
│   └── baselines/
│       └── baselines.py              # Random, Greedy, and Sequential heuristics
├── scripts/                    # Direct runnable entry points
│   ├── train.py                      # Primary MARL training CLI
│   ├── run_benchmark.py              # Multi-seed benchmark & evaluation pipeline
│   ├── web_ui.py                     # Web dashboard server backend
│   ├── train_ppo.py                  # Single-agent PPO trainer
│   ├── experiment_runner.py          # Baseline experiment runner
│   └── result_generator.py           # Benchmark figure generator
├── tests/                      # Unified automated test suite
├── tools/                      # Multi-seed tracking, evaluation, and statistical battery
├── webui/                      # Web dashboard frontend (HTML, CSS, JS)
├── results/                    # Saved models, training logs, and publication figures
├── Thesis Paper/               # LaTeX thesis & conference manuscript assets
├── network_params.json         # Root vehicular channel parameter profile
├── pyproject.toml              # Modern package manifest (PEP 621)
├── requirements.txt            # Pinned dependencies
├── PROJECT_LOG.md              # Executive milestone index
└── README.md
```

---

## Research Question & Hypotheses

Modern vehicles contain heterogeneous Electronic Control Units (ECUs) with distinct computational roles and divergent update preferences. This raises a fundamental research question:

> **Does heterogeneous MARL require explicit policy specialization, or can a fully shared policy resolve conflicting behaviors when provided with semantic information about each agent's role?**

We investigate this by comparing:
* **FP3O**: Partial parameter sharing via a shared representation backbone and role-specialized policy heads.
* **Type-Conditioned IPPO / MAPPO**: Fully shared actor-critic parameters conditioned on a semantic one-hot `ecu_type` input vector.
* **Unshared IPPO / Independent Networks**: Fully decentralized actor-critics without any parameter sharing.

### Hypotheses
* **$H_0$ (Null)**: Explicit policy specialization through type-specific action heads is necessary to resolve conflicting policies in a heterogeneous vehicular fleet.
* **$H_1$ (Alternative)**: A fully shared policy conditioned on semantic ECU type can achieve performance comparable to or better than an explicitly specialized policy architecture in the tested heterogeneous OTA environment.

---

## Vehicular Cyber-Physical Environment Dynamics

The environment models coordinated firmware updates across multiple ECUs sharing vehicle-level resources:

### 1. Firmware Update Operations
Each ECU processes incoming firmware blocks by selecting one of three operations:
* **Copy ($o=0$)**: Direct reuse of existing memory blocks (~64 B control hash). Exempt from data-plane downlink bandwidth contention.
* **Modify ($o=1$)**: Standard binary difference update requiring delta reconstruction and transmission.
* **Modify + Backup ($o=2$)**: Binary diff update with an in-place safety rollback backup buffer, requiring additional memory overhead and verification.

### 2. Coupled Gateway Channel Contention
When multiple ECUs transmit concurrently, the vehicle's shared gateway bandwidth ($B_{\mathrm{gw}} = 25\text{ Mbps}$) is partitioned across transmitting agents:
$$B_{\mathrm{eff}} = \frac{B_{\mathrm{gw}}}{\max(n_{\mathrm{transmitting}}, 1)}$$
This creates emergent pressure for temporal coordination: policies must learn to stagger heavy transmissions over time.

### 3. Vehicular Network Simulation Profile (`network_params.json`)
The physical wireless downlink is modeled through realistic vehicular CPS constraints:
* **Base Latency**: $120.0\text{ ms}$ (with stochastic truncated Gaussian jitter $[50\text{ ms}, 200\text{ ms}]$).
* **Packet Loss Rate**: $5\%$ stochastic drop rate.
* **Gateway Bandwidth**: $25.0\text{ Mbps}$ shared downlink.
* **Memory Budget Fraction**: $60\%$ of theoretical peak fleet buffer.

### 4. Autonomous Safety Shield & Metric
A supervisory safety shield enforces the modeled physical memory constraint during action selection. Rather than reporting mechanical "0% violations" (which is guaranteed by the shield's fallback), we measure the **Autonomous Shield Intervention Rate (%)** — the percentage of actions where the policy violated physical bounds and required emergency supervisor override.

### 5. Cooperative Credit Assignment
The joint fleet reward is distributed to individual ECUs using **Shapley Value** credit assignment, ensuring each agent receives reward proportional to its marginal contribution to fleet bandwidth, transmission efficiency, and memory preservation.

---

## Key Experimental Findings

### 1. Homogeneous Fleet Baseline
When ECU types do not introduce conflicting operation preferences, all algorithms converge to identical asymptotic returns (~$12.85$). Specialization provides no benefit when agents face identical objectives.

### 2. Operation-Specific Cost Heterogeneity
Uniform reward scaling failed to induce specialization because it left the ordinal ranking of operations unchanged (Copy remained universally cheapest). Structuring operation-specific multipliers (e.g. Engine penalized for standard Modify; Infotainment penalized for Modify+Backup) established genuine qualitative policy conflicts.

### 3. The Bijective Confound Discovery & Elimination
In 4-agent fleets with 4 distinct types ($N=4, K=4$), a 1:1 bijection existed between agent identity and type vector, allowing shared networks to memorize agent-specific lookups. We eliminated this confound by scaling to $N=8$ and $N=12$ with multiple interchangeable ECUs per type (`balanced_8`, `balanced_12`), observationally and mathematically verified via [`tests/test_bijective_confound.py`](file:///e:/Projects/Thesis/tests/test_bijective_confound.py).

### 4. Representation Interference under Channel Contention
Under shared gateway bottleneck contention ($B_{\mathrm{gw}} = 25\text{ Mbps}$):
* **Shared-Actor Baselines Degrade**: Adding type conditioning to standard shared IPPO/MAPPO reduced return (IPPO dropped by $27.37$ points; MAPPO dropped by $52.18$ points) due to conflicting gradient updates in a single shared actor.
* **FP3O Resolves Gradient Conflict**: FP3O demonstrated a massive positive gain of **+111.44 points** ($-213.64 \rightarrow -102.20$) upon enabling type conditioning, because its specialized actor heads isolate gradients between ECU classes.
* **Autonomous Constraint Compliance**: FP3O achieved an emergency shield intervention rate of **only 13.82%**, compared to **32.66%** for IPPO Blind — an absolute **57.7% reduction** in emergency safety overrides.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/saadmansakib47/Multiagent-ReLES-OTA.git
cd Multiagent-ReLES-OTA

# Install dependencies
pip install -r requirements.txt

# Install marl_ota package in editable development mode
pip install -e .
```

---

## Contributors & Roles

- **Saadman Sakib**: Lead Architect, policy routing, environment design, MARL integration, training infrastructure
- **Mohtasim Dipto**: Environment development, death masking, parallel vectorization
- **Mahin Islam**: Statistical analysis, Welch's t-tests, confidence intervals, result visualization

---

<p align="center">
  <i>Undergraduate Thesis Research — Software & Systems Lab</i>
</p>
