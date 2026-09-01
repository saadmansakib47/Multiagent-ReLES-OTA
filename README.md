# MA-ReLES-OTA

### Multi-Agent Reinforcement Learning for Coordinated Automotive OTA Updates

A research extension of the **ReLES-OTA** framework that models firmware update scheduling as a heterogeneous **Multi-Agent Reinforcement Learning (MARL)** problem for resource-constrained vehicular networks.

---

## Research Question

Modern vehicles contain heterogeneous Electronic Control Units (ECUs) with different computational roles and different preferences over firmware update operations. This raises a fundamental question:

> **Does heterogeneous MARL require explicit policy specialization, or can a fully shared policy resolve conflicting behaviors when provided with semantic information about each agent's role?**

We investigate this question by comparing:

- **FP3O**: partial parameter sharing with a shared backbone and specialized policy heads for different ECU types.
- **IPPO/MAPPO**: fully shared policy parameters, with experiments using semantic `ecu_type` conditioning.
- **Type-Conditioned IPPO**: a shared actor-critic policy receiving a one-hot `ecu_type` vector as part of its input.

The key distinction is between **parameter-based specialization** and **input-based specialization**.

---

## Hypotheses

### $H_0$ (Null)

Explicit policy specialization through type-specific action heads is necessary to resolve conflicting policies in a heterogeneous fleet.

### $H_1$ (Alternative)

A fully shared policy conditioned on semantic ECU type can achieve performance comparable to or better than an explicitly specialized policy architecture in the tested heterogeneous OTA environment.

The current study does **not** claim that either architecture is universally superior. Statistical tests are interpreted together with effect size, variance, and seed-level behavior.

---

## Key Experimental Findings

The experiments were conducted over a sequence of increasingly heterogeneous and stabilized MARL configurations, with headline comparisons trained for 2 million timesteps.

### 1. Homogeneous Fleet

In the homogeneous environment, ECU types did not introduce different operation preferences. All algorithms eventually converged to approximately the same return of **12.85**.

This provided a baseline showing that explicit specialization offers little advantage when the agents face effectively identical objectives.

### 2. Reward-Scale Heterogeneity Was Insufficient

We initially introduced different reward scales for different ECU types.

For example, one type could receive a larger penalty for the same operation than another type.

However, multiplying all operation costs by a constant does not change the relative ordering of those operations. Agents therefore continued to prefer the same low-cost operation.

This did not create the qualitative policy conflict required to meaningfully test heterogeneous specialization.

### 3. Operation-Specific Cost Heterogeneity

We then introduced operation-specific cost multipliers.

For example:

- Engine ECUs were given different costs for binary modification and multi-base verification.
- Infotainment ECUs were given different costs for verification and simpler copy operations.

This changed the relative preference between operations for different ECU types and created the intended heterogeneous decision problem.

The same action could therefore be desirable for one ECU type and undesirable for another.

### 4. MARL Stabilization

The initial FP3O implementation was highly unstable. Early runs produced extremely poor returns, including a collapse around **-1396**.

We progressively investigated and corrected numerical and architectural issues, including:

- invalid action handling,
- NaN policy logits,
- safety constraints,
- death masking,
- centralized critic behavior,
- shared feature extraction,
- vectorized environment behavior,
- policy routing.

After these changes, FP3O performance improved progressively from severe failure to approximately **-60**, then **-12**, then positive returns, eventually reaching the range of the other algorithms.

This stabilization process was important because it established that poor initial FP3O performance was not simply treated as evidence against the algorithm.

### 5. The `agent_id` Conditioning Confound

An important observation emerged during the IPPO experiments.

The initial shared IPPO policy received a unique `agent_id`. Since the physical agent identity was correlated with its fixed ECU type, the shared network could use the identifier to learn different behaviors for different agents.

Conceptually:

```text
Agent ID
   ↓
Shared Policy
   ↓
Different behavior for different physical agents
```

This provided a form of implicit specialization and made the initial comparison unfair.

We therefore replaced the identity information with a semantic one-hot `ecu_type` representation:

```text
ECU Type
   ↓
Shared Policy
   ↓
Role-dependent behavior
```

This separates **who the agent is** from **what role the agent performs**.

### 6. Type-Conditioned IPPO

The resulting Type-Conditioned IPPO policy uses the same actor-critic parameters for every agent while receiving the agent's semantic ECU type as an input.

This allows the network to condition its behavior on functional role without maintaining separate policy heads.

In the current four-agent heterogeneous environment, Type-Conditioned IPPO achieved a return of approximately **16.71** across the evaluated seeds, while FP3O achieved a lower mean with substantially greater seed-to-seed variation.

The current five-seed FP3O results include one severe collapse to approximately **-32.37**, while the Type-Conditioned IPPO runs remained tightly clustered around their converged return.

The Welch tests did not establish statistically significant superiority at the conventional $\alpha=0.05$ threshold. Therefore, the result is interpreted as evidence of **competitive performance and apparent seed robustness**, rather than definitive statistical proof that Type-Conditioned IPPO is universally superior.

---

## Current Interpretation

The central observation of this study is:

> **In the tested heterogeneous OTA environment, semantic type conditioning allowed a fully shared IPPO policy to achieve performance comparable to an explicitly specialized FP3O architecture.**

The two approaches represent specialization differently:

```text
FP3O
Shared representation
        +
Type-specific policy heads
        ↓
Parameter-based specialization


Type-Conditioned IPPO
Shared representation
        +
Semantic ECU type
        ↓
Input-conditioned specialization
```

The result suggests that explicit type-specific policy heads may not always be necessary when the information required to distinguish agent roles is provided directly to a sufficiently expressive shared policy.

However, the current experiments do not establish whether this remains true for larger fleets, unseen type compositions, or substantially greater levels of heterogeneity.

---

## Architectural Components

| Component | Concept | Implementation |
| :--- | :--- | :--- |
| **Parameter Sharing** | IPPO / MAPPO | Shared actor-critic parameters across agents |
| **Type Conditioning** | Semantic role conditioning | One-hot `ecu_type` input to the shared policy |
| **Partial Sharing** | FP3O | Shared backbone with specialized policy heads |
| **Stability** | CTDE / Death Masking | Centralized critic with terminated-agent handling |
| **Safety** | Safety Shield | Action filter preventing modeled fleet-memory constraint violations |
| **Credit Assignment** | Shapley Value | Monte Carlo Shapley calculation for marginal contribution |
| **Environment** | PettingZoo MARL | Parallel multi-agent OTA scheduling environment |
| **Training** | PyTorch / RL algorithms | PPO-based multi-agent policy training |

---

## Environment

The environment represents coordinated firmware updates across multiple ECUs.

Each agent observes its local firmware and resource state and selects firmware update operations such as:

- **Copy**
- **Modify**
- **Multi-Base**

Invalid action combinations are masked before policy execution.

The environment introduces stochastic transmission behavior, memory constraints, and type-dependent operation costs to create conflicting optimization preferences between ECU types.

The reward represents the simulated OTA scheduling objective, including update cost, transmission-related cost, and memory overhead.

The safety shield separately enforces the modeled fleet-memory constraint during action selection.

---

## Experimental Progression

The research was developed incrementally:

```text
ReLES-OTA SARL baseline
        ↓
Multi-Agent environment
        ↓
Homogeneous MARL
        ↓
Heterogeneous ECU objectives
        ↓
IPPO / MAPPO / FP3O comparison
        ↓
FP3O instability
        ↓
Numerical and architectural stabilization
        ↓
Shared backbone + safety shield + death masking
        ↓
Agent-ID conditioning discovered
        ↓
Agent-ID removed as a confound
        ↓
Semantic ECU-type conditioning
        ↓
Type-Conditioned IPPO
        ↓
Competitive performance against FP3O
```

---

## Reproducibility

The repository contains the environment, training implementations, experiment configurations, seed-level results, analysis scripts, and training logs used in the current study.

Headline comparisons currently use **five independent random seeds**. The individual seed results are retained rather than reporting only aggregate means.

The five-seed setting reflects the computational constraints of the current study. Larger seed counts, broader fleet configurations, and additional ablations are reserved for follow-up evaluation.

---

## Installation & Tests

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify environment compliance

```bash
python test_marl_env.py
```

### 3. Verify FP3O policy routing

```bash
python test_fp3o.py
```

---

## Contributors & Roles

- **Saadman Sakib**: Lead Architect, policy routing, environment design, MARL integration, training infrastructure
- **Mohtasim Dipto**: Environment development, death masking, parallel vectorization
- **Mahin Islam**: Statistical analysis, Welch's t-tests, confidence intervals, result visualization

---

## Future Research

The current result raises a broader question:

> **At what level of heterogeneity does explicit policy specialization become necessary?**

Future experiments can investigate:

- larger fleet sizes,
- increasing numbers of ECU types,
- unseen type compositions,
- additional random seeds,
- type conditioning versus agent-ID conditioning,
- safety shield ablations,
- death masking ablations,
- the boundary where FP3O begins to outperform Type-Conditioned IPPO.

These experiments are intended to determine the limits of shared type-conditioned policies rather than assume that one architecture is universally superior.

---

<p align="center">
  <i>Undergraduate Thesis Research — Software & Systems Lab</i>
</p>
