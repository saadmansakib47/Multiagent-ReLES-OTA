# Phase 10 — ICCIT 2026 Peer Review Audit & Remediation Roadmap

## 2026-11-20 (Phase 10) — ICCIT 2026 Peer Review Comprehensive Audit & Systematic Remediation Roadmap

**Context**: Formal receipt of peer review decisions from ICCIT 2026 (Paper ID 1206: *"All You Need Is Type Conditioning: Parameter Sharing vs. Specialization for Safety-Constrained Heterogeneous Automotive OTA Update Scheduling"*). While Reviewers #3 and #4 provided general conference remarks, Reviewer #2 delivered an exhaustive, highly technical, top-tier peer review. Reviewer #2 confirmed that the research question is valuable and commended the open reporting of limitations, but identified foundational experimental, statistical, architectural, and presentation weaknesses that precluded publication in its current form.

Below is the complete, granular audit of every single point raised across all reviewers, mapped to a systematic remediation roadmap.

### 1. Comprehensive Review Itemization & Action Matrix

| Item ID | Category | Reviewer Critique Summary | Root Cause & Scientific Requirement | Action Item & Implementation Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REV-01** | **Confound** | **Bijective Type-to-Identity Equivalence ($N=4$)** | With 4 agents and 4 types (1 agent/type), a one-hot type vector carries the identical information as a one-hot agent index. Identity leakage was not eliminated. | Benchmark on $N=8$ and $N=12$ with $\ge 2$ interchangeable ECUs per type (`balanced_8`, `balanced_12`), plus randomized agent-to-type assignments. | **COMPLETED IN CODE** (Presets implemented in Phase 9; needs 10-seed execution) |
| **REV-02** | **Confound** | **Lack of Factorial Isolation for Confound** | IPPO mean return was 16.71 in both S3 and S4, and the gap over FP3O increased from 8.03 to 11.62. Staged design changed multiple factors simultaneously. | Execute strict $2 \times 2$ factorial matrix (`{Agent ID: On/Off}` $\times$ `{Type: On/Off}`) across matched seeds and hyperparameter settings to isolate causality. | **PLANNED** |
| **REV-03** | **Theory** | **Nuanced Framing of Terry et al. [11]** | Terry et al. theoretically prove agent indication enables shared policies to represent heterogeneous behaviors; paper should not present agent IDs as intrinsically flawed. | Reframe discussion: evaluate where agent IDs fail empirically (e.g. inability to generalize to zero-shot fleet compositions or out-of-distribution vehicle sizes). | **PLANNED** (Paper Overhaul) |
| **REV-04** | **Baselines** | **Broadening Architectural Spectrum** | Input-based type conditioning and specialized heads are both forms of type conditioning. Missing true unshared and established heterogeneous baselines. | Add: 1) Type-blind shared policy (Phase 8), 2) Fully Independent Networks (unshared IPPO), 3) Heterogeneous MARL baseline (**HAPPO** or **SePS**). | **IN PROGRESS** (Type-blind done; HAPPO/Independent planned) |
| **REV-05** | **Architecture** | **FP3O Algorithmic vs Architectural Fidelity** | Paper labels the method FP3O, but published FP3O is a full-pipeline optimization method. If only multi-head architecture was used, labeling is inaccurate. | Explicitly clarify whether FP3O's specific optimization update rule was implemented vs. shared-backbone/multi-head architecture; decouple update rule from network structure. | **PLANNED** (Paper Overhaul & Code Audit) |
| **REV-06** | **Statistics** | **Insufficient Seed Sample Size ($N=5$)** | 5 seeds cannot establish stability, equivalence, or sufficiency. Exact 95% Clopper-Pearson CI for 1 failure in 5 is $[0.5\%, 71.6\%]$; 0 failures in 5 still allows up to $52\%$ failure rate. | Expand evaluation to $\ge 10$ seeds per configuration using the fault-tolerant atomic benchmark runner (`tools/benchmark_runner.py`). | **IN PROGRESS** (Runner built; executing full suite) |
| **REV-07** | **Statistics** | **Equivalence Testing & Robust Metrics** | $p = 0.079$ fails to reject null, which does NOT show equivalence. FP3O non-failed seeds average 14.46 vs 16.71. | Implement Two One-Sided Tests (**TOST**) with predefined equivalence margin $\varepsilon$; compute Interquartile Mean (**IQM**) and 10,000-sample bootstrap 95% CIs. | **PLANNED** (`tools/statistical_battery.py`) |
| **REV-08** | **Metrics** | **Return Definition & Evaluation Protocol** | IPPO/MAPPO cross-seed variation ($\pm 0.001$) is suspiciously small despite stochastic costs and MC Shapley. Table I shows positive returns while Fig 1 shows negative VecNormalize values. | Formalize exact mathematical return definition; explicitly state raw vs. normalized evaluation protocol; compare against Random, Greedy/Sequential, and Near-Optimal Heuristics. | **PLANNED** |
| **REV-09** | **Environment** | **Reconciling ReLES-OTA Grounding** | ReLES-OTA uses $-175$ to $-180$ rewards, MB = Modify + Backup, encoding/memory costs, and Alpine Linux images. Manuscript uses $-18$, Multi-Base, synthetic blocks, and coupled channel. | Formally reconcile differences in the manuscript: establish MA-ReLES-OTA as a multi-agent vehicular CPS extension benchmark with communication contention rather than an unverified clone. | **PLANNED** (Paper Section III rewrite) |
| **REV-10** | **Heterogeneity** | **Empirical Evidence of ECU-Type Policy Divergence** | Multipliers were reported without base costs; Copy remains globally cheapest. Do ECUs actually learn distinct behaviors? | Provide explicit base costs and multiplier formulas; plot per-type action selection frequency histograms proving divergent policy distributions across ECU classes; add sensitivity analysis. | **PLANNED** |
| **REV-11** | **Safety** | **Reframing "Zero Violations" & CBF Terminology** | A 0% violation rate is mechanical because the safety shield intervenes before boundary violations occur. Loose usage of "CBF" without formal barrier proofs. | Replace "zero violations" with **Autonomous Shield Intervention Rate (%)**, near-boundary margins, and unshielded ablation runs; drop "CBF" terminology unless formal barrier functions are derived. | **COMPLETED IN CODE** (Metric implemented in Phase 8; needs paper update) |
| **REV-12** | **Credit** | **Shapley Formulation & Multi-Agent Necessity** | Coalition value function $v(S)$ and Monte Carlo sample count were undefined. With $N=4$, exact Shapley needs only $2^4 = 16$ evaluations. Need proof MARL is necessary over centralized scheduler. | Formally define $v(S)$; benchmark exact Shapley (16 subsets for $N=4$) vs. MC Shapley; compare with Equal-Split / Global Team reward; benchmark against a centralized rule-based scheduler. | **PLANNED** |
| **REV-13** | **Stability** | **FP3O Catastrophic Collapse Diagnosis** | Paper claimed all subsequent runs were stable, which directly contradicts FP3O's catastrophic seed collapse ($-32.37$). | Perform post-mortem diagnostic on collapsed seed: log per-head gradient norms, KL divergence, policy entropy decay, and value loss spikes to isolate the mathematical breakdown. | **PLANNED** |
| **REV-14** | **Claims** | **Payload Target Compliance & Title Calibration** | Mean payload remains 7.7x to 9.0x above deployment target ($<5000\text{ B}$). Claim of "All You Need" is unjustified without out-of-distribution fleet generalization. | Tone down title and claims (e.g., *"Evaluating Parameter Sharing and Specialization for Safety-Constrained Heterogeneous Automotive OTA Updates"*); acknowledge payload gap openly. | **PLANNED** |
| **REV-15** | **Presentation** | **Figure 2 Contradictions & Citation Corrections** | Fig 2 text says FP3O entropy spread is wider, but plot shows IPPO/MAPPO wider; claim of determinism contradicts 1.6–2.3 nats final entropy. Citation [18] date error; misattribution of death masking. | Fix Fig 2 narrative and visuals; add payload error bars and shield units; fix citation dates (Dec 2025); attribute death masking to MAPPO (Yu et al.); cite Fu et al. (2022), Christianos et al. (2021), Alshiekh et al. (2018). | **PLANNED** |
| **REV-16** | **Reviewer #4** | **Intro Citations, Table Caption, & Model Comparison** | Reviewer #4 requested corrected citation indices in Intro, concise Table I caption, and comparison with existing model. | Align reference numbering in introduction; revise Table I caption for conciseness; include baseline comparisons against published ReLES-OTA SARL and heuristic baselines. | **PLANNED** |

---

### 2. Systematic 4-Stage Execution Roadmap

To address every single reviewer critique with undeniable empirical and theoretical rigor, the project will execute the following structured phases:

#### Stage A: Methodological & Algorithmic Expansion
1. **$2 \times 2$ Factorial Confound Benchmark ($N=4$)**:
   - Condition 1: `Agent_ID = False`, `Type = False` (Blind baseline).
   - Condition 2: `Agent_ID = True`, `Type = False` (Agent indication only — Terry et al.).
   - Condition 3: `Agent_ID = False`, `Type = True` (Semantic type conditioning).
   - Condition 4: `Agent_ID = True`, `Type = True` (Both active).
   - *Goal*: Isolate the exact causal effect of agent ID vs. type conditioning on identical random seeds.
2. **Scalable Fleet Benchmark ($N=8$ and $N=12$)**:
   - Execute balanced fleets (`balanced_8`, `balanced_12`) and asymmetric profiles (`safety_heavy_8`, `infotainment_heavy_8`) across $\ge 10$ seeds.
   - *Goal*: Eliminate the bijective confound completely ($>1$ agent per type) and test zero-shot generalization to unseen fleet compositions.
3. **Architectural Baseline Additions**:
   - Implement **Independent PPO (IPPO-Unshared)**: each agent trains its own independent neural network without parameter sharing.
   - Implement/integrate an established heterogeneous MARL baseline (**HAPPO** or **SePS**).
   - Clarify FP3O architecture vs. update rules in code and documentation.

#### Stage B: Empirical Diagnostics & Statistical Battery
1. **10-Seed Factorial Execution with Atomic Persistence**:
   - Run the automated fault-tolerant multi-seed runner (`tools/benchmark_runner.py`) across all algorithms and fleet sizes ($10$ seeds $\times 100\text{k}$ timesteps).
2. **Comprehensive Statistical Testing Suite (`tools/statistical_battery.py`)**:
   - Compute Welch's $t$-test, Mann-Whitney $U$ test, and Cohen's $d$ effect sizes.
   - Compute **TOST (Two One-Sided Tests)** to formally establish equivalence bounds.
   - Generate Interquartile Mean (**IQM**) and 10,000-sample bootstrap 95% confidence intervals.
   - Compute exact Clopper-Pearson 95% binomial confidence intervals for seed failure/collapse rates.
3. **Per-Type Policy Divergence & Action Frequency Profiling**:
   - Extract action selection distributions ($P(\text{Copy}), P(\text{Modify}), P(\text{MB})$) conditioned on ECU type.
   - Prove via histogram plots and Kolmogorov-Smirnov tests that different ECU types learn distinct, non-degenerate operational behaviors.
4. **Safety & Credit Assignment Ablations**:
   - Report Autonomous Shield Intervention Rate (%) across all algorithms.
   - Run unshielded benchmark ablation to measure raw boundary violation rates.
   - Benchmark exact Shapley ($2^4 = 16$ evaluations) vs. Monte Carlo Shapley vs. Global Team Reward vs. Centralized Greedy Scheduler.

#### Stage C: Mathematical Formalism & Environment Re-grounding
1. **Formal Environment Specification**:
   - Formally define the observation space $\mathcal{O}_i$, action space $\mathcal{A}_i$, state space $\mathcal{S}$, transition dynamics $\mathcal{P}$, and reward formulation $R(s, a)$.
   - Explicitly define the coalition value function $v(S)$ for Shapley calculations.
   - Reconcile differences with ReLES-OTA: document MA-ReLES-OTA as a multi-agent cyber-physical network extension addressing coupled wireless gateway contention.
2. **Safety Shield Terminology Precision**:
   - Formally define the safety shield as a projection-based supervisory action shield with safe fallback $\pi_{\text{safe}}$. Remove informal "CBF" designations unless full barrier invariance conditions are derived.

#### Stage D: Paper Overhaul & Camera-Ready Resubmission
1. **Title & Narrative Re-framing**:
   - Retitle to reflect rigorous empirical investigation: *"Parameter Sharing vs. Specialization for Safety-Constrained Heterogeneous Automotive OTA Update Scheduling"*.
   - Replace over-claiming language with balanced empirical synthesis.
2. **Visual & Figure Enhancements**:
   - Regenerate Fig 1 with raw returns and explicit VecNormalize documentation.
   - Fix Fig 2 entropy plots and align text with plotted confidence bands.
   - Add per-type action distribution plots and shield intervention rate bar charts.
   - Add error margins to all payload columns and clarify units.
3. **Reference Corrections**:
   - Fix citation dates (e.g. Dec 2025).
   - Correctly attribute death masking to MAPPO (Yu et al., 2022).
   - Correctly describe MAPPO-PIS.
   - Add citations to Fu et al. (2022), Christianos et al. (2021), and Alshiekh et al. (2018).

---
