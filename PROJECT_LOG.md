# Project Log & Engineering Milestone Directory

Welcome to the **MA-ReLES-OTA** project log directory. This file provides an executive summary of research milestones, engineering phases, and review audits. 

For full technical deep-dives, mathematical formulations, and step-by-step logs, refer to the modular milestone logs in [`docs/project_logs/`](docs/project_logs/).

---

## Milestone Timeline & Navigation Index

| Milestone | Date | Focus & Scope | Detailed Log | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 10** | **2026-11-20** | **ICCIT 2026 Peer Review Audit & Systematic Remediation Roadmap** | [Phase 10 Audit](docs/project_logs/phase_10_iccit_audit.md) | **ACTIVE** |
| **Phase 9** | 2026-09-21 | TMLR Desk Review & Multi-Agent Fleet Scalability ($N=8, 12$) | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-09-21-phase-9--tmlr-action-editor-desk-review--multi-agent-fleet-scalability-destroying-the-bijective-confound) | Completed |
| **Phase 8** | 2026-09-11 | 5-Seed Coupled Channel Benchmark & Representation Interference | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-09-11-phase-8--5-seed-coupled-channel-benchmark--representation-interference-analysis) | Completed |
| **Phase 7** | 2026-09-03 | Constrained Coupled Channel & Gateway Contention Architecture | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-09-03-phase-7--generalization-of-network-constraints--coupled-channel-shared-gateway-contention-architecture) | Completed |
| **Phase 6** | 2026-09-01 | Codebase Standardization & Type Conditioning Ablation Switch | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-09-01-phase-6--codebase-standardization--repository-cleanup) | Completed |
| **Phase 5** | 2026-08-29 | IEEE VTC 2027 Conference Paper Drafting & Log Extraction | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-08-29-phase-5--ieee-vtc-2027-conference-paper-drafting--log-extraction) | Completed |
| **Phase 4** | 2026-08-06 | 5-Seed Scaling, ECU Heterogeneity & Agent-ID Confound Discovery | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-08-26-phase-4-step-6--authoritative-5-seed-scaling--fp3o-seed-instability) | Completed |
| **Phase 3** | 2026-07-17 | SB3 VecNormalize Integration & Numerical Stability Success | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-07-17-phase-3-step-2--2m-timesteps-training-success) | Completed |
| **Phase 2** | 2026-07-15 | Action Masking, Entropy Schedules & Invalid Penalty Tuning | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-07-16-phase-2-step-3--ppo-hyperparameter-tuning-for-payload-optimization) | Completed |
| **Phase 1** | 2026-07-09 | PPO Rollout Bottleneck Fixes & Safety Shield Corrections | [Phases 1–9 Archive](docs/project_logs/phases_01_to_09.md#2026-07-09-session-5--ppo-update-frequency-bottleneck-fix--invalid-action-penalty-tuning) | Completed |

---

## Active Milestone: Phase 10 (ICCIT 2026 Peer Review Remediation)

The paper submitted to ICCIT 2026 (*Paper ID 1206: "All You Need Is Type Conditioning: Parameter Sharing vs. Specialization for Safety-Constrained Heterogeneous Automotive OTA Update Scheduling"*) was reviewed by three referees. Reviewer #2 provided an exceptionally thorough, constructive, high-rigor critique highlighting core areas for improvement.

### Summary of Priority Actions

1. **Bijective Confound Elimination**: Scale evaluations from $N=4$ to $N=8$ and $N=12$ with multiple interchangeable ECUs per type.
2. **Factorial Confound Isolation**: Execute a $2 \times 2$ factorial experiment (`{Agent ID: On/Off}` $\times$ `{Type: On/Off}`) on matched seeds to decouple identity memorization from type conditioning.
3. **Statistical Power Expansion**: Scale benchmarks from 5 to $\ge 10$ seeds; report TOST equivalence testing, Interquartile Mean (IQM), and bootstrap 95% confidence intervals.
4. **Architectural Baselines**: Add fully Independent Networks (unshared IPPO) and an established heterogeneous benchmark (HAPPO or SePS).
5. **Safety Metric Reframing**: Replace mechanical "0% violations" claims with the **Autonomous Shield Intervention Rate (%)** and unshielded boundary stress-tests.
6. **Empirical Policy Divergence**: Plot per-type action selection frequency distributions ($P(\text{Copy}), P(\text{Modify}), P(\text{MB})$) to prove that ECU classes learn distinct operational behaviors.

See the complete itemized matrix of all 16 reviewer findings and the 4-stage execution roadmap in:  
👉 **[`docs/project_logs/phase_10_iccit_audit.md`](docs/project_logs/phase_10_iccit_audit.md)**

---

## Documentation Quick Links

* [Research Overview & Domain Background](docs/research_explanation.md)
* [Key Empirical Research Findings](docs/research_findings.md)
* [Interpretation & Practical Discussion](docs/interpretation.md)
