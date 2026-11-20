## 2026-09-03 (Phase 7) — Generalization of Network Constraints & Coupled Channel (Shared Gateway Contention) Architecture
— Generalization of Network Constraints & Coupled Channel (Shared Gateway Contention) Architecture

**Context**: Addressing the empirical finding where Independent PPO (IPPO) matched FP3O asymptotically at 16.71 in decoupled channel settings, and transitioning from regional network parameters to a generalized constrained network model.

**Key Actions & Implementation Details**:
1. **Network Parameter Generalization & Repository Decoupling**:
   - Replaced all occurrences of `bd_params` and `bd_mode` with `network_params` and `constrained_network_mode` across all 15 project files (`marl_ota_env.py`, `ota_core.py`, `ota_env.py`, `train_mappo.py`, `train_ppo.py`, `baselines.py`, `config.py`, `experiment_runner.py`, `main.py`, `result_generator.py`, `test_env.py`, `test_marl_env.py`, `web_ui.py`, `webui/app.js`, `webui/index.html`).
   - Permanently deleted `bd_params.json` and standardized on `network_params.json` to ground the research in generic, internationally applicable constrained vehicular CPS conditions (120ms baseline latency, 5% loss, 25 Mbps bandwidth).

2. **Coupled Channel (Shared Gateway Downlink Contention)**:
   - Formulated the shared gateway bandwidth bottleneck: when multiple ECUs push OTA updates concurrently, the shared gateway bandwidth is partitioned:
     $$\text{effective\_bw\_mbps} = \frac{\text{gateway\_bw\_mbps}}{\max(n_{\text{transmitting}}, 1)}$$
   - Implemented `calculate_tx_cost_coupled()` in `ota_core.py` and wired the `coupled_channel: bool = False` toggle into `MultiAgentOTAEnv`.
   - Added an independent `gateway_bw_mbps: float = 50.0` parameter so gateway capacity can be evaluated independently of per-agent channel noise.
   - Injected a normalized `gateway_load` scalar ($n_{\text{transmitting}} / N_{\text{agents}}$) into the global state representation when coupled mode is active, providing the centralized critic and FP3O shared backbone with explicit contention feedback.

3. **Explicit Design Decision — Copy (op=0) Bandwidth Exemption**:
   - **Domain Rationale**: Only `Modify` (op=1) and `Modify+Backup` (op=2) operations increment $n_{\text{transmitting}}$. `Copy` operations are strictly exempt from bandwidth contention.
   - **Physical Justification**: A Copy block has a near-zero delta payload (~64 bytes vs. thousands of bytes for binary diffs). In real vehicular OTA architectures, a Copy operation represents a lightweight control-plane hash/CRC verification check, never traversing the shared data-plane downlink bottleneck.
   - Exempting Copy operations avoids penalizing policies for selecting non-transmitting actions and encourages emergent temporal coordination (e.g., staggering heavy updates while utilizing Copy during peak contention).

4. **Web UI & CLI Switch Integration**:
   - Added the **Coupled Channel** toggle and dynamic **Gateway BW (Mbps)** input to `webui/index.html` and `webui/app.js`.
   - Wired `--coupled_channel` and `--gateway_bw_mbps` CLI flags through `train_mappo.py` and `web_ui.py` for automated multi-seed execution.
   - Maintained full backward compatibility: default `coupled_channel=False` reproduces the exact isolated network slicing benchmark.

5. **Verification**:
   - Unit tests confirmed transmission costs scale by exactly $2.00\times$ when two agents transmit concurrently vs. one agent ($19.14 \rightarrow 38.27$).
   - Full validation suite in `test_marl_env.py` passed 100% (PettingZoo API compliance, death masking, stochastic latency, and agent scalability).
   - Test training runs confirmed seamless integration with SB3 PPO, vectorized Markov wrappers, and tensorboard logging.

---
