# Project Log

This file tracks the major implementation updates for the ReLES-OTA replication project. Add a new dated entry for each milestone so the team can keep a clean history of what changed, why it changed, and how it was verified.

## 2026-07-08 — Developer Q&A, GPU Fix, Config Centralisation & Chart Tooling

### Completed Work
- **GPU Fix:** Diagnosed that PyTorch was installed as the CPU-only build (`2.8.0+cpu`). CUDA 12.6 driver and RTX 3060 are fully functional — the issue was purely the wrong pip wheel. Reinstalled with `--index-url https://download.pytorch.org/whl/cu124` to get `torch 2.6.0+cu124`. After the install, `torch.cuda.is_available()` returns `True` and `train_mappo.py` automatically pins to GPU.
- **`config.py` (new):** Centralised all magic numbers into typed config dicts: `ENV_CFG`, `SAFETY_CFG`, `BD_CFG`, `SHAPLEY_CFG`, `NET_CFG`, `TRAIN_CFG`, `BENCHMARK_CFG`, `PATHS_CFG`. This eliminates scattered hard-codes like `0.85` (memory budget), `250` (monsoon jitter ms), `10` (n_envs), `128` (latent dim).
- **`tools/training_registry.py` (new):** Append-only JSON log of every training run with full metadata (algorithm, seeds, timesteps, mean return, CI, p-value, timestamp). Answers "how many times has this model been trained?" definitively. Viewable via `python tools/training_registry.py`.
- **`tools/plot_comparison.py` (new):** Auto-generates two-panel side-by-side comparison charts from the leaderboard. Left panel: bar chart of Mean Return ± 95% CI with benchmark target line. Right panel: metadata table + p-value significance verdict. Old charts are deleted automatically on each run (use `--keep-old` to accumulate). Called automatically from `main.py` after each run if ≥2 experiments exist.
- **`marl_ota_env.py` refactor:** Replaced hardcoded `0.25` memory limit fraction with `1 - SAFETY_CFG["memory_budget_frac"]` from `config.py`. Added `config.py` import with fallback defaults.
- **`tests/edge_case.py` refactor:** Replaced hardcoded `2.5` monsoon multiplier with `BD_CFG["monsoon_multiplier_high"]` from `config.py`.
- **`main.py` updates:** Wired in `training_registry.log_run()`, benchmark pass/fail check, and auto chart generation.
- **`interpretation.md` (new):** Comprehensive developer guide answering all key project questions: GPU setup, leaderboard column meanings, seed semantics, training enough determination, parameter effects, training history, and hardcoded value policy.

### Verification
- `python tools/plot_comparison.py --algo1 FP3O_Safety_True --algo2 IPPO_Safety_False` — Chart generated at `results/charts/compare_FP3O_Safety_True_vs_IPPO_Safety_False.png`.
- `python tools/training_registry.py` — Lists all recorded runs with full metadata.
- `python tests/edge_case.py` — Uses config values for monsoon parameters.
- All three test files pass without hardcoded domain constants.
- `python -c "import torch; print(torch.cuda.is_available())"` → `True`; `get_device_name(0)` → `NVIDIA GeForce RTX 3060`.
- `python train_mappo.py --mode train --algorithm fp3o --timesteps 10000` — Logs `Using cuda device`, achieves **4,500 fps** (vs ~3,600 fps on CPU). Model saved to `results/marl_models/fp3o_generic_final`.

### Notes
- The CUDA-enabled torch wheel is ~2.5 GB. This is a one-time download.
- After GPU install, every `train_algorithm()` call will automatically use CUDA without any code change.
- The `interpretation.md` file should be kept updated when new design decisions are made.

## 2026-07-07 (Session 2) — Full Research Pipeline Implementation

### Completed Work
- **CBF Safety Shield** (`marl_ota_env.py`): Implemented `safety_shield` flag in `step()`. When enabled, agents exceeding 85% of their memory budget have their actions overridden to a safe no-op (Copy, block 0), enforcing CBF constraint satisfaction without crashing.
- **Monte Carlo Shapley Credit Assignment** (`marl_ota_env.py`): Implemented `_compute_shapley()` using Monte Carlo sampling over random agent permutations. Each agent receives a reward equal to its marginal contribution to the coalition, ensuring cooperative credit assignment with provable efficiency axiom.
- **Death Masking Repair** (`marl_ota_env.py`): Ensured `_get_obs()` returns a consistent zero-vector (with only `agent_id` preserved) for terminated agents, guaranteeing SB3 vectorized env compatibility.
- **Global State Observation** (`marl_ota_env.py`): Added a `"state"` key to each agent's observation dict containing the concatenated global state for the centralized critic in MAPPO/FP3O.
- **Policy Gradient Interference Fix** (`fp3o_policy.py`): Added `share_features_extractor=False` support. `FP3OFeaturesExtractor` now accepts `is_critic` and `algorithm` flags, routing the critic to the global `"state"` observation and the actor to the local observation dict.
- **Algorithm Switch** (`fp3o_policy.py`): Policy supports `"ippo"`, `"mappo"`, and `"fp3o"` algorithms. IPPO/MAPPO use a single shared head while FP3O routes dynamically per ECU type from the `agent_id` one-hot in the observation batch.
- **Unified Training Script** (`train_mappo.py`): Refactored `train_ippo` into `train_algorithm(algorithm, ..., safety, ...)`. Adds `safety_shield` to env construction, passes `algorithm` and `share_features_extractor=False` in `policy_kwargs`, scales to `n_envs=10` parallel rollouts, and pins device to CUDA/CPU automatically.
- **`main.py` Entry Point** (new file): Implements multi-seed training loop (default 10 seeds). After each training run, computes mean return + 95% CI and writes/updates `results/leaderboard.csv` with p-value columns (Welch's t-test vs. IPPO baseline).
- **Automated Test Suite** (`tests/`):
  - `shapley_axiom.py` — Verifies agent Shapley rewards are computable without crash; checks efficiency axiom structure.
  - `edge_case.py` — Simulates 250ms "Monsoon" jitter by overriding `bd_params["monsoon_multiplier"]`; confirms env handles severe stochastic latency gracefully.
  - `efficiency_audit.py` — Benchmarks FP3O policy throughput across block sizes 8/16/24; reports samples/sec for forward-pass profiling.
- **TensorBoard installed** for SB3 logging.

### Verification
- `python fp3o_policy.py` — All component smoke tests pass (ValueNormalizer roundtrip error: 0.000000; backbone/head shapes correct; 277,997 total params).
- `python train_mappo.py --mode train --algorithm fp3o --timesteps 2500` — Training completes without errors; model saved.
- `python main.py --mode train --algorithm fp3o --safety True --seeds 1 --timesteps 2048` — Full pipeline runs; leaderboard CSV written with correct columns.
- `python tests/shapley_axiom.py` — PASS.
- `python tests/edge_case.py` — PASS (10 steps, monsoon jitter).
- `python tests/efficiency_audit.py` — PASS; 8-block: 27,838 sps / 16-block: 26,443 sps / 24-block: 24,743 sps (CPU).

### Notes
- `n_steps=2048` in the SB3 PPO model causes SB3 to collect `n_steps × n_envs = 81,920` steps before the first gradient update regardless of `total_timesteps`; for short smoke runs, use `--timesteps 81920` or reduce `n_steps`.
- The leaderboard p-value column is `N/A` for IPPO (baseline) and populated for MAPPO/FP3O once IPPO results are stored in `results/raw_seed_returns.json`.

## 2026-07-07 (Session 1) — Environment Validation & Research Pipeline Planning

### Completed Work
- **Environment Validation:** Comprehensively validated `MultiAgentOTAEnv` using `test_marl_env.py` to ensure API compliance, check stochastic latency behavior, and verify agent scalability.
- **Dependency Management:** Resolved build failures (specifically `tinyscaler` C++ requirements) by installing individual dependencies and utilizing precompiled wheels, avoiding build-from-source issues.
- **Architecture Audit:** Examined `Stable-Baselines3` source for `ActorCriticPolicy` to plan the `share_features_extractor=False` mechanism required by `FP3OPolicy`.
- **Pipeline Planning:** Drafted an implementation plan detailing the updates to `marl_ota_env.py`, `fp3o_policy.py`, `train_mappo.py`, and the creation of `main.py` and automated test cases.

### Verification
- `python test_marl_env.py` passes the dependency load and environmental sanity checks.

### Notes
- We have established a firm plan for adding the CBF Safety Shield, Monte Carlo Shapley values, and gradient interference fixes in the upcoming execution phase.

## 2026-06-07 — Phase 2, Step 2: FP3O Architecture & Environment Migration

### Completed Work
- **Conda Environment Migration**: Created dedicated `reles-ota` Conda environment (Python 3.12) to bypass Visual C++ build tool requirements on Windows; all dependencies install from precompiled binary wheels.
- **Shared Backbone**: Verified `SharedBackbone` (3-layer MLP: 256→256→128 with ReLU) processes flattened observations shared across all agents — firmware block masks, cumulative encoding/TX costs, memory budget, step counter, and agent one-hot ID.
- **Specialized Heads (FP3O)**: Completed Partial Parameter Sharing architecture in [fp3o_policy.py](file:///d:/Thesis/reles-ota-bd-replication/fp3o_policy.py):
  - `ActionHead` per ECU type (engine / braking / infotainment / generic) → logits over {Copy, Modify, Modify+Backup}
  - `PositionHead` per ECU type → logits over N firmware block positions
  - Dynamic routing in `_get_action_dist_from_latent`: extracts agent indices from `agent_id` one-hot in observation batch, selects the correct head per sample in a vectorized batch.
- **Value Normalization**: Implemented `ValueNormalizer` (PopArt-style EMA running mean/std) and `ValueNormalizationCallback` (SB3 `BaseCallback`):
  - `update()` called per rollout; normalizes rollout buffer returns and values before critic gradient steps.
  - `predict_values()` and `forward()` denormalize critic output to reward scale for GAE advantage estimation.
- **Training Integration**: [train_mappo.py](file:///d:/Thesis/reles-ota-bd-replication/train_mappo.py) IPPO path now uses `FP3OPolicy` and `ValueNormalizationCallback` instead of `MultiInputPolicy`.
- **Console Encoding Fix**: Added `sys.stdout.reconfigure(encoding='utf-8')` to `test_marl_env.py` and `train_mappo.py` to prevent `UnicodeEncodeError` on Windows cp1252 consoles.
- **Codebase Cleanup**: Removed duplicate `FP3OPolicy` stub class, unused imports (`F`, `FlattenExtractor`, `Type`, `List`), and moved all imports to module level.

### Verification
- `python test_fp3o.py` — ValueNormalizer roundtrip PASS; FP3O backbone/heads shape verification PASS.
- `python test_marl_env.py` — All 5 tests PASS (API compliance, random episodes, death masking, stochastic latency, scalability).
- `python train_mappo.py --mode ippo --n_agents 4 --n_blocks 6 --timesteps 5000` — IPPO training with FP3OPolicy completed successfully; model saved.

### Notes
- Dynamic head routing uses `agent_id` one-hot in the observation to route each sample in a vectorized batch to its correct ECU-type specialized head, enabling truly heterogeneous multi-agent training under a single policy instance.
- `ValueNormalizationCallback` operates on the rollout buffer's `returns` and `values` tensors after GAE computation, ensuring the critic regresses on z-scored targets while `predict_values()` always denormalizes back to raw reward scale.

## 2026-06-06 — Phase 2, Step 1 (Continuation): Dependency Setup & Wrapper Integration Fixes

### Completed Work
- Resolved dependencies on Windows Python 3.12, including precompiled `tinyscaler` and `supersuit` setup.
- Added `render_mode` parameter and attribute to [MultiAgentOTAEnv](file:///d:/Thesis/reles-ota-bd-replication/marl_ota_env.py#L44) for compatibility with SuperSuit's wrappers.
- Configured `MarkovVectorEnv` wrapper with `black_death=True` in [train_mappo.py](file:///d:/Thesis/reles-ota-bd-replication/train_mappo.py) to support variable agent lifetimes and align with our death masking strategy.
- Patched `ConcatVecEnv`'s missing `seed` method to resolve the AttributeError in Stable Baselines 3 (`set_random_seed`).

### Verification
- Ran `python test_marl_env.py` under Python 3.12 successfully (all Parallel API compliance and death masking checks passed).
- Successfully ran IPPO training via `python train_mappo.py --mode ippo --n_agents 2 --n_blocks 6 --timesteps 5000` (completed log stats and model saving without errors).

### Notes
- Seeding for Gym 0.26+ vectorized environments is now safely bypassed with a dummy `.seed()` method on the vectorized wrapper.
- `black_death=True` is required when wrapping multi-agent environments with varying agent counts under SuperSuit's Markov vector wrappers.

## 2026-06-06 — Phase 2, Step 1: Multi-Agent Environment Scaling

### Completed Work
- Added a new PettingZoo Parallel API multi-agent environment in `marl_ota_env.py`.
- Supported N concurrent ECU agents, each with its own firmware block list, mask, and episode state.
- Implemented death masking so finished agents keep returning a fixed zero-vector observation with only `agent_id` preserved.
- Added stochastic network latency support through a shared physics module in `ota_core.py`.
- Kept the original single-agent environment in `ota_env.py` intact for Phase 1 comparison.
- Added a multi-agent validation suite in `test_marl_env.py`.
- Added a training scaffold / random baseline runner in `train_mappo.py`.
- Updated `requirements.txt` with PettingZoo, SuperSuit, and SciPy dependencies.

### Verification
- Ran `python test_marl_env.py` successfully.
- Confirmed PettingZoo parallel API compliance.
- Confirmed death masking works for terminated agents.
- Confirmed stochastic latency differs from fixed-latency behavior under identical actions.
- Confirmed scalability checks pass for multiple agent counts.
- Ran `python train_mappo.py --mode random --n_agents 4 --n_blocks 6 --episodes 2` successfully.

### Notes
- The multi-agent environment is implemented as a new file rather than refactoring the single-agent env in place.
- The current training entrypoint supports random rollout verification and an IPPO scaffold, while a full MAPPO learner remains a later step.

## Suggested Update Format

Use this template for future entries:

### YYYY-MM-DD — Short Milestone Title

#### Completed Work
- Item 1
- Item 2
- Item 3

#### Verification
- Test or command run
- Result

#### Notes
- Any design decisions, caveats, or follow-up work
