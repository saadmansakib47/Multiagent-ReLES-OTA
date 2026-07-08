"""
config.py — Centralized Configuration for ReLES-OTA
=====================================================
All hard-coded constants, hyperparameters, and experiment defaults
live here. Never hard-code magic numbers directly in model or env code;
import from this file instead.

Usage
-----
    from config import ENV_CFG, TRAIN_CFG, NET_CFG, SAFETY_CFG, BD_CFG
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment Configuration
# ─────────────────────────────────────────────────────────────────────────────
ENV_CFG = dict(
    n_agents          = 4,      # default number of ECU agents
    n_blocks          = 16,     # firmware blocks per agent
    bd_mode           = True,   # Bangladesh network conditions
    stochastic_latency= True,   # random packet-loss / delay per step
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Safety Shield Configuration (CBF)
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_CFG = dict(
    enabled            = True,   # toggle CBF safety shield globally
    # fraction of memory_budget that triggers a safe override
    memory_budget_frac = 0.85,   # override when usage > 85% of budget
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bangladesh (BD) Network Conditions
# ─────────────────────────────────────────────────────────────────────────────
# These map directly to the bd_params.json values but are code-accessible.
BD_CFG = dict(
    # "Monsoon" burst-congestion scenario
    monsoon_jitter_ms       = 250,   # target one-way delay added by monsoon jitter
    monsoon_multiplier_low  = 1.5,   # minimum latency multiplier under monsoon
    monsoon_multiplier_high = 3.0,   # maximum latency multiplier under monsoon

    # Baseline congestion scalars
    base_latency_ms         = 50,    # expected RTT without congestion
    packet_loss_rate        = 0.08,  # 8% packet-loss under BD conditions
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Shapley Credit Assignment
# ─────────────────────────────────────────────────────────────────────────────
SHAPLEY_CFG = dict(
    n_mc_samples = 64,   # Monte Carlo permutation samples (higher = more accurate)
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Policy / Network Architecture
# ─────────────────────────────────────────────────────────────────────────────
NET_CFG = dict(
    latent_dim     = 128,   # SharedBackbone output dimension
    backbone_widths= (256, 256, 128),  # MLP hidden layers
    ecu_type       = "generic",        # default ECU type for single-head runs
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CFG = dict(
    # PPO on-policy
    learning_rate  = 3e-4,
    n_steps        = 2048,   # rollout horizon per env
    batch_size     = 64,
    gae_lambda     = 0.95,
    gamma          = 0.99,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,

    # Parallel collection
    n_envs         = 10,     # matches Ryzen 5 5600 12-thread budget

    # Experiment scale
    total_timesteps= 500_000,  # per seed
    n_seeds        = 10,

    # Convergence / budget stop
    # We consider a run "converged" when the rolling 100-episode mean return
    # stops improving by less than CONVERGENCE_DELTA over CONVERGENCE_WINDOW steps.
    convergence_delta  = 1.0,
    convergence_window = 50_000,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Research Benchmarks / Target Scores
# ─────────────────────────────────────────────────────────────────────────────
# These are the minimum thresholds that a trained model must EXCEED to be
# considered deployment-ready (based on Phase 1 single-agent baselines).
BENCHMARK_CFG = dict(
    # Mean episode return targets (higher / less-negative is better)
    # Phase 1 PPO single-agent reference: ~−18 on generic, ~−35 on BD conditions
    target_return_generic  = -20.0,  # fleet mean return must be > this
    target_return_bd       = -40.0,  # BD-mode fleet return must be > this

    # Safety hard constraint: shield activation rate must be < 5%
    max_shield_activation_rate = 0.05,

    # Statistical significance threshold
    p_value_threshold      = 0.05,   # Welch t-test vs IPPO baseline

    # Payload cost target (lower is better)
    target_payload_cost    = 5000,   # bits; informed by OTA spec
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Logging & Paths
# ─────────────────────────────────────────────────────────────────────────────
PATHS_CFG = dict(
    results_dir       = "results",
    models_dir        = "results/marl_models",
    leaderboard_csv   = "results/leaderboard.csv",
    raw_returns_json  = "results/raw_seed_returns.json",
    training_registry = "results/training_registry.json",
    charts_dir        = "results/charts",
)
