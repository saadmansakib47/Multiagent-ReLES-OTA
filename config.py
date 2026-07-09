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
    n_steps        = 256,    # rollout horizon per env (was 2048) — smaller horizon means more updates
    batch_size     = 128,    # must divide n_steps*n_envs*n_agents (4096); was 512
    n_epochs       = 10,     # PPO gradient passes per rollout
    gae_lambda     = 0.95,
    gamma          = 0.99,
    clip_range     = 0.2,
    ent_coef       = 0.01,   # lower entropy allows PPO to converge on valid block selections
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,

    # Parallel collection
    n_envs         = 4,      # was 10

    # Experiment scale
    total_timesteps= 500_000,  # per seed; now provides 488 updates (was only 12 at 1M steps!)
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
# considered deployment-ready. Targets are set relative to the IPPO baseline
# (mean ~−110 on BD conditions with 500k steps) and Phase 1 PPO single-agent
# reference (~−18 generic, ~−35 BD).
#
# Interpretation of return scale:
#   The reward is the NEGATIVE fleet cost (encoding + transmission cost summed
#   across all agents). Lower-magnitude (closer to 0) = lower cost = better.
#
#   Random policy:  ~−300 to −500  (no strategy, wastes memory and bandwidth)
#   IPPO baseline:  ~−110           (independent learning, no cooperation)
#   MAPPO target:   ~−60            (centralised critic helps coordination)
#   FP3O target:    ~−20            (specialised heads + safety = optimal)
#   Phase 1 PPO:    ~−18 (generic), ~−35 (BD)  ← single-agent upper bound
BENCHMARK_CFG = dict(
    # Mean episode return targets (higher / less-negative is better)
    # A model is deployment-ready when mean_return >= the target below.
    target_return_generic  = -25.0,   # generic network, 4 agents, 16 blocks
    target_return_bd       = -40.0,   # BD (Bangladesh) congestion conditions

    # Minimum improvement over IPPO to justify the added complexity of MAPPO/FP3O
    # FP3O should be at least 50% better than IPPO in mean return.
    min_improvement_over_ippo_pct = 50.0,

    # Safety hard constraint: shield activation rate must be < 5%
    max_shield_activation_rate = 0.05,

    # Statistical significance threshold
    p_value_threshold      = 0.05,   # Welch t-test vs IPPO baseline

    # Payload cost target (lower is better; informed by OTA spec)
    target_payload_cost    = 5000,   # bits
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
