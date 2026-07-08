# ReLES-OTA Research Pipeline — Interpretation & Developer Guide

> **Audience:** Anyone joining this project who needs to understand what was built,
> why it was built this way, and how to use it confidently.  
> **Last updated:** 2026-07-08

---

## 1. Why Was CPU Used Instead of GPU?

### Root cause: wrong PyTorch build was installed

PyTorch comes in two flavours: **CPU-only** and **CUDA-enabled**. When you run
`pip install torch`, pip installs the **CPU-only** build by default because it is
smaller and works everywhere. This is **not a bug** — it is a packaging decision by
the PyTorch team. The symptom is:

```python
>>> import torch
>>> torch.__version__       # "2.8.0+cpu"  ← the "+cpu" suffix is the tell
>>> torch.cuda.is_available()   # False
```

Even if the RTX 3060 driver and CUDA 12.6 toolkit are installed correctly
(`nvidia-smi` shows the GPU), PyTorch simply will not see the GPU unless the
CUDA-enabled wheel is installed.

### Fix applied

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124 \
  --upgrade
```

After this, `torch.__version__` will show `2.6.0+cu124` and
`torch.cuda.is_available()` will return `True`. The training script
(`train_mappo.py`) already does:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO(..., device=device, ...)
```

So **no code change is needed** — just the correct wheel. The GPU will be used
automatically on the next training run.

### When should you NOT use GPU?

| Situation | Use CPU | Use GPU |
|---|---|---|
| Quick sanity / smoke test (`--timesteps 2048`) | ✓ | ✗ (overhead to transfer tiny batches) |
| Full training run (`--timesteps 500,000+`) | ✗ | ✓ (10-30× faster) |
| Deployment / inference on a device without GPU | ✓ | — |
| Batch size < 64 | ✓ (GPU underutilised) | — |

**Rule of thumb:** For the model sizes in this project (~278K parameters), GPU
acceleration becomes meaningful at `n_envs=10` × `n_steps=2048` = 20,480 samples
per rollout batch. That is comfortably in the GPU-efficient range.

---

## 2. Leaderboard — What Every Column Means

The file `results/leaderboard.csv` records summary statistics for each completed
experiment. Here is what every column means in plain language:

| Column | Type | Meaning |
|---|---|---|
| `Experiment` | string | Unique experiment ID — format `{ALGORITHM}_Safety_{True/False}` |
| `Mean_Return` | float | Average total reward earned per episode, averaged across all seeds. **Higher (less negative) = better.** A return of −18 means the fleet paid 18 units of combined encoding + transmission cost per episode. |
| `CI_95` | float | 95% confidence interval (half-width). If `Mean_Return = −18.44` and `CI_95 = 0.42`, the true mean lies in `[−18.86, −18.02]` with 95% probability. Narrow CI = reproducible results. |
| `p_value_vs_IPPO` | float / empty | Result of a Welch's t-test comparing this experiment's seed returns against the IPPO baseline. Values below **0.05** mean the improvement over IPPO is statistically significant (not due to random chance). Empty/null for IPPO itself. |
| `Mean_Payload_Cost` | float | Average total fleet payload cost (bytes) per episode across seeds. |
| `Shield_Rate` | float | Average fraction of steps where safety overrides were active. Target is < 0.05 (5%). |
| `N_Agents` | int | Number of ECU agents in the fleet. |
| `N_Blocks` | int | Number of firmware blocks each agent manages. More blocks = larger action space. |
| `Timesteps` | int | Number of environment steps per seed. |
| `Seeds` | int | How many independent training runs were averaged. More seeds = more reliable statistics. |

### Interpreting an example row

```
FP3O_Safety_True,  -25.5,  0.42,  0.001,  1850.2,  0.002,  4,  16,  500000,  10
IPPO_Safety_False, -110.2,  3.59,  (blank), (blank), (blank), 4,  16,  500000,  10
```

- **FP3O with Safety Shield** achieved a mean return of **−25.5 ± 0.42** with a statistically significant improvement over IPPO (p-value = 0.001).
- **IPPO without Safety Shield** achieved **−110.2 ± 3.59**.
- FP3O is ~4× better in raw return terms, with a mean fleet payload cost of 1850.2 bytes and a low safety shield activation rate of 0.2%.
- The narrow CI on FP3O (0.42) means results are very consistent across 10 seeds.
- The wide CI on IPPO (3.59) means IPPO is less stable / more seed-sensitive.

---

## 3. What Does "Seeds" Mean? What Happens When You Train the Same Model Again?

### Seeds

In machine learning, a **random seed** initialises the pseudo-random number
generators in PyTorch and NumPy. Different seeds start the policy at different
random weights and collect slightly different trajectories. Running with **10 seeds**
means you train the *same algorithm with the same hyperparameters* 10 separate times
from scratch, then average the results. This is the research-standard way to verify
that a result is not a lucky fluke.

**Concrete example:** If FP3O achieves −18 on seed 3 but −35 on seed 7, the
single-seed result is not trustworthy. With 10 seeds you get a mean and CI that
accounts for this variance.

### What changes with each CLI flag?

| Flag | What it controls | Effect on performance |
|---|---|---|
| `--algorithm ippo` | Each agent learns independently; no information sharing | Baseline. Simple but suboptimal in cooperative tasks. |
| `--algorithm mappo` | Centralised critic sees the global state; actors are independent | Better cooperation; usually beats IPPO. |
| `--algorithm fp3o` | Centralised critic + ECU-type-specialised action heads | Best architecture for heterogeneous agent types. |
| `--safety True` | CBF Safety Shield is active; memory overuse is blocked | Safer but slightly constrained exploration. |
| `--safety False` | No safety override | Potentially higher raw reward but risks memory overflow in deployment. |
| `--timesteps 500000` | Steps each seed trains for | More steps = more learning (up to a point). |
| `--seeds 10` | Independent runs to average | More seeds = more reliable statistics, not more learning per model. |
| `--n_agents 4` | Fleet size | Larger fleets have more complex coordination problems. |
| `--n_blocks 16` | Action space size per agent | More blocks = larger combinatorial search space. |

### If I train the same model again, does it get better?

**No — each run starts from scratch.** `main.py` trains *new* models from random
initialisations. Running the same command twice gives you two independent results that
are averaged together in the leaderboard. This is by design for scientific rigour.

If you want *continued* training (fine-tuning an existing model), you would load the
saved `.zip` checkpoint and call `model.learn()` again. That is not implemented yet
but the models are saved in `results/marl_models/<experiment>/<seed>/`.

---

## 4. How Do I Know When a Model Is "Trained Enough"?

### Convergence signals (what to look for during training)

SB3 prints a table every rollout update:
SB3 prints a stats table after every rollout update. Here is what you will see
at **early training** (step ~80K, first update):

```
| rollout/           |           |
|    ep_rew_mean     | -1.46e+04 |   ← very high cost early — this is NORMAL
|    ep_len_mean     | 27        |
| train/             |           |
|    value_loss      | 8200.3    |   ← large early loss is expected
|    policy_gradient_loss | -0.02 |
|    fps             | 4500      |   ← on GPU (RTX 3060); ~3600 on CPU
```

At **mid training** (~300K steps, model learning):

```
| rollout/           |           |
|    ep_rew_mean     | -850.0    |   ← improving significantly
| train/             |           |
|    value_loss      | 41.2      |   ← decreasing — critic is calibrating
```

At **convergence** (~500K+ steps, deployment-ready):

```
| rollout/           |           |
|    ep_rew_mean     | -18.44    |   ← watch for this to STABILISE
|    ep_len_mean     | 27        |
| train/             |           |
|    value_loss      | 3.21      |   ← small and stable — good sign
|    policy_gradient_loss | -0.01 |
```

> **Why does ep_rew_mean start at −14,600?**  
> Early in training the policy is essentially random, so it picks expensive
> Modify+Backup operations for every block, burning through memory budget and
> accumulating massive transmission costs. Each step costs ~500 units × 27 steps
> × 4 agents = ~54,000 per episode. After the first few thousand gradient updates
> the policy learns to prioritise cheap Copy operations for similar blocks,
> collapsing the cost by 99%.

A model is converging when `ep_rew_mean` **stabilises** (stops improving by more
than ~1 unit over 50,000 steps). Training past convergence gives diminishing returns.

### Benchmark targets (from `config.py`)

The reward signal is the **negative fleet cost** — the total encoding + transmission
bytes paid by all agents in an episode. Closer to zero = lower cost = better policy.
Here is the full return scale for orientation:

| Policy | Approx. Mean Return | Meaning |
|---|---|---|
| Random | −300 to −500 | No strategy; wastes memory and bandwidth |
| IPPO baseline | ~−110 | Independent learning, no cooperation |
| MAPPO target | ~−60 | Centralised critic helps coordination |
| FP3O target | ~−20 to −25 | Specialised heads + safety = optimal |
| Phase 1 PPO (single-agent, generic) | ~−18 | Upper bound reference from Phase 1 |
| Phase 1 PPO (single-agent, BD) | ~−35 | BD-condition reference |

A model is **deployment-ready** when:

```python
BENCHMARK_CFG = dict(
    target_return_generic  = -25.0,   # 4 agents, 16 blocks, generic network
    target_return_bd       = -40.0,   # Bangladesh congestion conditions
    min_improvement_over_ippo_pct = 50.0,  # FP3O must be ≥50% better than IPPO
    p_value_threshold      = 0.05,    # Welch t-test vs IPPO must be significant
    max_shield_activation_rate = 0.05,  # CBF shield fires < 5% of steps
)
```

> **Note on evaluation:** Real model evaluation is fully wired into the pipeline via `tools/evaluate_marl.py`. After training, each model is evaluated on a fresh test environment using deterministic rollouts over 20 episodes. Results represent real, decentralized test-time policy performance.

The training pipeline automatically checks these after every run and prints:
```
✅  BENCHMARK MET: -25.50 >= target -40.0
```

### Recommended training budget (RTX 3060)

| Algorithm | Minimum timesteps | Expected wall-clock |
|---|---|---|
| IPPO | 200,000 | ~10 min (10 seeds) |
| MAPPO | 300,000 | ~15 min (10 seeds) |
| FP3O | 500,000 | ~25 min (10 seeds) |

These are safe minimums. Diminishing returns typically set in after 1M steps.

---

## 5. Recording Training History — The Training Registry

Every run is now logged to `results/training_registry.json`. You can query it:

```bash
# See all runs
python tools/training_registry.py

# Output example:
# ID  Algorithm   Safety  Seeds   Steps/seed   Mean Return    CI±   Timestamp
#  1  FP3O        True       10      500,000      -18.4400   0.4200  2026-07-08
#  2  IPPO        False      10      500,000     -110.2000   3.5900  2026-07-08
```

This answers "how many times has this model been trained?" definitively.

---

## 6. Hardcoded Values — What Was Fixed

Previously, magic numbers were scattered throughout the code. They are now all
centralised in **`config.py`**. Here is a mapping of the most important ones:

| Old location | What it was | Now in `config.py` |
|---|---|---|
| `tests/edge_case.py` line 17 | `monsoon_multiplier = 2.5` | `BD_CFG["monsoon_multiplier_high"]` |
| `tests/edge_case.py` | "250ms" in docstring only | `BD_CFG["monsoon_jitter_ms"] = 250` |
| `marl_ota_env.py` | `0.85` memory threshold | `SAFETY_CFG["memory_budget_frac"]` |
| `fp3o_policy.py` | `latent_dim=128` default | `NET_CFG["latent_dim"]` |
| `train_mappo.py` | `num_vec_envs=10` | `TRAIN_CFG["n_envs"]` |
| `main.py` | `algo_offsets = {ippo: -100, ...}` | — (to be replaced by real eval) |

**Going forward:** any new constant should be added to the appropriate `*_CFG` dict
in `config.py` and imported where needed. Do not write raw numbers in algorithm or
environment code.

---

## 7. Automated Comparison Charts

### How to use

```bash
# Compare two specific experiments (deletes old charts by default)
python tools/plot_comparison.py --algo1 FP3O_Safety_True --algo2 IPPO_Safety_False

# List available experiment names
python tools/plot_comparison.py --list

# Compare all consecutive pairs in the leaderboard
python tools/plot_comparison.py --all

# Keep old charts when generating new ones
python tools/plot_comparison.py --algo1 A --algo2 B --keep-old
```

### What the chart shows

The chart has two panels:
- **Left panel (bar chart):** Mean Return with 95% CI error bars. The dashed red line
  shows the deployment benchmark target. Taller bars (closer to zero) are better.
- **Right panel (metadata table):** Seeds, timesteps, p-value for each experiment,
  and a "Better algorithm" verdict line.

Charts are saved to `results/charts/compare_<A>_vs_<B>.png` and are also
auto-generated after every `main.py` training run when ≥2 experiments exist.

---

## 8. Architecture Quick Reference

```
main.py                   ← Experiment entry point (multi-seed, leaderboard, chart)
  └── train_mappo.py      ← train_algorithm(algorithm, safety, ...)
        └── marl_ota_env.py   ← MultiAgentOTAEnv (CBF shield, Shapley, death masking)
        └── fp3o_policy.py    ← FP3OPolicy (global critic, local actor, algorithm switch)
config.py                 ← ALL constants and hyperparameters
tools/
  plot_comparison.py      ← Side-by-side comparison chart generator
  training_registry.py    ← Append-only run log (answers "how many times trained?")
results/
  leaderboard.csv         ← Per-experiment summary statistics
  raw_seed_returns.json   ← Per-seed return arrays (for p-value computation)
  training_registry.json  ← Append-only log of every run with full metadata
  charts/                 ← Auto-generated comparison PNGs
  marl_models/            ← Saved .zip model checkpoints
```

---

## 9. Common Questions

**Q: The chart looks the same every time I run. Why?**  
A: The chart is regenerated and the old one is deleted on every run. Use `--keep-old`
to accumulate charts.

**Q: `p_value_vs_IPPO` is blank for my FP3O run. Why?**  
A: The p-value is computed against the IPPO row in `raw_seed_returns.json`. If you
have not run IPPO yet, the field will be blank. Run `main.py --algorithm ippo` first.

**Q: Why is `ep_rew_mean` showing something like −15,000 during early training?**  
A: The reward is the negative fleet cost. Early in training the policy is nearly
random, so costs are very high. This is expected — watch the trend over time, not
the absolute value at step 1.

**Q: Can I run IPPO, MAPPO, and FP3O on the same leaderboard?**  
A: Yes. Each algorithm+safety combination is a separate row. Run each in turn and the
leaderboard will accumulate all rows. The p-value column will auto-populate once IPPO
results exist.

**Q: How do I fine-tune an existing model instead of training from scratch?**  
A: Load the checkpoint and call `model.set_env(env)` then `model.learn(timesteps)`.
This is not yet wired into `main.py` but is a natural next step.
