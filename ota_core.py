"""
ota_core.py — Shared OTA Physics Module
========================================
Shared cost/physics functions used by BOTH:
  - ota_env.py       (Phase 1 single-agent)
  - marl_ota_env.py  (Phase 2 multi-agent)

Keeping these here ensures both environments model identical firmware
update economics, making single-vs-multi comparisons scientifically valid.
"""

import numpy as np
# pyrefly: ignore [missing-import]
from scipy.stats import truncnorm


# ─────────────────────────────────────────────────────────────
#  Delta-size estimation  (mirrors paper's bsdiff-style model)
# ─────────────────────────────────────────────────────────────

def estimate_delta_size(
    block_idx: int,
    operation: int,
    similarity_bias: np.ndarray,
    block_size: int = 4096,
) -> float:
    """
    Estimate the compressed delta payload (in bytes) for a block.

    Operations
    ----------
    0 → Copy   (block unchanged — very small delta)
    1 → M      (standard binary diff)
    2 → MB     (multi-base diff — larger but may save memory on ECU)

    Parameters
    ----------
    block_idx      : index into the firmware block list
    operation      : 0, 1, or 2
    similarity_bias: per-block pre-computed similarity scores [0, 1]
    block_size     : bytes per block (default 4096)

    Returns
    -------
    delta_size : float  (bytes, minimum 64)
    """
    similarity = float(similarity_bias[block_idx])

    if operation == 0:          # Copy — treat as very high similarity
        similarity = min(0.95, similarity + 0.25)
    elif operation == 2:        # MB — penalise similarity (larger diff)
        similarity -= 0.18

    base_delta = block_size * (1.0 - max(0.0, similarity))

    if operation == 2:          # MB has encoding overhead
        base_delta *= 1.45

    return max(64.0, base_delta)


# ─────────────────────────────────────────────────────────────
#  Transmission-cost model
# ─────────────────────────────────────────────────────────────

# Stochastic latency parameters (RSR-RSMARL approach)
# Truncated Gaussian: mean=120ms, std=40ms, clipped to [50, 200]ms
_LATENCY_MEAN_MS   = 120.0
_LATENCY_STD_MS    =  40.0
_LATENCY_LOW_MS    =  50.0
_LATENCY_HIGH_MS   = 200.0

_a = (_LATENCY_LOW_MS  - _LATENCY_MEAN_MS) / _LATENCY_STD_MS   # = -1.75
_b = (_LATENCY_HIGH_MS - _LATENCY_MEAN_MS) / _LATENCY_STD_MS   # =  2.0
_TRUNCNORM = truncnorm(_a, _b, loc=_LATENCY_MEAN_MS, scale=_LATENCY_STD_MS)


def sample_latency_ms() -> float:
    """
    Draw one stochastic latency sample (ms) from a Truncated Gaussian.
    Range: [50, 200] ms  —  mean ≈ 120 ms (urban Dhaka 4G baseline).

    Used in the MARL env to force agents to learn latency-robust policies.
    """
    return float(_TRUNCNORM.rvs())


def calculate_tx_cost(
    payload_bytes: float,
    bd_params: dict,
    stochastic: bool = False,
) -> float:
    """
    Model the effective transmission cost of sending `payload_bytes`.

    Parameters
    ----------
    payload_bytes : size of the compressed delta in bytes
    bd_params     : dict with keys:
                      latency_base_ms      (fixed mode base latency)
                      packet_loss_rate     (0–1 fraction)
                      bandwidth_mbps       (available bandwidth)
    stochastic    : if True, latency is sampled from Truncated Gaussian
                    (Phase 2 MARL); if False, uses latency_base_ms (Phase 1)

    Returns
    -------
    tx_cost : float  (arbitrary cost units consistent with Phase 1)
    """
    if stochastic:
        latency_ms = sample_latency_ms()
    else:
        latency_ms = bd_params.get("latency_base_ms", 60.0)

    latency_factor    = 1.0 + (latency_ms / 800.0)
    loss_factor       = 1.0 + (bd_params.get("packet_loss_rate", 0.01) * 6.0)
    bandwidth_factor  = 800.0 / max(bd_params.get("bandwidth_mbps", 50.0), 5.0)

    return payload_bytes * latency_factor * loss_factor * bandwidth_factor * 0.0008


# ─────────────────────────────────────────────────────────────
#  BD params loader  (shared utility)
# ─────────────────────────────────────────────────────────────

import json
from pathlib import Path

_DEFAULT_PARAMS = {
    "latency_base_ms":      60.0,
    "packet_loss_rate":     0.01,
    "memory_budget_fraction": 1.0,
    "bandwidth_mbps":       50.0,
    "jitter_ms":            10.0,
    "monsoon_multiplier":   1.0,
}


def load_network_params(path: str = "network_params.json") -> dict:
    """Load vehicular network channel parameters, falling back to defaults."""
    params = dict(_DEFAULT_PARAMS)
    p = Path(path)
    if not p.exists() and path == "network_params.json":
        # Fallback to legacy bd_params.json if present
        p = Path("bd_params.json")
    if p.exists():
        try:
            with open(p, "r") as f:
                params.update(json.load(f))
        except Exception:
            pass
    return params


def load_bd_params(path: str = "network_params.json") -> dict:
    """Backward-compatible alias for load_network_params."""
    return load_network_params(path)

