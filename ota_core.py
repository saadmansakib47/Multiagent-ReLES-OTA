"""
ota_core.py - Shared OTA Physics Module
========================================
Shared cost/physics functions used by BOTH:
  - ota_env.py       (Phase 1 single-agent)
  - marl_ota_env.py  (Phase 2 multi-agent)

Keeping these here ensures both environments model identical firmware
update economics, making single-vs-multi comparisons scientifically valid.
"""

import json
from pathlib import Path
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.stats import truncnorm


# -------------------------------------------------------------
#  Delta-size estimation  (mirrors paper's bsdiff-style model)
# -------------------------------------------------------------

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
    0 -> Copy   (block unchanged - very small delta)
    1 -> M      (standard binary diff)
    2 -> MB     (multi-base diff - larger but may save memory on ECU)

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

    if operation == 0:          # Copy - treat as very high similarity
        similarity = min(0.95, similarity + 0.25)
    elif operation == 2:        # MB - penalise similarity (larger diff)
        similarity -= 0.18

    base_delta = block_size * (1.0 - max(0.0, similarity))

    if operation == 2:          # MB has encoding overhead
        base_delta *= 1.45

    return max(64.0, base_delta)


# -------------------------------------------------------------
#  Transmission-cost model (Decoupled / Dedicated Channel)
# -------------------------------------------------------------

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
    Range: [50, 200] ms  -  mean ~ 120 ms (constrained vehicular 4G baseline).

    Used in the MARL env to force agents to learn latency-robust policies.
    """
    return float(_TRUNCNORM.rvs())


def calculate_tx_cost(
    payload_bytes: float,
    net_params: dict,
    stochastic: bool = False,
) -> float:
    """
    Model the effective transmission cost of sending payload_bytes over an isolated channel.

    Parameters
    ----------
    payload_bytes : size of the compressed delta in bytes
    net_params    : dict with keys:
                      latency_base_ms      (fixed mode base latency)
                      packet_loss_rate     (0-1 fraction)
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
        latency_ms = net_params.get("latency_base_ms", 60.0)

    latency_factor    = 1.0 + (latency_ms / 800.0)
    loss_factor       = 1.0 + (net_params.get("packet_loss_rate", 0.01) * 6.0)
    bandwidth_factor  = 800.0 / max(net_params.get("bandwidth_mbps", 50.0), 5.0)

    return payload_bytes * latency_factor * loss_factor * bandwidth_factor * 0.0008


# -------------------------------------------------------------
#  Coupled-channel transmission cost (shared gateway contention)
# -------------------------------------------------------------

def calculate_tx_cost_coupled(
    payload_bytes: float,
    net_params: dict,
    n_transmitting: int,
    gateway_bw_mbps: float,
    stochastic: bool = False,
) -> float:
    """
    Coupled-channel transmission cost model for shared gateway contention.

    When multiple ECU agents transmit simultaneously over a shared vehicle
    gateway (e.g., a single cellular/V2X downlink), the available bandwidth
    per agent is divided by the number of concurrently transmitting agents:

        effective_bw_mbps = gateway_bw_mbps / max(n_transmitting, 1)

    This creates emergent temporal coordination pressure: agents must learn
    to stagger large transmissions rather than all pushing Modify/MB blocks
    at the same timestep.

    Design decision - Copy (op=0) exemption
    ----------------------------------------
    Only Modify (op=1) and Modify+Backup (op=2) operations are counted toward
    n_transmitting. Copy blocks have a near-zero compressed delta payload
    (~64 bytes vs. potentially thousands of bytes for Modify). Their gateway
    bandwidth consumption is negligible and treating them as 'transmitting'
    would unfairly penalise agents that choose the cheapest valid operation.
    This exemption is physically correct: in OTA practice, a Copy block diff
    is a small hash/CRC check that traverses the control plane, not the data
    plane where the bandwidth bottleneck lives.

    Parameters
    ----------
    payload_bytes   : compressed delta size in bytes (from estimate_delta_size)
    net_params      : network parameter dict (latency_base_ms, packet_loss_rate, etc.)
    n_transmitting  : number of agents transmitting non-Copy blocks this step
    gateway_bw_mbps : total gateway downlink bandwidth shared across all ECUs
                      (separate override from net_params['bandwidth_mbps'] so that
                       coupled vs decoupled experiments can differ in gateway capacity
                       while using identical per-agent channel parameters)
    stochastic      : if True, sample latency from Truncated Gaussian (Phase 2 MARL)

    Returns
    -------
    tx_cost : float (same cost units as calculate_tx_cost, comparable across modes)
    """
    if stochastic:
        latency_ms = sample_latency_ms()
    else:
        latency_ms = net_params.get("latency_base_ms", 60.0)

    effective_bw_mbps = gateway_bw_mbps / max(n_transmitting, 1)

    latency_factor   = 1.0 + (latency_ms / 800.0)
    loss_factor      = 1.0 + (net_params.get("packet_loss_rate", 0.01) * 6.0)
    bandwidth_factor = 800.0 / max(effective_bw_mbps, 5.0)

    return payload_bytes * latency_factor * loss_factor * bandwidth_factor * 0.0008


# -------------------------------------------------------------
#  Network params loader (shared utility)
# -------------------------------------------------------------

_DEFAULT_PARAMS = {
    "latency_base_ms":        60.0,
    "packet_loss_rate":       0.01,
    "memory_budget_fraction": 1.0,
    "bandwidth_mbps":         50.0,
    "jitter_ms":              10.0,
    "monsoon_multiplier":     1.0,
}


def load_network_params(path: str = "network_params.json") -> dict:
    """Load vehicular network channel parameters, falling back to defaults."""
    params = dict(_DEFAULT_PARAMS)
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                params.update(json.load(f))
        except Exception:
            pass
    return params


def load_bd_params(path: str = "network_params.json") -> dict:
    """Deprecated alias for load_network_params. Kept for backwards compatibility."""
    return load_network_params(path)