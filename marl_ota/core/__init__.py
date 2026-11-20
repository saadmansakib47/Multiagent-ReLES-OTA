"""
Core CPS vehicular and network primitives for OTA updates.
"""
from marl_ota.core.ota_core import (
    estimate_delta_size,
    calculate_encoding_cost,
    calculate_tx_cost,
    calculate_tx_cost_coupled,
    calculate_memory_overhead,
    generate_firmware_blocks,
    load_network_params,
    load_bd_params,
)

__all__ = [
    "estimate_delta_size",
    "calculate_encoding_cost",
    "calculate_tx_cost",
    "calculate_tx_cost_coupled",
    "calculate_memory_overhead",
    "generate_firmware_blocks",
    "load_network_params",
    "load_bd_params",
]
