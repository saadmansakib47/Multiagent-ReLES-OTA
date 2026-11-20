"""
marl_ota: Multi-Agent Reinforcement Learning for Coordinated Automotive OTA Updates.
"""

from marl_ota.envs.marl_ota_env import MultiAgentOTAEnv
from marl_ota.envs.ota_env import OTAEnv
from marl_ota.policies.fp3o_policy import FP3OPolicy, SharedBackbone, make_fp3o_policy_kwargs
from marl_ota.config import (
    FLEET_PRESETS,
    ENV_CFG,
    TRAIN_CFG,
    SAFETY_CFG,
    NET_CFG,
    ECU_CFG,
    BENCHMARK_CFG,
    PATHS_CFG,
)

__version__ = "0.2.0"
__all__ = [
    "MultiAgentOTAEnv",
    "OTAEnv",
    "FP3OPolicy",
    "SharedBackbone",
    "make_fp3o_policy_kwargs",
    "FLEET_PRESETS",
    "ENV_CFG",
    "TRAIN_CFG",
    "SAFETY_CFG",
    "NET_CFG",
    "ECU_CFG",
    "BENCHMARK_CFG",
    "PATHS_CFG",
]
