"""
Root backward-compatibility shim for marl_ota.envs.marl_ota_env.
Allows legacy scripts and tests to continue doing 'from marl_ota_env import ...' seamlessly.
"""
from marl_ota.envs.marl_ota_env import *
from marl_ota.envs.marl_ota_env import MultiAgentOTAEnv
