"""
Root backward-compatibility shim for marl_ota.envs.ota_env.
Allows legacy scripts to continue doing 'from ota_env import ...' seamlessly.
"""
from marl_ota.envs.ota_env import *
