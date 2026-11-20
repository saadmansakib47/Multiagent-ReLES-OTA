"""
Root backward-compatibility shim for marl_ota.baselines.baselines.
Allows legacy scripts to continue doing 'from baselines import ...' seamlessly.
"""
from marl_ota.baselines.baselines import *
from marl_ota.baselines.baselines import BaselineRunner
