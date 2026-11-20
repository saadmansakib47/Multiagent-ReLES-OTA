"""
Root backward-compatibility shim for marl_ota.policies.fp3o_policy.
Allows legacy scripts and tests to continue doing 'from fp3o_policy import ...' seamlessly.
"""
from marl_ota.policies.fp3o_policy import *
from marl_ota.policies.fp3o_policy import (
    FP3OPolicy,
    SharedBackbone,
    ActionHead,
    PositionHead,
    make_fp3o_policy_kwargs,
    ECU_TYPES,
    ECU_TYPE_TO_IDX,
    ecu_type_to_idx,
)
