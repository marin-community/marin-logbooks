# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Snowball 67B-A2B model as the cold-start SFT reproduction ran it on H100s (marin#10799).

The values are the model block of that run's production launch config. The parameter tree of
``moe_hero_ep.model.Transformer`` has not changed since that run's fork of it, so a checkpoint it wrote
restores into this config.
"""

import jmp
from levanter.grug.attention import RotaryConfig

from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator

SNOWBALL_MAX_SEQ_LEN = 262_144
SNOWBALL_67B_MODEL_CONFIG = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=2560,
    intermediate_dim=1280,
    shared_expert_intermediate_dim=2560,
    num_shared_experts=1,
    num_experts=256,
    num_experts_per_token=4,
    latent_dim=None,
    qb_estimator=QbEstimator.TOPK,
    qb_hist_bins=1000,
    num_layers=26,
    num_heads=20,
    num_kv_heads=5,
    local_kv_heads=None,
    global_kv_heads=None,
    head_dim=128,
    max_seq_len=SNOWBALL_MAX_SEQ_LEN,
    sliding_window=2048,
    global_every=4,
    capacity_factor=1.0,
    layer_norm_eps=1e-5,
    initializer_std=0.009882117688026186,
    qk_mult=1.75,
    qk_mult_long_scale=1.0,
    sconv=False,
    attention_implementation="gpu_fa4_cute",
    # The reproduction's FA4 always used the zigzag query layout under context parallelism.
    attention_context_layout="zigzag",
    moe_implementation="sonic",
    expert_chunks=1,
    num_expert_waves=1,
    remat_mode="recompute_all",
    rope=RotaryConfig(theta=10000.0, scaling_factor=None),
    rope_fused=False,
)

# The reproduction stored and computed in bfloat16; float32 parameters would not fit one node.
SNOWBALL_MIXED_PRECISION = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")

SNOWBALL_GPUS_PER_NODE = 8
SNOWBALL_NODE_CPU = 64
SNOWBALL_NODE_RAM = "1500g"
SNOWBALL_NODE_DISK = "500g"
