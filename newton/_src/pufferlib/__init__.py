# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PufferLib PPO implementation built on Warp.

All kernels use ``enable_backward=False`` with hand-written backward passes
for fast compile times.

Modules
-------
- :mod:`kernels` — Core GEMM, activations, elementwise ops, weight init.
- :mod:`network` — SimpleMLP policy (encoder → ReLU → hidden → ReLU → decoder).
- :mod:`ppo` — Fused PPO loss+grad, action sampling, GAE+V-Trace.
- :mod:`optimizer` — Muon (Newton-Schulz) and AdamW optimizers.
- :mod:`trainer` — Shared PPO training loop with CUDA graph capture.
- :mod:`reduce` — Parallel array reductions.
"""

from newton._src.pufferlib.kernels import (  # noqa: F401
    relu,
    sigmoid,
    softplus,
    fast_tanh,
    gelu,
    matmul,
)
from newton._src.pufferlib.network import SimpleMLP  # noqa: F401
from newton._src.pufferlib.ppo import (  # noqa: F401
    sample_actions_discrete,
    sample_actions_continuous,
    compute_gae_vtrace,
    ppo_loss_and_grad,
    priority_sample,
    PrioritySampler,
    PPOLossBuffers,
)
from newton._src.pufferlib.reduce import ArraySum, array_prefix_sum  # noqa: F401
from newton._src.pufferlib.optimizer import Muon, AdamW  # noqa: F401
from newton._src.pufferlib.trainer import PPOTrainer, PPOConfig  # noqa: F401
