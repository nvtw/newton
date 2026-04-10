# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ONNX export for PufferLib SimpleMLP policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from newton._src.pufferlib.network import SimpleMLP


def export_policy_to_onnx(
    policy: SimpleMLP,
    obs_dim: int,
    num_actions: int,
    path: str,
    obs_mean: np.ndarray | None = None,
    obs_var: np.ndarray | None = None,
) -> None:
    """Export a trained SimpleMLP actor to ONNX (strips value head).

    The resulting ONNX model takes ``(batch, obs_dim)`` float observations
    and produces ``(batch, num_actions)`` float actions (the policy mean).
    Uses ReLU activations matching SimpleMLP's architecture.

    When *obs_mean* and *obs_var* are provided, observation normalization
    is baked into the model: ``(obs - mean) / sqrt(var + eps)``, clipped to [-10, 10].

    Args:
        policy: The trained SimpleMLP policy.
        obs_dim: Observation dimension.
        num_actions: Number of action outputs (value column is stripped).
        path: Output ``.onnx`` file path.
        obs_mean: Running mean from :class:`ObsNormalizer`.
        obs_var: Running variance from :class:`ObsNormalizer`.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    initializers = []
    prev_output = "observation"

    # Bake observation normalization
    if obs_mean is not None and obs_var is not None:
        inv_std = (1.0 / np.sqrt(obs_var + 1e-4)).astype(np.float32)
        initializers.append(numpy_helper.from_array(obs_mean.astype(np.float32), name="obs_mean"))
        initializers.append(numpy_helper.from_array(inv_std, name="obs_inv_std"))
        nodes.append(helper.make_node("Sub", [prev_output, "obs_mean"], ["/norm/sub"]))
        nodes.append(helper.make_node("Mul", ["/norm/sub", "obs_inv_std"], ["/norm/mul"]))
        initializers.append(numpy_helper.from_array(np.float32(-10.0), name="clip_min"))
        initializers.append(numpy_helper.from_array(np.float32(10.0), name="clip_max"))
        nodes.append(helper.make_node("Clip", ["/norm/mul", "clip_min", "clip_max"], ["/norm/out"]))
        prev_output = "/norm/out"

    # Determine ONNX activation op matching the training network
    act_op = "Elu" if getattr(policy, "_activation", "relu") == "elu" else "Relu"

    # Layer 1: x @ w1^T -> activation
    w1 = policy.w1.numpy()  # (hidden, obs_dim)
    initializers.append(numpy_helper.from_array(w1.astype(np.float32), name="w1"))
    nodes.append(helper.make_node("Gemm", [prev_output, "w1"], ["pre_h1"], alpha=1.0, beta=0.0, transB=1))
    nodes.append(helper.make_node(act_op, ["pre_h1"], ["h1"]))
    prev_output = "h1"

    # Layer 2: h1 @ w2^T -> activation
    w2 = policy.w2.numpy()  # (hidden, hidden)
    initializers.append(numpy_helper.from_array(w2.astype(np.float32), name="w2"))
    nodes.append(helper.make_node("Gemm", [prev_output, "w2"], ["pre_h2"], alpha=1.0, beta=0.0, transB=1))
    nodes.append(helper.make_node(act_op, ["pre_h2"], ["h2"]))
    prev_output = "h2"

    # Layer 3: h2 @ w3^T (decoder, out_dim = num_actions + 1)
    # Strip value column: only take first num_actions columns of w3
    w3_full = policy.w3.numpy()  # (num_actions + 1, hidden)
    w3_actor = w3_full[:num_actions, :]  # (num_actions, hidden)
    initializers.append(numpy_helper.from_array(w3_actor.astype(np.float32), name="w3"))
    nodes.append(helper.make_node("Gemm", [prev_output, "w3"], ["action"], alpha=1.0, beta=0.0, transB=1))

    graph = helper.make_graph(
        nodes,
        "pufferlib_actor",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch_size", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch_size", num_actions])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)


def save_checkpoint(policy: SimpleMLP, path: str) -> None:
    """Save policy weights as numpy arrays for resuming training."""
    weights = {
        "w1": policy.w1.numpy(),
        "w2": policy.w2.numpy(),
        "w3": policy.w3.numpy(),
    }
    if policy.logstd is not None:
        weights["logstd"] = policy.logstd.numpy()
    np.savez(path, **weights)


def load_checkpoint(policy: SimpleMLP, path: str) -> None:
    """Load policy weights from a checkpoint file."""
    import warp as wp

    data = np.load(path)
    wp.copy(policy.w1, wp.array(data["w1"], dtype=float, device=policy.device))
    wp.copy(policy.w2, wp.array(data["w2"], dtype=float, device=policy.device))
    wp.copy(policy.w3, wp.array(data["w3"], dtype=float, device=policy.device))
    if "logstd" in data and policy.logstd is not None:
        wp.copy(policy.logstd, wp.array(data["logstd"], dtype=float, device=policy.device))
