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

    # Build layers dynamically from the policy's weight list
    weights = policy._weights
    biases = getattr(policy, "_biases", [None] * len(weights))
    num_layers = len(weights)

    for i in range(num_layers):
        w_np = weights[i].numpy().astype(np.float32)
        is_last = (i == num_layers - 1)

        # For the last layer, strip the value column (keep only action dims)
        if is_last:
            w_np = w_np[:num_actions, :]

        w_name = f"w{i}"
        out_name = "action" if is_last else f"h{i}"
        gemm_out = f"pre_{out_name}" if not is_last else out_name

        initializers.append(numpy_helper.from_array(w_np, name=w_name))

        if biases[i] is not None:
            b_np = biases[i].numpy().astype(np.float32)
            if is_last:
                b_np = b_np[:num_actions]
            b_name = f"b{i}"
            initializers.append(numpy_helper.from_array(b_np, name=b_name))
            nodes.append(helper.make_node("Gemm", [prev_output, w_name, b_name], [gemm_out],
                                          alpha=1.0, beta=1.0, transB=1))
        else:
            nodes.append(helper.make_node("Gemm", [prev_output, w_name], [gemm_out],
                                          alpha=1.0, beta=0.0, transB=1))

        if not is_last:
            act_out = f"h{i}"
            nodes.append(helper.make_node(act_op, [gemm_out], [act_out]))
            prev_output = act_out
        else:
            prev_output = out_name

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
    if policy.std is not None:
        weights["std"] = policy.std.numpy()
    np.savez(path, **weights)


def load_checkpoint(policy: SimpleMLP, path: str) -> None:
    """Load policy weights from a checkpoint file."""
    import warp as wp

    data = np.load(path)
    wp.copy(policy.w1, wp.array(data["w1"], dtype=float, device=policy.device))
    wp.copy(policy.w2, wp.array(data["w2"], dtype=float, device=policy.device))
    wp.copy(policy.w3, wp.array(data["w3"], dtype=float, device=policy.device))
    if "std" in data and policy.std is not None:
        wp.copy(policy.std, wp.array(data["std"], dtype=float, device=policy.device))
