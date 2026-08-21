# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ONNX interchange and graph-captured replay for FlashSAC actors."""

from __future__ import annotations

import importlib
import os

import numpy as np
import warp as wp

from .flash_sac import TrainerFlashSAC
from .flash_sac_networks import NetworkFlashSAC

_MODEL_KIND = "phoenx_flash_sac_actor"
_SCHEMA_VERSION = 1


def _require_onnx():
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as error:
        raise ImportError("FlashSAC ONNX support requires the 'onnx' extra") from error
    return onnx


def _require_onnx_runtime():
    try:
        OnnxRuntime = importlib.import_module("warp_nn.runtime").OnnxRuntime
    except ImportError as error:
        raise ImportError("FlashSAC ONNX replay requires Warp-NN with ONNX support") from error
    required_ops = {
        "Add",
        "BatchNormalization",
        "Div",
        "Gemm",
        "Mul",
        "ReduceMean",
        "Relu",
        "Sqrt",
        "Tanh",
    }
    if not required_ops <= getattr(OnnxRuntime, "supported_ops", frozenset()):
        raise ImportError("FlashSAC ONNX replay requires Warp-NN 0.3.1 or newer")
    return OnnxRuntime


def _network_initializers(network: NetworkFlashSAC) -> dict[str, np.ndarray]:
    arrays = {
        "embed.weight": network.embed_weight.numpy().T,
        "embed.bias": np.zeros(network.hidden_dim, dtype=np.float32),
        "input_norm.scale": network.input_norm.scale.numpy(),
        "input_norm.bias": network.input_norm.bias.numpy(),
        "input_norm.mean": network.input_norm.running_mean.numpy(),
        "input_norm.variance": network.input_norm.running_variance.numpy(),
        "rms.scale": network.rms_scale.numpy(),
        "mean.weight": network.head_weights[0].numpy().T,
        "mean.bias": network.head_biases[0].numpy(),
        "log_std.weight": network.head_weights[1].numpy().T,
        "log_std.bias": network.head_biases[1].numpy(),
    }
    for index, ((weight1, weight2), (norm1, norm2)) in enumerate(
        zip(network.block_weights, network.block_norms, strict=True)
    ):
        arrays[f"block.{index}.weight1"] = weight1.numpy().T
        arrays[f"block.{index}.bias1"] = np.zeros(network.hidden_dim * 4, dtype=np.float32)
        arrays[f"block.{index}.weight2"] = weight2.numpy().T
        arrays[f"block.{index}.bias2"] = np.zeros(network.hidden_dim, dtype=np.float32)
        for norm_index, norm in enumerate((norm1, norm2), start=1):
            prefix = f"block.{index}.norm{norm_index}"
            arrays[f"{prefix}.scale"] = norm.scale.numpy()
            arrays[f"{prefix}.bias"] = norm.bias.numpy()
            arrays[f"{prefix}.mean"] = norm.running_mean.numpy()
            arrays[f"{prefix}.variance"] = norm.running_variance.numpy()
    return {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}


def _metadata(network: NetworkFlashSAC) -> dict[str, str]:
    return {
        "newton.model_kind": _MODEL_KIND,
        "newton.schema_version": str(_SCHEMA_VERSION),
        "newton.input_dim": str(network.input_dim),
        "newton.action_dim": str(network.output_dim // 2),
        "newton.hidden_dim": str(network.hidden_dim),
        "newton.num_blocks": str(network.num_blocks),
        "newton.input_norm_epsilon": repr(network.input_norm.eps),
        "newton.rms_norm_epsilon": repr(1.0e-6),
        "newton.log_std_min": repr(network.log_std_min),
        "newton.log_std_max": repr(network.log_std_max),
    }


def export_flash_sac_actor_onnx(
    trainer: TrainerFlashSAC,
    path: str | os.PathLike[str],
    *,
    opset_version: int = 17,
) -> None:
    """Export a trained FlashSAC actor as a standard ONNX model.

    Args:
        trainer: Trainer containing the reference FlashSAC actor.
        path: Destination ONNX file.
        opset_version: ONNX operator-set version. Version 17 or newer is required.
    """

    onnx = _require_onnx()
    if opset_version < 17:
        raise ValueError("opset_version must be at least 17")
    network = trainer.actor.net
    if not isinstance(network, NetworkFlashSAC) or not network.actor_heads:
        raise ValueError("trainer must use the reference FlashSAC actor")

    helper = onnx.helper
    initializers = _network_initializers(network)
    initializers.update(
        {
            "rms.epsilon": np.asarray([1.0e-6], dtype=np.float32),
            "log_std.half_range": np.asarray([(network.log_std_max - network.log_std_min) * 0.5], dtype=np.float32),
            "log_std.midpoint": np.asarray([(network.log_std_max + network.log_std_min) * 0.5], dtype=np.float32),
        }
    )
    nodes = []

    def add_normalization(source: str, prefix: str, destination: str, epsilon: float) -> None:
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                [source, f"{prefix}.scale", f"{prefix}.bias", f"{prefix}.mean", f"{prefix}.variance"],
                [destination],
                epsilon=epsilon,
            )
        )

    add_normalization("observations", "input_norm", "input_normalized", network.input_norm.eps)
    nodes.append(helper.make_node("Gemm", ["input_normalized", "embed.weight", "embed.bias"], ["hidden.0"], transB=1))
    hidden = "hidden.0"
    for index, (norm1, norm2) in enumerate(network.block_norms):
        prefix = f"block.{index}"
        nodes.append(
            helper.make_node("Gemm", [hidden, f"{prefix}.weight1", f"{prefix}.bias1"], [f"{prefix}.pre1"], transB=1)
        )
        add_normalization(f"{prefix}.pre1", f"{prefix}.norm1", f"{prefix}.normalized1", norm1.eps)
        nodes.append(helper.make_node("Relu", [f"{prefix}.normalized1"], [f"{prefix}.activated1"]))
        nodes.append(
            helper.make_node(
                "Gemm",
                [f"{prefix}.activated1", f"{prefix}.weight2", f"{prefix}.bias2"],
                [f"{prefix}.pre2"],
                transB=1,
            )
        )
        add_normalization(f"{prefix}.pre2", f"{prefix}.norm2", f"{prefix}.normalized2", norm2.eps)
        nodes.append(helper.make_node("Relu", [f"{prefix}.normalized2"], [f"{prefix}.activated2"]))
        next_hidden = f"hidden.{index + 1}"
        nodes.append(helper.make_node("Add", [hidden, f"{prefix}.activated2"], [next_hidden]))
        hidden = next_hidden

    nodes.extend(
        (
            helper.make_node("Mul", [hidden, hidden], ["rms.squared"]),
            helper.make_node("ReduceMean", ["rms.squared"], ["rms.mean_square"], axes=[1], keepdims=1),
            helper.make_node("Add", ["rms.mean_square", "rms.epsilon"], ["rms.mean_square_epsilon"]),
            helper.make_node("Sqrt", ["rms.mean_square_epsilon"], ["rms.root"]),
            helper.make_node("Div", [hidden, "rms.root"], ["rms.unit"]),
            helper.make_node("Mul", ["rms.unit", "rms.scale"], ["normalized"]),
            helper.make_node("Gemm", ["normalized", "mean.weight", "mean.bias"], ["mean"], transB=1),
            helper.make_node("Gemm", ["normalized", "log_std.weight", "log_std.bias"], ["log_std.raw"], transB=1),
            helper.make_node("Tanh", ["log_std.raw"], ["log_std.tanh"]),
            helper.make_node("Mul", ["log_std.tanh", "log_std.half_range"], ["log_std.offset"]),
            helper.make_node("Add", ["log_std.offset", "log_std.midpoint"], ["log_std"]),
            helper.make_node("Tanh", ["mean"], ["actions"]),
        )
    )

    batch = "batch"
    input_info = helper.make_tensor_value_info("observations", onnx.TensorProto.FLOAT, [batch, network.input_dim])
    output_infos = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [batch, network.output_dim // 2])
        for name in ("actions", "mean", "log_std")
    ]
    graph = helper.make_graph(
        nodes,
        "PhoenX FlashSAC actor",
        [input_info],
        output_infos,
        [onnx.numpy_helper.from_array(value, name=name) for name, value in initializers.items()],
    )
    model = helper.make_model(
        graph,
        producer_name="newton.phoenx",
        opset_imports=[helper.make_opsetid("", int(opset_version))],
    )
    model.ir_version = 8
    for key, value in _metadata(network).items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save_model(model, os.fspath(path))


class PolicyFlashSACONNX:
    """Preallocated PhoenX replay policy loaded from a FlashSAC ONNX model.

    Args:
        path: FlashSAC actor ONNX file.
        batch_size: Fixed replay batch size.
        device: Warp device used for inference.
    """

    class Graph:
        """Captured fixed-input replay graph."""

        def __init__(self, graph: object, actions: wp.array2d[wp.float32]):
            self._graph = graph
            self.actions = actions

        def launch(self) -> wp.array2d[wp.float32]:
            """Replay the captured policy and return its fixed action buffer."""

            wp.capture_launch(self._graph)
            return self.actions

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        batch_size: int,
        device: wp.context.Devicelike = None,
    ):
        onnx = _require_onnx()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = onnx.load_model(os.fspath(path), load_external_data=True)
        onnx.checker.check_model(model)
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        if metadata.get("newton.model_kind") != _MODEL_KIND:
            raise ValueError("ONNX model is not a PhoenX FlashSAC actor")
        if int(metadata.get("newton.schema_version", "-1")) != _SCHEMA_VERSION:
            raise ValueError("unsupported FlashSAC ONNX schema version")

        self.device = wp.get_device(device)
        self.batch_size = int(batch_size)
        self.input_dim = int(metadata["newton.input_dim"])
        self.action_dim = int(metadata["newton.action_dim"])
        OnnxRuntime = _require_onnx_runtime()
        self._runtime = OnnxRuntime(
            os.fspath(path),
            device=self.device,
            batch_size=self.batch_size,
            input_batch_axes={"observations": 0},
        )
        if self._runtime.input_names != ["observations"] or "actions" not in self._runtime.output_names:
            raise ValueError("FlashSAC ONNX model has an invalid input or output interface")

    def act(self, observations: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        """Evaluate deterministic actions into the runtime-owned output buffer."""

        if observations.device != self.device:
            raise ValueError("observations and policy must use the same device")
        if observations.dtype != wp.float32 or tuple(observations.shape) != (self.batch_size, self.input_dim):
            raise ValueError(f"observations must have shape {(self.batch_size, self.input_dim)} and dtype float32")
        return self._runtime({"observations": observations})["actions"]

    def capture(self, observations: wp.array2d[wp.float32]) -> Graph:
        """Capture fixed-input deterministic replay as one CUDA graph."""

        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        actions = self.act(observations)
        with wp.ScopedCapture(device=self.device) as capture:
            self.act(observations)
        return self.Graph(capture.graph, actions)


def load_flash_sac_actor_onnx(
    path: str | os.PathLike[str],
    *,
    batch_size: int,
    device: wp.context.Devicelike = None,
) -> PolicyFlashSACONNX:
    """Load a preallocated FlashSAC actor replay policy from ONNX.

    Args:
        path: FlashSAC actor ONNX file.
        batch_size: Fixed replay batch size.
        device: Warp device used for inference.

    Returns:
        Loaded graph-compatible policy.
    """

    return PolicyFlashSACONNX(path, batch_size=batch_size, device=device)
