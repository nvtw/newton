# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import (
    _LegacyLstmTorchAdapter,
    _load_legacy_lstm_torch_checkpoint,
    _TorchModuleAdapter,
    load_checkpoint,
    load_metadata,
)
from .base import Controller


@wp.kernel
def _compute_errors_kernel(
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    pos_scale: float,
    vel_scale: float,
    out: wp.array3d[float],
):
    i = wp.tid()
    pi = pos_indices[i]
    vi = vel_indices[i]
    tpi = target_pos_indices[i]
    tvi = target_vel_indices[i]
    out[0, i, 0] = (target_pos[tpi] - positions[pi]) * pos_scale
    out[0, i, 1] = (target_vel[tvi] - velocities[vi]) * vel_scale


@wp.kernel
def _scale_effort_to_forces_kernel(
    src: wp.array2d[float],
    dst: wp.array[float],
    scale: float,
    cols: int,
):
    i = wp.tid()
    row = i // cols
    col = i % cols
    dst[i] = src[row, col] * scale


@wp.kernel
def _zero_masked_3d_kernel(buf: wp.array3d[float], mask: wp.array[wp.bool]):
    layer, b, h = wp.tid()
    if mask[b]:
        buf[layer, b, h] = 0.0


class ControllerNeuralLSTM(Controller):
    """LSTM-based neural network controller, ONNX-backed.

    Uses a pre-trained LSTM loaded from an ``.onnx`` file to compute joint
    effort from position error and velocity error. Hidden and cell state are
    maintained across timesteps.

    The exported ONNX model must have three inputs: input, initial hidden, and
    initial cell. It must have three graph outputs: effort, hidden output, and
    cell output. Metadata properties map those names to controller roles.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(Controller.State):
        """LSTM hidden and cell state."""

        hidden: wp.array3d[float] | None = None
        """LSTM hidden state, shape [num_layers, actuator_count, hidden_size]."""
        cell: wp.array3d[float] | None = None
        """LSTM cell state, shape [num_layers, actuator_count, hidden_size]."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.hidden.zero_()
                self.cell.zero_()
            else:
                wp.launch(
                    _zero_masked_3d_kernel,
                    dim=self.hidden.shape,
                    inputs=[self.hidden, mask],
                    device=self.hidden.device,
                )
                wp.launch(
                    _zero_masked_3d_kernel,
                    dim=self.cell.shape,
                    inputs=[self.cell, mask],
                    device=self.cell.device,
                )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "model_path" not in args:
            raise ValueError("ControllerNeuralLSTM requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("ControllerNeuralLSTM requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize LSTM controller from a checkpoint file.

        Args:
            model_path: Path to the ``.onnx`` checkpoint. Legacy ``.pt`` and
                ``.pth`` checkpoints are accepted during the deprecation window.
        """
        self.model_path = model_path

        self._legacy_adapter: _LegacyLstmTorchAdapter | None = None
        if model_path.lower().endswith((".pt", ".pth")):
            adapter, metadata = _load_legacy_lstm_torch_checkpoint(model_path)
            self._legacy_adapter = adapter
        else:
            metadata = load_metadata(model_path)

        self.pos_scale = float(metadata.get("pos_scale", 1.0))
        self.vel_scale = float(metadata.get("vel_scale", 1.0))
        self.effort_scale = float(metadata.get("effort_scale", metadata.get("torque_scale", 1.0)))

        for key in (
            "input_name",
            "hidden_in_name",
            "cell_in_name",
            "output_name",
            "hidden_out_name",
            "cell_out_name",
            "num_layers",
            "hidden_size",
        ):
            if key not in metadata:
                raise ValueError(f"ONNX metadata missing required key '{key}'")

        self._input_name = metadata["input_name"]
        self._hidden_in_name = metadata["hidden_in_name"]
        self._cell_in_name = metadata["cell_in_name"]
        self._output_name = metadata["output_name"]
        self._hidden_out_name = metadata["hidden_out_name"]
        self._cell_out_name = metadata["cell_out_name"]

        self._num_layers = int(metadata["num_layers"])
        self._hidden_size = int(metadata["hidden_size"])

        self._network = None
        self._device: wp.Device | None = None
        self._num_actuators = 0
        self._net_input: wp.array3d[float] | None = None
        self._next_hidden: wp.array3d[float] | None = None
        self._next_cell: wp.array3d[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._device = device
        self._num_actuators = num_actuators

        if self._legacy_adapter is not None:
            self._legacy_adapter._device = device
            self._legacy_adapter._torch_device = self._legacy_adapter._resolve_torch_device(device)
            self._legacy_adapter._model = self._legacy_adapter._model.to(self._legacy_adapter._torch_device)
            self._network = self._legacy_adapter
        else:
            runtime, _ = load_checkpoint(
                self.model_path,
                device=device,
                batch_size=num_actuators,
                input_batch_axes={
                    self._input_name: 1,
                    self._hidden_in_name: 1,
                    self._cell_in_name: 1,
                },
            )
            self._network = runtime

            out_shape = runtime._shapes[self._output_name]
            if out_shape != (num_actuators, 1):
                raise ValueError(
                    f"ControllerNeuralLSTM: ONNX output '{self._output_name}' has shape {out_shape}, "
                    f"expected {(num_actuators, 1)} (one scalar effort per actuator)"
                )

            for role, name in (
                ("hidden_out", self._hidden_out_name),
                ("cell_out", self._cell_out_name),
            ):
                if name not in runtime._shapes:
                    raise ValueError(
                        f"ControllerNeuralLSTM: ONNX model is missing the '{role}' output '{name}'; "
                        f"available outputs: {sorted(runtime._shapes)}"
                    )
                state_shape = runtime._shapes[name]
                expected_state_shape = (self._num_layers, num_actuators, self._hidden_size)
                if tuple(state_shape) != expected_state_shape:
                    raise ValueError(
                        f"ControllerNeuralLSTM: ONNX output '{name}' has shape {tuple(state_shape)}, "
                        f"expected {expected_state_shape} (num_layers, num_actuators, hidden_size)"
                    )

        self._net_input = wp.zeros((1, num_actuators, 2), dtype=wp.float32, device=device)
        self._next_hidden = wp.zeros(
            (self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device
        )
        self._next_cell = wp.zeros(
            (self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device
        )

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return not isinstance(self._network, (_TorchModuleAdapter, _LegacyLstmTorchAdapter))

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralLSTM.State:
        return ControllerNeuralLSTM.State(
            hidden=wp.zeros((self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device),
            cell=wp.zeros((self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerNeuralLSTM.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        device = device or self._device
        n = self._num_actuators

        wp.launch(
            _compute_errors_kernel,
            dim=n,
            inputs=[
                target_pos,
                target_vel,
                positions,
                velocities,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self.pos_scale,
                self.vel_scale,
                self._net_input,
            ],
            device=device,
        )

        out = self._network(
            {
                self._input_name: self._net_input,
                self._hidden_in_name: state.hidden,
                self._cell_in_name: state.cell,
            }
        )
        effort = out[self._output_name]
        hidden_new = out[self._hidden_out_name]
        cell_new = out[self._cell_out_name]

        wp.copy(self._next_hidden, hidden_new.reshape((self._num_layers, n, self._hidden_size)))
        wp.copy(self._next_cell, cell_new.reshape((self._num_layers, n, self._hidden_size)))

        wp.launch(
            _scale_effort_to_forces_kernel,
            dim=len(forces),
            inputs=[effort, forces, self.effort_scale, 1],
            device=device,
        )

    def update_state(
        self,
        current_state: ControllerNeuralLSTM.State,
        next_state: ControllerNeuralLSTM.State,
    ) -> None:
        if next_state is None:
            return
        wp.copy(next_state.hidden, self._next_hidden)
        wp.copy(next_state.cell, self._next_cell)
