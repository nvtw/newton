# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..utils import load_checkpoint, load_metadata
from .base import Controller


@wp.kernel
def _compute_errors_and_zero_history_kernel(
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    pos_error: wp.array[float],
    vel_error: wp.array[float],
):
    i = wp.tid()
    pi = pos_indices[i]
    vi = vel_indices[i]
    tpi = target_pos_indices[i]
    tvi = target_vel_indices[i]
    pos_error[i] = target_pos[tpi] - positions[pi]
    vel_error[i] = target_vel[tvi] - velocities[vi]


@wp.kernel
def _scale_pair_kernel(
    pos_error: wp.array[float],
    vel_error: wp.array[float],
    pos_scale: float,
    vel_scale: float,
    out: wp.array2d[float],
    n_per_block: int,
    pos_first: int,
):
    i, k = wp.tid()  # i in [0, N), k in [0, 2*history_length)
    block = k // n_per_block          # 0 -> first half, 1 -> second half
    is_pos = block == 0 if pos_first != 0 else block == 1
    if is_pos:
        out[i, k] = pos_error[i] * pos_scale
    else:
        out[i, k] = vel_error[i] * vel_scale


@wp.kernel
def _scale_and_copy_kernel(
    src: wp.array2d[float],
    dst: wp.array[float],
    scale: float,
    n: int,
):
    i = wp.tid()
    if i < n:
        dst[i] = src[0, i] * scale


@wp.kernel
def _zero_masked_2d_kernel(buf: wp.array2d[float], mask: wp.array[wp.bool]):
    i, j = wp.tid()
    if mask[j]:
        buf[i, j] = 0.0


class ControllerNeuralMLP(Controller):
    """MLP-based neural network controller, ONNX-backed.

    Uses a pre-trained MLP (loaded from an ``.onnx`` file) to compute joint
    effort from concatenated, scaled position-error and velocity-error
    history.  The output is multiplied by ``effort_scale`` to convert from
    network units to physical effort [N or N·m].

    Configuration parameters (``input_order``, ``input_idx``,
    ``pos_scale``, ``vel_scale``, ``effort_scale``) are read from the ONNX
    model's metadata properties (a single ``metadata`` JSON property is
    preferred), falling back to defaults when absent.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(Controller.State):
        """History buffers for MLP controller."""

        pos_error_history: wp.array2d[float] | None = None
        """Position error history, shape (history_length, N)."""
        vel_error_history: wp.array2d[float] | None = None
        """Velocity error history, shape (history_length, N)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.pos_error_history.zero_()
                self.vel_error_history.zero_()
            else:
                wp.launch(
                    _zero_masked_2d_kernel,
                    dim=self.pos_error_history.shape,
                    inputs=[self.pos_error_history, mask],
                    device=self.pos_error_history.device,
                )
                wp.launch(
                    _zero_masked_2d_kernel,
                    dim=self.vel_error_history.shape,
                    inputs=[self.vel_error_history, mask],
                    device=self.vel_error_history.device,
                )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "model_path" not in args:
            raise ValueError("ControllerNeuralMLP requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("ControllerNeuralMLP requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize MLP controller from an ONNX checkpoint file.

        Configuration is read from the model's metadata properties:

        - ``input_order`` (str): ``"pos_vel"`` or ``"vel_pos"`` (default ``"pos_vel"``).
        - ``input_idx`` (list[int]): history timestep indices (default ``[0]``).
        - ``pos_scale`` (float): position-error scaling (default ``1.0``).
        - ``vel_scale`` (float): velocity-error scaling (default ``1.0``).
        - ``effort_scale`` (float): output effort scaling (default ``1.0``).

        Args:
            model_path: Path to the ``.onnx`` checkpoint.
        """
        self.model_path = model_path

        metadata = load_metadata(model_path)

        self.input_order = metadata.get("input_order", "pos_vel")
        if self.input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{self.input_order}'")

        self.input_idx = metadata.get("input_idx", [0])
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        self.pos_scale = float(metadata.get("pos_scale", 1.0))
        self.vel_scale = float(metadata.get("vel_scale", 1.0))
        self.effort_scale = float(metadata.get("effort_scale", metadata.get("torque_scale", 1.0)))

        self._network = None
        self._device: wp.Device | None = None
        self._num_actuators = 0

        self._pos_error: wp.array[float] | None = None
        self._vel_error: wp.array[float] | None = None
        self._net_input: wp.array2d[float] | None = None
        self._net_output_name: str | None = None
        self._net_input_name: str | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._device = device
        self._num_actuators = num_actuators

        runtime, _ = load_checkpoint(self.model_path, device=str(device), batch_size=num_actuators)
        self._network = runtime
        self._net_input_name = runtime.input_names[0]
        self._net_output_name = runtime.output_names[0]

        # input shape: (N, 2 * history_length)  (pos history, vel history)
        feat = 2 * len(self.input_idx)
        self._net_input = wp.zeros((num_actuators, feat), dtype=wp.float32, device=device)
        self._pos_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._vel_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralMLP.State:
        return ControllerNeuralMLP.State(
            pos_error_history=wp.zeros((self.history_length, num_actuators), dtype=wp.float32, device=device),
            vel_error_history=wp.zeros((self.history_length, num_actuators), dtype=wp.float32, device=device),
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
        state: ControllerNeuralMLP.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        device = device or self._device
        n = self._num_actuators

        wp.launch(
            _compute_errors_and_zero_history_kernel,
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
                self._pos_error,
                self._vel_error,
            ],
            device=device,
        )

        # Build net_input: stack [pos_error_history[i] for i in input_idx] then vel ones
        # We do this on host since input_idx is small and may be non-contiguous.
        pos_err_np = self._pos_error.numpy()
        vel_err_np = self._vel_error.numpy()
        history_pos = state.pos_error_history.numpy() if self.history_length > 1 else None
        history_vel = state.vel_error_history.numpy() if self.history_length > 1 else None

        pos_cols = []
        vel_cols = []
        for i in self.input_idx:
            if i == 0:
                pos_cols.append(pos_err_np)
                vel_cols.append(vel_err_np)
            else:
                pos_cols.append(history_pos[i - 1])
                vel_cols.append(history_vel[i - 1])
        pos_stack = np.stack(pos_cols, axis=1).astype(np.float32) * self.pos_scale  # (N, k)
        vel_stack = np.stack(vel_cols, axis=1).astype(np.float32) * self.vel_scale  # (N, k)

        if self.input_order == "pos_vel":
            net_input_np = np.concatenate([pos_stack, vel_stack], axis=1)
        else:
            net_input_np = np.concatenate([vel_stack, pos_stack], axis=1)

        net_input_wp = wp.array(np.ascontiguousarray(net_input_np), dtype=wp.float32, device=device)
        out = self._network({self._net_input_name: net_input_wp})
        effort = out[self._net_output_name]

        wp.launch(
            _scale_and_copy_kernel,
            dim=len(forces),
            inputs=[effort, forces, self.effort_scale, len(forces)],
            device=device,
        )

    def update_state(
        self,
        current_state: ControllerNeuralMLP.State,
        next_state: ControllerNeuralMLP.State,
    ) -> None:
        if next_state is None:
            return
        # Roll history along axis 0 (oldest sample drops out, newest at index 0).
        cur_pos_np = current_state.pos_error_history.numpy()
        cur_vel_np = current_state.vel_error_history.numpy()
        rolled_pos = np.roll(cur_pos_np, 1, axis=0)
        rolled_vel = np.roll(cur_vel_np, 1, axis=0)
        rolled_pos[0] = self._pos_error.numpy()
        rolled_vel[0] = self._vel_error.numpy()

        wp.copy(
            next_state.pos_error_history,
            wp.array(np.ascontiguousarray(rolled_pos), dtype=wp.float32, device=self._device),
        )
        wp.copy(
            next_state.vel_error_history,
            wp.array(np.ascontiguousarray(rolled_vel), dtype=wp.float32, device=self._device),
        )
