# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data containers for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.size import SizeKamino
from ...core.types import float32, int32, vec2f
from ...linalg import DenseLinearOperatorData
from ..padmm.types import PADMMSolution

wp.set_module_options({"enable_backward": False})


@wp.struct
class DVIConfigStruct:
    """On-device DVI solver configuration."""

    tolerance: float32
    """Natural-map convergence tolerance."""

    regularization: float32
    """Diagonal regularization used by projected Gauss-Seidel updates."""

    omega: float32
    """Projected Gauss-Seidel update relaxation."""

    max_iterations: int32
    """Maximum number of projected Gauss-Seidel iterations."""


@wp.struct
class DVIStatus:
    """Per-world DVI solver status.

    Field names intentionally match :class:`PADMMStatus` so existing benchmark
    code can read either solver through the same status path.
    """

    converged: int32
    iterations: int32
    r_p: float32
    r_d: float32
    r_c: float32


class DVIState:
    """Scratch arrays used by the DVI solver."""

    def __init__(self, size: SizeKamino | None = None):
        self.sigma: wp.array | None = None
        self.v_aug: wp.array | None = None
        self.s: wp.array | None = None
        self.scratch: wp.array | None = None
        self.bilateral_rhs: wp.array | None = None
        self.bilateral_solution: wp.array | None = None
        if size is not None:
            self.finalize(size)

    def finalize(self, size: SizeKamino):
        """Allocate scratch arrays for the supplied model size."""
        self.sigma = wp.zeros(size.num_worlds, dtype=vec2f)
        self.v_aug = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.scratch = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.bilateral_rhs = wp.zeros(size.sum_of_num_joint_cts, dtype=float32)
        self.bilateral_solution = wp.zeros(size.sum_of_num_joint_cts, dtype=float32)

    def reset(self):
        """Reset scratch arrays to zero."""
        self.sigma.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.scratch.zero_()
        self.bilateral_rhs.zero_()
        self.bilateral_solution.zero_()


class DVIData:
    """High-level DVI solver data."""

    def __init__(
        self,
        size: SizeKamino | None = None,
        device: wp.DeviceLike = None,
    ):
        self.config: wp.array | None = None
        self.status: wp.array | None = None
        self.state: DVIState | None = None
        self.solution: PADMMSolution | None = None
        self.bilateral_operator: DenseLinearOperatorData | None = None
        if size is not None:
            self.finalize(size=size, device=device)

    def finalize(self, size: SizeKamino, device: wp.DeviceLike = None):
        """Allocate DVI data arrays."""
        with wp.ScopedDevice(device):
            self.config = wp.zeros(shape=(size.num_worlds,), dtype=DVIConfigStruct)
            self.status = wp.zeros(shape=(size.num_worlds,), dtype=DVIStatus)
            self.state = DVIState(size)
            self.solution = PADMMSolution(size)
            self.bilateral_operator = None


def convert_config_to_struct(config: DVISolverConfig) -> DVIConfigStruct:
    """Convert a host-side DVI config to an on-device struct."""
    config_struct = DVIConfigStruct()
    config_struct.tolerance = config.tolerance
    config_struct.regularization = config.regularization
    config_struct.omega = config.omega
    config_struct.max_iterations = config.max_iterations
    return config_struct
