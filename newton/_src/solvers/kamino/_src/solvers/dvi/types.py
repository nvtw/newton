# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data containers for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.size import SizeKamino
from ...linalg import DenseLinearOperatorData
from ..padmm.types import PADMMSolution

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
mat33f = wp.mat33f
vec2f = wp.vec2f


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
    """Maximum projected Gauss-Seidel iterations for the fallback path."""

    block_iterations: int32
    """Outer direct-bilateral/projected-inequality block iterations."""

    contact_iterations: int32
    """Projected sweeps for unilateral inequalities in each direct-bilateral block."""

    bilateral_solve_period: int32
    """Block iteration period for repeated direct bilateral solves."""

    contact_jacobi_omega: float32
    """Step size for contact Jacobi and block-preconditioned updates."""

    contact_jacobi_relaxation: float32
    """Solution mixing for contact Jacobi and block-preconditioned updates."""

    contact_block_preconditioner: wp.bool
    """Whether to use the full contact 3x3 diagonal block for projected updates."""


@wp.struct
class DVIStatus:
    """Per-world DVI solver status.

    Field names intentionally match :class:`PADMMStatus` so existing benchmark
    code can read either solver through the same status path.
    """

    converged: int32
    iterations: int32
    """Projected iteration count; direct-bilateral solves report block sweeps."""
    r_p: float32
    r_d: float32
    r_c: float32
    r_b: float32


class DVIState:
    """Scratch arrays used by the DVI solver."""

    def __init__(self, size: SizeKamino | None = None):
        self.sigma: wp.array[vec2f] | None = None
        self.v_aug: wp.array[float32] | None = None
        self.s: wp.array[float32] | None = None
        self.scratch: wp.array[float32] | None = None
        self.bilateral_rhs: wp.array[float32] | None = None
        self.bilateral_solution: wp.array[float32] | None = None
        self.bilateral_preconditioner: wp.array[float32] | None = None
        self.bilateral_active_dim: wp.array[int32] | None = None
        self.contact_block_inv: wp.array[mat33f] | None = None
        self.contact_colors: wp.array[int32] | None = None
        self.contact_num_colors: wp.array[int32] | None = None
        self.world_mask: wp.array[wp.bool] | None = None
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
        self.bilateral_preconditioner = wp.zeros(size.sum_of_num_joint_cts, dtype=float32)
        self.bilateral_active_dim = wp.zeros(size.num_worlds, dtype=int32)
        self.contact_block_inv = wp.zeros(max(1, size.sum_of_max_contacts), dtype=mat33f)
        self.contact_colors = wp.full(max(1, size.sum_of_max_contacts), -1, dtype=int32)
        self.contact_num_colors = wp.zeros(max(1, size.num_worlds), dtype=int32)
        self.world_mask = wp.full(max(1, size.num_worlds), True, dtype=wp.bool)

    def reset(self):
        """Reset scratch arrays to zero."""
        self.sigma.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.scratch.zero_()
        self.bilateral_rhs.zero_()
        self.bilateral_solution.zero_()
        self.bilateral_preconditioner.zero_()
        self.bilateral_active_dim.zero_()
        self.contact_block_inv.zero_()
        self.contact_colors.fill_(-1)
        self.contact_num_colors.zero_()


class DVIData:
    """High-level DVI solver data."""

    def __init__(
        self,
        size: SizeKamino | None = None,
        device: wp.DeviceLike = None,
    ):
        self.config: wp.array[DVIConfigStruct] | None = None
        self.status: wp.array[DVIStatus] | None = None
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
    config_struct.block_iterations = config.block_iterations
    config_struct.contact_iterations = config.contact_iterations
    config_struct.bilateral_solve_period = config.bilateral_solve_period
    config_struct.contact_jacobi_omega = config.contact_jacobi_omega
    config_struct.contact_jacobi_relaxation = config.contact_jacobi_relaxation
    config_struct.contact_block_preconditioner = config.contact_block_preconditioner
    return config_struct
