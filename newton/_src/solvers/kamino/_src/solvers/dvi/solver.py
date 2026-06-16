# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""DVI-style projected solver for Kamino's dual dynamics system."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.data import DataKamino
from ...core.model import ModelKamino
from ...core.size import SizeKamino
from ...dynamics.dual import DualProblem
from ...geometry.contacts import ContactsKamino
from ...kinematics.limits import LimitsKamino
from ..base import ForwardDynamicsSolver
from ..padmm.types import PADMMWarmStartMode
from ..padmm.kernels import (
    _apply_dual_preconditioner_to_solution,
    _warmstart_contact_constraints,
    _warmstart_joint_constraints,
    _warmstart_limit_constraints,
)
from .kernels import (
    _compute_dvi_desaxce_corrections,
    _compute_dvi_solution_vectors,
    _reset_dvi_solver_data,
    _solve_dvi_pgs,
    _unprecondition_dvi_solution,
)
from .types import DVIConfigStruct, DVIData, convert_config_to_struct

wp.set_module_options({"enable_backward": False})


class DVISolver(ForwardDynamicsSolver):
    """Projected Gauss-Seidel DVI solver for Kamino ``DualProblem`` systems."""

    Config = DVISolverConfig

    def __init__(
        self,
        model: ModelKamino | None = None,
        config: list[DVISolver.Config] | DVISolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        collect_info: bool = False,
    ):
        self._config: list[DVISolver.Config] = []
        self._warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE
        self._collect_info: bool = False
        self._size: SizeKamino | None = None
        self._data: DVIData | None = None
        self._device: wp.DeviceLike = None

        if model is not None:
            self.finalize(model=model, config=config, warmstart=warmstart, collect_info=collect_info)

    @property
    def config(self) -> list[DVISolver.Config]:
        """Host-side per-world DVI configs."""
        return self._config

    @property
    def size(self) -> SizeKamino:
        """Model size cache."""
        return self._size

    @property
    def data(self) -> DVIData:
        """Solver data arrays."""
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `finalize()` first.")
        return self._data

    @property
    def device(self) -> wp.DeviceLike:
        """Device on which solver data is allocated."""
        return self._device

    def finalize(
        self,
        model: ModelKamino,
        config: list[DVISolver.Config] | DVISolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        collect_info: bool = False,
    ):
        """Allocate DVI solver data for ``model``."""
        if model is None or not isinstance(model, ModelKamino):
            raise ValueError("A model of type `ModelKamino` must be provided.")

        self._size = model.size
        self._device = model.device
        self._config = self._check_config(model, config)
        self._warmstart = warmstart
        self._collect_info = collect_info
        self._data = DVIData(size=self._size, device=self._device)

        configs = [convert_config_to_struct(c) for c in self._config]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=DVIConfigStruct)

    @staticmethod
    def _check_config(
        model: ModelKamino | None = None, config: list[DVISolver.Config] | DVISolver.Config | None = None
    ) -> list[DVISolver.Config]:
        if config is None:
            config = [DVISolver.Config()] * (model.info.num_worlds if model else 1)
        elif isinstance(config, DVISolver.Config):
            config = [config] * (model.info.num_worlds if model else 1)
        elif isinstance(config, list):
            if model is not None and len(config) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} configs, got {len(config)}")
            if not all(isinstance(c, DVISolver.Config) for c in config):
                raise TypeError("All configs must be instances of DVISolver.Config")
        else:
            raise TypeError(f"Expected a single object or list of `DVISolver.Config`, got {type(config)}")
        return config

    def reset(self, problem: DualProblem | None = None, world_mask: wp.array | None = None):
        """Reset scratch state and cached solution data."""
        self._data.state.reset()
        if world_mask is None:
            self._data.solution.zero()
        else:
            if problem is None:
                raise ValueError("A `DualProblem` instance must be provided when a world mask is used.")
            wp.launch(
                kernel=_reset_dvi_solver_data,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    world_mask,
                    problem.data.vio,
                    problem.data.maxdim,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

    def coldstart(self):
        """Prepare a cold-start solve."""
        self._data.state.reset()
        self._data.solution.zero()

    def warmstart(
        self,
        problem: DualProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """Prepare a warm-start solve."""
        self._data.state.reset()

        match self._warmstart:
            case PADMMWarmStartMode.NONE:
                self._data.solution.zero()
            case PADMMWarmStartMode.INTERNAL:
                self._warmstart_from_solution(problem)
            case PADMMWarmStartMode.CONTAINERS:
                self._warmstart_from_containers(problem, model, data, limits, contacts)
            case _:
                raise ValueError(f"Invalid warmstart mode: {self._warmstart}")

    def solve(self, problem: DualProblem):
        """Solve ``problem`` using projected Gauss-Seidel over its dense Delassus matrix."""
        if problem.sparse:
            raise ValueError("The DVI solver currently requires `sparse_dynamics=False`.")

        wp.launch(
            kernel=_solve_dvi_pgs,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.njc,
                problem.data.nl,
                problem.data.nc,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.cio,
                problem.data.mu,
                problem.data.D,
                problem.data.v_f,
                self._data.config,
                self._data.status,
                self._data.solution.lambdas,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_compute_dvi_solution_vectors,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.D,
                problem.data.v_f,
                self._data.state.s,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

        if self._size.max_of_max_contacts > 0:
            wp.launch(
                kernel=_compute_dvi_desaxce_corrections,
                dim=(self._size.num_worlds, self._size.max_of_max_contacts),
                inputs=[
                    problem.data.nc,
                    problem.data.ccgo,
                    problem.data.cio,
                    problem.data.vio,
                    problem.data.mu,
                    self._data.state.s,
                    self._data.state.v_aug,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

        wp.launch(
            kernel=_unprecondition_dvi_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.state.s,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

    def _warmstart_from_solution(self, problem: DualProblem):
        wp.launch(
            kernel=_apply_dual_preconditioner_to_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

    def _warmstart_from_containers(
        self,
        problem: DualProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        self._data.solution.zero()
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=_warmstart_joint_constraints,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    model.time.dt,
                    model.joints.wid,
                    model.joints.num_dynamic_cts,
                    model.joints.num_kinematic_cts,
                    model.joints.dynamic_cts_offset_joint_cts,
                    model.joints.kinematic_cts_offset_joint_cts,
                    model.joints.dynamic_cts_offset_total_cts,
                    model.joints.kinematic_cts_offset_total_cts,
                    data.joints.lambda_j,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                kernel=_warmstart_limit_constraints,
                dim=limits.model_max_limits_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.limit_cts_group_offset,
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.reaction,
                    limits.velocity,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                kernel=_warmstart_contact_constraints,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.material,
                    contacts.reaction,
                    contacts.velocity,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
