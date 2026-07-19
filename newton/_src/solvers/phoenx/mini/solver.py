# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host orchestration for the throughput-oriented PhoenX mini solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.sim import Contacts, Control, JointType, Model, State
from newton._src.solvers.solver import SolverBase

from .kernels import (
    color_world_constraints_kernel,
    gather_contact_constraints_kernel,
    gather_revolute_constraints_kernel,
    integrate_poses_kernel,
    integrate_velocities_kernel,
    solve_worlds_kernel,
)
from .mixed_kernels import prepare_mixed_constraints_kernel, solve_mixed_constraints_kernel
from .packed_kernels import (
    compute_color_offsets_kernel,
    integrate_poses_packed_kernel,
    integrate_velocities_packed_kernel,
    make_solve_packed_contacts_shared_kernel,
    prepare_packed_contacts_kernel,
    prepare_world_contacts_kernel,
    solve_packed_contacts_kernel,
    solve_serial_worlds_kernel,
)


@dataclass(frozen=True)
class MiniSolverConfig:
    """Configuration for :class:`MiniSolver`."""

    substeps: int = 4
    iterations: int = 4
    block_dim: int = 32
    max_colors: int = 64
    max_constraints_per_world: int = 256
    max_constraints_per_color: int = 32
    contact_beta: float = 0.2
    contact_slop: float = 0.001
    joint_beta: float = 0.2
    angular_damping: float = 0.05
    shared_body_cache: bool = False
    solve_layout: str = "colored"


@dataclass(frozen=True)
class MiniSolverStats:
    """Synchronous diagnostic snapshot for the most recent step."""

    overflow_constraints: int
    max_constraints_in_world: int
    total_constraints: int
    max_colors_in_world: int
    max_constraints_in_color: int
    gather_overflow: int
    color_overflow: int


class MiniSolver(SolverBase):
    """Experimental graph-colored PGS rigid-body solver.

    The solver intentionally supports only rigid contacts and revolute joints.
    It maps one CUDA block to each replicated world and performs every PGS
    color and iteration inside that block, avoiding global atomics during the
    solve and avoiding one kernel launch per color.
    """

    def __init__(self, model: Model, config: MiniSolverConfig | None = None):
        super().__init__(model)
        self.config = config or MiniSolverConfig()
        self._validate_model()
        self._validate_config()
        self._init_kinematic_state()
        self._has_revolute = bool(model.joint_count and (model.joint_type.numpy() == int(JointType.REVOLUTE)).any())
        self._packed_contacts = not self._has_revolute
        self._packed_mixed = self._has_revolute

        world_count = max(1, int(model.world_count))
        bodies_per_world = model.body_count // world_count
        expected_worlds = np.repeat(np.arange(world_count, dtype=np.int32), bodies_per_world)
        contiguous_worlds = expected_worlds.shape[0] == model.body_count and np.array_equal(
            model.body_world.numpy(), expected_worlds
        )
        self._shared_solve_kernel = None
        if (
            self._packed_contacts
            and self.config.shared_body_cache
            and 0 < bodies_per_world <= self.config.block_dim
            and self.config.max_constraints_per_color <= self.config.block_dim
            and contiguous_worlds
        ):
            self._shared_solve_kernel = make_solve_packed_contacts_shared_kernel(bodies_per_world)
        capacity = self.config.max_constraints_per_world
        contact_capacity = max(1, int(model.rigid_contact_max))
        self._world_count = world_count
        self._world_constraint_count = wp.zeros(world_count, dtype=wp.int32, device=model.device)
        self._world_constraints = wp.empty(world_count * capacity, dtype=wp.int32, device=model.device)
        self._world_num_colors = wp.zeros(world_count, dtype=wp.int32, device=model.device)
        self._world_color_count = wp.zeros(world_count * self.config.max_colors, dtype=wp.int32, device=model.device)
        self._color_constraints = wp.empty(
            world_count * self.config.max_colors * self.config.max_constraints_per_color,
            dtype=wp.int32,
            device=model.device,
        )
        self._body_color_owner = wp.full(
            max(1, model.body_count * self.config.max_colors), -1, dtype=wp.int32, device=model.device
        )
        self._overflow = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._gather_overflow = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._lambda_n = wp.zeros(contact_capacity, dtype=wp.float32, device=model.device)
        self._lambda_t1 = wp.zeros(contact_capacity, dtype=wp.float32, device=model.device)
        self._lambda_t2 = wp.zeros(contact_capacity, dtype=wp.float32, device=model.device)
        packed_capacity = world_count * capacity
        self._world_color_offset = wp.zeros(world_count * self.config.max_colors, dtype=wp.int32, device=model.device)
        self._linear_velocity = wp.empty(model.body_count, dtype=wp.vec4, device=model.device)
        self._angular_velocity = wp.empty(model.body_count, dtype=wp.vec4, device=model.device)
        self._inertia0 = wp.empty(model.body_count, dtype=wp.vec4, device=model.device)
        self._inertia1 = wp.empty(model.body_count, dtype=wp.vec4, device=model.device)
        self._inertia2 = wp.empty(model.body_count, dtype=wp.vec4, device=model.device)
        self._packed_body_pair = wp.empty(packed_capacity, dtype=wp.vec4i, device=model.device)
        self._packed_normal_bias = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_tangent_mu = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_arm_mass_a = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_arm_mass_b = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_effective_mass = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_impulse = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_row0 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_row1 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_row2 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_row3 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_row4 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_mass = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_mass4 = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        self._packed_joint_motor = wp.empty(packed_capacity, dtype=wp.vec4, device=model.device)
        # Warp kernel parameters cannot be None. Contact-only experiments use
        # these one-element sentinels without paying for a second kernel flavor.
        self._joint_parent = model.joint_parent or wp.zeros(1, dtype=wp.int32, device=model.device)
        self._joint_child = model.joint_child or wp.zeros(1, dtype=wp.int32, device=model.device)
        self._joint_x_p = model.joint_X_p or wp.zeros(1, dtype=wp.transform, device=model.device)
        self._joint_x_c = model.joint_X_c or wp.zeros(1, dtype=wp.transform, device=model.device)
        self._joint_qd_start = model.joint_qd_start or wp.zeros(1, dtype=wp.int32, device=model.device)
        self._joint_axis = model.joint_axis or wp.zeros(1, dtype=wp.vec3, device=model.device)
        self._joint_act = model.joint_act or wp.zeros(1, dtype=wp.float32, device=model.device)
        self._joint_effort_limit = model.joint_effort_limit or wp.zeros(1, dtype=wp.float32, device=model.device)

    def _validate_config(self) -> None:
        c = self.config
        if c.substeps < 1 or c.iterations < 1:
            raise ValueError("MiniSolver substeps and iterations must be positive")
        if c.block_dim not in (8, 16, 32, 64, 128, 256):
            raise ValueError("MiniSolver block_dim must be one of 8, 16, 32, 64, 128, or 256")
        if c.max_colors < 1 or c.max_colors > 64:
            raise ValueError("MiniSolver max_colors must be in [1, 64]")
        if c.max_constraints_per_world < 1:
            raise ValueError("MiniSolver max_constraints_per_world must be positive")
        if c.max_constraints_per_color < 1:
            raise ValueError("MiniSolver max_constraints_per_color must be positive")
        if c.solve_layout not in ("colored", "serial_world"):
            raise ValueError("MiniSolver solve_layout must be 'colored' or 'serial_world'")

    def _validate_model(self) -> None:
        model = self.model
        if model.particle_count:
            raise NotImplementedError("MiniSolver supports rigid bodies only")
        if model.world_count < 1:
            raise ValueError("MiniSolver requires ModelBuilder worlds (model.world_count >= 1)")
        if model.joint_count:
            joint_types = model.joint_type.numpy()
            enabled = model.joint_enabled.numpy()
            supported = (int(JointType.FREE), int(JointType.REVOLUTE))
            unsupported = sorted(
                {int(t) for t, on in zip(joint_types, enabled, strict=True) if on and int(t) not in supported}
            )
            if unsupported:
                raise NotImplementedError(f"MiniSolver supports only revolute joints; found joint types {unsupported}")

    def _build_schedule(self, contacts: Contacts, *, color: bool) -> None:
        c = self.config
        model = self.model
        self._world_constraint_count.zero_()
        self._world_num_colors.zero_()
        self._world_color_count.zero_()
        self._body_color_owner.fill_(-1)
        self._overflow.zero_()
        self._gather_overflow.zero_()
        if not self._packed_contacts and not self._packed_mixed:
            self._lambda_n.zero_()
            self._lambda_t1.zero_()
            self._lambda_t2.zero_()
        if self._has_revolute:
            wp.launch(
                gather_revolute_constraints_kernel,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_world,
                    c.max_constraints_per_world,
                ],
                outputs=[self._world_constraint_count, self._world_constraints, self._gather_overflow],
                device=model.device,
            )
        wp.launch(
            gather_contact_constraints_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                model.shape_body,
                model.body_world,
                c.max_constraints_per_world,
            ],
            outputs=[self._world_constraint_count, self._world_constraints, self._gather_overflow],
            device=model.device,
        )
        if color:
            wp.launch(
                color_world_constraints_kernel,
                dim=self._world_count,
                inputs=[
                    c.max_constraints_per_world,
                    c.max_colors,
                    c.max_constraints_per_color,
                    self._world_constraint_count,
                    self._world_constraints,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self._joint_parent,
                    self._joint_child,
                ],
                outputs=[
                    self._body_color_owner,
                    self._world_num_colors,
                    self._world_color_count,
                    self._color_constraints,
                    self._overflow,
                ],
                device=model.device,
            )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one step using collision-pipeline contacts supplied by the caller."""
        if contacts is None:
            raise ValueError("MiniSolver requires contacts from newton.CollisionPipeline")
        if contacts.rigid_contact_max > self._lambda_n.shape[0]:
            raise ValueError("Contacts capacity exceeds the capacity used to construct MiniSolver")
        if control is None and self.model.joint_count:
            control = self.model.control(clone_variables=False)

        self._refresh_kinematic_state()
        substep_dt = float(dt) / self.config.substeps
        for substep in range(self.config.substeps):
            q_in = state_in.body_q if substep == 0 else state_out.body_q
            qd_in = state_in.body_qd if substep == 0 else state_out.body_qd
            if self._packed_contacts or self._packed_mixed:
                wp.launch(
                    integrate_velocities_packed_kernel,
                    dim=self.model.body_count,
                    inputs=[
                        q_in,
                        qd_in,
                        state_in.body_f,
                        self.model.body_inertia,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        self.model.body_flags,
                        self.model.body_world,
                        self.model.gravity,
                        self.config.angular_damping,
                        substep_dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        self._linear_velocity,
                        self._angular_velocity,
                        self._inertia0,
                        self._inertia1,
                        self._inertia2,
                    ],
                    device=self.model.device,
                )
            else:
                wp.launch(
                    integrate_velocities_kernel,
                    dim=self.model.body_count,
                    inputs=[
                        q_in,
                        qd_in,
                        state_in.body_f,
                        self.model.body_com,
                        self.model.body_inertia,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        self.model.body_flags,
                        self.model.body_world,
                        self.model.gravity,
                        self.config.angular_damping,
                        substep_dt,
                    ],
                    outputs=[state_out.body_q, state_out.body_qd],
                    device=self.model.device,
                )

            self._build_schedule(contacts, color=(not self._packed_contacts or self.config.solve_layout == "colored"))
            if self._packed_mixed:
                wp.launch(
                    compute_color_offsets_kernel,
                    dim=self._world_count,
                    inputs=[
                        self.config.max_colors,
                        self.config.max_constraints_per_color,
                        self._world_num_colors,
                        self._world_color_count,
                    ],
                    outputs=[self._world_color_offset],
                    device=self.model.device,
                )
                joint_act = control.joint_act if control is not None else self._joint_act
                mixed_block_dim = 128 if self.config.block_dim <= 32 else self.config.block_dim
                wp.launch(
                    prepare_mixed_constraints_kernel,
                    dim=self._world_count * self.config.block_dim,
                    inputs=[
                        self.config.block_dim,
                        self.config.max_constraints_per_world,
                        self.config.max_colors,
                        self.config.max_constraints_per_color,
                        substep_dt,
                        self.config.contact_beta,
                        self.config.contact_slop,
                        self.config.joint_beta,
                        self._world_num_colors,
                        self._world_color_count,
                        self._world_color_offset,
                        self._color_constraints,
                        self.model.shape_body,
                        self.model.shape_material_mu,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        self._joint_parent,
                        self._joint_child,
                        self._joint_x_p,
                        self._joint_x_c,
                        self._joint_qd_start,
                        self._joint_axis,
                        joint_act,
                        self._joint_effort_limit,
                        state_out.body_q,
                        self.model.body_com,
                        self.body_inv_mass_effective,
                        self._inertia0,
                        self._inertia1,
                        self._inertia2,
                    ],
                    outputs=[
                        self._packed_body_pair,
                        self._packed_normal_bias,
                        self._packed_tangent_mu,
                        self._packed_arm_mass_a,
                        self._packed_arm_mass_b,
                        self._packed_effective_mass,
                        self._packed_impulse,
                        self._packed_joint_row0,
                        self._packed_joint_row1,
                        self._packed_joint_row2,
                        self._packed_joint_row3,
                        self._packed_joint_row4,
                        self._packed_joint_mass,
                        self._packed_joint_mass4,
                        self._packed_joint_motor,
                    ],
                    block_dim=mixed_block_dim,
                    device=self.model.device,
                )
                wp.launch(
                    solve_mixed_constraints_kernel,
                    dim=self._world_count * self.config.block_dim,
                    inputs=[
                        self.config.block_dim,
                        self.config.max_constraints_per_world,
                        self.config.max_colors,
                        self.config.max_constraints_per_color,
                        self.config.iterations,
                        self._world_num_colors,
                        self._world_color_count,
                        self._world_color_offset,
                        self._packed_body_pair,
                        self._packed_normal_bias,
                        self._packed_tangent_mu,
                        self._packed_arm_mass_a,
                        self._packed_arm_mass_b,
                        self._packed_effective_mass,
                        self._packed_impulse,
                        self._packed_joint_row0,
                        self._packed_joint_row1,
                        self._packed_joint_row2,
                        self._packed_joint_row3,
                        self._packed_joint_row4,
                        self._packed_joint_mass,
                        self._packed_joint_mass4,
                        self._packed_joint_motor,
                        self._inertia0,
                        self._inertia1,
                        self._inertia2,
                    ],
                    outputs=[self._linear_velocity, self._angular_velocity],
                    block_dim=mixed_block_dim,
                    device=self.model.device,
                )
                wp.launch(
                    integrate_poses_packed_kernel,
                    dim=self.model.body_count,
                    inputs=[
                        state_out.body_q,
                        self.model.body_com,
                        self.model.body_flags,
                        self._linear_velocity,
                        self._angular_velocity,
                        substep_dt,
                    ],
                    outputs=[state_out.body_qd],
                    device=self.model.device,
                )
            elif self._packed_contacts:
                if self.config.solve_layout == "serial_world":
                    wp.launch(
                        prepare_world_contacts_kernel,
                        dim=self._world_count * self.config.block_dim,
                        inputs=[
                            self.config.block_dim,
                            self.config.max_constraints_per_world,
                            substep_dt,
                            self.config.contact_beta,
                            self.config.contact_slop,
                            self._world_constraint_count,
                            self._world_constraints,
                            self.model.shape_body,
                            self.model.shape_material_mu,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            state_out.body_q,
                            self.model.body_com,
                            self.body_inv_mass_effective,
                            self._inertia0,
                            self._inertia1,
                            self._inertia2,
                        ],
                        outputs=[
                            self._packed_body_pair,
                            self._packed_normal_bias,
                            self._packed_tangent_mu,
                            self._packed_arm_mass_a,
                            self._packed_arm_mass_b,
                            self._packed_effective_mass,
                            self._packed_impulse,
                        ],
                        block_dim=self.config.block_dim,
                        device=self.model.device,
                    )
                    wp.launch(
                        solve_serial_worlds_kernel,
                        dim=self._world_count,
                        inputs=[
                            self.config.max_constraints_per_world,
                            self.config.iterations,
                            self._world_constraint_count,
                            self._packed_body_pair,
                            self._packed_normal_bias,
                            self._packed_tangent_mu,
                            self._packed_arm_mass_a,
                            self._packed_arm_mass_b,
                            self._packed_effective_mass,
                            self._inertia0,
                            self._inertia1,
                            self._inertia2,
                            self._packed_impulse,
                        ],
                        outputs=[self._linear_velocity, self._angular_velocity],
                        device=self.model.device,
                    )
                else:
                    wp.launch(
                        compute_color_offsets_kernel,
                        dim=self._world_count,
                        inputs=[
                            self.config.max_colors,
                            self.config.max_constraints_per_color,
                            self._world_num_colors,
                            self._world_color_count,
                        ],
                        outputs=[self._world_color_offset],
                        device=self.model.device,
                    )
                    wp.launch(
                        prepare_packed_contacts_kernel,
                        dim=self._world_count * self.config.block_dim,
                        inputs=[
                            self.config.block_dim,
                            self.config.max_constraints_per_world,
                            self.config.max_colors,
                            self.config.max_constraints_per_color,
                            substep_dt,
                            self.config.contact_beta,
                            self.config.contact_slop,
                            self._world_num_colors,
                            self._world_color_count,
                            self._world_color_offset,
                            self._color_constraints,
                            self.model.shape_body,
                            self.model.shape_material_mu,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            state_out.body_q,
                            self.model.body_com,
                            self.body_inv_mass_effective,
                            self._inertia0,
                            self._inertia1,
                            self._inertia2,
                        ],
                        outputs=[
                            self._packed_body_pair,
                            self._packed_normal_bias,
                            self._packed_tangent_mu,
                            self._packed_arm_mass_a,
                            self._packed_arm_mass_b,
                            self._packed_effective_mass,
                            self._packed_impulse,
                        ],
                        block_dim=self.config.block_dim,
                        device=self.model.device,
                    )
                    if self._shared_solve_kernel is not None:
                        wp.launch_tiled(
                            self._shared_solve_kernel,
                            dim=self._world_count,
                            inputs=[
                                self.config.max_constraints_per_world,
                                self.config.max_colors,
                                self.config.max_constraints_per_color,
                                self.config.iterations,
                                self._world_num_colors,
                                self._world_color_count,
                                self._world_color_offset,
                                self._packed_body_pair,
                                self._packed_normal_bias,
                                self._packed_tangent_mu,
                                self._packed_arm_mass_a,
                                self._packed_arm_mass_b,
                                self._packed_effective_mass,
                                self._packed_impulse,
                                self._linear_velocity,
                                self._angular_velocity,
                                self._inertia0,
                                self._inertia1,
                                self._inertia2,
                            ],
                            block_dim=self.config.block_dim,
                            device=self.model.device,
                        )
                    else:
                        wp.launch(
                            solve_packed_contacts_kernel,
                            dim=self._world_count * self.config.block_dim,
                            inputs=[
                                self.config.block_dim,
                                self.config.max_constraints_per_world,
                                self.config.max_colors,
                                self.config.max_constraints_per_color,
                                self.config.iterations,
                                self._world_num_colors,
                                self._world_color_count,
                                self._world_color_offset,
                                self._packed_body_pair,
                                self._packed_normal_bias,
                                self._packed_tangent_mu,
                                self._packed_arm_mass_a,
                                self._packed_arm_mass_b,
                                self._packed_effective_mass,
                                self._inertia0,
                                self._inertia1,
                                self._inertia2,
                                self._packed_impulse,
                            ],
                            outputs=[self._linear_velocity, self._angular_velocity],
                            block_dim=self.config.block_dim,
                            device=self.model.device,
                        )
                wp.launch(
                    integrate_poses_packed_kernel,
                    dim=self.model.body_count,
                    inputs=[
                        state_out.body_q,
                        self.model.body_com,
                        self.model.body_flags,
                        self._linear_velocity,
                        self._angular_velocity,
                        substep_dt,
                    ],
                    outputs=[state_out.body_qd],
                    device=self.model.device,
                )
            else:
                joint_act = control.joint_act if control is not None else self._joint_act
                wp.launch(
                    solve_worlds_kernel,
                    dim=self._world_count * self.config.block_dim,
                    inputs=[
                        self.config.block_dim,
                        self.config.max_constraints_per_color,
                        self.config.max_colors,
                        self.config.iterations,
                        substep_dt,
                        self.config.contact_beta,
                        self.config.contact_slop,
                        self.config.joint_beta,
                        self._world_num_colors,
                        self._world_color_count,
                        self._color_constraints,
                        self.model.shape_body,
                        self.model.shape_material_mu,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        self._joint_parent,
                        self._joint_child,
                        self._joint_x_p,
                        self._joint_x_c,
                        self._joint_qd_start,
                        self._joint_axis,
                        joint_act,
                        self._joint_effort_limit,
                        state_out.body_q,
                        self.model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        self._lambda_n,
                        self._lambda_t1,
                        self._lambda_t2,
                    ],
                    outputs=[state_out.body_qd],
                    block_dim=self.config.block_dim,
                    device=self.model.device,
                )
                wp.launch(
                    integrate_poses_kernel,
                    dim=self.model.body_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        self.model.body_com,
                        self.model.body_flags,
                        substep_dt,
                    ],
                    device=self.model.device,
                )

    def stats(self) -> MiniSolverStats:
        """Return a synchronous diagnostic snapshot."""
        counts = self._world_constraint_count.numpy()
        gather_overflow = int((counts - self.config.max_constraints_per_world).clip(min=0).sum())
        overflow = int(self._overflow.numpy()[0]) + gather_overflow
        color_counts = self._world_color_count.numpy()
        num_colors = self._world_num_colors.numpy()
        return MiniSolverStats(
            overflow_constraints=overflow,
            max_constraints_in_world=int(counts.max(initial=0)),
            total_constraints=int(counts.sum()),
            max_colors_in_world=int(num_colors.max(initial=0)),
            max_constraints_in_color=int(color_counts.max(initial=0)),
            gather_overflow=gather_overflow,
            color_overflow=int(self._overflow.numpy()[0]),
        )
