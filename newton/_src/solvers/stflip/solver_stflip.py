# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capacity-bounded sparse ST-FLIP solver.

The temporal transfer is adapted from st-flip-blender; see
``newton/licenses/st-flip-blender-LICENSE.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import warp as wp

import newton

from ..semi_implicit.kernels_contact import eval_body_contact_forces, eval_particle_body_contact_forces
from ..solver import SolverBase
from .kernels import (
    advect_particles,
    apply_pressure,
    build_particle_active_mask,
    build_pressure_system,
    constrain_particles,
    grid_to_particles,
    initialize_temporal_offsets,
    normalize_grid,
    particle_faces_to_grid,
    particles_to_grid,
    pressure_jacobi,
    reconstruct_affine_rows,
    sample_grid_velocity,
    store_affine,
    update_particle_clocks,
)
from .sparse_grid import SparseGrid


class SolverSTFLIP(SolverBase):
    """Sparse temporally staggered FLIP fluid solver.

    Grid fields contain packed core cells only. Cross-tile stencils use a
    27-neighbor table and never allocate halo cells.

    Call :meth:`register_custom_attributes` before finalizing the model.
    CUDA graph capture is supported after one warm-up step when the model,
    solver configuration, state buffers, and step arguments remain fixed.
    Reserve enough :attr:`Config.max_active_tile_count` headroom for motion
    and call :meth:`check_status` after graph replay to detect overflow.

    Args:
        model: Model containing the fluid particles.
        config: Solver configuration.
    """

    @dataclass
    class Config:
        """Configure :class:`SolverSTFLIP`."""

        cell_size: float = 0.08
        """MAC-grid cell edge length [m]."""
        tile_size: int = 8
        """Number of core cells along each packed tile axis."""
        max_active_tile_count: int = 512
        """Maximum number of simultaneously active sparse tiles."""
        padding_tiles: int = 1
        """Number of core-tile layers activated around occupied tiles."""
        pressure_iterations: int = 80
        """Number of fixed Jacobi pressure iterations."""
        liquid_density: float = 1000.0
        """Liquid rest density [kg/m³]."""
        particles_per_cell: float = 8.0
        """Target particle count represented by a full liquid cell."""
        transfer_scheme: Literal["flip", "pic", "apic"] = "apic"
        """Particle/grid transfer method."""
        flip_blend: float = 0.97
        """FLIP fraction used by ``"flip"`` and ``"apic"`` transfers."""
        min_cell_mass_fraction: float = 0.01
        """Minimum fraction of nominal cell mass treated as liquid."""
        domain_lower: tuple[float, float, float] | None = None
        """Optional closed-domain lower bound [m]."""
        domain_upper: tuple[float, float, float] | None = None
        """Optional closed-domain upper bound [m]."""
        max_velocity: float = 50.0
        """Maximum particle speed [m/s]."""
        temporal_staggering: bool = True
        """Use deterministic per-particle temporal offsets during P2G."""
        seed: int = 42
        """Seed for deterministic temporal phase offsets."""

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register ST-FLIP particle state attributes.

        Args:
            builder: Model builder receiving the attributes.
        """
        for name, dtype, default in (
            ("particle_affine", wp.mat33, wp.mat33(0.0)),
            ("particle_time_residual", wp.float32, 0.0),
            ("particle_age", wp.float32, 0.0),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    assignment=newton.Model.AttributeAssignment.STATE,
                    dtype=dtype,
                    default=default,
                    namespace="stflip",
                )
            )

    def __init__(self, model: newton.Model, config: Config | None = None):
        super().__init__(model)
        self.config = config if config is not None else self.Config()
        if model.particle_count == 0:
            raise ValueError("SolverSTFLIP requires at least one particle")
        if self.config.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if self.config.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be positive")
        if self.config.liquid_density <= 0.0:
            raise ValueError("liquid_density must be positive")
        if self.config.particles_per_cell <= 0.0:
            raise ValueError("particles_per_cell must be positive")
        if self.config.transfer_scheme not in ("flip", "pic", "apic"):
            raise ValueError(f"unsupported transfer_scheme {self.config.transfer_scheme!r}")
        if not 0.0 <= self.config.flip_blend <= 1.0:
            raise ValueError("flip_blend must be in [0, 1]")
        if self.config.min_cell_mass_fraction < 0.0:
            raise ValueError("min_cell_mass_fraction must be non-negative")
        if (self.config.domain_lower is None) != (self.config.domain_upper is None):
            raise ValueError("domain_lower and domain_upper must be specified together")
        if self.config.domain_lower is not None and any(
            lower >= upper for lower, upper in zip(self.config.domain_lower, self.config.domain_upper, strict=True)
        ):
            raise ValueError("each domain_lower component must be less than domain_upper")
        if self.config.max_velocity <= 0.0:
            raise ValueError("max_velocity must be positive")
        if model.world_count > 1:
            raise ValueError("SolverSTFLIP currently supports single-world models only")

        self.grid = SparseGrid(
            point_capacity=model.particle_count,
            tile_capacity=self.config.max_active_tile_count,
            tile_size=self.config.tile_size,
            cell_size=self.config.cell_size,
            padding_tiles=self.config.padding_tiles,
            device=model.device,
        )
        capacity = self.grid.cell_capacity
        self.cell_mass = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self.face_mass = wp.zeros(3 * capacity, dtype=wp.float32, device=model.device)
        self.face_momentum = wp.zeros(3 * capacity, dtype=wp.float32, device=model.device)
        self.face_velocity = wp.zeros(3 * capacity, dtype=wp.float32, device=model.device)
        self.face_velocity_old = wp.zeros(3 * capacity, dtype=wp.float32, device=model.device)
        self.pressure = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self.pressure_scratch = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self.pressure_rhs = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self.pressure_diag = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self._zero_affine = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self._particle_velocity_old = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._affine_rows = wp.zeros(9 * model.particle_count, dtype=wp.float32, device=model.device)
        self._temporal_offsets = wp.zeros(model.particle_count, dtype=wp.float32, device=model.device)
        self._contact_body_force_dummy = wp.zeros(1, dtype=wp.spatial_vector, device=model.device)
        self._active_mask = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
        if self.config.temporal_staggering:
            wp.launch(
                initialize_temporal_offsets,
                dim=model.particle_count,
                inputs=[self.config.seed, self._temporal_offsets],
                device=model.device,
            )

    def check_status(self) -> None:
        """Raise if the most recent sparse-grid rebuild exceeded capacity."""
        self.grid.check_status()

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance the fluid by one time step.

        Args:
            state_in: Input particle state.
            state_out: Output particle state.
            control: Unused control input.
            contacts: Particle-rigid and rigid-rigid contacts.
            dt: Time step [s].
        """
        del control
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        try:
            affine_in = state_in.stflip.particle_affine
            affine_out = state_out.stflip.particle_affine
        except AttributeError as error:
            raise RuntimeError(
                "Call SolverSTFLIP.register_custom_attributes(builder) before finalizing the model"
            ) from error

        if contacts is not None and contacts.soft_contact_max:
            body_force = state_in.body_f
            if body_force is None:
                body_force = self._contact_body_force_dummy
            eval_particle_body_contact_forces(
                self.model,
                state_in,
                contacts,
                state_in.particle_f,
                body_force,
            )
        if self.model.body_count and contacts is not None and contacts.rigid_contact_max:
            eval_body_contact_forces(self.model, state_in, contacts)

        wp.launch(
            build_particle_active_mask,
            dim=self.model.particle_count,
            inputs=[self.model.particle_flags, self._active_mask],
            device=self.device,
        )
        self.grid.build(state_in.particle_q, self._active_mask)
        self.cell_mass.zero_()
        self.face_mass.zero_()
        self.face_momentum.zero_()
        self.pressure.zero_()
        self.pressure_scratch.zero_()
        affine_transfer = affine_in if self.config.transfer_scheme == "apic" else self._zero_affine

        wp.launch(
            particles_to_grid,
            dim=self.model.particle_count,
            inputs=[
                self.grid.data,
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                affine_transfer,
                self._temporal_offsets,
                1.0 / self.config.cell_size,
                dt,
                self.cell_mass,
                self.face_mass,
                self.face_momentum,
            ],
            device=self.device,
        )
        wp.launch(
            particle_faces_to_grid,
            dim=3 * self.model.particle_count,
            inputs=[
                self.grid.data,
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                affine_transfer,
                self._temporal_offsets,
                1.0 / self.config.cell_size,
                dt,
                self.face_mass,
                self.face_momentum,
            ],
            device=self.device,
        )
        wp.launch(
            normalize_grid,
            dim=self.grid.cell_capacity,
            inputs=[
                self.face_mass,
                self.face_momentum,
                self.model.gravity,
                dt,
                self.face_velocity,
                self.face_velocity_old,
            ],
            device=self.device,
        )

        nominal_mass = self.config.liquid_density * self.config.cell_size**3
        min_mass = self.config.min_cell_mass_fraction * nominal_mass
        wp.launch(
            build_pressure_system,
            dim=self.grid.cell_capacity,
            inputs=[
                self.grid.data,
                self.cell_mass,
                self.face_velocity,
                min_mass,
                self.config.liquid_density * self.config.cell_size / dt,
                self.pressure_rhs,
                self.pressure_diag,
            ],
            device=self.device,
        )
        pressure_in = self.pressure
        pressure_out = self.pressure_scratch
        for _ in range(self.config.pressure_iterations):
            wp.launch(
                pressure_jacobi,
                dim=self.grid.cell_capacity,
                inputs=[
                    self.grid.data,
                    self.cell_mass,
                    min_mass,
                    self.pressure_rhs,
                    self.pressure_diag,
                    pressure_in,
                    pressure_out,
                ],
                device=self.device,
            )
            pressure_in, pressure_out = pressure_out, pressure_in

        wp.launch(
            apply_pressure,
            dim=self.grid.cell_capacity,
            inputs=[
                self.grid.data,
                self.cell_mass,
                min_mass,
                pressure_in,
                dt / (self.config.liquid_density * self.config.cell_size),
                self.face_mass,
                self.face_velocity,
            ],
            device=self.device,
        )
        flip_blend = 0.0 if self.config.transfer_scheme == "pic" else self.config.flip_blend
        wp.launch(
            sample_grid_velocity,
            dim=self.model.particle_count,
            inputs=[
                self.grid.data,
                state_in.particle_q,
                self.model.particle_flags,
                1.0 / self.config.cell_size,
                self.face_velocity_old,
                self._particle_velocity_old,
            ],
            device=self.device,
        )
        wp.launch(
            grid_to_particles,
            dim=self.model.particle_count,
            inputs=[
                self.grid.data,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_flags,
                1.0 / self.config.cell_size,
                flip_blend,
                self.face_velocity,
                self._particle_velocity_old,
                state_out.particle_q,
                state_out.particle_qd,
                affine_out,
            ],
            device=self.device,
        )
        wp.launch(
            advect_particles,
            dim=self.model.particle_count,
            inputs=[
                self.grid.data,
                state_in.particle_q,
                state_out.particle_qd,
                self.model.particle_flags,
                1.0 / self.config.cell_size,
                dt,
                self.face_velocity,
                state_out.particle_q,
            ],
            device=self.device,
        )
        if self.config.domain_lower is not None:
            wp.launch(
                constrain_particles,
                dim=self.model.particle_count,
                inputs=[
                    wp.vec3(self.config.domain_lower),
                    wp.vec3(self.config.domain_upper),
                    self.config.max_velocity,
                    self.model.particle_flags,
                    state_out.particle_q,
                    state_out.particle_qd,
                ],
                device=self.device,
            )
        if self.config.transfer_scheme == "apic":
            wp.launch(
                reconstruct_affine_rows,
                dim=3 * self.model.particle_count,
                inputs=[
                    self.grid.data,
                    state_in.particle_q,
                    self.model.particle_flags,
                    1.0 / self.config.cell_size,
                    self.face_velocity,
                    self._affine_rows,
                ],
                device=self.device,
            )
            wp.launch(
                store_affine,
                dim=self.model.particle_count,
                inputs=[self._affine_rows, affine_out],
                device=self.device,
            )
        wp.launch(
            update_particle_clocks,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_flags,
                self._temporal_offsets,
                dt,
                state_in.stflip.particle_age,
                state_out.stflip.particle_time_residual,
                state_out.stflip.particle_age,
            ],
            device=self.device,
        )
        self.integrate_bodies(self.model, state_in, state_out, dt)
