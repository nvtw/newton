# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared PhoenX soft-beam setup for stretch and twist examples."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _get_arg(args, name: str, default):
    return default if args is None or not hasattr(args, name) else getattr(args, name)


def _deform_stretch(rest: np.ndarray, x_min: float, length: float, poisson_ratio: float, axial_strain: float):
    deformed = rest.copy()
    center_y = 0.5 * (float(rest[:, 1].min()) + float(rest[:, 1].max()))
    center_z = 0.5 * (float(rest[:, 2].min()) + float(rest[:, 2].max()))
    t = np.clip((rest[:, 0] - x_min) / length, 0.0, 1.0)
    lateral_scale = np.maximum(0.35, 1.0 - poisson_ratio * axial_strain * t)
    deformed[:, 0] = rest[:, 0] + axial_strain * length * t
    deformed[:, 1] = center_y + (rest[:, 1] - center_y) * lateral_scale
    deformed[:, 2] = center_z + (rest[:, 2] - center_z) * lateral_scale
    return deformed


def _deform_twist(rest: np.ndarray, x_min: float, length: float, twist_radians: float):
    deformed = rest.copy()
    center_y = 0.5 * (float(rest[:, 1].min()) + float(rest[:, 1].max()))
    center_z = 0.5 * (float(rest[:, 2].min()) + float(rest[:, 2].max()))
    t = np.clip((rest[:, 0] - x_min) / length, 0.0, 1.0)
    angle = twist_radians * t
    c = np.cos(angle)
    s = np.sin(angle)
    y = rest[:, 1] - center_y
    z = rest[:, 2] - center_z
    deformed[:, 1] = center_y + y * c - z * s
    deformed[:, 2] = center_z + y * s + z * c
    return deformed


class SoftBeamExample:
    """Soft tetrahedral beam with prescribed endpoint deformation."""

    def __init__(
        self,
        viewer,
        args=None,
        *,
        mode: str,
        length: float = 1.8,
        width: float = 0.32,
        depth: float = 0.22,
        dim_x: int = 12,
        dim_y: int = 3,
        dim_z: int = 3,
        youngs_modulus: float = 1.0e6,
        poisson_ratio: float = 0.45,
        density: float = 500.0,
        axial_strain: float = 0.25,
        twist_degrees: float = 270.0,
        substeps: int = 8,
        solver_iterations: int = 20,
        beta: float = 1.0,
    ):
        if mode not in ("stretch", "twist"):
            raise ValueError(f"mode must be 'stretch' or 'twist', got {mode!r}")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {poisson_ratio}")
        if min(length, width, depth) <= 0.0:
            raise ValueError("beam dimensions must be positive")
        if min(dim_x, dim_y, dim_z) < 1:
            raise ValueError("beam grid dimensions must be at least 1")

        self.viewer = viewer
        self.device = wp.get_device()
        self.mode = mode
        self.length = float(length)
        self.width = float(width)
        self.depth = float(depth)
        self.axial_strain = float(axial_strain)
        self.twist_degrees = float(twist_degrees)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)

        k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(float(youngs_modulus), float(poisson_ratio))

        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * self.length, -0.5 * self.width, -0.5 * self.depth),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(dim_x),
            dim_y=int(dim_y),
            dim_z=int(dim_z),
            cell_x=self.length / int(dim_x),
            cell_y=self.width / int(dim_y),
            cell_z=self.depth / int(dim_z),
            density=float(density),
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            fix_left=True,
            fix_right=True,
            add_surface_mesh_edges=False,
            particle_radius=0.004,
        )
        self.model = builder.finalize(device=self.device)

        rest = self.model.particle_q.numpy().astype(np.float32, copy=True)
        x_min = float(rest[:, 0].min())
        x_max = float(rest[:, 0].max())
        grid_length = x_max - x_min
        tol = 1.0e-5 * max(1.0, grid_length)
        self._left_indices = np.flatnonzero(rest[:, 0] <= x_min + tol).astype(np.int32)
        self._right_indices = np.flatnonzero(rest[:, 0] >= x_max - tol).astype(np.int32)
        self._pinned_indices = np.concatenate((self._left_indices, self._right_indices)).astype(np.int32)

        if mode == "stretch":
            initial = _deform_stretch(rest, x_min, grid_length, float(poisson_ratio), self.axial_strain)
        else:
            initial = _deform_twist(rest, x_min, grid_length, math.radians(self.twist_degrees))

        self._rest_positions = rest
        self._pinned_targets = initial[self._pinned_indices].copy()

        bodies = body_container_zeros(1, device=self.device)
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=0,
            rigid_contact_max=0,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=12,
            partitioner_algorithm="greedy",
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(
            self.model,
            beta_lambda=float(beta),
            beta_mu=float(beta),
        )

        self.state = self.model.state()
        initial_wp = wp.array(initial, dtype=wp.vec3f, device=self.device)
        wp.copy(self.world.particles.position, initial_wp)
        wp.copy(self.world.particles.position_prev_substep, initial_wp)
        wp.copy(self.state.particle_q, initial_wp)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.6, -3.0, 1.0), pitch=-12.0, yaw=130.0)
        self._frame_index = 0
        self._capture()

    def _capture(self) -> None:
        if self.device.is_cuda:
            self._simulate_one_frame()
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        self.world.step(self.frame_dt)
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
            wp.copy(self.state.particle_q, self.world.particles.position)
            wp.copy(self.state.particle_qd, self.world.particles.velocity)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.isfinite(positions).all():
            raise AssertionError(f"non-finite particle position at frame {self._frame_index}")
        pin_drift = np.linalg.norm(positions[self._pinned_indices] - self._pinned_targets, axis=1)
        if float(pin_drift.max()) > 1.0e-4:
            raise AssertionError(f"pinned beam endpoints drifted by {float(pin_drift.max()):.6f} m")

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        velocities = self.state.particle_qd.numpy()
        assert np.isfinite(positions).all(), "non-finite particle positions"
        assert np.isfinite(velocities).all(), "non-finite particle velocities"
        assert np.linalg.norm(positions - self._rest_positions, axis=1).max() > 0.05, "beam did not deform"
        bounds = 4.0 * max(self.length, self.width, self.depth)
        assert float(np.abs(positions).max()) < bounds, "beam escaped the expected scene bounds"


def create_soft_beam_parser(*, mode: str):
    parser = newton.examples.create_parser()
    parser.add_argument("--length", type=float, default=1.8)
    parser.add_argument("--width", type=float, default=0.32)
    parser.add_argument("--depth", type=float, default=0.22)
    parser.add_argument("--dim-x", type=int, default=12)
    parser.add_argument("--dim-y", type=int, default=3)
    parser.add_argument("--dim-z", type=int, default=3)
    parser.add_argument("--youngs-modulus", type=float, default=1.0e6)
    parser.add_argument("--poisson-ratio", type=float, default=0.45)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    if mode == "stretch":
        parser.add_argument("--axial-strain", type=float, default=0.25)
    elif mode == "twist":
        parser.add_argument("--twist-degrees", type=float, default=270.0)
    else:
        raise ValueError(f"unknown beam mode {mode!r}")
    return parser


def soft_beam_kwargs_from_args(args) -> dict:
    return {
        "length": float(_get_arg(args, "length", 1.8)),
        "width": float(_get_arg(args, "width", 0.32)),
        "depth": float(_get_arg(args, "depth", 0.22)),
        "dim_x": int(_get_arg(args, "dim_x", 12)),
        "dim_y": int(_get_arg(args, "dim_y", 3)),
        "dim_z": int(_get_arg(args, "dim_z", 3)),
        "youngs_modulus": float(_get_arg(args, "youngs_modulus", 1.0e6)),
        "poisson_ratio": float(_get_arg(args, "poisson_ratio", 0.45)),
        "density": float(_get_arg(args, "density", 500.0)),
        "axial_strain": float(_get_arg(args, "axial_strain", 0.25)),
        "twist_degrees": float(_get_arg(args, "twist_degrees", 270.0)),
        "substeps": int(_get_arg(args, "substeps", 8)),
        "solver_iterations": int(_get_arg(args, "solver_iterations", 20)),
        "beta": float(_get_arg(args, "beta", 1.0)),
    }
