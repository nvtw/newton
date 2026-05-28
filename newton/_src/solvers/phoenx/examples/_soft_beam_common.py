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
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SoftBodyConstraintType,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _get_arg(args, name: str, default):
    return default if args is None or not hasattr(args, name) else getattr(args, name)


_TAU = 2.0 * math.pi
_STRETCH_MODE = 0
_TWIST_MODE = 1


def _stretch_displacement(length: float, axial_strain: float, time: float, motion_period: float) -> float:
    return axial_strain * length * math.sin(_TAU * time / motion_period)


def _twist_angle(twist_degrees: float, time: float, motion_period: float) -> float:
    return math.radians(twist_degrees) * math.sin(_TAU * time / motion_period)


def _right_face_stretch_targets(rest: np.ndarray, displacement: float) -> np.ndarray:
    target = rest.copy()
    target[:, 0] += displacement
    return target


def _right_face_twist_targets(rest: np.ndarray, center_y: float, center_z: float, angle: float) -> np.ndarray:
    target = rest.copy()
    c = np.cos(angle)
    s = np.sin(angle)
    y = rest[:, 1] - center_y
    z = rest[:, 2] - center_z
    target[:, 1] = center_y + y * c - z * s
    target[:, 2] = center_z + y * s + z * c
    return target


@wp.kernel
def _animate_right_face_kernel(
    indices: wp.array[wp.int32],
    rest_positions: wp.array[wp.vec3f],
    mode: wp.int32,
    center_y: wp.float32,
    center_z: wp.float32,
    stretch_displacement: wp.float32,
    twist_angle: wp.float32,
    inv_dt: wp.float32,
    particles: ParticleContainer,
    state_q: wp.array[wp.vec3f],
    state_qd: wp.array[wp.vec3f],
):
    tid = wp.tid()
    particle_id = indices[tid]
    previous = particles.position[particle_id]
    rest = rest_positions[tid]

    target = rest
    if mode == wp.int32(_STRETCH_MODE):
        target = wp.vec3(rest[0] + stretch_displacement, rest[1], rest[2])
    else:
        s = wp.sin(twist_angle)
        c = wp.cos(twist_angle)
        y = rest[1] - center_y
        z = rest[2] - center_z
        target = wp.vec3(rest[0], center_y + y * c - z * s, center_z + y * s + z * c)

    velocity = (target - previous) * inv_dt
    particles.position[particle_id] = target
    particles.position_prev_substep[particle_id] = previous
    particles.velocity[particle_id] = velocity
    state_q[particle_id] = target
    state_qd[particle_id] = velocity


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
        youngs_modulus: float = 1.0e8,
        poisson_ratio: float = 0.45,
        density: float = 500.0,
        axial_strain: float = 0.10,
        twist_degrees: float = 120.0,
        motion_period: float = 6.0,
        substeps: int = 16,
        solver_iterations: int = 64,
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
        if motion_period <= 0.0:
            raise ValueError(f"motion_period must be positive, got {motion_period}")

        self.viewer = viewer
        self.device = wp.get_device()
        self.mode = mode
        self.length = float(length)
        self.width = float(width)
        self.depth = float(depth)
        self.axial_strain = float(axial_strain)
        self.twist_degrees = float(twist_degrees)
        self.motion_period = float(motion_period)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)

        max_thread_blocks = 8 * self.device.sm_count if self.device.is_cuda else None
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

        self._rest_positions = rest
        self._right_rest_positions = rest[self._right_indices].copy()
        self._right_indices_wp = wp.array(self._right_indices, dtype=wp.int32, device=self.device)
        self._right_rest_wp = wp.array(self._right_rest_positions, dtype=wp.vec3f, device=self.device)
        self._right_count = int(self._right_indices.shape[0])
        self._mode_id = _STRETCH_MODE if mode == "stretch" else _TWIST_MODE
        self._center_y = 0.5 * (float(rest[:, 1].min()) + float(rest[:, 1].max()))
        self._center_z = 0.5 * (float(rest[:, 2].min()) + float(rest[:, 2].max()))
        self._pinned_targets = self._pinned_targets_for_time(0.0)
        self._max_dynamic_rest_drift = 0.0

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
            max_colored_partitions=8,
            mass_splitting_unrolled=True,
            max_thread_blocks=max_thread_blocks,
            mass_splitting_batch_size=1,
            sor_boost=0.1,
            partitioner_algorithm="greedy",
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(
            self.model,
            constraint_type=SoftBodyConstraintType.BLOCK_NEOHOOKEAN,
            beta_lambda=float(beta),
            beta_mu=float(beta),
        )

        self.state = self.model.state()
        rest_wp = wp.array(rest, dtype=wp.vec3f, device=self.device)
        wp.copy(self.world.particles.position, rest_wp)
        wp.copy(self.world.particles.position_prev_substep, rest_wp)
        wp.copy(self.state.particle_q, rest_wp)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.6, -3.0, 1.0), pitch=-12.0, yaw=130.0)
        self._frame_index = 0
        self._capture()

    def _right_targets_for_time(self, time: float) -> np.ndarray:
        if self.mode == "stretch":
            displacement = _stretch_displacement(self.length, self.axial_strain, time, self.motion_period)
            return _right_face_stretch_targets(self._right_rest_positions, displacement)
        angle = _twist_angle(self.twist_degrees, time, self.motion_period)
        return _right_face_twist_targets(self._right_rest_positions, self._center_y, self._center_z, angle)

    def _pinned_targets_for_time(self, time: float) -> np.ndarray:
        targets = self._rest_positions[self._pinned_indices].copy()
        targets[self._left_indices.shape[0] :] = self._right_targets_for_time(time)
        return targets

    def _animate_right_face(self, time: float) -> None:
        displacement = 0.0
        angle = 0.0
        if self.mode == "stretch":
            displacement = _stretch_displacement(self.length, self.axial_strain, time, self.motion_period)
        else:
            angle = _twist_angle(self.twist_degrees, time, self.motion_period)

        wp.launch(
            _animate_right_face_kernel,
            dim=self._right_count,
            inputs=[
                self._right_indices_wp,
                self._right_rest_wp,
                wp.int32(self._mode_id),
                wp.float32(self._center_y),
                wp.float32(self._center_z),
                wp.float32(displacement),
                wp.float32(angle),
                wp.float32(1.0 / self.frame_dt),
                self.world.particles,
                self.state.particle_q,
                self.state.particle_qd,
            ],
            device=self.device,
        )
        self._pinned_targets = self._pinned_targets_for_time(time)

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
        next_time = self.sim_time + self.frame_dt
        self._animate_right_face(next_time)
        if self.graph is not None:
            wp.capture_launch(self.graph)
            wp.copy(self.state.particle_q, self.world.particles.position)
            wp.copy(self.state.particle_qd, self.world.particles.velocity)
        else:
            self._simulate_one_frame()
        self.sim_time = next_time
        self._frame_index += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.isfinite(positions).all():
            raise AssertionError(f"non-finite particle position at frame {self._frame_index}")
        dynamic_indices = np.setdiff1d(np.arange(self.model.particle_count), self._pinned_indices)
        dynamic_drift = np.linalg.norm(positions[dynamic_indices] - self._rest_positions[dynamic_indices], axis=1)
        self._max_dynamic_rest_drift = max(self._max_dynamic_rest_drift, float(dynamic_drift.max()))
        pin_drift = np.linalg.norm(positions[self._pinned_indices] - self._pinned_targets, axis=1)
        if float(pin_drift.max()) > 1.0e-4:
            raise AssertionError(f"pinned beam endpoints drifted by {float(pin_drift.max()):.6f} m")

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        velocities = self.state.particle_qd.numpy()
        assert np.isfinite(positions).all(), "non-finite particle positions"
        assert np.isfinite(velocities).all(), "non-finite particle velocities"
        assert self._max_dynamic_rest_drift > 0.01, "beam did not deform"
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
    parser.add_argument("--youngs-modulus", type=float, default=1.0e8)
    parser.add_argument("--poisson-ratio", type=float, default=0.45)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--solver-iterations", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--motion-period", type=float, default=6.0)
    if mode == "stretch":
        parser.add_argument("--axial-strain", type=float, default=0.10)
    elif mode == "twist":
        parser.add_argument("--twist-degrees", type=float, default=120.0)
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
        "youngs_modulus": float(_get_arg(args, "youngs_modulus", 1.0e8)),
        "poisson_ratio": float(_get_arg(args, "poisson_ratio", 0.45)),
        "density": float(_get_arg(args, "density", 500.0)),
        "axial_strain": float(_get_arg(args, "axial_strain", 0.10)),
        "twist_degrees": float(_get_arg(args, "twist_degrees", 120.0)),
        "motion_period": float(_get_arg(args, "motion_period", 6.0)),
        "substeps": int(_get_arg(args, "substeps", 16)),
        "solver_iterations": int(_get_arg(args, "solver_iterations", 64)),
        "beta": float(_get_arg(args, "beta", 1.0)),
    }
