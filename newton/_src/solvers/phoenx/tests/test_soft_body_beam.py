# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX soft-beam stability regressions."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SoftBodyConstraintType,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

_TAU = 2.0 * math.pi
_STRETCH_MODE = 0
_TWIST_MODE = 1


@wp.kernel
def _drive_right_face_kernel(
    indices: wp.array[wp.int32],
    rest_positions: wp.array[wp.vec3f],
    mode: wp.int32,
    center_y: wp.float32,
    center_z: wp.float32,
    stretch_displacement: wp.float32,
    twist_angle: wp.float32,
    inv_dt: wp.float32,
    particles: ParticleContainer,
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

    particles.position[particle_id] = target
    particles.position_prev_substep[particle_id] = previous
    particles.velocity[particle_id] = (target - previous) * inv_dt


class _BeamStats:
    def __init__(self) -> None:
        self.max_speed = 0.0
        self.max_frame_displacement = 0.0
        self.max_pinned_drift = 0.0
        self.max_dynamic_rest_drift = 0.0


class _BeamScene:
    def __init__(
        self,
        *,
        mode: str,
        youngs_modulus: float = 1.0e8,
        axial_strain: float = 0.10,
        twist_degrees: float = 120.0,
        motion_period: float = 6.0,
        substeps: int = 16,
        solver_iterations: int = 64,
        device: wp.Device,
    ) -> None:
        if mode not in ("stretch", "twist"):
            raise ValueError(f"unexpected beam mode {mode!r}")

        self.device = device
        self.mode = mode
        self.length = 1.8
        self.width = 0.32
        self.depth = 0.22
        self.axial_strain = float(axial_strain)
        self.twist_degrees = float(twist_degrees)
        self.motion_period = float(motion_period)
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(youngs_modulus, 0.45)
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * self.length, -0.5 * self.width, -0.5 * self.depth),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=12,
            dim_y=3,
            dim_z=3,
            cell_x=self.length / 12,
            cell_y=self.width / 3,
            cell_z=self.depth / 3,
            density=500.0,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            fix_left=True,
            fix_right=True,
            add_surface_mesh_edges=False,
            particle_radius=0.004,
        )
        self.model = builder.finalize(device=device)
        self.rest = self.model.particle_q.numpy().astype(np.float32, copy=True)

        x_min = float(self.rest[:, 0].min())
        x_max = float(self.rest[:, 0].max())
        tol = 1.0e-5 * max(1.0, x_max - x_min)
        self.left_indices = np.flatnonzero(self.rest[:, 0] <= x_min + tol).astype(np.int32)
        self.right_indices = np.flatnonzero(self.rest[:, 0] >= x_max - tol).astype(np.int32)
        self.pinned_indices = np.concatenate((self.left_indices, self.right_indices)).astype(np.int32)
        self.dynamic_indices = np.setdiff1d(np.arange(self.model.particle_count), self.pinned_indices)
        self.right_rest = self.rest[self.right_indices].copy()
        self.right_indices_wp = wp.array(self.right_indices, dtype=wp.int32, device=device)
        self.right_rest_wp = wp.array(self.right_rest, dtype=wp.vec3f, device=device)
        self.center_y = 0.5 * (float(self.rest[:, 1].min()) + float(self.rest[:, 1].max()))
        self.center_z = 0.5 * (float(self.rest[:, 2].min()) + float(self.rest[:, 2].max()))
        self.mode_id = _STRETCH_MODE if mode == "stretch" else _TWIST_MODE

        bodies = body_container_zeros(1, device=device)
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=0,
            rigid_contact_max=0,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=8,
            mass_splitting_unrolled=True,
            max_thread_blocks=8 * device.sm_count,
            mass_splitting_batch_size=1,
            sor_boost=0.1,
            partitioner_algorithm="greedy",
            device=device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(
            self.model,
            constraint_type=SoftBodyConstraintType.BLOCK_NEOHOOKEAN,
            beta_lambda=1.0,
            beta_mu=1.0,
        )

        rest_wp = wp.array(self.rest, dtype=wp.vec3f, device=device)
        wp.copy(self.world.particles.position, rest_wp)
        wp.copy(self.world.particles.position_prev_substep, rest_wp)
        self.state = self.model.state()
        wp.copy(self.state.particle_q, rest_wp)
        self._capture()

    def _right_targets(self, time: float) -> np.ndarray:
        targets = self.right_rest.copy()
        if self.mode == "stretch":
            targets[:, 0] += self.axial_strain * self.length * math.sin(_TAU * time / self.motion_period)
            return targets
        angle = math.radians(self.twist_degrees) * math.sin(_TAU * time / self.motion_period)
        c = math.cos(angle)
        s = math.sin(angle)
        y = self.right_rest[:, 1] - self.center_y
        z = self.right_rest[:, 2] - self.center_z
        targets[:, 1] = self.center_y + y * c - z * s
        targets[:, 2] = self.center_z + y * s + z * c
        return targets

    def _drive_right_face(self, time: float) -> None:
        displacement = 0.0
        angle = 0.0
        if self.mode == "stretch":
            displacement = self.axial_strain * self.length * math.sin(_TAU * time / self.motion_period)
        else:
            angle = math.radians(self.twist_degrees) * math.sin(_TAU * time / self.motion_period)

        wp.launch(
            _drive_right_face_kernel,
            dim=int(self.right_indices.shape[0]),
            inputs=[
                self.right_indices_wp,
                self.right_rest_wp,
                wp.int32(self.mode_id),
                wp.float32(self.center_y),
                wp.float32(self.center_z),
                wp.float32(displacement),
                wp.float32(angle),
                wp.float32(1.0 / self.frame_dt),
                self.world.particles,
            ],
            device=self.device,
        )

    def _step_world(self) -> None:
        self.world.step(self.frame_dt)
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def _capture(self) -> None:
        self._step_world()
        with wp.ScopedCapture(device=self.device) as capture:
            self._step_world()
        self.graph = capture.graph

    def step(self) -> None:
        self.sim_time += self.frame_dt
        self._drive_right_face(self.sim_time)
        wp.capture_launch(self.graph)
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def run(self, frames: int) -> _BeamStats:
        stats = _BeamStats()
        previous = self.state.particle_q.numpy().copy()
        pinned_targets = self.rest[self.pinned_indices].copy()
        for _ in range(frames):
            self.step()
            positions = self.state.particle_q.numpy()
            velocities = self.state.particle_qd.numpy()
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise AssertionError("beam produced non-finite particle state")
            pinned_targets[self.left_indices.shape[0] :] = self._right_targets(self.sim_time)
            stats.max_pinned_drift = max(
                stats.max_pinned_drift,
                float(np.linalg.norm(positions[self.pinned_indices] - pinned_targets, axis=1).max()),
            )
            stats.max_speed = max(
                stats.max_speed, float(np.linalg.norm(velocities[self.dynamic_indices], axis=1).max())
            )
            stats.max_frame_displacement = max(
                stats.max_frame_displacement,
                float(np.linalg.norm((positions - previous)[self.dynamic_indices], axis=1).max()),
            )
            stats.max_dynamic_rest_drift = max(
                stats.max_dynamic_rest_drift,
                float(np.linalg.norm(positions[self.dynamic_indices] - self.rest[self.dynamic_indices], axis=1).max()),
            )
            previous = positions.copy()
        return stats


class TestSoftBodyBeam(unittest.TestCase):
    def test_high_youngs_modulus_rest_beam_is_stable(self) -> None:
        device = require_cuda_graph_capture("PhoenX soft-beam tests")
        scene = _BeamScene(mode="stretch", youngs_modulus=1.0e10, axial_strain=0.0, device=device)
        stats = scene.run(60)
        self.assertLess(stats.max_speed, 1.0e-3)
        self.assertLess(stats.max_frame_displacement, 1.0e-5)
        self.assertLess(stats.max_pinned_drift, 1.0e-5)

    def test_driven_stretch_beam_is_stable(self) -> None:
        device = require_cuda_graph_capture("PhoenX soft-beam tests")
        scene = _BeamScene(mode="stretch", device=device)
        stats = scene.run(120)
        self.assertLess(stats.max_speed, 5.0)
        self.assertLess(stats.max_frame_displacement, 0.03)
        self.assertLess(stats.max_pinned_drift, 1.0e-4)
        self.assertGreater(stats.max_dynamic_rest_drift, 0.01)

    def test_driven_twist_beam_is_stable(self) -> None:
        device = require_cuda_graph_capture("PhoenX soft-beam tests")
        scene = _BeamScene(mode="twist", device=device)
        stats = scene.run(120)
        self.assertLess(stats.max_speed, 5.0)
        self.assertLess(stats.max_frame_displacement, 0.03)
        self.assertLess(stats.max_pinned_drift, 1.0e-4)
        self.assertGreater(stats.max_dynamic_rest_drift, 0.005)


if __name__ == "__main__":
    unittest.main()
