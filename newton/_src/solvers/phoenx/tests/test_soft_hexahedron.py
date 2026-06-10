# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the 8-node soft-hexahedron mixed strain/volume constraint.

CUDA-only tests, mirroring
:mod:`test_soft_tet_neohookean`'s pattern:

1. **rest_pose_remains_bounded**: at the rest pose with zero gravity
   the mixed strain/volume energy gradient vanishes
   (mu*I + lambda*(1 - gamma)*I = 0 by construction), so the cube must
   remain spatially bounded.

2. **free_fall_integrates_gravity**: a soft cube in vacuum under
   gravity must fall ballistically without exploding; the block
   constraint is the only thing active.

3. **uniform_compression_recovers_volume**: initialize an isotropically
   compressed cube under zero gravity, verify the center determinant
   moves back toward rest volume.

4. **rigid_rotation_remains_bounded**: initialize a rigidly rotated
   cube and verify the default strain row remains shape preserving.

5. **hourglass_mode_is_corrected**: initialize the canonical
   ``sx * sy * sz`` hourglass displacement. A center-only hex strain
   row sees exactly ``F = I`` for this mode and leaves it untouched; the
   integrated strain row must reduce it.

6. **arap_model_*:** repeat the key deformation tests with the optional
   integrated ARAP strain model.

7. **invariance / high-stiffness / multi-hex tests:** cover absolute
   translation invariance, high-stiffness compression recovery, invalid
   strain-model rejection, and a two-hex shared-face patch.

8. **top_face_pin_hang_stable**: pin the 4 top-face corners, apply
   gravity, verify the bottom face hangs in a bounded elastic
   configuration (edge ratios stay within +/- 50% of rest, no NaN)
   over a long-ish horizon.

The long-running equilibrium tests replay through ``wp.capture_launch``;
targeted deformation tests also exercise the direct ``world.step`` path.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float32,
)

_HOURGLASS_X = np.prod(_CORNER_SIGNS, axis=1).astype(np.float32)

_HEX_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


def _hourglass_x_amplitude(positions: np.ndarray, rest: np.ndarray) -> float:
    """Return the x-amplitude of the canonical trilinear hourglass mode."""
    displacement_x = positions[:, 0] - rest[:, 0]
    return float(np.dot(displacement_x, _HOURGLASS_X) / np.dot(_HOURGLASS_X, _HOURGLASS_X))


def _center_jacobian(corners: np.ndarray) -> np.ndarray:
    """Return the center Jacobian for a canonical 8-node hex."""
    jac = np.zeros((3, 3), dtype=np.float64)
    for corner, sign in zip(corners.astype(np.float64), _CORNER_SIGNS.astype(np.float64), strict=True):
        jac += np.outer(corner, sign)
    return 0.125 * jac


def _center_volume_ratio(positions: np.ndarray, rest: np.ndarray) -> float:
    """Return det(F) at the hex center against the rest element."""
    rest_det = np.linalg.det(_center_jacobian(rest))
    current_det = np.linalg.det(_center_jacobian(positions))
    return float(current_det / rest_det)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3D rotation matrix from an axis and angle."""
    axis = axis.astype(np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def _assign_particle_positions(world: PhoenXWorld, positions: np.ndarray, device) -> None:
    """Assign particle positions and previous-substep positions."""
    positions_wp = wp.array(positions, dtype=wp.vec3f, device=device)
    wp.copy(world.particles.position, positions_wp)
    wp.copy(world.particles.position_prev_substep, positions_wp)


def _hex_edge_ratios(positions: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Return current/rest edge-length ratios for a canonical hex."""
    rest_lens = np.linalg.norm(rest[_HEX_EDGES[:, 1]] - rest[_HEX_EDGES[:, 0]], axis=1)
    now_lens = np.linalg.norm(positions[_HEX_EDGES[:, 1]] - positions[_HEX_EDGES[:, 0]], axis=1)
    return now_lens / rest_lens


def _build_world_from_arrays(
    device,
    *,
    rest_positions: np.ndarray,  # [num_particles, 3]
    hex_indices_np: np.ndarray,  # [num_hexes, 8]
    inv_mass: np.ndarray,  # [num_particles]
    k_mu: float = 1.0e3,
    k_lambda: float = 1.0e5,  # high lambda / low mu -> gamma ~ 1 -> small rest drift
    beta_h: float = 0.0,
    beta_d: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    substeps: int = 4,
    solver_iterations: int = 8,
    strain_model: str | int = "trace",
) -> PhoenXWorld:
    """Construct a PhoenX world from hex indices and rest positions."""
    num_particles = int(rest_positions.shape[0])
    num_hexes = int(hex_indices_np.shape[0])
    bodies = body_container_zeros(1, device=device)
    bodies.orientation.assign(
        wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (1, 1)).astype(np.float32), dtype=wp.quatf, device=device)
    )
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0,
        num_soft_hexahedra=num_hexes,
        device=device,
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=num_particles,
        num_soft_hexahedra=num_hexes,
        num_worlds=1,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=0,
        rigid_contact_max=0,
        step_layout="single_world",
        mass_splitting=False,
        partitioner_algorithm="greedy",
        device=device,
    )
    world.gravity.assign(np.array([list(gravity)], dtype=np.float32))

    hex_indices = wp.array(hex_indices_np.astype(np.int32), dtype=wp.int32, device=device)
    particle_q = wp.array(rest_positions, dtype=wp.vec3f, device=device)
    particle_qd = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=device)
    hex_materials = wp.array(
        np.tile(np.array([[k_mu, k_lambda, beta_h, beta_d]], dtype=np.float32), (num_hexes, 1)),
        dtype=wp.float32,
        device=device,
    )
    world.populate_soft_hexahedra_from_arrays(
        hex_indices=hex_indices,
        particle_q=particle_q,
        hex_materials=hex_materials,
        particle_qd=particle_qd,
        particle_inv_mass=particle_inv_mass,
        strain_model=strain_model,
    )
    return world


def _build_world(
    device,
    *,
    rest_corners: np.ndarray,  # [8, 3]
    inv_mass: np.ndarray,  # [8]
    k_mu: float = 1.0e3,
    k_lambda: float = 1.0e5,
    beta_h: float = 0.0,
    beta_d: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    substeps: int = 4,
    solver_iterations: int = 8,
    strain_model: str | int = "trace",
) -> PhoenXWorld:
    """Construct a 1-hex / 8-particle PhoenX world."""
    return _build_world_from_arrays(
        device,
        rest_positions=rest_corners,
        hex_indices_np=np.arange(8, dtype=np.int32).reshape(1, 8),
        inv_mass=inv_mass,
        k_mu=k_mu,
        k_lambda=k_lambda,
        beta_h=beta_h,
        beta_d=beta_d,
        gravity=gravity,
        substeps=substeps,
        solver_iterations=solver_iterations,
        strain_model=strain_model,
    )


def _build_two_hex_bar() -> tuple[np.ndarray, np.ndarray]:
    """Return rest positions and indices for two hexes sharing a face."""
    xs = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    ys = np.array([-0.05, 0.05], dtype=np.float32)
    zs = np.array([0.45, 0.55], dtype=np.float32)

    positions = []
    for ix in range(3):
        for iy in range(2):
            for iz in range(2):
                positions.append([xs[ix], ys[iy], zs[iz]])

    def idx(ix: int, iy: int, iz: int) -> int:
        return ix * 4 + iy * 2 + iz

    hex_indices = []
    for ix in range(2):
        hex_indices.append(
            [
                idx(ix, 0, 0),
                idx(ix + 1, 0, 0),
                idx(ix + 1, 1, 0),
                idx(ix, 1, 0),
                idx(ix, 0, 1),
                idx(ix + 1, 0, 1),
                idx(ix + 1, 1, 1),
                idx(ix, 1, 1),
            ]
        )
    return np.asarray(positions, dtype=np.float32), np.asarray(hex_indices, dtype=np.int32)


def _capture_step_graph(world: PhoenXWorld, dt: float, device) -> wp.Graph:
    """Warm up (1 un-captured step) then capture a single step graph."""
    world.step(dt)
    with wp.ScopedCapture(device=device) as cap:
        world.step(dt)
    return cap.graph


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-hexahedron tests are CUDA-only (need graph capture).",
)
class TestSoftHexahedron(unittest.TestCase):
    def test_rest_pose_remains_bounded(self):
        """Rest pose under zero gravity -- bounded equilibrium drift."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS  # 0.1 m cube
        rest[:, 2] += 0.5  # lift off origin (irrelevant; just hygiene)
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        # High lambda / low mu so gamma ~ 1 and the natural inflation is
        # sub-percent.
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, 0.0),
        )
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        p_initial = world.particles.position.numpy().copy()
        for _ in range(20):
            wp.capture_launch(graph)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        # Bound the drift to a fraction of the cube edge. The mixed
        # strain/volume equilibrium produces only a small expansion at
        # lambda/mu = 100; 0.05 * cube_size leaves generous headroom for
        # the explicit-Euler equilibration mode.
        drift = float(np.linalg.norm(positions - p_initial, axis=1).max())
        self.assertLess(
            drift,
            0.05 * 0.1,
            f"rest-pose drift {drift:.4f} m exceeds bound",
        )

    def test_free_fall_integrates_gravity(self):
        """Cube in vacuum under gravity falls ballistically (~exact)."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 1.0
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, -9.81),
        )
        initial_mean_z = float(world.particles.position.numpy()[:, 2].mean())
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        T = 0.25
        n_frames = int(T / dt)
        for _ in range(n_frames):
            wp.capture_launch(graph)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_mean_z = float(positions[:, 2].mean())
        elapsed = (n_frames + 1) * dt  # +1 for the warm-up step inside _capture_step_graph
        expected_drop = 0.5 * 9.81 * elapsed * elapsed
        actual_drop = initial_mean_z - final_mean_z
        # The cube also breathes during self-equilibration. 15% tolerance
        # covers the explicit-Euler bias + equilibration.
        self.assertAlmostEqual(actual_drop, expected_drop, delta=0.15 * expected_drop)

    def test_hourglass_mode_is_corrected(self):
        """Integrated strain sees the center-gradient hourglass null mode."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            gravity=(0.0, 0.0, 0.0),
            substeps=5,
            solver_iterations=16,
        )

        deformed = rest.copy()
        deformed[:, 0] += 0.025 * _HOURGLASS_X
        deformed_wp = wp.array(deformed, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, deformed_wp)
        wp.copy(world.particles.position_prev_substep, deformed_wp)

        initial_amp = abs(_hourglass_x_amplitude(deformed, rest))
        dt = 1.0 / 60.0
        for _ in range(12):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_amp = abs(_hourglass_x_amplitude(positions, rest))
        self.assertLess(
            final_amp,
            0.9 * initial_amp,
            f"hourglass amplitude was not reduced enough: initial={initial_amp:.5f}, final={final_amp:.5f}",
        )

    def test_uniform_compression_recovers_volume(self):
        """Volume row expands an initially compressed cube toward rest."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            gravity=(0.0, 0.0, 0.0),
            substeps=5,
            solver_iterations=20,
        )

        center = rest.mean(axis=0)
        compressed = center + 0.65 * (rest - center)
        compressed_wp = wp.array(compressed, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, compressed_wp)
        wp.copy(world.particles.position_prev_substep, compressed_wp)

        initial_ratio = _center_volume_ratio(compressed, rest)
        dt = 1.0 / 60.0
        for _ in range(16):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_ratio = _center_volume_ratio(positions, rest)
        self.assertGreater(
            final_ratio,
            initial_ratio + 0.1,
            f"compressed volume did not recover enough: initial={initial_ratio:.4f}, final={final_ratio:.4f}",
        )
        self.assertLess(
            abs(1.0 - final_ratio),
            abs(1.0 - initial_ratio),
            f"volume ratio moved away from rest: initial={initial_ratio:.4f}, final={final_ratio:.4f}",
        )

    def test_rigid_rotation_remains_bounded(self):
        """Pure rotation stays shape-preserving under zero gravity."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, 0.0),
            substeps=4,
            solver_iterations=8,
        )

        theta = np.deg2rad(37.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.array(
            [
                [cos_t, -sin_t, 0.0],
                [sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = rest.mean(axis=0)
        rotated = (rest - center) @ rotation.T + center
        rotated_wp = wp.array(rotated, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, rotated_wp)
        wp.copy(world.particles.position_prev_substep, rotated_wp)

        dt = 1.0 / 60.0
        for _ in range(20):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        drift = float(np.linalg.norm(positions - rotated, axis=1).max())
        self.assertLess(
            drift,
            0.06 * 0.1,
            f"rigidly rotated hex drifted too far: drift={drift:.4f} m",
        )
        rest_lens = np.linalg.norm(rest[_HEX_EDGES[:, 1]] - rest[_HEX_EDGES[:, 0]], axis=1)
        now_lens = np.linalg.norm(positions[_HEX_EDGES[:, 1]] - positions[_HEX_EDGES[:, 0]], axis=1)
        ratios = now_lens / rest_lens
        self.assertTrue(
            np.all((ratios > 0.95) & (ratios < 1.05)),
            f"rigid rotation changed edge lengths: ratios min={ratios.min():.3f} max={ratios.max():.3f}",
        )

    def test_arap_model_hourglass_mode_is_corrected(self):
        """Integrated ARAP sees the center-gradient hourglass null mode."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            gravity=(0.0, 0.0, 0.0),
            substeps=5,
            solver_iterations=16,
            strain_model="arap",
        )

        deformed = rest.copy()
        deformed[:, 0] += 0.025 * _HOURGLASS_X
        deformed_wp = wp.array(deformed, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, deformed_wp)
        wp.copy(world.particles.position_prev_substep, deformed_wp)

        initial_amp = abs(_hourglass_x_amplitude(deformed, rest))
        dt = 1.0 / 60.0
        for _ in range(12):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_amp = abs(_hourglass_x_amplitude(positions, rest))
        self.assertLess(
            final_amp,
            0.9 * initial_amp,
            f"ARAP hourglass amplitude was not reduced enough: initial={initial_amp:.5f}, final={final_amp:.5f}",
        )

    def test_arap_model_uniform_compression_recovers_volume(self):
        """Integrated ARAP still couples to the center volume row."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            gravity=(0.0, 0.0, 0.0),
            substeps=5,
            solver_iterations=20,
            strain_model="arap",
        )

        center = rest.mean(axis=0)
        compressed = center + 0.65 * (rest - center)
        compressed_wp = wp.array(compressed, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, compressed_wp)
        wp.copy(world.particles.position_prev_substep, compressed_wp)

        initial_ratio = _center_volume_ratio(compressed, rest)
        dt = 1.0 / 60.0
        for _ in range(16):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_ratio = _center_volume_ratio(positions, rest)
        self.assertGreater(
            final_ratio,
            initial_ratio + 0.1,
            f"ARAP compressed volume did not recover enough: initial={initial_ratio:.4f}, final={final_ratio:.4f}",
        )
        self.assertLess(
            abs(1.0 - final_ratio),
            abs(1.0 - initial_ratio),
            f"ARAP volume ratio moved away from rest: initial={initial_ratio:.4f}, final={final_ratio:.4f}",
        )

    def test_arap_model_rigid_rotation_is_shape_preserving(self):
        """Integrated ARAP treats pure rotation as zero strain."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, 0.0),
            substeps=4,
            solver_iterations=8,
            strain_model="arap",
        )

        theta = np.deg2rad(37.0)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = rest.mean(axis=0)
        rotated = (rest - center) @ rotation.T + center
        rotated_wp = wp.array(rotated, dtype=wp.vec3f, device=device)
        wp.copy(world.particles.position, rotated_wp)
        wp.copy(world.particles.position_prev_substep, rotated_wp)

        dt = 1.0 / 60.0
        for _ in range(20):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        drift = float(np.linalg.norm(positions - rotated, axis=1).max())
        self.assertLess(
            drift,
            1.0e-3,
            f"ARAP rigidly rotated hex drifted too far: drift={drift:.4f} m",
        )
        rest_lens = np.linalg.norm(rest[_HEX_EDGES[:, 1]] - rest[_HEX_EDGES[:, 0]], axis=1)
        now_lens = np.linalg.norm(positions[_HEX_EDGES[:, 1]] - positions[_HEX_EDGES[:, 0]], axis=1)
        ratios = now_lens / rest_lens
        self.assertTrue(
            np.all((ratios > 0.995) & (ratios < 1.005)),
            f"ARAP rigid rotation changed edge lengths: ratios min={ratios.min():.3f} max={ratios.max():.3f}",
        )

    def test_arap_model_large_3d_rotation_is_shape_preserving(self):
        """Integrated ARAP stays invariant under a large arbitrary rotation."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            gravity=(0.0, 0.0, 0.0),
            substeps=5,
            solver_iterations=16,
            strain_model="arap",
        )

        rotation = _axis_angle_rotation(np.array([1.0, 2.0, 3.0], dtype=np.float32), np.deg2rad(82.0))
        center = rest.mean(axis=0)
        rotated = (rest - center) @ rotation.T + center
        _assign_particle_positions(world, rotated, device)

        dt = 1.0 / 60.0
        for _ in range(20):
            world.step(dt)

        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        drift = float(np.linalg.norm(positions - rotated, axis=1).max())
        self.assertLess(drift, 1.0e-3, f"large rigid rotation drifted by {drift:.4f} m")
        ratios = _hex_edge_ratios(positions, rest)
        self.assertTrue(
            np.all((ratios > 0.995) & (ratios < 1.005)),
            f"large ARAP rotation changed edge lengths: ratios min={ratios.min():.3f} max={ratios.max():.3f}",
        )

    def test_invalid_strain_model_is_rejected(self):
        """Bad strain-model names fail before stamping hex rows."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "strain_model"):
            _build_world(
                device,
                rest_corners=rest,
                inv_mass=inv_mass,
                gravity=(0.0, 0.0, 0.0),
                strain_model="center_arap",
            )

    def test_translation_invariance_for_strain_models(self):
        """Absolute position offsets do not change internal hex response."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        offset = np.array([3.0, -2.0, 4.0], dtype=np.float32)
        translated = rest + offset
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)

        for strain_model in ("trace", "arap"):
            with self.subTest(strain_model=strain_model):
                world = _build_world(
                    device,
                    rest_corners=rest,
                    inv_mass=inv_mass,
                    k_mu=1.0e3,
                    k_lambda=1.0e5,
                    gravity=(0.0, 0.0, 0.0),
                    substeps=4,
                    solver_iterations=8,
                    strain_model=strain_model,
                )
                _assign_particle_positions(world, translated, device)
                initial_centroid = translated.mean(axis=0)

                dt = 1.0 / 60.0
                for _ in range(10):
                    world.step(dt)

                positions = world.particles.position.numpy()
                self.assertTrue(np.isfinite(positions).all())
                centroid_drift = float(np.linalg.norm(positions.mean(axis=0) - initial_centroid))
                self.assertLess(
                    centroid_drift,
                    1.0e-4,
                    f"{strain_model} model changed centroid under pure translation by {centroid_drift:.3e} m",
                )
                shape_drift = float(
                    np.linalg.norm((positions - positions.mean(axis=0)) - (translated - initial_centroid), axis=1).max()
                )
                drift_bound = 0.006 if strain_model == "trace" else 1.0e-3
                self.assertLess(
                    shape_drift,
                    drift_bound,
                    f"{strain_model} model response changed under pure translation by {shape_drift:.4f} m",
                )

    def test_elevated_stiffness_compression_stays_finite_for_strain_models(self):
        """Both strain models stay finite and non-inverted at elevated stiffness."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 0.5
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)

        for strain_model in ("trace", "arap"):
            with self.subTest(strain_model=strain_model):
                world = _build_world(
                    device,
                    rest_corners=rest,
                    inv_mass=inv_mass,
                    k_mu=1.0e5,
                    k_lambda=1.0e7,
                    gravity=(0.0, 0.0, 0.0),
                    substeps=8,
                    solver_iterations=24,
                    strain_model=strain_model,
                )
                center = rest.mean(axis=0)
                compressed = center + 0.85 * (rest - center)
                _assign_particle_positions(world, compressed, device)

                initial_ratio = _center_volume_ratio(compressed, rest)
                dt = 1.0 / 60.0
                for _ in range(8):
                    world.step(dt)

                positions = world.particles.position.numpy()
                self.assertTrue(np.isfinite(positions).all())
                final_ratio = _center_volume_ratio(positions, rest)
                self.assertGreater(
                    final_ratio,
                    0.25,
                    f"{strain_model} elevated-stiffness compression inverted or collapsed: "
                    f"initial={initial_ratio:.4f}, final={final_ratio:.4f}",
                )
                self.assertLess(
                    final_ratio,
                    2.0,
                    f"{strain_model} elevated-stiffness compression expanded too far: "
                    f"initial={initial_ratio:.4f}, final={final_ratio:.4f}",
                )

    def test_two_hex_patch_compression_recovers_volume_for_strain_models(self):
        """Adjacent hexes sharing a face recover volume without separating."""
        device = wp.get_preferred_device()
        rest, hex_indices = _build_two_hex_bar()
        inv_mass = np.full(rest.shape[0], 1.0 / (200.0 * 0.002 / rest.shape[0]), dtype=np.float32)

        for strain_model in ("trace", "arap"):
            with self.subTest(strain_model=strain_model):
                world = _build_world_from_arrays(
                    device,
                    rest_positions=rest,
                    hex_indices_np=hex_indices,
                    inv_mass=inv_mass,
                    k_mu=1.0e4,
                    k_lambda=1.0e6,
                    gravity=(0.0, 0.0, 0.0),
                    substeps=5,
                    solver_iterations=20,
                    strain_model=strain_model,
                )
                center = rest.mean(axis=0)
                compressed = center + 0.72 * (rest - center)
                _assign_particle_positions(world, compressed, device)

                initial_ratios = np.array(
                    [_center_volume_ratio(compressed[hex_indices[h]], rest[hex_indices[h]]) for h in range(2)]
                )
                dt = 1.0 / 60.0
                for _ in range(12):
                    world.step(dt)

                positions = world.particles.position.numpy()
                self.assertTrue(np.isfinite(positions).all())
                final_ratios = np.array(
                    [_center_volume_ratio(positions[hex_indices[h]], rest[hex_indices[h]]) for h in range(2)]
                )
                self.assertTrue(
                    np.all(final_ratios > initial_ratios + 0.05),
                    f"{strain_model} two-hex volumes did not recover: initial={initial_ratios}, final={final_ratios}",
                )
                self.assertTrue(
                    np.all(np.abs(1.0 - final_ratios) < np.abs(1.0 - initial_ratios)),
                    f"{strain_model} two-hex volumes moved away from rest: "
                    f"initial={initial_ratios}, final={final_ratios}",
                )

    def test_top_face_pin_hang_stable(self):
        """Top-face pin + gravity -> bottom face hangs without exploding."""
        device = wp.get_preferred_device()
        cube_size = 0.1
        rest = 0.5 * cube_size * _CORNER_SIGNS
        rest[:, 2] += 0.5
        # Pin top face (corners 4..7).
        density = 200.0
        cube_volume = cube_size**3
        corner_mass = density * cube_volume / 8.0
        inv_mass = np.full(8, 1.0 / corner_mass, dtype=np.float32)
        inv_mass[4:8] = 0.0
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            beta_h=5.0,
            beta_d=5.0,
            gravity=(0.0, 0.0, -9.81),
            substeps=5,
            solver_iterations=8,
        )
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        for _ in range(120):  # 2 s at 60 Hz
            wp.capture_launch(graph)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())

        # Pinned top corners haven't drifted (inv_mass=0 => no update).
        pinned_drift = float(np.linalg.norm(positions[4:8] - rest[4:8], axis=1).max())
        self.assertLess(pinned_drift, 1e-4, f"pinned-corner drift {pinned_drift:.2e}")

        # Hex didn't explode: every edge stays within +/- 50% of rest.
        rest_lens = np.linalg.norm(rest[_HEX_EDGES[:, 1]] - rest[_HEX_EDGES[:, 0]], axis=1)
        now_lens = np.linalg.norm(positions[_HEX_EDGES[:, 1]] - positions[_HEX_EDGES[:, 0]], axis=1)
        ratios = now_lens / rest_lens
        self.assertTrue(
            np.all((ratios > 0.5) & (ratios < 1.5)),
            f"hex edges blew up: ratios min={ratios.min():.3f} max={ratios.max():.3f}",
        )

        # Bottom face hangs below the pinned top.
        free_z = float(positions[:4, 2].mean())
        pin_z = float(positions[4:8, 2].mean())
        self.assertLess(free_z, pin_z, "bottom face didn't hang below pinned top")


if __name__ == "__main__":
    unittest.main()
