# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stacking-stability tests for :mod:`solver_phoenx`.

Covers the fundamental solver behaviours the
``example_phoenx_tower`` scene relies on, scaled down so each test
runs in a few frames:

* Free-fall under gravity (no contacts): validates
  :func:`_phoenx_integrate_gravity_kernel` and the position
  integrator.
* Single box on a ground plane: basic contact + penetration
  recovery.
* Two-body vertical stack: contact propagation between two
  dynamics bodies.
* Five-body vertical stack: graph-colouring across a small
  kinematic chain of contacts.
* Mini circular tower (3 layers, 6 planks per ring): miniature of
  the :mod:`example_phoenx_tower` scene, shape and mass ratios
  chosen to roughly match one ring of the full 40-layer tower.

The tests use the same Newton ``CollisionPipeline`` path the
example does so any solver regression that would tip the tower
over shows up here first.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
)
from newton._src.solvers.jitter.constraints.constraint_contact import (
    CONTACT_DWORDS,
)
from newton._src.solvers.jitter.constraints.constraint_container import (
    constraint_container_zeros,
)
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.solver_phoenx import PhoenXWorld

_G = 9.81


# ---------------------------------------------------------------------------
# State sync kernels -- moved from example_phoenx_tower to avoid the
# cost of importing the example module (which builds a 1280-body
# model at import time would not be great).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _newton_to_phoenx_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    i = wp.tid()
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[i] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[i] = rot
    qd = body_qd[i]
    velocity[i] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[i] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def _phoenx_to_newton_kernel(
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    rot = orientation[i]
    com_world = wp.quat_rotate(rot, body_com[i])
    pos_body = position[i] - com_world
    body_q[i] = wp.transform(pos_body, rot)
    body_qd[i] = wp.spatial_vector(velocity[i], angular_velocity[i])


@wp.kernel(enable_backward=False)
def _init_phoenx_bodies_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33f],
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia: wp.array[wp.mat33f],
    inverse_inertia_world: wp.array[wp.mat33f],
    motion_type: wp.array[wp.int32],
):
    """Copy a Newton model's body state into PhoenX layout.

    Slot 0 is the static world anchor; Newton body ``i`` lands at
    ``i + 1``. See ``example_phoenx_tower`` for the long-form
    commentary; this duplicates the kernel so the test module is
    import-independent of the example.
    """
    i = wp.tid()
    dst = i + 1
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[dst] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[dst] = rot
    qd = body_qd[i]
    velocity[dst] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[dst] = wp.vec3f(qd[3], qd[4], qd[5])

    inv_m = body_inv_mass[i]
    inv_I = body_inv_inertia[i]
    inverse_mass[dst] = inv_m
    inverse_inertia[dst] = inv_I
    r = wp.quat_to_matrix(rot)
    inverse_inertia_world[dst] = r * inv_I * wp.transpose(r)
    if inv_m > 0.0:
        motion_type[dst] = wp.int32(MOTION_DYNAMIC)
    else:
        motion_type[dst] = wp.int32(MOTION_STATIC)


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


class _PhoenXScene:
    """Drives a Newton ``ModelBuilder``-built scene through
    :class:`PhoenXWorld`.

    Consumers call :meth:`add_box`, :meth:`add_ground_plane`, then
    :meth:`finalize`; :meth:`step` advances one render frame
    (collide + solve) and is CUDA-graph captured after the first
    call. State queries go through :meth:`body_position` /
    :meth:`body_velocity`.

    Keeps the tests short -- each test case sets up a geometry, runs
    N frames, and asserts a final-state predicate.
    """

    def __init__(
        self,
        *,
        fps: int = 60,
        substeps: int = 20,
        solver_iterations: int = 3,
        velocity_iterations: int = 0,
        friction: float = 0.5,
    ) -> None:
        self.device = wp.get_device("cuda:0")
        self.fps = int(fps)
        self.frame_dt = 1.0 / self.fps
        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.velocity_iterations = int(velocity_iterations)
        self.friction = float(friction)

        self.mb = newton.ModelBuilder()
        self._newton_body_ids: list[int] = []
        self._finalized = False

    # -- scene construction --

    def add_ground_plane(self) -> None:
        self.mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    def add_box(
        self,
        position: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        *,
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float = 1000.0,
    ) -> int:
        """Add a dynamic box; returns the Newton body index."""
        body = self.mb.add_body(
            xform=wp.transform(
                p=wp.vec3(*position),
                q=wp.quat(*orientation),
            ),
        )
        self.mb.add_shape_box(
            body,
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        self._newton_body_ids.append(body)
        return body

    def finalize(self) -> None:
        """Build the Newton model and the PhoenX solver state."""
        assert not self._finalized
        self._finalized = True

        self.model = self.mb.finalize()
        self.state = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state
        )
        self.model.body_q.assign(self.state.body_q)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=JITTER_CONTACT_MATCHING
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        # PhoenX body container: slot 0 = static world anchor.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(
                    np.float32
                ),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        if self.model.body_count > 0:
            wp.launch(
                _init_phoenx_bodies_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_q,
                    self.state.body_qd,
                    self.model.body_com,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                ],
                outputs=[
                    bodies.position,
                    bodies.orientation,
                    bodies.velocity,
                    bodies.angular_velocity,
                    bodies.inverse_mass,
                    bodies.inverse_inertia,
                    bodies.inverse_inertia_world,
                    bodies.motion_type,
                ],
                device=self.device,
            )
        self.bodies = bodies

        # Contact-only constraint container.
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        self.constraints = constraint_container_zeros(
            num_constraints=max_contact_columns,
            num_dwords=CONTACT_DWORDS,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_phoenx, dtype=wp.int32, device=self.device
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            default_friction=self.friction,
            device=self.device,
        )

        self._sync_newton_to_phoenx()
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate()
        self._graph = capture.graph

    # -- per-frame plumbing --

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            _newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def _simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        wp.capture_launch(self._graph)

    # -- state queries --

    def body_position(self, newton_idx: int) -> np.ndarray:
        return self.bodies.position.numpy()[newton_idx + 1].copy()

    def body_velocity(self, newton_idx: int) -> np.ndarray:
        return self.bodies.velocity.numpy()[newton_idx + 1].copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping"
)
class TestPhoenXSolverStacking(unittest.TestCase):
    """End-to-end regression tests for :class:`PhoenXWorld`."""

    # ------------------------------------------------------------------
    # Free-fall (no contacts)
    # ------------------------------------------------------------------

    def test_free_fall_no_contacts(self) -> None:
        """A dynamic box far above the plane free-falls under gravity.

        After ``t`` seconds the COM must sit at
        ``z0 - 0.5 * g * t^2`` (bounded error). No contacts, so this
        isolates :func:`_phoenx_integrate_gravity_kernel` plus the
        position integrator.
        """
        scene = _PhoenXScene(substeps=4, solver_iterations=4)
        scene.add_ground_plane()
        body = scene.add_box(
            position=(0.0, 0.0, 50.0),
            half_extents=(0.5, 0.5, 0.5),
        )
        scene.finalize()

        n_frames = 30  # 0.5 s @ 60 Hz
        for _ in range(n_frames):
            scene.step()

        pos = scene.body_position(body)
        t = n_frames * scene.frame_dt
        expected_z = 50.0 - 0.5 * _G * t * t
        # Allow 3% tolerance for integrator drift across 30 frames.
        self.assertAlmostEqual(float(pos[2]), expected_z, delta=abs(expected_z) * 0.03)
        # XY should stay zero -- no force sideways.
        self.assertLess(float(np.hypot(pos[0], pos[1])), 1e-3)

    # ------------------------------------------------------------------
    # Single box on plane
    # ------------------------------------------------------------------

    def test_single_box_rests_on_plane(self) -> None:
        """One dynamic box dropped near the plane settles on it.

        Tests the plane-vs-box contact path end-to-end: contact
        generation (CollisionPipeline), ingest, warm-start across
        substeps, PGS iterate on the normal row.
        """
        scene = _PhoenXScene(substeps=8, solver_iterations=8)
        scene.add_ground_plane()
        box = scene.add_box(
            position=(0.0, 0.0, 1.2),
            half_extents=(0.5, 0.5, 0.5),
        )
        scene.finalize()

        for _ in range(120):  # 2 s -- plenty to settle
            scene.step()

        pos = scene.body_position(box)
        vel = scene.body_velocity(box)
        # Settled height: half_extent above the plane, allow a small
        # positive margin (collision margin + numerical slop).
        self.assertGreater(float(pos[2]), 0.45)
        self.assertLess(float(pos[2]), 0.6)
        # Settled velocity magnitude should be << gravity-impulse per
        # frame (``g * dt = 9.81 / 60 ≈ 0.16``). Anything above that
        # signals the contact row isn't holding.
        self.assertLess(float(np.linalg.norm(vel)), 0.05)

    # ------------------------------------------------------------------
    # Two-body stack
    # ------------------------------------------------------------------

    def test_two_body_stack(self) -> None:
        """Two equal-mass boxes stacked vertically settle without
        tipping.

        Catches regressions in the graph colouring / warm-start
        chain where the lower box's contact with the plane and its
        contact with the upper box would both need to converge.
        """
        scene = _PhoenXScene(substeps=8, solver_iterations=12)
        scene.add_ground_plane()
        he = 0.5
        gap = 5.0e-3
        bottom = scene.add_box(
            position=(0.0, 0.0, he + 0.1),  # small initial penetration margin
            half_extents=(he, he, he),
        )
        top = scene.add_box(
            position=(0.0, 0.0, 2 * he + he + 0.1 + gap),
            half_extents=(he, he, he),
        )
        scene.finalize()

        for _ in range(180):  # 3 s
            scene.step()

        p_bottom = scene.body_position(bottom)
        p_top = scene.body_position(top)
        v_bottom = scene.body_velocity(bottom)
        v_top = scene.body_velocity(top)

        # Positions: bottom sits around he, top around 3*he (1 he
        # above the bottom's top face). Allow ±0.15 m slack for
        # collision margin + settle slop.
        self.assertAlmostEqual(float(p_bottom[2]), he, delta=0.15)
        self.assertAlmostEqual(float(p_top[2]), 3 * he, delta=0.2)
        # XY drift must stay tiny -- any sideways slip leaks
        # momentum out through friction.
        self.assertLess(float(np.hypot(p_bottom[0], p_bottom[1])), 0.05)
        self.assertLess(float(np.hypot(p_top[0], p_top[1])), 0.1)
        # Velocities must damp to near zero.
        self.assertLess(float(np.linalg.norm(v_bottom)), 0.05)
        self.assertLess(float(np.linalg.norm(v_top)), 0.1)

    # ------------------------------------------------------------------
    # Five-body stack
    # ------------------------------------------------------------------

    def test_five_body_stack(self) -> None:
        """A 5-cube vertical column settles without collapse.

        Probes contact propagation through a small kinematic chain
        under gravity; the bottom cube carries 5x its own weight
        once settled.
        """
        scene = _PhoenXScene(substeps=10, solver_iterations=12)
        scene.add_ground_plane()
        he = 0.5
        gap = 5.0e-3
        n = 5
        bodies: list[int] = []
        z = he + 0.1  # small initial gap to plane to avoid start-penetrating
        for _ in range(n):
            bodies.append(
                scene.add_box(
                    position=(0.0, 0.0, z),
                    half_extents=(he, he, he),
                )
            )
            z += 2 * he + gap
        scene.finalize()

        z0_top = z - (2 * he + gap)  # initial top COM
        for _ in range(240):  # 4 s
            scene.step()

        positions = [scene.body_position(b) for b in bodies]
        top_z = float(positions[-1][2])
        # Top cube must stay within a tight envelope of the expected
        # settled height ``(n - 0.5) * 2 * he = 4.5`` (for he=0.5).
        expected_top_z = (n - 0.5) * 2 * he
        self.assertAlmostEqual(top_z, expected_top_z, delta=0.3)
        # Stack must not lean: the top cube's XY drift from its spawn
        # should stay under half a cube.
        xy_drift = float(np.hypot(positions[-1][0], positions[-1][1]))
        self.assertLess(xy_drift, he)
        # All cubes finite -- no NaN blow-ups.
        for i, p in enumerate(positions):
            self.assertTrue(np.isfinite(p).all(), f"cube {i} non-finite: {p}")
        # Z ordering preserved -- no cube passed through another.
        for i in range(1, n):
            self.assertGreater(
                float(positions[i][2]),
                float(positions[i - 1][2]) + he,
                f"cube {i} sank into cube {i - 1}",
            )

    # ------------------------------------------------------------------
    # Mini circular tower
    # ------------------------------------------------------------------

    def test_mini_circular_tower(self) -> None:
        """A 3-layer, 6-planks-per-ring circular tower stays upright
        over a short settle.

        Geometry scaled down from :mod:`example_phoenx_tower` but
        keeps the same plank aspect ratio (thin in the radial
        direction, wide tangentially), so contact topology matches
        the full tower's. A settled tower should have every plank
        within an envelope roughly 2x the ring radius.
        """
        scene = _PhoenXScene(
            fps=120, substeps=20, solver_iterations=3, velocity_iterations=1
        )
        scene.add_ground_plane()

        tower_height_layers = 3
        boxes_per_ring = 6
        plank_hx = 1.5
        plank_hy = 0.1
        plank_hz = 0.5
        ring_radius = 3.5  # smaller than full tower (19.5) so 6 planks form a ring
        full_rotation_step = 2.0 * math.pi / boxes_per_ring
        half_rotation_step = 0.5 * full_rotation_step

        plank_ids: list[int] = []
        orientation_rad = 0.0
        for e in range(tower_height_layers):
            orientation_rad += half_rotation_step
            for _ in range(boxes_per_ring):
                cos_o = math.cos(orientation_rad)
                sin_o = math.sin(orientation_rad)
                local_y = ring_radius
                world_x = -sin_o * local_y
                world_y = cos_o * local_y
                world_z = plank_hz + e * 2.0 * plank_hz  # base of plank on ground for e=0
                # Rotation quaternion about +Z by orientation_rad.
                half = 0.5 * orientation_rad
                quat = (0.0, 0.0, math.sin(half), math.cos(half))
                plank_ids.append(
                    scene.add_box(
                        position=(float(world_x), float(world_y), float(world_z)),
                        half_extents=(plank_hx, plank_hy, plank_hz),
                        orientation=quat,
                    )
                )
                orientation_rad += full_rotation_step

        scene.finalize()

        # Run 1.5 s at 120 Hz -- enough for the tower to settle onto
        # its neighbours and for any instability to show itself.
        for _ in range(180):
            scene.step()

        envelope = ring_radius * 2.5
        z_max_expected = tower_height_layers * 2 * plank_hz + 0.5
        for pid in plank_ids:
            p = scene.body_position(pid)
            self.assertTrue(np.isfinite(p).all(), f"plank {pid} non-finite: {p}")
            r_xy = float(np.hypot(p[0], p[1]))
            self.assertLess(
                r_xy,
                envelope,
                f"plank {pid} escaped the tower envelope (r_xy={r_xy:.2f})",
            )
            self.assertLess(
                float(p[2]),
                z_max_expected,
                f"plank {pid} rose above its spawn layer (z={p[2]:.2f})",
            )


if __name__ == "__main__":
    unittest.main()
