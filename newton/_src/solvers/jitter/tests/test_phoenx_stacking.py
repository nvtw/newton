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
from newton._src.solvers.jitter.examples.example_phoenx_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.jitter.solver_phoenx import PhoenXWorld

_G = 9.81


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
        # Pick up the PhoenX contact-ahead-of-impact default so every
        # shape added to the scene detects contacts a few cm before
        # penetration. Keeps the scene-level contact settings in one
        # place and lets the tests exercise the same defaults that
        # user-facing PhoenX scenes pick up via ``make_phoenx_shape_cfg``.
        from newton._src.solvers.jitter.solver_phoenx import DEFAULT_SHAPE_GAP
        self.mb.default_shape_cfg.gap = DEFAULT_SHAPE_GAP
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
        mass: float | None = None,
    ) -> int:
        """Add a dynamic box; returns the Newton body index.

        If ``mass`` is provided, the body is created with an
        analytically-correct inertia tensor for a uniform-density
        box (``I_xx = m/3 * (hy² + hz²)`` etc.) and the shape is
        added with ``density=0`` so Newton's shape-inertia path
        doesn't double-count. Without this the ``density=0`` path
        leaves the body with zero inertia, which Newton's inertia
        validator then corrects to a near-zero-but-not-zero
        substitute (``inv_I ≈ 1e6``); the resulting body is
        effectively a point mass with infinite angular mass and
        stacks refuse to settle. Use ``mass`` when a test needs an
        exact analytical mass for a contact-force comparison; use
        ``density`` otherwise.
        """
        if mass is not None:
            hx, hy, hz = (
                float(half_extents[0]),
                float(half_extents[1]),
                float(half_extents[2]),
            )
            ixx = float(mass) / 3.0 * (hy * hy + hz * hz)
            iyy = float(mass) / 3.0 * (hx * hx + hz * hz)
            izz = float(mass) / 3.0 * (hx * hx + hy * hy)
            body = self.mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(*position),
                    q=wp.quat(*orientation),
                ),
                mass=float(mass),
                inertia=((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz)),
            )
            self.mb.add_shape_box(
                body,
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            )
        else:
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

    def add_static_box(
        self,
        position: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        *,
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Add a box-shape attached to the static world body at slot 0.

        Instead of creating a new Newton body the shape is attached
        to Newton body index ``-1`` (the world anchor), which on the
        PhoenX side maps to slot 0 (the static anchor). Useful for
        building ramps / walls / arenas that are not themselves
        dynamic but need rigid-box collision geometry, which Newton's
        :func:`add_shape_plane` alone can't express (plane is always
        horizontal and unbounded).
        """
        # ``body = -1`` attaches the shape to Newton's world body.
        self.mb.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(*position), q=wp.quat(*orientation)),
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
        )
        return -1

    def add_sphere(
        self,
        position: tuple[float, float, float],
        radius: float,
        *,
        mass: float | None = None,
        density: float = 1000.0,
    ) -> int:
        """Add a dynamic sphere; returns the Newton body index.

        Sphere-on-plane is the classic analytical contact-force test
        (single-point manifold, trivial expected F=mg). See
        :meth:`add_box` for ``mass`` vs ``density`` semantics.
        """
        if mass is not None:
            # Solid-sphere inertia: I = 2/5 * m * r^2 along every axis.
            # Same rationale as :meth:`add_box`: supply explicit
            # inertia when passing ``mass`` so Newton's validator
            # doesn't corrupt a zero-inertia body into an
            # effectively-massless angular body.
            r = float(radius)
            i_sphere = 0.4 * float(mass) * r * r
            body = self.mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(*position),
                    q=wp.quat_identity(),
                ),
                mass=float(mass),
                inertia=(
                    (i_sphere, 0.0, 0.0),
                    (0.0, i_sphere, 0.0),
                    (0.0, 0.0, i_sphere),
                ),
            )
            self.mb.add_shape_sphere(
                body,
                radius=float(radius),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            )
        else:
            body = self.mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(*position),
                    q=wp.quat_identity(),
                ),
            )
            self.mb.add_shape_sphere(
                body,
                radius=float(radius),
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

        # Defer the CUDA graph capture until the first :meth:`step`
        # call. Tests that install materials or change other solver
        # state after :meth:`finalize` can do so before the graph
        # records its kernel launches; otherwise the captured graph
        # would bind stale pointers (e.g. the pre-``set_materials``
        # ``_materials = None`` fallback) and replay with incorrect
        # behaviour.
        self._sync_newton_to_phoenx()
        self._graph = None

    def _capture_graph(self) -> None:
        """Record the per-frame pipeline into a CUDA graph.

        Called lazily by :meth:`step`. Runs :meth:`_simulate` once
        eagerly (so every kernel involved compiles before capture
        begins), then records a second invocation into the graph.
        """
        # Warm-up launch: compiles every kernel Warp might touch.
        self._simulate()
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
        if self._graph is None:
            self._capture_graph()
        wp.capture_launch(self._graph)

    # -- state queries --

    def body_position(self, newton_idx: int) -> np.ndarray:
        return self.bodies.position.numpy()[newton_idx + 1].copy()

    def body_velocity(self, newton_idx: int) -> np.ndarray:
        return self.bodies.velocity.numpy()[newton_idx + 1].copy()

    def set_body_velocity(
        self,
        newton_idx: int,
        velocity: tuple[float, float, float],
    ) -> None:
        """Overwrite ``state.body_qd`` (linear only) for one body.

        Used by slam / impact tests to prime the scene with a known
        initial velocity. Must be called after :meth:`finalize` (so
        ``state.body_qd`` exists) and before the first :meth:`step`;
        the velocity is picked up by the first ``_sync_newton_to_phoenx``
        at the start of :meth:`step`.
        """
        body_qd_np = self.state.body_qd.numpy()
        body_qd_np[newton_idx, 0] = float(velocity[0])
        body_qd_np[newton_idx, 1] = float(velocity[1])
        body_qd_np[newton_idx, 2] = float(velocity[2])
        self.state.body_qd.assign(body_qd_np)

    # -- contact force reporting --

    def gather_contact_wrench_on_body(
        self, newton_idx: int
    ) -> tuple[np.ndarray, int, int]:
        """Return ``(force[3], n_pairs, n_contact_points)`` for one
        Newton body, summed from the per-pair wrench API.

        Sign convention: every column's wrench is reported as the
        force body 2 felt from body 1. Entries where ``newton_idx``
        is body 2 contribute verbatim; entries where it is body 1
        contribute with the sign flipped. Inactive columns
        (``pair_count == 0``) are skipped.

        Returns:
            force: 3-vector [N] -- net contact force on
                ``newton_idx`` from every contact pair involving it
                during the last substep.
            n_pairs: number of distinct contact columns that
                contributed.
            n_contact_points: sum of ``pair_count`` across those
                columns -- the actual number of contact manifold
                points folded into ``force``.
        """
        n_cols = self.world.max_contact_columns
        if n_cols == 0:
            return np.zeros(3, dtype=np.float32), 0, 0
        pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=self.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_count = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        self.world.gather_contact_pair_wrenches(
            pair_w, pair_b1, pair_b2, pair_count
        )
        pw = pair_w.numpy()[:, :3]
        b1 = pair_b1.numpy()
        b2 = pair_b2.numpy()
        cnt = pair_count.numpy()

        phoenx_slot = int(newton_idx) + 1  # slot 0 is the static world
        total = np.zeros(3, dtype=np.float32)
        npairs = 0
        npoints = 0
        for i in range(n_cols):
            if cnt[i] <= 0:
                continue
            if b2[i] == phoenx_slot:
                total += pw[i]
                npairs += 1
                npoints += int(cnt[i])
            elif b1[i] == phoenx_slot:
                total -= pw[i]
                npairs += 1
                npoints += int(cnt[i])
        return total, npairs, npoints

    def active_rigid_contact_count(self) -> int:
        """Raw narrow-phase contact count, pre-ingest. Compared
        against the summed ``n_contact_points`` from
        :meth:`gather_contact_wrench_on_body` to detect ingest
        duplication / loss.
        """
        return int(self.contacts.rigid_contact_count.numpy()[0])

    def gather_pair_wrenches_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the raw per-pair wrench arrays: ``(force_xyz,
        body1_slot, body2_slot, pair_count)``.

        ``body{1,2}_slot`` are PhoenX slot indices (slot 0 = static
        world). Inactive columns have ``pair_count == 0``. Callers
        that want per-body sums can use
        :meth:`gather_contact_wrench_on_body` instead; this raw view
        is useful for per-body up/down decomposition (pyramid force
        balance) where we need to distinguish "pairs where this
        body is body1" from "pairs where this body is body2".
        """
        n_cols = self.world.max_contact_columns
        if n_cols == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )
        pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=self.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_count = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        self.world.gather_contact_pair_wrenches(
            pair_w, pair_b1, pair_b2, pair_count
        )
        return (
            pair_w.numpy()[:, :3],
            pair_b1.numpy(),
            pair_b2.numpy(),
            pair_count.numpy(),
        )

    def gather_per_contact_wrenches(self) -> np.ndarray:
        """Return the per-rigid-contact 3-vector force array.

        Indexed by ``rigid_contact_index`` -- row ``k`` is the
        force body 2 of the ``k``-th contact felt from that contact
        in the last substep. Inactive indices (``k >=
        rigid_contact_count[0]``) are zeroed because
        :meth:`PhoenXWorld.gather_contact_wrenches` pre-zeroes the
        output; callers that want only the active prefix slice
        down to ``active_rigid_contact_count()``.
        """
        per = wp.zeros(
            self.world.rigid_contact_max,
            dtype=wp.spatial_vector,
            device=self.device,
        )
        self.world.gather_contact_wrenches(per)
        return per.numpy()[:, :3]

    # -- per-frame actuation -------------------------------------------

    def apply_body_force(
        self,
        newton_idx: int,
        force: tuple[float, float, float] = (0.0, 0.0, 0.0),
        torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Overwrite ``bodies.force`` / ``bodies.torque`` for one body.

        PhoenX consumes the force/torque accumulators each substep via
        ``_phoenx_apply_external_forces_kernel`` and zeroes them at
        the end of each :meth:`PhoenXWorld.step` call, so this writes
        a per-frame impulse. Call before :meth:`step` every frame you
        want the force applied; zero-ing / stopping is the default
        (force stays cleared after the step finishes).
        """
        slot = int(newton_idx) + 1
        f = self.bodies.force.numpy()
        t = self.bodies.torque.numpy()
        f[slot] = force
        t[slot] = torque
        self.bodies.force.assign(f)
        self.bodies.torque.assign(t)

    # -- optional material installation --------------------------------

    def install_materials(
        self,
        materials: wp.array,
        shape_material: wp.array,
    ) -> None:
        """Install a per-shape material table on the solver.

        Pass-through to :meth:`PhoenXWorld.set_materials`. Kept on
        the scene for symmetry with the jitter-side harness.
        """
        self.world.set_materials(materials, shape_material)


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


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping"
)
class TestPhoenXSolverRobustness(unittest.TestCase):
    """Stress tests for the PhoenX solver's Baumgarte / bias handling.

    The tower example used to explode any time a user-picked plank
    was yanked hard enough to deeply penetrate its neighbours: the
    contact prepare kernel computed ``bias = 0.2 * penetration /
    substep_dt`` which at substep_dt=1/1200 s turned a 10 cm
    penetration into a 24 m/s velocity correction, and with only 3
    PGS iterations that correction couldn't be resolved in one
    substep. Sub-sequent substeps saw the injected velocity but
    with a new (post-integration) penetration that was even larger,
    producing runaway.

    These tests deliberately cause deep penetration / high-velocity
    contact to verify the solver stays bounded; they failed at
    factor-of-1000 velocity amplifications on the pre-fix code.
    """

    def test_slam_two_boxes_head_on(self) -> None:
        """Two boxes launched at each other at 20 m/s.

        Pre-fix: the pair's velocity would amplify past 100 m/s and
        NaN out within ~10 frames. Post-fix: the normal row clamps
        the relative velocity and the pair either rebounds (inelastic
        collision with warm-start) or stops; either way the peak
        velocity never exceeds a few times the impact speed.
        """
        scene = _PhoenXScene(substeps=20, solver_iterations=3)
        scene.add_ground_plane()
        left = scene.add_box(position=(-2.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        right = scene.add_box(position=(2.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        scene.set_body_velocity(left, (20.0, 0.0, 0.0))
        scene.set_body_velocity(right, (-20.0, 0.0, 0.0))

        peak_v = 0.0
        for _ in range(120):
            scene.step()
            v_left = scene.body_velocity(left)
            v_right = scene.body_velocity(right)
            peak_v = max(
                peak_v,
                float(np.linalg.norm(v_left)),
                float(np.linalg.norm(v_right)),
            )
            self.assertTrue(
                np.isfinite(v_left).all() and np.isfinite(v_right).all(),
                "velocity went non-finite",
            )

        # Even with a perfectly elastic rebound the pair can never
        # exceed the 20 m/s impact speed (momentum conservation).
        # We allow a 2x margin for Baumgarte-generated overshoot and
        # any normal-row divergence.
        self.assertLess(peak_v, 40.0, f"peak velocity {peak_v:.1f} m/s blew past 2x impact speed")

    def test_slam_ball_into_stack(self) -> None:
        """A heavy box launched horizontally into a 3-cube vertical
        stack. Common picking-demo scenario: yanking a plank sideways
        into its neighbour drives the neighbour's COM past the next
        plank's face, producing a multi-body cascade.

        The test verifies the stack responds but doesn't explode
        (velocities stay bounded, no NaNs).
        """
        scene = _PhoenXScene(substeps=20, solver_iterations=3)
        scene.add_ground_plane()
        he = 0.5
        gap = 5.0e-3
        stack_ids: list[int] = []
        z = he + 0.05
        for _ in range(3):
            stack_ids.append(
                scene.add_box(position=(0.0, 0.0, z), half_extents=(he, he, he))
            )
            z += 2 * he + gap
        slammer = scene.add_box(
            position=(-4.0, 0.0, he + 0.05), half_extents=(he, he, he)
        )
        scene.finalize()
        scene.set_body_velocity(slammer, (15.0, 0.0, 0.0))

        peak_v = 0.0
        peak_z = 0.0
        peak_r = 0.0
        for _ in range(240):
            scene.step()
            for b in [*stack_ids, slammer]:
                v = scene.body_velocity(b)
                p = scene.body_position(b)
                self.assertTrue(
                    np.isfinite(v).all() and np.isfinite(p).all(),
                    f"body {b} went non-finite",
                )
                peak_v = max(peak_v, float(np.linalg.norm(v)))
                peak_z = max(peak_z, float(p[2]))
                peak_r = max(peak_r, float(np.linalg.norm(p)))

        # Impact speed 15 m/s; with a 1:1 mass ratio the stack can
        # pick up roughly the impact speed and then shed it through
        # friction+contacts. 3x impact speed is a generous cap that
        # still catches explosions (pre-fix runs hit 2000+ m/s).
        self.assertLess(peak_v, 45.0, f"peak velocity {peak_v:.1f} m/s indicates explosion")
        # No cube should end up more than 20 m from the origin --
        # deflation goes through the ground plane (z stays bounded)
        # and sliding drifts ~a few meters at worst.
        self.assertLess(peak_r, 20.0, f"peak radius {peak_r:.2f} m indicates ejected body")
        self.assertLess(peak_z, 10.0, f"peak height {peak_z:.2f} m indicates upward blow-up")

    def test_dropped_box_deep_penetration(self) -> None:
        """A box spawned intersecting the ground plane by half its
        half-extent should *not* explode on the first step.

        Worst case for the Baumgarte bias: penetration is large
        (he/2 = 25 cm for a 0.5 m cube) and the initial velocity is
        zero, so a naive ``bias = 0.2 * 0.25 / substep_dt`` injects
        60 m/s of velocity in one substep. The solver must either
        cap the bias or spread the resolution over many substeps;
        either way the cube's settled velocity must stay bounded.
        """
        scene = _PhoenXScene(substeps=20, solver_iterations=3)
        scene.add_ground_plane()
        he = 0.5
        # Spawn so the cube's bottom face sits 25 cm below z=0.
        box = scene.add_box(position=(0.0, 0.0, he * 0.5), half_extents=(he, he, he))
        scene.finalize()

        peak_v = 0.0
        for _ in range(120):
            scene.step()
            v = scene.body_velocity(box)
            p = scene.body_position(box)
            self.assertTrue(np.isfinite(v).all() and np.isfinite(p).all())
            peak_v = max(peak_v, float(np.linalg.norm(v)))

        # Pre-fix peak was ~3000 m/s. Post-fix should keep it under
        # a few m/s (just the gravity + pushout transient).
        self.assertLess(peak_v, 20.0, f"peak velocity {peak_v:.1f} m/s indicates bias runaway")


if __name__ == "__main__":
    unittest.main()
