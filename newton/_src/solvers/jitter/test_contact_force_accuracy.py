# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Debug suite for contact-force reporting accuracy.

This file is deliberately minimal and exists purely to pin down a
suspected bug in :meth:`World.gather_contact_pair_wrenches`. The
existing :mod:`test_actuated_double_ball_socket_physics` tests showed
the reported sphere-on-ground normal force coming in well above the
analytical ``m*g`` baseline even before any motor pushed on the
sphere; this file isolates that.

Test plan
---------

``TestContactForceAccuracy.test_static_sphere_weight``
    A single sphere rests on an infinite ground plane under gravity.
    After the scene settles, the per-contact wrench reported by
    :meth:`World.gather_contact_pair_wrenches` must equal ``(0, 0, m*g)``
    (normal force only, magnitude equal to the sphere's weight).

``TestContactForceAccuracy.test_prismatic_drive_pushes_sphere``
    Same scene as above, but a vertical prismatic joint uses a
    **position** PD drive to press the sphere into the ground below
    its rest position. At steady state the drive force is predictable
    from the PD gains:

    .. math::

        F_{\\text{drive}} = k_p \\, (x_{\\text{eq}} - x_{\\text{target}})

    where ``x_eq`` is the sphere's equilibrium slide coordinate (its
    height above the ground, since the slider is axis-aligned).
    Ground contact must carry ``m*g + F_drive``. We test a sweep of
    ``(kp, kd)`` pairs to confirm the relationship holds across
    stiffnesses.

The tests use :meth:`World.gather_contact_pair_wrenches` as the
ground truth (it's the solver's own output of the PGS-applied
impulse divided by the substep dt) -- the whole point is to decide
whether that API is correct.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)
from newton._src.solvers.jitter.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

_G = 9.81  # [m/s^2]


# ---------------------------------------------------------------------------
# Minimal Newton-builder + Jitter-world wrapper (duplicated from the
# actuator physics test so this file stands on its own without a
# helper-module dependency)
# ---------------------------------------------------------------------------


class _JitterScene:
    def __init__(
        self,
        model_builder: newton.ModelBuilder,
        world_builder_factory,
        *,
        fps: int,
        substeps: int,
        solver_iterations: int,
        gravity: tuple[float, float, float] = (0.0, 0.0, -_G),
        expected_masses: dict[int, float] | None = None,
    ):
        # CUDA-only path. The contact-force debug scenes are too heavy
        # to run on CPU in a reasonable time; graph capture below
        # assumes CUDA as well. Callers are expected to
        # :meth:`unittest.TestCase.skipTest` the test before
        # constructing a scene on non-CUDA hardware.
        self.device = wp.get_device("cuda:0")
        self.fps = int(fps)
        self.substeps = int(substeps)
        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.substeps

        self.model = model_builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state
        )
        self.model.body_q.assign(self.state.body_q)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=True
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        # ``expected_masses`` lets each test assert the Newton-side
        # ingested mass matches the analytical value it's about to
        # validate against. This catches "``add_body(mass=M)`` was
        # silently inflated by shape density" bugs up front, before
        # the contact solver gets a chance to produce mystery
        # multipliers on top.
        builder, newton_to_jitter = build_jitter_world_from_model(
            self.model, expected_masses=expected_masses
        )
        self.newton_to_jitter = newton_to_jitter
        world_builder_factory(newton_to_jitter, builder)

        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        num_shapes = int(self.model.shape_count)

        self.world = builder.finalize(
            substeps=self.substeps,
            solver_iterations=int(solver_iterations),
            gravity=gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()

        # Record the full per-frame pipeline (sync -> collide -> step
        # -> sync) as a single CUDA graph. Calling ``step()`` from
        # now on replays the graph with one ``wp.capture_launch`` --
        # no Python-side launch overhead, and crucially no
        # module-load / JIT compile cost after warm-up. We follow the
        # :mod:`example_pyramid` pattern: the ``with ScopedCapture``
        # block runs one full frame eagerly (so every kernel is
        # compiled before capture begins) and Warp records it. Each
        # subsequent ``step()`` call replays the same graph.
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate()
        self._graph = capture.graph

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            newton_to_jitter_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def _simulate(self) -> None:
        """Inner per-frame pipeline -- captured once into a CUDA graph.

        Pure device-side work: collision pipeline + world.step +
        Newton<->Jitter state sync. No host-side branching, no
        readbacks.
        """
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            picking=None,
        )
        self._sync_jitter_to_newton()

    def step(self) -> None:
        """Advance one frame by replaying the captured CUDA graph."""
        wp.capture_launch(self._graph)

    def gather_contact_wrench_on_body(
        self, newton_body: int
    ) -> tuple[np.ndarray, int, int]:
        """Return ``(force[3], n_pairs, n_contact_points)`` for
        ``newton_body`` using the solver's per-pair summary API.

        * ``force``: net 3-vector [N] the body felt from all contact
          pairs in the last substep (body1 entries negated per
          Newton's third law).
        * ``n_pairs``: number of distinct shape-pair *columns* that
          contributed, i.e. the number of packed contact columns whose
          ``pair_count > 0`` and that mention ``newton_body``.
        * ``n_contact_points``: sum of ``pair_count`` across the same
          columns -- the actual number of discrete contact manifold
          points folded into ``force``. For a sphere-vs-plane pair we
          expect exactly one manifold point; anything more would mean
          the narrow phase is over-reporting and ``force`` is a multi-
          count sum.

        Sign convention: every pair's wrench is reported as the force
        body 2 felt. Pairs where ``newton_body`` is body 2 contribute
        verbatim; where it is body 1 they contribute negated.
        """
        n_cols = self.world.max_contact_columns
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

        j = self.newton_to_jitter[int(newton_body)]
        total = np.zeros(3, dtype=np.float32)
        npairs = 0
        npoints = 0
        for i in range(n_cols):
            if cnt[i] <= 0:
                continue
            if b2[i] == j:
                total += pw[i]
                npairs += 1
                npoints += int(cnt[i])
            elif b1[i] == j:
                total -= pw[i]
                npairs += 1
                npoints += int(cnt[i])
        return total, npairs, npoints

    def active_rigid_contact_count(self) -> int:
        """Number of active contact points the upstream narrow phase
        emitted for the most recent frame. Read from
        ``contacts.rigid_contact_count[0]``; this is the *raw* count
        before the per-pair packing/deduplication done by the Jitter
        ingest stage, so comparing it with ``n_contact_points`` from
        :meth:`gather_contact_wrench_on_body` tells us whether the
        ingest stage is losing or duplicating points.
        """
        return int(self.contacts.rigid_contact_count.numpy()[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContactForceAccuracy(unittest.TestCase):
    """Isolate the contact-force reporting bug on the simplest possible
    scenes: a sphere on a plane, with and without a predictable axial
    push-down drive.
    """

    MASS = 2.0
    SPHERE_RADIUS = 0.1

    FPS = 120
    SUBSTEPS = 8
    SOLVER_ITERATIONS = 40
    # 1 s at 120 fps -- at SOLVER_ITERATIONS=40 and SUBSTEPS=8 this is
    # enough to bleed the drop transient on every ``(kp, kd)`` case we
    # sweep below down to <1 cm/s. The scene is CUDA-graph-captured
    # so each frame is essentially a single ``cudaGraphLaunch``.
    SETTLE_FRAMES = 120

    @classmethod
    def setUpClass(cls):
        # These tests rely on CUDA graph capture for acceptable
        # runtime; CPU would be several orders of magnitude slower.
        # Skip cleanly on machines without a CUDA device so CI on
        # CPU-only runners still passes.
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "contact-force accuracy tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Scene builders
    # ------------------------------------------------------------------

    def _build_static_sphere_on_ground(self) -> _JitterScene:
        """Single sphere released just above the ground, no joint,
        gravity only. After settling the only non-gravity force on the
        sphere is the ground's normal contact -- so the reported
        contact force must equal ``+m g`` along +z.
        """
        mb = newton.ModelBuilder()
        # Start 5 cm above the ground so the drop transient is tiny
        # and settles within a fraction of a second.
        start_z = self.SPHERE_RADIUS + 0.05
        sphere = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, start_z), q=wp.quat_identity()
            ),
            mass=self.MASS,
        )
        # ``density=0`` so the shape contributes no implicit mass/inertia --
        # otherwise Newton's ModelBuilder adds the sphere's default density
        # mass (4/3 pi r^3 * 1000 kg/m^3 ~ 4 kg for r=0.1) *on top of* the
        # body's explicit ``mass=self.MASS`` and the analytical expected
        # contact force no longer matches ``m * g``.
        mb.add_shape_sphere(
            sphere,
            radius=self.SPHERE_RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        def _factory(n2j, builder):
            # No joints -- the sphere is purely free-falling / resting
            # on the ground plane. All contact force diagnostics flow
            # through the contact pipeline alone.
            del n2j, builder

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            # Lock in the exact mass the analytical F=m*g check relies on.
            expected_masses={sphere: self.MASS},
        )
        scene.sphere = sphere
        return scene

    def _build_prismatic_pd_press(
        self, *, kp: float, kd: float, x_target: float
    ) -> _JitterScene:
        """Vertical prismatic joint with a PD position drive pushing
        the sphere into the ground.

        Geometry: ``anchor1 = (0, 0, 0)`` on the world, ``anchor2 =
        (0, 0, -1)`` so the slide axis ``n_hat`` points along ``-z``.
        The slide coordinate ``x = n_hat . (p2 - p1) = -p2.z`` grows
        as the sphere moves down. The drive target ``x_target`` is
        *below* the ground (``x_target > SPHERE_RADIUS``), so in steady
        state the sphere sits on the ground at ``p2.z = SPHERE_RADIUS``
        (i.e. ``x_eq = -SPHERE_RADIUS``... sign flip: since n_hat = -z,
        ``x = -p2.z = -SPHERE_RADIUS``. Drive push force is
        ``kp * (x_target - x_eq)``).

        Wait -- sign check. The drive applies
            ``tau = -kp * (x - x_target) - kd * (v - 0)``
        along ``+n_hat``. If ``x_target > x_eq``, the drive pushes
        along ``+n_hat`` (downward here), loading the ground.
        Magnitude at rest: ``|F_drive| = kp * (x_target - x_eq)``.
        """
        start_z = self.SPHERE_RADIUS + 0.05

        mb = newton.ModelBuilder()
        sphere = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, start_z), q=wp.quat_identity()
            ),
            mass=self.MASS,
        )
        # See the comment in ``_build_static_sphere_on_ground``: ``density=0``
        # prevents ``add_shape_sphere`` from stacking its default density
        # mass on top of the body's explicit ``mass=self.MASS``.
        mb.add_shape_sphere(
            sphere,
            radius=self.SPHERE_RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[sphere],
                # ``n_hat`` = (anchor2 - anchor1) normalised = (0, 0, -1).
                # Initial slide ``x0 = n_hat . (p2 - p1) = -start_z``.
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 0.0, -1.0),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.POSITION,
                target=float(x_target),
                target_velocity=0.0,
                # No cap -- we want the PD to *set* the force, not
                # saturate at some external limit.
                max_force_drive=1.0e6,
                stiffness_drive=float(kp),
                damping_drive=float(kd),
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            expected_masses={sphere: self.MASS},
        )
        scene.sphere = sphere
        return scene

    # ------------------------------------------------------------------
    # Post-settle helpers
    # ------------------------------------------------------------------

    def _settle(self, scene: _JitterScene) -> None:
        """Step for ``SETTLE_FRAMES`` frames and assert the sphere has
        come to rest to within 5 mm/s -- the contact-force readout
        below assumes quasi-static equilibrium.
        """
        for _ in range(self.SETTLE_FRAMES):
            scene.step()
        v = scene.world.bodies.velocity.numpy()[
            scene.newton_to_jitter[scene.sphere]
        ]
        speed = float(np.linalg.norm(v))
        self.assertLess(
            speed,
            0.05,
            f"sphere still moving at end of settle: |v|={speed:.4f} m/s",
        )

    def _sphere_slide(self, scene: _JitterScene) -> float:
        """Slide coordinate ``x = n_hat . (p2 - p1)`` with
        ``n_hat = (0, 0, -1)``, i.e. ``x = -p2.z``.
        """
        p = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[scene.sphere]
        ]
        return -float(p[2])

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_static_sphere_weight(self):
        """Zero-joint sphere-on-ground: contact force == m*g.

        Also validates the contact *cardinality*: a sphere touching a
        plane in static equilibrium must produce exactly one contact
        point. If the narrow phase emits duplicates (e.g. two slightly-
        offset points that the pair-summary kernel sums together), the
        reported force gets multiplied by the duplicate count. That's
        the bug we're hunting in this suite, so we gate it explicitly.
        """
        scene = self._build_static_sphere_on_ground()
        self._settle(scene)

        F, npairs, npoints = scene.gather_contact_wrench_on_body(scene.sphere)
        self.assertEqual(
            npairs,
            1,
            f"expected exactly one sphere-ground contact pair, got {npairs}",
        )
        # A sphere sitting on a plane is a single-point contact. Any
        # value > 1 here means the narrow phase is over-reporting or
        # the pair-summary kernel is double-counting.
        self.assertEqual(
            npoints,
            1,
            f"sphere-plane should produce exactly 1 contact point in the "
            f"pair-summary output; got {npoints}",
        )
        # Cross-check against the raw narrow-phase count.
        raw_count = scene.active_rigid_contact_count()
        self.assertEqual(
            raw_count,
            1,
            f"narrow phase emitted {raw_count} contacts for a sphere-on-"
            f"plane at rest; expected exactly 1",
        )

        expected_Fz = self.MASS * _G
        self.assertGreater(
            float(F[2]),
            0.0,
            f"contact Fz should be upward but got F={F}",
        )
        rel_err = abs(float(F[2]) - expected_Fz) / expected_Fz
        self.assertLess(
            rel_err,
            0.05,
            f"static contact force off: measured Fz={float(F[2]):.3f} N, "
            f"expected m*g = {expected_Fz:.3f} N (rel_err={rel_err:.2%})",
        )
        # Lateral components must be negligible.
        lateral = math.hypot(float(F[0]), float(F[1]))
        self.assertLess(
            lateral,
            0.05 * expected_Fz,
            f"lateral contact force should be ~0 but got F_xy=({F[0]:.3f}, "
            f"{F[1]:.3f}) N",
        )

    def test_prismatic_drive_pushes_sphere(self):
        """Vertical prismatic PD drive: contact force == m*g + kp*dx.

        We sweep a range of ``(kp, kd)`` pairs. For each one:

        1. Build the scene with a drive target placed ``dx_target = 0.5``
           below the ground-contact equilibrium.
        2. Settle.
        3. Measure the actual slide ``x_eq`` (so ``dx = x_target - x_eq``
           includes any tracking error the PD leaves behind).
        4. Predicted drive force: ``F_drive = kp * dx``.
        5. Predicted contact force: ``m*g + F_drive``.
        6. Compare against ``gather_contact_pair_wrenches``.

        The per-kp tolerance is kept at 5 % because the PD residual at
        finite ``kp`` can smear the equilibrium position by sub-mm; we
        compensate by *measuring* ``x_eq`` rather than assuming it.
        """
        # Slide coordinate of the sphere-on-ground equilibrium
        # (``n_hat . (p2 - p1)`` with n_hat=-z and p2.z = SPHERE_RADIUS
        # → x_contact = -SPHERE_RADIUS). The drive target is placed
        # one half-metre further along ``+n_hat`` (further below the
        # ground) so the drive presses hard and the PD residual is
        # tiny compared to the commanded offset.
        x_contact = -self.SPHERE_RADIUS
        dx_target = 0.5  # [m] below the contact equilibrium
        x_target = x_contact + dx_target

        # (kp, kd) pairs. Cover two orders of magnitude of stiffness.
        # Critical damping for unit mass: ``kd = 2*sqrt(kp*m)``; we pick
        # damping somewhat above critical so the drop settles inside
        # SETTLE_FRAMES without ringing.
        m = self.MASS
        cases = []
        for kp in (50.0, 200.0, 1000.0, 5000.0):
            kd = 3.0 * math.sqrt(kp * m)  # ~1.5x critical
            cases.append((kp, kd))

        for kp, kd in cases:
            with self.subTest(kp=kp, kd=kd):
                scene = self._build_prismatic_pd_press(
                    kp=kp, kd=kd, x_target=x_target
                )
                self._settle(scene)

                x_eq = self._sphere_slide(scene)
                dx = x_target - x_eq  # positive -> drive pushes along +n_hat
                F_drive = kp * dx
                expected_Fz = m * _G + F_drive

                F, npairs, npoints = scene.gather_contact_wrench_on_body(
                    scene.sphere
                )
                self.assertEqual(
                    npairs,
                    1,
                    f"[kp={kp}] expected one sphere-ground contact pair, got "
                    f"{npairs}",
                )
                # A sphere pressed onto a plane is still a single-
                # point contact in the narrow phase. Anything > 1
                # would inflate ``F`` by the duplicate count.
                self.assertEqual(
                    npoints,
                    1,
                    f"[kp={kp}] sphere-plane should produce 1 contact point "
                    f"in the pair-summary; got {npoints}",
                )
                raw_count = scene.active_rigid_contact_count()
                self.assertEqual(
                    raw_count,
                    1,
                    f"[kp={kp}] narrow phase emitted {raw_count} contacts "
                    f"for a sphere-on-plane; expected exactly 1",
                )
                self.assertGreater(
                    float(F[2]),
                    0.0,
                    f"[kp={kp}] drive pushes down; contact Fz should be "
                    f"upward but got F={F}",
                )
                rel_err = abs(float(F[2]) - expected_Fz) / expected_Fz
                self.assertLess(
                    rel_err,
                    0.05,
                    f"[kp={kp}, kd={kd:.2f}] contact force off: measured "
                    f"Fz={float(F[2]):.3f} N, expected m*g + kp*dx = "
                    f"{expected_Fz:.3f} N "
                    f"(m*g={m * _G:.3f} N, kp*dx={F_drive:.3f} N, "
                    f"dx={dx:.4f} m, rel_err={rel_err:.2%})",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
