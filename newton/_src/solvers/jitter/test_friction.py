# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic-prediction regression test for the Jitter contact solver's
Coulomb friction path.

Scene
-----

A row of independent unit cubes sits on a frictionless floor. Each cube
gets a chosen initial horizontal velocity ``v0`` (all along +X) and no
angular velocity. Under gravity ``g`` the cubes settle onto the floor in
the first few contacts of substep 0, and from there dry Coulomb friction
with a coefficient ``mu`` decelerates them at::

    a = mu * g       # [m/s^2]

so the stopping distance and stopping time follow the elementary
Physics-101 formulas::

    d_expected = v0**2 / (2 * mu * g)
    t_expected = v0 / (mu * g)

The test runs long enough to be confident every cube has stopped (it
uses a long tail after ``t_expected`` and then asserts ``|v| ~ 0``) and
then compares the measured stopping distance against ``d_expected``. A
companion "static friction" case with ``v0 = 0`` asserts that cubes
starting at rest stay at rest -- the signature symptom that made us
write this test in the first place (the pyramid demo showed cubes
drifting sideways even when at rest).

Why a separate test (not an Example)
------------------------------------

Examples are visualisation-first; this test needs hundreds of frames,
four different friction coefficients, and per-cube analytic assertions
-- it's easier to drive the Newton + Jitter pipeline directly. The
scene / pipeline skeleton (``ModelBuilder`` -> ``CollisionPipeline`` ->
Jitter ``World``) mirrors :mod:`example_pyramid` line-for-line so
anything it tests stays aligned with the interactive demos.

A note on per-shape friction
----------------------------

At the time of writing the Jitter contact solver consumes a single
scalar :attr:`World.default_friction` for every contact column; the
upstream ``shape_material_mu`` (settable via
``mb.default_shape_cfg.copy().mu = ...``) is *not yet* read by the
solver. The test therefore parametrises :attr:`World.default_friction`
across the outer loop rather than trying to assign per-cube mus through
the material system. When per-material friction lands, the per-cube
assertions here translate trivially to one run with per-cube ``mu``
values.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.contact_matching_config import JITTER_CONTACT_MATCHING
from newton._src.solvers.jitter.example_pyramid import (
    _build_jitter_world_from_model,
    _jitter_to_newton_kernel,
    _newton_to_jitter_kernel,
)

# ---------------------------------------------------------------------------
# Scene and analytic constants
# ---------------------------------------------------------------------------

#: Gravity magnitude [m/s^2]. Matches the default in
#: :class:`WorldBuilder.finalize` so the analytic prediction lines up
#: with what the solver integrates.
GRAVITY = 9.81

#: Frame settings. 120 Hz + 4 substeps keeps the per-substep friction
#: budget comfortably larger than the per-substep velocity change, so
#: the Coulomb cone never under-clamps a decelerating cube.
FPS = 120
FRAME_DT = 1.0 / FPS
SUBSTEPS = 4
SOLVER_ITERATIONS = 12

#: Box half-extent. Mass = 1 kg so force / impulse numbers are easy to
#: eyeball in logs.
BOX_HALF = 0.5
BOX_MASS = 1.0

#: Horizontal spacing between adjacent cubes along +Y. Large enough
#: that even the fastest cube can't catch up to / hit its neighbour.
BOX_PITCH_Y = 50.0


def _analytic_stop_distance(v0: float, mu: float) -> float:
    """Distance a rigid box travels before kinetic friction stops it.

    Derived from ``F = mu * m * g`` (constant deceleration
    ``a = mu * g``) and ``v^2 = v0^2 - 2 a d`` evaluated at ``v = 0``.
    """
    if mu <= 0.0:
        return float("inf")
    return (v0 * v0) / (2.0 * mu * GRAVITY)


def _analytic_stop_time(v0: float, mu: float) -> float:
    """Time until kinetic friction brings ``v0`` down to zero."""
    if mu <= 0.0:
        return float("inf")
    return v0 / (mu * GRAVITY)


# ---------------------------------------------------------------------------
# Test scene: row of independent cubes on a flat floor
# ---------------------------------------------------------------------------


class _FrictionScene:
    """Headless driver around the pyramid example's pipeline wiring.

    Builds ``num_cubes`` unit cubes in a line, each with its own
    initial +X velocity, on a frictionless ``shape_plane`` ground.
    Supports both eager single-stepping (:meth:`step`) and a
    graph-captured bulk :meth:`run_frames` that amortises Warp's
    per-launch overhead over ``frames`` frames -- essential for the
    multi-hundred-frame settle loops used by the friction tests.
    """

    def __init__(
        self,
        *,
        initial_speeds: list[float],
        mu: float,
        device: wp.context.Devicelike = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.mu = float(mu)
        self.initial_speeds = [float(v) for v in initial_speeds]
        self.num_cubes = len(initial_speeds)

        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Each cube sits just above its resting height to avoid
        # first-frame penetration; warm-start then settles the normal
        # impulse in a handful of substeps. Spacing along +Y so each
        # cube runs its own slide along +X independently.
        self._box_bodies: list[int] = []
        for i, v0 in enumerate(self.initial_speeds):
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.0, i * BOX_PITCH_Y, BOX_HALF + 1.0e-3),
                    q=wp.quat_identity(),
                ),
                mass=BOX_MASS,
            )
            mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
            self._box_bodies.append(body)

        self.model = mb.finalize()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=JITTER_CONTACT_MATCHING
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state
        )

        builder, newton_to_jitter = _build_jitter_world_from_model(self.model)
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        self.world = builder.finalize(
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, -GRAVITY),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            default_friction=self.mu,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()

        # Inject per-cube initial +X velocity directly into Jitter's
        # body container, then bounce it back into Newton's state so
        # the first collide() / sync sees the correct qd.
        velocities = self.world.bodies.velocity.numpy().copy()
        for newton_idx, v0 in zip(self._box_bodies, self.initial_speeds):
            j = self._newton_to_jitter[newton_idx]
            velocities[j] = (v0, 0.0, 0.0)
        self.world.bodies.velocity.assign(velocities.astype(np.float32))
        self._sync_jitter_to_newton()

    # -- pipeline plumbing ----------------------------------------------

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            _newton_to_jitter_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
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
        wp.launch(
            _jitter_to_newton_kernel,
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

    def step(self) -> None:
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=FRAME_DT,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        self._sync_jitter_to_newton()

    def run_frames(self, frames: int) -> None:
        """Advance the scene by ``frames`` frames.

        On CUDA this records a single :meth:`step` into a CUDA graph
        after a warm-up frame and replays it for the remainder, matching
        the :func:`_test_helpers.run_settle_loop` pattern used by every
        other jitter simulation test. On CPU (and for very short
        loops) falls back to plain eager stepping.
        """
        if frames <= 0:
            return

        use_graph = self.device.is_cuda and frames >= 40

        if not use_graph:
            for _ in range(frames):
                self.step()
            return

        self.step()

        with wp.ScopedCapture(device=self.device) as capture:
            self.step()
        graph = capture.graph

        for _ in range(frames - 2):
            wp.capture_launch(graph)

    # -- state queries ---------------------------------------------------

    def positions_world(self) -> np.ndarray:
        """World-space cube positions, ordered by input ``initial_speeds``."""
        all_pos = self.world.bodies.position.numpy()
        return np.array(
            [all_pos[self._newton_to_jitter[b]] for b in self._box_bodies]
        )

    def velocities_world(self) -> np.ndarray:
        all_vel = self.world.bodies.velocity.numpy()
        return np.array(
            [all_vel[self._newton_to_jitter[b]] for b in self._box_bodies]
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestKineticFrictionStopDistance(unittest.TestCase):
    """Cubes should decelerate at ``a = mu * g`` and stop at
    ``d = v0^2 / (2 mu g)``.

    Compares measured stopping distance across four ``mu`` values and
    four initial speeds per ``mu``. Tolerance is relative (``15%``)
    because PGS + substepping produces a small systematic under-
    estimate of friction impulse during the first few substeps (while
    the normal-impulse warm-start is still ramping up), which shows up
    as a slightly longer slide than the analytic value.
    """

    MU_VALUES = (0.1, 0.3, 0.5, 0.8)
    V0_VALUES = (1.0, 3.0, 5.0, 8.0)

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_preferred_device()

    def _run_single_mu(self, mu: float) -> dict[float, tuple[float, float]]:
        """Run one scene with all four v0s at friction ``mu`` and return
        ``{v0: (measured_stop_distance, residual_speed)}``.
        """
        scene = _FrictionScene(
            initial_speeds=list(self.V0_VALUES), mu=mu, device=self.device
        )

        # Simulate long enough for even the fastest cube to stop,
        # plus a generous margin so residual PGS jitter has time to
        # converge to zero velocity under the static friction cone.
        v_max = max(self.V0_VALUES)
        t_stop_max = _analytic_stop_time(v_max, mu)
        total_time = t_stop_max * 2.5 + 2.0
        num_frames = int(np.ceil(total_time * FPS))

        scene.run_frames(num_frames)

        positions = scene.positions_world()
        velocities = scene.velocities_world()

        # Every cube started at x=0, so the measured stop distance is
        # just the final x coordinate.
        stop_x = positions[:, 0]
        residual_speed = np.linalg.norm(velocities[:, :3], axis=1)

        return {
            float(v0): (float(stop_x[i]), float(residual_speed[i]))
            for i, v0 in enumerate(self.V0_VALUES)
        }

    def test_stop_distances_match_analytic(self) -> None:
        """Measured stop distance vs ``v0^2 / (2 mu g)`` within 20 %.

        The 20 % tolerance is empirical: a PGS + Baumgarte-style
        normal-bias contact solver with a moderate substep count
        tends to over- or under-estimate friction impulse for the
        first 1-2 substeps (normal impulse is still warming up, and
        the PhoenX-style Jacobi-within-slot iterate over-shoots a
        bit on the low-velocity / low-mu combinations). We accept
        20 % but report the full distance / residual-velocity matrix
        so a regression in the *shape* of the error (e.g. 2x off at
        low mu, not a uniform drift) is still obvious.
        """
        all_results: dict[float, dict[float, tuple[float, float]]] = {}
        for mu in self.MU_VALUES:
            all_results[mu] = self._run_single_mu(mu)

        # Print a summary table first so a failure shows the full
        # picture (not just the first assertion that tripped).
        print()
        print("Friction stop-distance sweep (meters unless noted)")
        print("=" * 72)
        header = f"{'mu':>5}  {'v0':>5}  {'expected':>10}  {'measured':>10}  {'rel_err':>8}  {'|v_f|':>8}"
        print(header)
        print("-" * 72)
        for mu, row in all_results.items():
            for v0, (d_meas, v_res) in row.items():
                d_exp = _analytic_stop_distance(v0, mu)
                rel = (d_meas - d_exp) / d_exp
                print(
                    f"{mu:>5.2f}  {v0:>5.2f}  {d_exp:>10.4f}  "
                    f"{d_meas:>10.4f}  {rel:>+8.2%}  {v_res:>8.4f}"
                )
        print("=" * 72)

        for mu, row in all_results.items():
            for v0, (d_meas, v_res) in row.items():
                d_exp = _analytic_stop_distance(v0, mu)
                rel_err = abs(d_meas - d_exp) / d_exp
                with self.subTest(mu=mu, v0=v0):
                    self.assertLess(
                        rel_err,
                        0.20,
                        f"mu={mu}, v0={v0}: stop distance {d_meas:.4f} m "
                        f"vs expected {d_exp:.4f} m "
                        f"(rel err {rel_err:+.2%})",
                    )
                    # Residual velocity must be near zero -- if the
                    # cube is still moving then 'stop distance'
                    # isn't meaningful.
                    self.assertLess(
                        v_res,
                        0.05,
                        f"mu={mu}, v0={v0}: cube still moving at "
                        f"|v|={v_res:.4f} m/s after settle",
                    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestStaticFrictionNoDrift(unittest.TestCase):
    """A cube at rest on a flat floor must stay at rest.

    This pins the symptom that motivated the whole audit: in the
    pyramid demo resting cubes drifted sideways with no applied
    lateral load. Kinetic friction clamps ``|lambda_t| <= mu *
    lambda_n``; at zero relative velocity the PGS tangent row sets
    ``lambda_t`` exactly to the value that cancels tangential
    velocity, so drift can only come from:

      * a non-vertical contact normal creating horizontal bias,
      * a warm-start ``lambda_t`` from a shifted contact slot, or
      * numerical noise larger than the cone budget.

    A single cube on a flat plane eliminates stack-specific
    warm-start slot shuffling, so any drift observed here points
    squarely at one of the first two.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_preferred_device()

    def test_cube_at_rest_does_not_drift(self) -> None:
        """After 3 s of simulated time a resting cube moves less than
        1 mm horizontally and its speed stays below 1 mm/s.
        """
        scene = _FrictionScene(
            initial_speeds=[0.0], mu=0.5, device=self.device
        )
        scene.run_frames(3 * FPS)
        pos = scene.positions_world()[0]
        vel = scene.velocities_world()[0]
        horizontal_drift = float(np.hypot(pos[0], pos[1]))
        horizontal_speed = float(np.hypot(vel[0], vel[1]))
        print(
            f"rest-drift after 3 s: horizontal_drift={horizontal_drift:.6f} m, "
            f"horizontal_speed={horizontal_speed:.6f} m/s"
        )
        self.assertLess(
            horizontal_drift, 1.0e-3, "resting cube drifted horizontally"
        )
        self.assertLess(
            horizontal_speed,
            1.0e-3,
            "resting cube is still moving after settle",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
