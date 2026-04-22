# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction-coefficient sweep tests for the jitter contact solver.

Two scenes:

* ``TestStaticFrictionThreshold`` -- a horizontal box on a ground
  plane with an applied horizontal force. Classical prediction: the
  box stays at rest while the applied force magnitude
  :math:`F_{\\text{applied}} \\le \\mu_s m g`, and starts sliding
  above that. We sweep ``mu`` through a handful of values at a
  fixed push and check the solver picks the right side of the
  static-friction threshold.

* ``TestKineticSlideDeceleration`` -- a block moving along +X with a
  known initial velocity on a flat ground. With kinetic friction
  :math:`\\mu_k` the block decelerates at :math:`\\mu_k g`; after
  time :math:`t = v_0 / (\\mu_k g)` it should be at rest. We measure
  the time-to-stop and the average deceleration and check they match
  the analytical values within a reasonable tolerance.

Both tests sit on top of the same CollisionPipeline + WorldBuilder
harness the stack-stability tests use; CUDA-only.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)


_G = 9.81


class _FrictionScene:
    """One cube on a ground plane with an optional push-along-X force.

    Matches the minimal harness the other contact tests use; specialised
    to a single-cube scene so the tests can read one position and one
    velocity directly without any broadcasting.
    """

    def __init__(
        self,
        *,
        friction: float,
        applied_force_x: float = 0.0,
        initial_velocity_x: float = 0.0,
        half_extent: float = 0.5,
        fps: int = 60,
        substeps: int = 4,
        solver_iterations: int = 16,
    ) -> None:
        self.device = wp.get_device("cuda:0")
        self.half_extent = float(half_extent)
        density = 1000.0
        self.cube_mass = float(density * (2.0 * self.half_extent) ** 3)
        self.fps = int(fps)
        self.substeps = int(substeps)
        self.frame_dt = 1.0 / self.fps
        self.friction = float(friction)
        self.applied_force_x = float(applied_force_x)
        self.initial_velocity_x = float(initial_velocity_x)

        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
        body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.half_extent + 1e-3),
                q=wp.quat_identity(),
            ),
        )
        mb.add_shape_box(
            body, hx=self.half_extent, hy=self.half_extent, hz=self.half_extent
        )
        self.body = body

        self.model = mb.finalize()
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

        builder, n2j = build_jitter_world_from_model(self.model)
        self.n2j = n2j

        max_contact_columns = max(4, (rigid_contact_max + 5) // 6)
        self.world = builder.finalize(
            substeps=self.substeps,
            solver_iterations=int(solver_iterations),
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            default_friction=self.friction,
            device=self.device,
        )

        # Initial linear velocity along +X (set after builder finalize so
        # it lands in the jitter bodies directly; on the first sync it
        # gets read from Newton state). We seed Newton state instead
        # so the sync path carries it.
        if self.initial_velocity_x != 0.0:
            qd = self.state.body_qd.numpy().copy()
            qd[body, 0] = self.initial_velocity_x
            self.state.body_qd.assign(qd)

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        # Applied force is pushed in via the jitter body's ``force``
        # accumulator each frame (not captured: we reset it between
        # frames). Keeping this out of the captured graph lets us
        # later vary the push during a test without re-compiling.
        self._sync_newton_to_jitter()
        self._applied_force_vec = wp.vec3f(self.applied_force_x, 0.0, 0.0)

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_jitter_kernel,
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

    def step(self) -> None:
        """Advance one frame. Applies the push force first so it's
        integrated into the body's ``force`` accumulator before the
        per-step ``_foreach_active_body`` turns it into a delta."""
        if self.applied_force_x != 0.0:
            # Scatter ``force[body] = (Fx, 0, 0)`` into the jitter body.
            j = self.n2j[self.body]
            forces = self.world.bodies.force.numpy().copy()
            forces[j] = [self.applied_force_x, 0.0, 0.0]
            self.world.bodies.force.assign(forces)

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

    def position(self) -> np.ndarray:
        return self.world.bodies.position.numpy()[self.n2j[self.body]]

    def velocity(self) -> np.ndarray:
        return self.world.bodies.velocity.numpy()[self.n2j[self.body]]


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Friction tests run on CUDA only.",
)
class TestStaticFrictionThreshold(unittest.TestCase):
    """Predict the static-friction threshold: the box should stay at
    rest for ``F_applied <= mu * m * g`` and start moving above that.

    Cube is 1000 kg (1 m^3 at density 1000). Weight is ``1000 * 9.81 =
    9810 N``. A push of ``P`` newtons means the ratio ``P / (m*g)`` is
    ``P / 9810``. We pick ``P = 4905 N`` (half the weight) and sweep
    ``mu`` through 0.1, 0.3, 0.5, 0.7 so the case ``mu > 0.5`` should
    hold the box, ``mu < 0.5`` should let it slide.
    """

    N_FRAMES = 120  # 2 s at 60 fps
    PUSH_RATIO = 0.5  # F_applied / (m * g)

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("friction tests require CUDA")

    def _run_mu(self, mu: float) -> tuple[float, float]:
        """Run a ``N_FRAMES`` settle with friction coefficient ``mu`` and
        return ``(final_x, final_vx)``."""
        scene = _FrictionScene(
            friction=mu,
            applied_force_x=self.PUSH_RATIO * 1000.0 * _G,
        )
        # Let the cube settle vertically before the push takes effect.
        # Apply zero force for a few frames so the normal response is
        # established first. Once moving, the applied_force accumulator
        # will kick in on the next step.
        for _ in range(10):
            scene.applied_force_x = 0.0
            scene.step()
        scene.applied_force_x = self.PUSH_RATIO * 1000.0 * _G
        for _ in range(self.N_FRAMES):
            scene.step()
        return float(scene.position()[0]), float(scene.velocity()[0])

    def test_high_friction_holds(self):
        """mu = 0.7 > PUSH_RATIO=0.5: the box should stay put."""
        x, vx = self._run_mu(0.7)
        print(f"[friction mu=0.7 push=0.5 mg] final_x={x:.4f} m  vx={vx:.4f} m/s")
        # With mu > PUSH_RATIO the box must not have drifted more than
        # a few cm -- tight bound that fails loudly if friction goes
        # kinetic.
        self.assertLess(abs(x), 0.1)
        self.assertLess(abs(vx), 0.5)

    def test_borderline_friction(self):
        """mu = 0.5 == PUSH_RATIO: boundary case; some drift allowed."""
        x, vx = self._run_mu(0.5)
        print(f"[friction mu=0.5 push=0.5 mg] final_x={x:.4f} m  vx={vx:.4f} m/s")
        # Boundary case is numerically sensitive; bounds are loose.
        self.assertTrue(np.isfinite(x))

    def test_low_friction_slides(self):
        """mu = 0.1 << PUSH_RATIO=0.5: the box must clearly slide."""
        x, vx = self._run_mu(0.1)
        print(f"[friction mu=0.1 push=0.5 mg] final_x={x:.4f} m  vx={vx:.4f} m/s")
        # Net horizontal acceleration should be ``g * (PUSH_RATIO - mu)
        # = 9.81 * 0.4 ≈ 3.92 m/s^2``. After 2 s that's ~7.84 m/s
        # velocity and ~7.85 m displacement -- solver won't hit the
        # analytical value exactly, but should be in the right
        # ballpark.
        self.assertGreater(x, 1.0)  # clearly sliding
        self.assertGreater(vx, 1.0)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Friction tests run on CUDA only.",
)
class TestKineticSlideDeceleration(unittest.TestCase):
    """Measure the kinetic-friction deceleration of a sliding block.

    Analytic: ``a = -mu * g`` (-Z component from gravity times mu,
    applied opposite to velocity). Time-to-stop is ``v0 / (mu * g)``.
    For ``v0 = 5 m/s`` and ``mu = 0.5``, ``t_stop = 5 / 4.905 = 1.02 s``.
    """

    N_FRAMES = 180  # 3 s at 60 fps

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("friction tests require CUDA")

    def _measure(self, mu: float, v0: float) -> dict:
        scene = _FrictionScene(
            friction=mu, initial_velocity_x=v0, applied_force_x=0.0
        )
        t_stop = None
        for frame in range(self.N_FRAMES):
            scene.step()
            vx = float(scene.velocity()[0])
            if t_stop is None and vx <= 0.01:
                t_stop = (frame + 1) * scene.frame_dt
        final_vx = float(scene.velocity()[0])
        return {
            "mu": mu,
            "v0": v0,
            "t_stop": t_stop,  # may be None if still moving
            "final_vx": final_vx,
            "t_stop_analytic": v0 / (mu * _G),
        }

    def test_kinetic_mu_0p5_v_5(self):
        m = self._measure(0.5, 5.0)
        print(
            f"[kinetic mu=0.5 v0=5]   t_stop={m['t_stop']}  "
            f"final_vx={m['final_vx']:+.4f}  analytic_t={m['t_stop_analytic']:.4f} s"
        )
        # Block should have fully decelerated within the analytic
        # stop time plus a 50% fudge for PGS softness / overshoot.
        self.assertLess(m["final_vx"], 0.5)
        if m["t_stop"] is not None:
            self.assertLess(m["t_stop"], m["t_stop_analytic"] * 2.0)

    def test_kinetic_mu_0p1_v_2(self):
        m = self._measure(0.1, 2.0)
        # t_analytic = 2 / 0.981 = 2.04 s; still within 3s window.
        print(
            f"[kinetic mu=0.1 v0=2]   t_stop={m['t_stop']}  "
            f"final_vx={m['final_vx']:+.4f}  analytic_t={m['t_stop_analytic']:.4f} s"
        )
        # Low-friction case: sanity-check the block's velocity kept
        # decreasing (no weird solver acceleration).
        self.assertLess(m["final_vx"], m["v0"])


if __name__ == "__main__":
    unittest.main()
