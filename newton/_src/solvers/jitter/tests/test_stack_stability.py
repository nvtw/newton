# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Automated stack-stability + side-slip diagnostics for the jitter solver.

Builds a simple vertical column of cubes (heights 2, 5, 10, 20),
settles it on an infinite ground plane under gravity, and records:

* **Side-slip** -- lateral XY drift of each cube's COM after settle,
  measured both peak-over-time and final-state. Ideal is ~0 m; a
  friction model that leaks tangential momentum will show up as a
  monotonic drift.
* **Stack toppling** -- max drop in the top cube's Z coordinate.
  A collapsing stack drops fast, a stable stack stays at the
  settled height (``row_index * 2 * half_extent + half_extent``).
* **Linear momentum drift** -- net horizontal momentum of the stack
  across settled frames. With vertical gravity and no external
  impulses in XY, this should stay ~0. Any systematic leak shows
  up as lateral velocity accumulation and points at a momentum
  non-conserving friction or contact formulation.

The tests don't hard-fail on tight numeric thresholds yet -- the
goal at this stage is producing reproducible numbers so we can
compare before/after when we try friction-formulation
experiments. Each test prints a one-line diagnostic summary
(``[stack N=10 slip=... top_drop=... momentum=...]``).

Runs on CUDA only because the full collide+step pipeline is
graph-captured; CPU would take ~100x longer.
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
from newton._src.solvers.jitter.world_builder import WorldBuilder


_G = 9.81


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _StackScene:
    """Reusable jitter+Newton harness for a cube column on a ground plane.

    Matches the pattern in :mod:`test_contact_force_accuracy` but
    specialised to stacks so the tests can focus on the metrics.
    CUDA-graph captures the inner per-frame pipeline
    (``collide -> step``) so a full settle loop runs in milliseconds
    even at ``solver_iterations = 16`` and ``substeps = 4``.
    """

    def __init__(
        self,
        *,
        num_cubes: int,
        half_extent: float = 0.5,
        # Rely on Newton's default shape density (1000 kg/m^3) so the
        # cube mass scales with volume -- matches :mod:`example_pyramid`
        # and keeps the current bias-factor calibration in range. A
        # unit-mass variant would need a stiffer normal bias to stay
        # equilibrated (see the diagnostic script in
        # ``debug_stack2.py`` for the derivation); kept out of the
        # default path here so the baseline doesn't require tuning
        # knobs the user hasn't asked for.
        gap: float = 5.0e-3,
        friction: float = 0.5,
        fps: int = 60,
        substeps: int = 4,
        solver_iterations: int = 16,
        seed_jitter: float = 0.0,
    ) -> None:
        self.device = wp.get_device("cuda:0")
        self.num_cubes = int(num_cubes)
        self.half_extent = float(half_extent)
        # Density-derived mass keeps parity with the pyramid example;
        # derived here so the momentum metric can scale by it.
        density = 1000.0
        volume = (2.0 * self.half_extent) ** 3
        self.cube_mass = float(density * volume)
        self.fps = int(fps)
        self.substeps = int(substeps)
        self.frame_dt = 1.0 / self.fps
        self.friction = float(friction)

        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Seed jitter lets us kick the initial stack off-axis by a
        # tiny random amount so the solver can't fall into perfectly
        # symmetric numerics. 0 by default; use >0 to stress-test.
        rng = np.random.default_rng(0)
        newton_body_ids: list[int] = []
        expected_masses: dict[int, float] = {}
        for row in range(self.num_cubes):
            z = self.half_extent + row * (2.0 * self.half_extent + gap)
            offset = (
                rng.normal(scale=seed_jitter, size=2)
                if seed_jitter > 0
                else (0.0, 0.0)
            )
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(offset[0]), float(offset[1]), z),
                    q=wp.quat_identity(),
                ),
            )
            # Default density (1000 kg/m^3) so the effective-mass
            # scaling matches the pyramid example and the baseline
            # stays in the calibrated regime.
            mb.add_shape_box(
                body,
                hx=self.half_extent,
                hy=self.half_extent,
                hz=self.half_extent,
            )
            newton_body_ids.append(body)
            expected_masses[body] = self.cube_mass
        self.newton_body_ids = newton_body_ids

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

        builder, newton_to_jitter = build_jitter_world_from_model(
            self.model, expected_masses=expected_masses
        )
        self.newton_to_jitter = newton_to_jitter

        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        num_shapes = int(self.model.shape_count)
        self.world = builder.finalize(
            substeps=self.substeps,
            solver_iterations=int(solver_iterations),
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            default_friction=self.friction,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()
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
        wp.capture_launch(self._graph)

    def body_positions(self) -> np.ndarray:
        """Per-stack-cube world positions, shape ``(num_cubes, 3)``."""
        pos = self.world.bodies.position.numpy()
        out = np.zeros((self.num_cubes, 3), dtype=np.float32)
        for i, nb in enumerate(self.newton_body_ids):
            out[i] = pos[self.newton_to_jitter[nb]]
        return out

    def body_velocities(self) -> np.ndarray:
        """Per-stack-cube linear velocities, shape ``(num_cubes, 3)``."""
        vel = self.world.bodies.velocity.numpy()
        out = np.zeros((self.num_cubes, 3), dtype=np.float32)
        for i, nb in enumerate(self.newton_body_ids):
            out[i] = vel[self.newton_to_jitter[nb]]
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _run_stack_and_measure(
    num_cubes: int, *, n_frames: int = 240, seed_jitter: float = 0.0
) -> dict:
    """Settle a ``num_cubes``-tall stack and return diagnostic metrics.

    Metrics:

    * ``final_slip_max_xy`` -- max |XY| drift of any cube's COM from
      its spawn position after ``n_frames``. Measures absolute
      side-slip of the worst offender.
    * ``final_slip_mean_xy`` -- mean across the stack, same quantity.
    * ``top_drop`` -- ``initial_top_z - final_top_z``. Positive means
      the top cube sank (stack compressed or collapsed). Large
      positive = catastrophic topple.
    * ``peak_horizontal_momentum`` -- max over settled frames
      (``frame >= n_frames // 2``) of ``|sum(mass_i * v_i_xy)|``.
      With vertical gravity and no external XY impulses, this must
      be ~0 for a momentum-conserving friction solve.
    * ``rms_horizontal_velocity`` -- RMS of horizontal speed across
      all cubes in the last frame. A stable stack should settle to
      <~ 1 mm/s.
    """
    scene = _StackScene(num_cubes=num_cubes, seed_jitter=seed_jitter)

    initial_positions = scene.body_positions()
    initial_top_z = float(initial_positions[-1, 2])

    peak_horizontal_momentum = 0.0
    for frame in range(n_frames):
        scene.step()
        # Sample horizontal momentum during the settled tail.
        if frame >= n_frames // 2:
            vels = scene.body_velocities()
            p_xy = np.sum(vels[:, :2] * scene.cube_mass, axis=0)
            mag = float(np.linalg.norm(p_xy))
            if mag > peak_horizontal_momentum:
                peak_horizontal_momentum = mag

    final_positions = scene.body_positions()
    final_velocities = scene.body_velocities()

    xy_drift = np.linalg.norm(
        (final_positions[:, :2] - initial_positions[:, :2]), axis=1
    )
    final_top_z = float(final_positions[-1, 2])
    horiz_speed = np.linalg.norm(final_velocities[:, :2], axis=1)

    return {
        "num_cubes": num_cubes,
        "final_slip_max_xy": float(np.max(xy_drift)),
        "final_slip_mean_xy": float(np.mean(xy_drift)),
        "top_drop": float(initial_top_z - final_top_z),
        "peak_horizontal_momentum": peak_horizontal_momentum,
        "rms_horizontal_velocity": float(np.sqrt(np.mean(horiz_speed**2))),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter stack tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestStackStability(unittest.TestCase):
    """Record diagnostic numbers on 2, 5, 10, 20-cube stacks.

    Asserts are intentionally loose: the goal is reproducible
    measurements we can diff against experimental improvements.
    Tighten thresholds after a friction-formulation change lands.
    """

    STACK_HEIGHTS = (2, 5, 10, 20)
    # 4 seconds at 60 fps -- long enough for a 20-cube tower to
    # either settle or visibly slip, short enough to stay fast.
    N_FRAMES = 240

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "stack stability tests require CUDA (graph capture)"
            )

    def _run_and_print(self, num_cubes: int) -> dict:
        m = _run_stack_and_measure(num_cubes, n_frames=self.N_FRAMES)
        print(
            f"[stack N={m['num_cubes']:>2d}] "
            f"slip_max={m['final_slip_max_xy']:.4f} m  "
            f"slip_mean={m['final_slip_mean_xy']:.4f} m  "
            f"top_drop={m['top_drop']:+.4f} m  "
            f"p_xy_peak={m['peak_horizontal_momentum']:.4f}  "
            f"v_rms={m['rms_horizontal_velocity']:.4f} m/s"
        )
        return m

    def test_stack_2(self):
        m = self._run_and_print(2)
        # Very loose sanity bounds -- anything worse is a huge
        # regression and almost certainly a solver bug rather than a
        # friction-tuning issue.
        self.assertLess(m["final_slip_max_xy"], 1.0)
        self.assertLess(m["top_drop"], 0.5)

    def test_stack_5(self):
        m = self._run_and_print(5)
        self.assertLess(m["final_slip_max_xy"], 1.0)
        self.assertLess(m["top_drop"], 0.5)

    def test_stack_10(self):
        m = self._run_and_print(10)
        self.assertLess(m["final_slip_max_xy"], 2.0)
        self.assertLess(m["top_drop"], 1.5)

    def test_stack_20(self):
        m = self._run_and_print(20)
        # 20-cube towers frequently topple today; this test logs
        # how bad it is but doesn't enforce an upper bound yet.
        # Tighten once we've landed the friction-formulation fix.
        self.assertTrue(np.isfinite(m["final_slip_max_xy"]))
        self.assertTrue(np.isfinite(m["top_drop"]))


if __name__ == "__main__":
    unittest.main()
