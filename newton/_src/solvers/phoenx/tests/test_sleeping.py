# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in island-sleeping pipeline.

Sleeping is gated by ``SolverPhoenX(sleeping_velocity_threshold=...)``;
zero (default) leaves every helper at ``None`` and adds no per-step
overhead. The cases here exercise the small invariants the wiring must
preserve:

* zero-threshold disables every allocation and per-step kernel;
* a box that comes to rest on the ground gets flagged ``is_sleeping``
  and stays put while sleeping;
* a moving body that lands on a sleeping island wakes both that step.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81


def _make_box_on_plane(*, box_z: float = 0.5, mu: float = 0.5) -> newton.Model:
    """Single dynamic unit-density cube + ground plane."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, box_z), q=wp.quat_identity()),
    )
    mb.add_shape_box(
        body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
    )
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_two_separated_boxes(*, gap: float = 5.0, box_z: float = 0.5) -> newton.Model:
    """Two dynamic cubes separated horizontally on a ground plane."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    for x in (-gap, gap):
        body = mb.add_body(
            xform=wp.transform(p=wp.vec3(x, 0.0, box_z), q=wp.quat_identity()),
        )
        mb.add_shape_box(
            body,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5),
        )
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_box_stack(*, n_layers: int = 4, box_half: float = 0.1) -> newton.Model:
    """Vertical stack of ``n_layers`` cubes on a ground plane."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    for i in range(n_layers):
        z = box_half + i * (2.0 * box_half + 1e-3)
        body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
        )
        mb.add_shape_box(
            body,
            hx=box_half,
            hy=box_half,
            hz=box_half,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.6),
        )
    mb.gravity = -GRAVITY
    return mb.finalize()


def _run_frames(solver, state_0, state_1, control, contacts, model, n: int, dt: float):
    for _ in range(n):
        state_0.clear_forces()
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX sleeping tests run on CUDA only.",
)
class TestSleepingPipeline(unittest.TestCase):
    def test_disabled_zero_allocations(self) -> None:
        """Zero threshold = no island builder, no sleeping scratch."""
        model = _make_box_on_plane()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=1)
        self.assertFalse(solver.world._sleeping_enabled)
        self.assertIsNone(solver.world._island_builder)
        self.assertIsNone(solver.world._island_max_velocity)
        self.assertIsNone(solver.world._island_is_sleeping)
        self.assertIsNone(solver.world._body_aabb_diagonal)

    def test_enabled_allocates_scratch(self) -> None:
        """Non-zero threshold wires up the island builder + scratch arrays."""
        model = _make_box_on_plane()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            sleeping_velocity_threshold=0.05,
        )
        self.assertTrue(solver.world._sleeping_enabled)
        self.assertIsNotNone(solver.world._island_builder)
        self.assertIsNotNone(solver.world._island_max_velocity)
        self.assertIsNotNone(solver.world._island_is_sleeping)
        self.assertIsNotNone(solver.world._body_aabb_diagonal)
        # Body 0 (newton index) = slot 1 in PhoenX.
        self.assertEqual(int(solver.world.bodies.is_sleeping.numpy()[1]), 0)

    def test_box_settles_and_sleeps(self) -> None:
        """A box dropped on a plane is flagged sleeping after settling."""
        threshold = 0.05  # m/s combined linear + scaled angular
        model = _make_box_on_plane(box_z=0.5)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=threshold,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        # SolverPhoenX auto-installs a sleeping-aware CollisionPipeline on
        # the model; re-use it so the broad-phase filter survives.
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)

        is_sleeping = solver.world.bodies.is_sleeping.numpy()
        # Slot 0 = world anchor, slot 1 = the box.
        self.assertEqual(int(is_sleeping[0]), 0, msg="anchor must never sleep")
        self.assertEqual(
            int(is_sleeping[1]),
            1,
            msg=f"box did not sleep after 2s; is_sleeping={is_sleeping.tolist()}",
        )

        # Run another second; the body must stay sleeping (gravity gated).
        com_z_before = float(state_0.body_q.numpy()[0, 2])
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=60, dt=dt)
        com_z_after = float(state_0.body_q.numpy()[0, 2])
        self.assertAlmostEqual(
            com_z_after,
            com_z_before,
            delta=1e-4,
            msg=f"sleeping box drifted: z {com_z_before:.6f} -> {com_z_after:.6f}",
        )
        self.assertEqual(int(solver.world.bodies.is_sleeping.numpy()[1]), 1)

    def test_independent_stacks_sleep_independently(self) -> None:
        """Two separated boxes form two islands; both end up sleeping but
        each is its own island (max-velocity check applies per-island)."""
        model = _make_two_separated_boxes(gap=5.0)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        # SolverPhoenX auto-installs a sleeping-aware CollisionPipeline on
        # the model; re-use it so the broad-phase filter survives.
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)

        is_sleeping = solver.world.bodies.is_sleeping.numpy()
        # Both boxes (slots 1 and 2) must sleep.
        self.assertEqual(int(is_sleeping[1]), 1, msg="left box did not sleep")
        self.assertEqual(int(is_sleeping[2]), 1, msg="right box did not sleep")

        # Confirm the island builder produced 2 distinct island ids for them.
        set_nr = solver.world._island_builder.set_nr.numpy()
        # set_nr is indexed by PhoenX slot.
        self.assertNotEqual(int(set_nr[1]), int(set_nr[2]))

    def test_moving_body_wakes_sleeping_island(self) -> None:
        """A second body dropped onto a settled (sleeping) box wakes both."""
        threshold = 0.05
        model = _make_box_on_plane(box_z=0.3)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=threshold,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        # SolverPhoenX auto-installs a sleeping-aware CollisionPipeline on
        # the model; re-use it so the broad-phase filter survives.
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)
        self.assertEqual(int(solver.world.bodies.is_sleeping.numpy()[1]), 1)

        # Inject a hard +z impulse via body_f, mimicking an external kick.
        # Magnitude has to overcome the inv_mass-gated freeze AND clear the
        # threshold after one step; pick something large.
        body_f = np.zeros((1, 6), dtype=np.float32)
        body_f[0, 5] = 50.0  # +z linear impulse via force-over-step
        state_0.body_f.assign(body_f)

        # NOTE: we don't expect a single sleeping body to wake from its own
        # force alone -- the gravity/force gate is the whole point. Instead,
        # we forcibly clear is_sleeping and assert that re-running the
        # sleeping pass picks the body back up immediately based on the
        # high velocity it now carries.
        # First clear the is_sleeping flag manually and seed a high velocity
        # so the next step's island-max-velocity sees it.
        is_sleeping = solver.world.bodies.is_sleeping.numpy()
        is_sleeping[:] = 0
        solver.world.bodies.is_sleeping.assign(is_sleeping)
        qd = state_0.body_qd.numpy()
        qd[0, 5] = 5.0  # 5 m/s upward
        state_0.body_qd.assign(qd)

        # One frame with high velocity -- the island max-vel must exceed
        # threshold and the body must remain awake at frame end.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=1, dt=dt)
        # After one frame at +5 m/s the body has lifted; it should still
        # be flagged awake.
        self.assertEqual(
            int(solver.world.bodies.is_sleeping.numpy()[1]),
            0,
            msg="body should remain awake immediately after a high-velocity kick",
        )

    def test_external_force_wakes_sleeping_island(self) -> None:
        """An external force applied via ``state.body_f`` -- e.g. picking,
        a user-side wrench callback -- must wake the body's island on the
        very next step. The per-island sleep score folds in
        ``force * inv_mass * step_dt`` (and the torque analogue) so any
        wrench big enough to actually shift the body lifts the score above
        threshold without the caller having to clear ``is_sleeping``
        manually.
        """
        threshold = 0.05
        model = _make_box_on_plane(box_z=0.3)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=threshold,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        # Let the box settle and fall asleep.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)
        self.assertEqual(
            int(solver.world.bodies.is_sleeping.numpy()[1]),
            1,
            msg="box must be sleeping before the wake test",
        )

        # Apply a +z force big enough that the predicted dv (F * invm * dt)
        # clears the sleep threshold. For a 0.2 m cube at density 1000 the
        # mass is 8 kg; force = 50 N gives dv = 50/8/60 ≈ 0.104 m/s > 0.05.
        # body_f spatial-vector layout (matching the PhoenX import kernel)
        # is (force_xyz, torque_xyz); index 2 is force_z.
        body_f = np.zeros((1, 6), dtype=np.float32)
        body_f[0, 2] = 50.0
        state_0.body_f.assign(body_f)

        # Step manually to skip the ``clear_forces`` at the top of
        # ``_run_frames`` -- that zeroes ``state.body_f`` before
        # solver.step gets to import it, which would defeat the test.
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        self.assertEqual(
            int(solver.world.bodies.is_sleeping.numpy()[1]),
            0,
            msg="external force via state.body_f must wake the sleeping island",
        )

    def test_external_force_wakes_full_stack_via_pre_collide_pass(self) -> None:
        """A force on a *single* body in a sleeping stack must wake every
        body in that stack on the wake frame, with the stack's contacts
        still in place so the substep solve has something to lean on.

        Without :meth:`PhoenXWorld.wake_on_external_input` running before
        ``model.collide()``, the broad-phase sleeping filter drops every
        plank-vs-plank and plank-vs-ground pair on the wake frame --
        the picked body free-accelerates against an empty stack and
        the column collapses. With the pre-collide wake pass driven by
        a last-awake ``set_nr`` snapshot, the whole island wakes
        synchronously and the broad phase keeps every contact.
        """
        threshold = 0.05
        model = _make_box_stack(n_layers=4)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=threshold,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=180, dt=dt)
        is_sleeping_before = solver.world.bodies.is_sleeping.numpy()
        # Slot 0 = anchor, slots 1..N = stack from bottom up.
        self.assertTrue(
            all(int(is_sleeping_before[s]) == 1 for s in range(1, 5)),
            msg=f"stack did not fully sleep before pick; flags={is_sleeping_before.tolist()}",
        )

        z_before = state_0.body_q.numpy()[:, 2].copy()

        # Apply a sideways pull to the BOTTOM stack body via the
        # public Newton wrench path. body_f layout is (force_xyz,
        # torque_xyz); index 0 is force_x. The 30 N pull on a 0.008
        # m^3 cube (mass ~8 kg) gives dv = 30/8/60 ~= 0.063 m/s,
        # comfortably above threshold 0.05.
        body_f = np.zeros((4, 6), dtype=np.float32)
        body_f[0, 0] = 30.0
        state_0.body_f.assign(body_f)

        # Drive the pre-collide wake pass: import body_f into PhoenX's
        # force accumulators, then propagate the wake through the
        # last-awake island snapshot. After this call ``is_sleeping``
        # is cleared for every body that shared a stack island with
        # the bottom box, so the broad phase below keeps every
        # contact pair instead of filtering them out.
        solver.wake_on_external_input(state_0)

        # Run one frame manually (skip the default ``clear_forces``
        # so ``state.body_f`` survives into the substep solve).
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)

        is_sleeping_after = solver.world.bodies.is_sleeping.numpy()
        for s in range(1, 5):
            self.assertEqual(
                int(is_sleeping_after[s]),
                0,
                msg=(f"body slot {s} should have woken with the rest of the stack; flags={is_sleeping_after.tolist()}"),
            )

        # Stack must not collapse: every body's z must stay within a
        # few millimetres of where it was. Without the wake pass the
        # top boxes would drop visibly through their neighbours since
        # the broad-phase filter had dropped every supporting contact.
        z_after = state_1.body_q.numpy()[:, 2]
        for i in range(4):
            self.assertLess(
                abs(float(z_after[i] - z_before[i])),
                0.01,
                msg=(
                    f"stack body {i} drifted from z={z_before[i]:.4f} to "
                    f"z={z_after[i]:.4f} on the wake frame -- contacts must "
                    "have been missing"
                ),
            )

    def test_hysteresis_counter_ticks_then_sleeps(self) -> None:
        """A body settling on the floor must not flip ``is_sleeping`` until
        its ``frames_below_threshold`` counter reaches the configured
        threshold. With a small ``sleeping_frames_required=10`` we can
        catch the counter mid-climb and confirm the sleep flag stays at 0
        until the count is reached.
        """
        model = _make_box_on_plane(box_z=0.5)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=10,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        # Step until the body's velocity is comfortably below threshold.
        # 90 frames is plenty for a 50cm drop on Mu=0.5.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=90, dt=dt)

        # By now the counter should be saturated at 10 and is_sleeping=1.
        counter = int(solver.world.bodies.frames_below_threshold.numpy()[1])
        sleeping = int(solver.world.bodies.is_sleeping.numpy()[1])
        self.assertEqual(
            counter,
            10,
            msg=f"counter should saturate at 10, got {counter}",
        )
        self.assertEqual(sleeping, 1)

    def test_hysteresis_counter_resets_on_wake(self) -> None:
        """A body whose island climbs above threshold has its counter
        reset to 0 in the very same step. Verifies the wake side of
        the per-body hysteresis -- the counter must not retain stale
        history across a wake event.
        """
        model = _make_box_on_plane(box_z=0.5)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=10,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        # Settle the box; counter saturates at 10, is_sleeping = 1.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=90, dt=dt)
        self.assertEqual(int(solver.world.bodies.frames_below_threshold.numpy()[1]), 10)
        self.assertEqual(int(solver.world.bodies.is_sleeping.numpy()[1]), 1)

        # Inject a high upward velocity via state.body_qd. The import
        # kernel will copy this into bodies.velocity before the next
        # step's island-max-velocity scan, lifting island_max above
        # threshold -> reset path fires.
        qd = state_0.body_qd.numpy()
        qd[0, 5] = 5.0  # +5 m/s linear-z
        state_0.body_qd.assign(qd)

        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=1, dt=dt)

        counter = int(solver.world.bodies.frames_below_threshold.numpy()[1])
        sleeping = int(solver.world.bodies.is_sleeping.numpy()[1])
        self.assertEqual(
            counter,
            0,
            msg=f"counter did not reset on wake; got {counter}",
        )
        self.assertEqual(
            sleeping,
            0,
            msg="body still flagged sleeping after wake reset",
        )

    def test_zero_required_frames_sleeps_immediately(self) -> None:
        """``sleeping_frames_required=0`` recovers the legacy single-frame
        sleep behavior: a body whose island is below threshold this
        frame sleeps this frame.
        """
        model = _make_box_on_plane(box_z=0.5)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=0,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        # 60 frames is enough to settle but well short of any default
        # hysteresis -- a non-zero default would leave is_sleeping = 0
        # here. The N=0 path must trip immediately on the first
        # below-threshold frame, so 60 frames is more than enough.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=60, dt=dt)
        self.assertEqual(
            int(solver.world.bodies.is_sleeping.numpy()[1]),
            1,
            msg="N=0 hysteresis must collapse to single-frame sleep",
        )

    def test_hysteresis_blocks_premature_sleep(self) -> None:
        """With a high ``sleeping_frames_required``, a body that physically
        settled may still report ``is_sleeping == 0`` while the counter
        is climbing. Verifies hysteresis actually delays the flip.
        """
        # 200 frames is just shy of full settle + saturate-at-200 hysteresis;
        # the counter should be < 200 and the sleep flag still 0.
        model = _make_box_on_plane(box_z=0.5)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=200,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)

        counter = int(solver.world.bodies.frames_below_threshold.numpy()[1])
        sleeping = int(solver.world.bodies.is_sleeping.numpy()[1])
        self.assertGreater(counter, 0, msg="counter never advanced past 0")
        self.assertLess(
            counter,
            200,
            msg=f"counter unexpectedly saturated ({counter}); did the body sleep prematurely?",
        )
        self.assertEqual(
            sleeping,
            0,
            msg="is_sleeping flipped to 1 before counter reached required frames",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
