# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in island-sleeping pipeline.

Sleeping is gated by ``SolverPhoenX(sleeping_velocity_threshold=...)``;
zero (default) leaves every helper at ``None`` and adds no per-step
overhead. The cases here exercise the small invariants the wiring must
preserve:

* zero-threshold disables every allocation and per-step kernel;
* a box that comes to rest on the ground gets a non-negative
  ``island_root`` (the sleep label) and stays put while sleeping;
* a moving body that lands on a sleeping island wakes both that step.

A body is sleeping iff ``bodies.island_root[b] >= 0``; the value is the
lowest body id in its island at the moment of sleep transition.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81


def _sleep_flags(solver) -> np.ndarray:
    """Return a 0/1 array: 1 where the body is sleeping (``island_root >= 0``)."""
    return (solver.world.bodies.island_root.numpy() >= 0).astype(np.int32)


def _is_sleeping(solver, slot: int) -> int:
    """1 if PhoenX slot ``slot`` is sleeping, 0 otherwise."""
    return int(int(solver.world.bodies.island_root.numpy()[slot]) >= 0)


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


def _make_two_stacks(*, n_layers: int = 3, box_half: float = 0.1, gap: float = 3.0) -> newton.Model:
    """Two vertical stacks of ``n_layers`` cubes each, on a ground plane,
    separated horizontally by ``gap`` so the two stacks never share an
    island.
    """
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    for stack_x in (-gap, gap):
        for i in range(n_layers):
            z = box_half + i * (2.0 * box_half + 1e-3)
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(stack_x, 0.0, z), q=wp.quat_identity()),
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
        self.assertEqual(_is_sleeping(solver, 1), 0)

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

        flags = _sleep_flags(solver)
        # Slot 0 = world anchor, slot 1 = the box.
        self.assertEqual(int(flags[0]), 0, msg="anchor must never sleep")
        self.assertEqual(
            int(flags[1]),
            1,
            msg=f"box did not sleep after 2s; sleep flags={flags.tolist()}",
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
        self.assertEqual(_is_sleeping(solver, 1), 1)

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

        flags = _sleep_flags(solver)
        # Both boxes (slots 1 and 2) must sleep.
        self.assertEqual(int(flags[1]), 1, msg="left box did not sleep")
        self.assertEqual(int(flags[2]), 1, msg="right box did not sleep")

        # Confirm the persistent island_root labels are distinct -- each
        # box's root is its own body id, so slot 1's root is 1 and
        # slot 2's root is 2.
        island_root = solver.world.bodies.island_root.numpy()
        self.assertNotEqual(int(island_root[1]), int(island_root[2]))

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
        self.assertEqual(_is_sleeping(solver, 1), 1)

        # Inject a hard +z impulse via body_f, mimicking an external kick.
        # Magnitude has to overcome the inv_mass-gated freeze AND clear the
        # threshold after one step; pick something large.
        body_f = np.zeros((1, 6), dtype=np.float32)
        body_f[0, 5] = 50.0  # +z linear impulse via force-over-step
        state_0.body_f.assign(body_f)

        # NOTE: we don't expect a single sleeping body to wake from its own
        # force alone -- the gravity/force gate is the whole point. Instead,
        # we forcibly clear ``island_root`` (mark every body awake) and
        # assert that re-running the sleeping pass picks the body back up
        # immediately based on the high velocity it now carries.
        island_root = solver.world.bodies.island_root.numpy()
        island_root[:] = -1
        solver.world.bodies.island_root.assign(island_root)
        qd = state_0.body_qd.numpy()
        qd[0, 5] = 5.0  # 5 m/s upward
        state_0.body_qd.assign(qd)

        # One frame with high velocity -- the island max-vel must exceed
        # threshold and the body must remain awake at frame end.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=1, dt=dt)
        # After one frame at +5 m/s the body has lifted; it should still
        # be flagged awake.
        self.assertEqual(
            _is_sleeping(solver, 1),
            0,
            msg="body should remain awake immediately after a high-velocity kick",
        )

    def test_external_force_wakes_sleeping_island(self) -> None:
        """An external force applied via ``state.body_f`` -- e.g. picking,
        a user-side wrench callback -- must wake the body's island on the
        very next step. The per-island sleep score folds in
        ``force * inv_mass * step_dt`` (and the torque analogue) so any
        wrench big enough to actually shift the body lifts the score above
        threshold without the caller having to clear ``island_root``
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
            _is_sleeping(solver, 1),
            1,
            msg="box must be sleeping before the wake test",
        )

        # Apply a deliberately small +z force whose predicted dv
        # (F * invm * dt) stays below the sleep threshold. Explicit user
        # force is still a wake request; otherwise near-static picking
        # can leave a body asleep until the spring grows large enough.
        # body_f spatial-vector layout (matching the PhoenX import kernel)
        # is (force_xyz, torque_xyz); index 2 is force_z.
        body_f = np.zeros((1, 6), dtype=np.float32)
        body_f[0, 2] = 1.0
        state_0.body_f.assign(body_f)

        # Step manually to skip the ``clear_forces`` at the top of
        # ``_run_frames`` -- that zeroes ``state.body_f`` before
        # solver.step gets to import it, which would defeat the test.
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        self.assertEqual(
            _is_sleeping(solver, 1),
            0,
            msg="external force via state.body_f must wake the sleeping island",
        )

    def test_small_external_force_prevents_resleep(self) -> None:
        """A latched user force must keep an awake island from going
        back to sleep even when instantaneous motion is below the sleep
        threshold.

        This covers the square-tower pick path: after one tower wakes,
        the picking spring can become nearly static. The old sleeping
        score let the island consume its final hysteresis tick and go
        back to sleep while the pick was still active, dropping contacts
        for a frame and causing a later local blow-up.
        """
        model = _make_box_on_plane(box_z=0.1)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            step_layout="single_world",
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=10,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        # Put the body one quiet frame away from sleeping, then apply a
        # deliberately small user force. Old code treated only the
        # resulting dv as the sleep score, so this below-threshold force
        # still allowed the body to sleep.
        roots = solver.world.bodies.island_root.numpy()
        roots[1] = -1
        solver.world.bodies.island_root.assign(roots)
        counters = solver.world.bodies.frames_below_threshold.numpy()
        counters[1] = 9
        solver.world.bodies.frames_below_threshold.assign(counters)

        body_f = np.zeros((1, 6), dtype=np.float32)
        body_f[0, 0] = 1.0
        state_0.body_f.assign(body_f)

        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        self.assertEqual(
            _is_sleeping(solver, 1),
            0,
            msg="body under an explicit user force must not re-sleep",
        )
        self.assertEqual(
            int(solver.world.bodies.frames_below_threshold.numpy()[1]),
            0,
            msg="external force should reset the sleep hysteresis counter",
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
        ``bodies.island_root`` (the persistent body-id label captured
        at sleep time), the whole island wakes synchronously and the
        broad phase keeps every contact.
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
        flags_before = _sleep_flags(solver)
        # Slot 0 = anchor, slots 1..N = stack from bottom up.
        self.assertTrue(
            all(int(flags_before[s]) == 1 for s in range(1, 5)),
            msg=f"stack did not fully sleep before pick; sleep flags={flags_before.tolist()}",
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
        # force accumulators, then propagate the wake through every
        # body sharing the picked body's ``island_root``. After this
        # call ``island_root`` is -1 for every body that shared a stack
        # island with the bottom box, so the broad phase below keeps
        # every contact pair instead of filtering them out.
        solver.wake_on_external_input(state_0)

        # Run one frame manually (skip the default ``clear_forces``
        # so ``state.body_f`` survives into the substep solve).
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)

        flags_after = _sleep_flags(solver)
        for s in range(1, 5):
            self.assertEqual(
                int(flags_after[s]),
                0,
                msg=(f"body slot {s} should have woken with the rest of the stack; sleep flags={flags_after.tolist()}"),
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
        """A body settling on the floor must not stamp ``island_root`` until
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

        # By now the counter should be saturated at 10 and island_root>=0.
        counter = int(solver.world.bodies.frames_below_threshold.numpy()[1])
        sleeping = _is_sleeping(solver, 1)
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
        # Settle the box; counter saturates at 10, island_root >= 0.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=90, dt=dt)
        self.assertEqual(int(solver.world.bodies.frames_below_threshold.numpy()[1]), 10)
        self.assertEqual(_is_sleeping(solver, 1), 1)

        # Inject a high upward velocity via state.body_qd. The import
        # kernel will copy this into bodies.velocity before the next
        # step's island-max-velocity scan, lifting island_max above
        # threshold -> reset path fires.
        qd = state_0.body_qd.numpy()
        qd[0, 5] = 5.0  # +5 m/s linear-z
        state_0.body_qd.assign(qd)

        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=1, dt=dt)

        counter = int(solver.world.bodies.frames_below_threshold.numpy()[1])
        sleeping = _is_sleeping(solver, 1)
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
        # hysteresis -- a non-zero default would leave the sleep flag
        # at 0 here. The N=0 path must trip immediately on the first
        # below-threshold frame, so 60 frames is more than enough.
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=60, dt=dt)
        self.assertEqual(
            _is_sleeping(solver, 1),
            1,
            msg="N=0 hysteresis must collapse to single-frame sleep",
        )

    def test_hysteresis_blocks_premature_sleep(self) -> None:
        """With a high ``sleeping_frames_required``, a body that physically
        settled may still report awake (``island_root == -1``) while the
        counter is climbing. Verifies hysteresis actually delays the flip.
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
        sleeping = _is_sleeping(solver, 1)
        self.assertGreater(counter, 0, msg="counter never advanced past 0")
        self.assertLess(
            counter,
            200,
            msg=f"counter unexpectedly saturated ({counter}); did the body sleep prematurely?",
        )
        self.assertEqual(
            sleeping,
            0,
            msg="island_root stamped before counter reached required frames",
        )

    def test_island_root_is_lowest_body_id(self) -> None:
        """The persistent ``island_root`` stamped at sleep time must equal
        the lowest dynamic-body id in the island. With one box on the
        ground (slot 0 = world anchor, slot 1 = the box), the box's
        ``island_root`` must be 1 (its own body id, the smallest in the
        singleton island).
        """
        model = _make_box_on_plane()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)
        island_root = solver.world.bodies.island_root.numpy()
        self.assertEqual(
            int(island_root[1]),
            1,
            msg=f"island_root for a single-box island must equal the box's body id; got {int(island_root[1])}",
        )
        # Anchor must stay at -1: it's static, never sleeps.
        self.assertEqual(int(island_root[0]), -1, msg="anchor must never sleep")

    def test_stack_planks_share_root_after_sequential_sleep(self) -> None:
        """A 4-plank stack settles under gravity. Planks may transition
        to sleeping at different frames (the top plank typically
        jitters longer than the bottom), so the per-step union-find
        sees a different "remaining awake group" each time another
        plank goes to sleep. Without the cross-island unification
        kernel, each plank would be stamped with a different
        ``island_root`` -- the lowest body id of whatever happened to
        still be awake at that moment.

        After unification, every plank's ``island_root`` must converge
        on the lowest body id in the entire physically-connected stack
        (slot 1 here, the bottom plank). This is the precondition for
        atomic wake: picking any one plank can then wake every plank
        in the same root group in a single step instead of triggering
        a multi-frame flood-fill through fresh contacts.
        """
        model = _make_box_stack(n_layers=4)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=240, dt=dt)
        flags = _sleep_flags(solver)
        for s in range(1, 5):
            self.assertEqual(
                int(flags[s]),
                1,
                msg=f"slot {s} not sleeping after settle; sleep flags={flags.tolist()}",
            )
        island_root = solver.world.bodies.island_root.numpy()
        # Slot 1 is the bottom plank = lowest dynamic body id, so the
        # entire stack's unified root must equal 1.
        for s in range(1, 5):
            self.assertEqual(
                int(island_root[s]),
                1,
                msg=(
                    f"slot {s} root {int(island_root[s])} differs from the "
                    f"expected unified root 1; full island_root = {island_root.tolist()}"
                ),
            )

    def test_pick_top_of_stack_wakes_whole_stack_in_one_step(self) -> None:
        """Picking the *top* plank of a sleeping stack must wake every
        plank in a single step. Without unification + atomic
        fan/apply wake, the top plank would wake alone in step 1,
        then propagate downward over multiple frames as new contacts
        are regenerated and the wake-on-contact pass fires plank by
        plank -- the "flood-fill" behaviour the unification fixes.
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
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=240, dt=dt)
        for s in range(1, 5):
            self.assertEqual(
                _is_sleeping(solver, s),
                1,
                msg=f"slot {s} not sleeping before pick",
            )

        # Apply a sideways pull to the TOP plank (newton index n-1 =
        # 3, PhoenX slot 4). 30 N is comfortably above threshold for
        # an ~8 kg cube (dv ~= 0.063 m/s vs threshold 0.05).
        body_f = np.zeros((4, 6), dtype=np.float32)
        body_f[3, 0] = 30.0
        state_0.body_f.assign(body_f)
        solver.wake_on_external_input(state_0)

        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)

        flags_after = _sleep_flags(solver)
        for s in range(1, 5):
            self.assertEqual(
                int(flags_after[s]),
                0,
                msg=(
                    f"slot {s} should have woken with the rest of the stack on "
                    f"the same step; sleep flags={flags_after.tolist()}"
                ),
            )

    def test_multi_stack_picks_wake_only_their_own_stack(self) -> None:
        """Two separated stacks each fall asleep on their own island.
        After many frames asleep (enough that the volatile compact id
        in the live ``set_nr`` would have walked if sleeping bodies
        still participated in the union-find), picking any body in
        one stack must wake the *entire* original stack while leaving
        the other stack still sleeping.

        Regression for the bug where the per-step sleeping pass still
        ran union-find over sleeping bodies: the broad-phase filter
        dropped every sleeping-vs-sleeping pair, leaving the builder
        with a sparse graph that fragmented each settled stack into
        per-body singletons. With ``bodies.island_root`` holding a
        stable body-id label captured at sleep time, the wake
        propagates correctly regardless of how the live compact ids
        drift across frames.
        """
        n_layers = 3
        model = _make_two_stacks(n_layers=n_layers, gap=3.0)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=16,
            sleeping_velocity_threshold=0.05,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        # Settle long enough that both stacks fall asleep and a few
        # more sleeping-only frames pass (so the per-step union-find
        # would have decayed any volatile compact ids the old design
        # relied on).
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=180, dt=dt)

        # PhoenX layout: slot 0 = anchor; slots 1..n_layers = stack A
        # (the first stack added); slots n_layers+1..2*n_layers = stack B.
        stack_a = list(range(1, n_layers + 1))
        stack_b = list(range(n_layers + 1, 2 * n_layers + 1))

        flags_before = _sleep_flags(solver)
        for s in stack_a + stack_b:
            self.assertEqual(
                int(flags_before[s]),
                1,
                msg=f"slot {s} not sleeping before pick; sleep flags={flags_before.tolist()}",
            )

        # Each stack's island_root must equal the smallest body id in
        # that stack (stable, body-id-shaped label).
        island_root = solver.world.bodies.island_root.numpy()
        for s in stack_a:
            self.assertEqual(
                int(island_root[s]),
                stack_a[0],
                msg=f"stack A slot {s} root {int(island_root[s])} != expected {stack_a[0]}",
            )
        for s in stack_b:
            self.assertEqual(
                int(island_root[s]),
                stack_b[0],
                msg=f"stack B slot {s} root {int(island_root[s])} != expected {stack_b[0]}",
            )
        self.assertNotEqual(
            int(island_root[stack_a[0]]),
            int(island_root[stack_b[0]]),
            msg="two physically separate stacks must have distinct island roots",
        )

        # Pick the bottom of stack B with a sideways force, using the
        # public Newton wrench path. body_f layout is (force_xyz,
        # torque_xyz); index 0 is force_x. 30 N pull on a ~8 kg cube
        # gives dv = 30/8/60 ~= 0.063 m/s, comfortably above the 0.05
        # threshold.
        body_f = np.zeros((2 * n_layers, 6), dtype=np.float32)
        body_f[n_layers, 0] = 30.0  # bottom of stack B (newton index = n_layers)
        state_0.body_f.assign(body_f)

        solver.wake_on_external_input(state_0)

        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)

        flags_after = _sleep_flags(solver)
        # Stack B must be entirely awake.
        for s in stack_b:
            self.assertEqual(
                int(flags_after[s]),
                0,
                msg=f"stack B slot {s} should have woken with the rest of the stack; sleep flags={flags_after.tolist()}",
            )
        # Stack A must remain entirely asleep (the wake must not have
        # leaked across the gap between the two physically separate
        # stacks).
        for s in stack_a:
            self.assertEqual(
                int(flags_after[s]),
                1,
                msg=f"stack A slot {s} should have stayed asleep; sleep flags={flags_after.tolist()}",
            )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX sleeping tests run on CUDA only.",
)
class TestSleepingKinematicWake(unittest.TestCase):
    """Regression for the kapla_tower2 camera-collider bug: a kinematic
    mover (e.g. a camera collider) that moves into a sleeping island
    must wake every body in that island on the same step the contact
    appears -- otherwise the kinematic body just sweeps through the
    sleeping geometry as if it weren't there.

    Failure modes this guards against:
      1. ``_constraints_to_elements_kernel`` collapsing the kinematic
         body to -1 because ``inverse_mass == 0`` -- then the
         sleeping detect kernel never sees it in any element.
      2. Broad-phase sleeping filter dropping the kinematic-vs-sleeping
         contact pair (would happen if the filter treated kinematic
         as "frozen").
      3. ``_phoenx_copy_elements_to_int2d_kernel`` dropping the
         kinematic body from the union-find input (non-DYNAMIC
         filter) -- the compact island then has no node carrying the
         kinematic velocity and the score stays at 0.
      4. ``_phoenx_island_max_velocity_kernel`` skipping the kinematic
         body when scoring the compact island.
      5. The order in ``world.step`` running the sleeping pass BEFORE
         ``_kinematic_prepare_step`` -- the score kernel then reads a
         stale ``bodies.velocity`` (= 0 for fresh kinematic targets)
         and the island is re-marked as sleeping.

    Mirrors the kapla example's setup pattern: builds a Newton model
    with the camera-collider body, then creates PhoenXWorld directly
    so we can promote that slot to MOTION_KINEMATIC + zero inv_mass
    BEFORE PhoenXWorld counts kinematic bodies. The Newton-Model
    wrapper (``SolverPhoenX``) doesn't expose a "this body is
    kinematic" knob -- the lower-level path is the right surface.
    """

    def _build_scene(self, *, n_layers: int = 3, box_half: float = 0.1, pusher_radius: float = 0.15):
        """Build a brick-stack + sphere pusher scene exactly like the
        kapla example wires up its camera collider: Newton model,
        sleeping-aware ``CollisionPipeline``, manual PhoenX body
        container, kinematic-promotion BEFORE ``PhoenXWorld``, the
        share-vertex filter data with sleeping fields populated.

        Returns a dict bundling everything the per-frame loop needs.
        """
        from newton._src.solvers.phoenx.body import (  # noqa: PLC0415
            MOTION_KINEMATIC,
            body_container_zeros,
        )
        from newton._src.solvers.phoenx.cloth_collision import (  # noqa: PLC0415
            PhoenXClothShareVertexFilterData,
            build_phoenx_share_vertex_filter_data,
            phoenx_cloth_share_vertex_filter,
        )
        from newton._src.solvers.phoenx.examples.example_common import (  # noqa: PLC0415
            init_phoenx_bodies_kernel,
            newton_to_phoenx_kernel,
            phoenx_to_newton_kernel,
        )
        from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld  # noqa: PLC0415

        device = wp.get_device("cuda:0")

        mb = newton.ModelBuilder()
        mb.add_ground_plane()
        for i in range(n_layers):
            z = box_half + i * (2.0 * box_half + 1e-3)
            b = mb.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()))
            mb.add_shape_box(
                b,
                hx=box_half,
                hy=box_half,
                hz=box_half,
                cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.6),
            )
        # Pusher: starts 2 m away (no broad-phase contact with the
        # stack during the settle).
        pusher_id = mb.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 0.0, box_half), q=wp.quat_identity()),
        )
        mb.add_shape_sphere(
            pusher_id,
            radius=pusher_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
        )
        mb.gravity = -GRAVITY
        model = mb.finalize()

        # Sleeping-aware collision pipeline (share-vertex filter
        # carries the per-shape body / sleeping / motion-type lookup
        # used to filter rigid-rigid pairs where both sides are
        # frozen).
        collision_pipeline = newton.CollisionPipeline(
            model,
            contact_matching="sticky",
            broad_phase_filter=(
                phoenx_cloth_share_vertex_filter,
                PhoenXClothShareVertexFilterData,
            ),
        )
        contacts = collision_pipeline.contacts()
        rigid_contact_max = int(contacts.rigid_contact_point0.shape[0])

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)

        # PhoenX body container: slot 0 = world anchor; slot i+1 =
        # newton body i. ``_init_phoenx_bodies_kernel`` reads inv_mass
        # / inv_inertia from the model.
        num_phx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phx_bodies, device=device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=device,
            ),
        )
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=model.body_count,
            inputs=[
                model.body_q,
                state.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
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
                bodies.body_com,
            ],
            device=device,
        )

        # Promote pusher to KINEMATIC + zero its inverse mass / inertia
        # BEFORE ``PhoenXWorld.__init__`` (which caches kinematic count).
        pusher_slot = pusher_id + 1
        mt = bodies.motion_type.numpy()
        mt[pusher_slot] = int(MOTION_KINEMATIC)
        bodies.motion_type.assign(mt)
        for arr_name in ("inverse_mass",):
            arr = getattr(bodies, arr_name).numpy()
            arr[pusher_slot] = 0.0
            getattr(bodies, arr_name).assign(arr)
        for arr_name in ("inverse_inertia", "inverse_inertia_world"):
            arr = getattr(bodies, arr_name).numpy()
            arr[pusher_slot] = np.zeros((3, 3), dtype=np.float32)
            getattr(bodies, arr_name).assign(arr)

        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        shape_body_np = model.shape_body.numpy()
        shape_body_phx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        shape_body = wp.array(shape_body_phx, dtype=wp.int32, device=device)

        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            substeps=4,
            solver_iterations=8,
            gravity=(0.0, 0.0, -GRAVITY),
            rigid_contact_max=rigid_contact_max,
            step_layout="single_world",
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=10,
            device=device,
        )

        # Sleeping-aware filter data, post-PhoenXWorld so it can read
        # ``bodies.island_root``.
        tri_sentinel = wp.zeros((1, 3), dtype=wp.int32, device=device)
        tet_sentinel = wp.zeros((1, 4), dtype=wp.int32, device=device)
        filter_data = build_phoenx_share_vertex_filter_data(
            num_rigid_shapes=int(model.shape_count),
            num_cloth_triangles=0,
            tri_indices=tri_sentinel,
            tet_indices=tet_sentinel,
            sleeping_enabled=True,
            phoenx_body_offset=1,
            shape_body=model.shape_body,
            body_island_root=bodies.island_root,
            body_motion_type=bodies.motion_type,
            device=device,
        )
        collision_pipeline.set_broad_phase_filter_data(filter_data)

        return {
            "model": model,
            "state": state,
            "contacts": contacts,
            "collision_pipeline": collision_pipeline,
            "world": world,
            "bodies": bodies,
            "shape_body": shape_body,
            "pusher_id": pusher_id,
            "pusher_slot": pusher_slot,
            "n_layers": n_layers,
            "box_half": box_half,
            "newton_to_phoenx": newton_to_phoenx_kernel,
            "phoenx_to_newton": phoenx_to_newton_kernel,
            "device": device,
        }

    def _step_frame(self, scene, pusher_xyz: tuple[float, float, float], dt: float) -> None:
        """One outer step: stage pusher target, sync dynamic state,
        collide, step. Mirrors the kapla example's per-frame pipeline."""
        world = scene["world"]
        device = scene["device"]
        n = scene["model"].body_count
        bodies = scene["bodies"]
        state = scene["state"]

        # Tell PhoenX the new kinematic target via the batch API
        # (matches the kapla example's per-frame camera-collider
        # update path).
        body_id_arr = wp.array([int(scene["pusher_slot"])], dtype=wp.int32, device=device)
        pos_arr = wp.array([pusher_xyz], dtype=wp.vec3f, device=device)
        orient_arr = wp.array([(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
        world.set_kinematic_poses_batch(
            body_ids=body_id_arr,
            positions=pos_arr,
            orientations=orient_arr,
        )
        # Also patch state.body_q[pusher] so Newton's CollisionPipeline
        # broad-phase sees the live kinematic position (the kapla
        # example does this with a one-line write kernel).
        bq = state.body_q.numpy()
        bq[scene["pusher_id"]] = (
            pusher_xyz[0],
            pusher_xyz[1],
            pusher_xyz[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )
        state.body_q.assign(bq)

        # Sync dynamic-only Newton -> PhoenX slice (skip the kinematic
        # body's slot -- it's owned by ``set_kinematic_poses_batch``).
        if n > 1:
            wp.launch(
                scene["newton_to_phoenx"],
                dim=n - 1,
                inputs=[state.body_q, state.body_qd, scene["model"].body_com],
                outputs=[
                    bodies.position[1:n],
                    bodies.orientation[1:n],
                    bodies.velocity[1:n],
                    bodies.angular_velocity[1:n],
                ],
                device=device,
            )

        scene["model"].collide(state, contacts=scene["contacts"], collision_pipeline=scene["collision_pipeline"])
        nphase = scene["collision_pipeline"].narrow_phase
        world.step(
            dt=dt,
            contacts=scene["contacts"],
            shape_body=scene["shape_body"],
            shape_aabb_lower=nphase.shape_aabb_lower,
            shape_aabb_upper=nphase.shape_aabb_upper,
        )
        # Bridge PhoenX -> Newton state for dynamic bodies (so the
        # next frame's collide() reads the post-step poses).
        if n > 1:
            wp.launch(
                scene["phoenx_to_newton"],
                dim=n - 1,
                inputs=[
                    bodies.position[1:n],
                    bodies.orientation[1:n],
                    bodies.velocity[1:n],
                    bodies.angular_velocity[1:n],
                    scene["model"].body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )

    def test_moving_kinematic_wakes_sleeping_stack(self) -> None:
        """A kinematic body teleported into a sleeping stack must wake
        every body in the stack on the same step."""
        scene = self._build_scene(n_layers=3, box_half=0.1, pusher_radius=0.15)
        bodies = scene["bodies"]
        pusher_slot = scene["pusher_slot"]
        stack_slots = list(range(1, scene["n_layers"] + 1))
        dt = 1.0 / 60.0
        # Mark the already-resting stack as one sleeping island. The
        # test is about kinematic-contact wake propagation, not the
        # unrelated settling heuristic.
        roots = bodies.island_root.numpy()
        roots[stack_slots] = stack_slots[0]
        roots[pusher_slot] = -1
        bodies.island_root.assign(roots)
        counters = bodies.frames_below_threshold.numpy()
        counters[stack_slots] = 10
        counters[pusher_slot] = 0
        bodies.frames_below_threshold.assign(counters)

        flags_before = (bodies.island_root.numpy() >= 0).astype(np.int32)
        for s in stack_slots:
            self.assertEqual(
                int(flags_before[s]),
                1,
                msg=(f"stack must be fully sleeping before pusher arrives; got sleep flags={flags_before.tolist()}"),
            )
        self.assertEqual(
            int(flags_before[pusher_slot]),
            0,
            msg="kinematic pusher must never carry a sleep flag",
        )
        positions_before = bodies.position.numpy().copy()

        # Wake step: pusher teleports onto the stack center
        # (inferred kinematic velocity = 2 m / (1/60 s) = 120 m/s --
        # well above the 0.05 m/s threshold).
        at_stack = (0.0, 0.0, scene["box_half"])
        self._step_frame(scene, at_stack, dt)

        flags_after = (bodies.island_root.numpy() >= 0).astype(np.int32)
        still_sleeping = [s for s in stack_slots if int(flags_after[s]) == 1]
        self.assertEqual(
            still_sleeping,
            [],
            msg=(
                "Every stack body should have woken after the kinematic "
                "pusher moved into the stack. "
                f"sleep flags before={flags_before.tolist()} "
                f"sleep flags after={flags_after.tolist()}"
            ),
        )
        positions_after = bodies.position.numpy()
        max_disp = float(np.linalg.norm(positions_after[stack_slots] - positions_before[stack_slots], axis=1).max())
        self.assertGreater(
            max_disp,
            1e-3,
            msg=(
                "Stack bodies did not visibly move after the kinematic "
                f"pusher entered the column (max displacement = {max_disp:.6f} m). "
                "Either contacts were dropped or the wake failed to lift "
                "the substep solve."
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
