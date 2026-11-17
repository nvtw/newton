# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check parallel contact preparation with sleeping islands."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.sleeping_filter import PhoenXSleepingFilterData, phoenx_sleeping_filter


def _build_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.6)
    # A resting stack sleeps with contacts against the ground and between boxes.
    for i in range(3):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.1 + 0.201 * i), q=wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    # A hanging pendulum sleeps with loaded joint rows that hold gravity.
    parent = -1
    joints = []
    for i in range(2):
        link = builder.add_link(xform=wp.transform(p=wp.vec3(2.0, 0.0, 2.0 - 0.4 * i), q=wp.quat_identity()))
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.15, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(p=wp.vec3(2.0, 0.0, 2.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.2)),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.2)),
                axis=newton.Axis.Y,
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize()


def _run(parallel: bool, frames: int):
    model = _build_model()
    pipeline = newton.CollisionPipeline(
        model, contact_matching="sticky", broad_phase_filter=(phoenx_sleeping_filter, PhoenXSleepingFilterData)
    )
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        step_layout="single_world",
        joint_mode="maximal_pgs",
        substeps=2,
        solver_iterations=8,
        parallel_contact_prepare=parallel,
        sleeping_velocity_threshold=0.05,
        sleeping_frames_required=10,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = pipeline.contacts()
    push = np.zeros((model.body_count, 6), dtype=np.float32)
    push[:, 0] = 2.0
    history = []
    sleeping = None
    for frame in range(frames + 30):
        state_0.clear_forces()
        if frame == frames:
            sleeping = solver.world.bodies.island_root.numpy()[1 : model.body_count + 1] >= 0
        if frame >= frames:
            # Waking exposes any velocity that frozen rows changed while asleep.
            state_0.body_f.assign(push)
            solver.wake_on_external_input(state_0)
        solver.collide(state_0, contacts, dt=1.0 / 60.0)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
        history.append((state_0.body_q.numpy(), state_0.body_qd.numpy()))
    return history, sleeping


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "Parallel contact preparation requires CUDA.")
class TestParallelPrepareSleeping(unittest.TestCase):
    def test_sleeping_matches_serial_prepare(self):
        """Frozen joint rows stay unsolved, and every frame matches serial preparation, through waking."""
        frames = 240
        baseline, baseline_sleeping = _run(False, frames)
        candidate, candidate_sleeping = _run(True, frames)
        self.assertTrue(np.all(baseline_sleeping), baseline_sleeping)
        np.testing.assert_array_equal(candidate_sleeping, baseline_sleeping)
        for frame, ((q0, qd0), (q1, qd1)) in enumerate(zip(baseline, candidate, strict=True)):
            with self.subTest(frame=frame):
                np.testing.assert_array_equal(q0.view(np.uint8), q1.view(np.uint8))
                np.testing.assert_array_equal(qd0.view(np.uint8), qd1.view(np.uint8))
        # Sleeping bodies hold their poses exactly until the push wakes them.
        np.testing.assert_array_equal(candidate[frames - 30][0], candidate[frames - 1][0])
        self.assertFalse(np.array_equal(candidate[frames - 1][0], candidate[-1][0]))


if __name__ == "__main__":
    unittest.main()
