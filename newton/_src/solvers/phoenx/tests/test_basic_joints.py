# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-limit regression test mirroring ``example_basic_joints``.

Reproduces the prismatic-joint scene from
``newton/examples/basic/example_basic_joints.py`` exactly (revolute +
prismatic + ball arms anchored to fixed parents) and runs it through
``SolverPhoenX``. The prismatic joint has a tight ``[-0.3, 0.3] m``
slide window; under gravity the slider must hit the lower stop and
stay inside the window.

Before the fix, Newton's default ``limit_ke = 1e4`` / ``limit_kd = 10``
were forwarded to PhoenX as soft PD gains via :mod:`model_adapter`,
which on a 1 kg body with gravity produced a barely-engaged spring
that let the slider overshoot the lower limit by ~0.2 m and oscillate
indefinitely. The fix routes Newton joint limits through PhoenX's
rigid Box2D path (``hertz_limit = 1e9``) -- matching the ``SolverXPBD``
contract that limits are hard positional constraints, not soft
penalties.

Runs on CUDA only -- same rationale as the other PhoenX tests.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests._test_helpers import make_solver_graph_stepper


def _build_basic_joints_model() -> tuple[newton.Model, dict[str, int]]:
    """Build the full ``example_basic_joints`` scene.

    Returns the finalised model plus a name -> body-index map for the
    bodies the assertions inspect.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    cuboid_hx = 0.1
    cuboid_hy = 0.1
    cuboid_hz = 0.75
    upper_hz = 0.25 * cuboid_hz

    rows = [-3.0, 0.0, 3.0]
    drop_z = 2.0

    # Revolute (hinge) arm.
    y = rows[0]
    a_rev = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
    b_rev = builder.add_link(
        xform=wp.transform(
            p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15)
        ),
        label="b_rev",
    )
    builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
    builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)
    j_fixed_rev = builder.add_joint_fixed(
        parent=-1,
        child=a_rev,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j_revolute = builder.add_joint_revolute(
        parent=a_rev,
        child=b_rev,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
    )
    builder.add_articulation([j_fixed_rev, j_revolute])
    builder.joint_q[-1] = wp.pi * 0.5

    # Prismatic (slider) arm with tight limits.
    y = rows[1]
    a_pri = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
    b_pri = builder.add_link(
        xform=wp.transform(
            p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.12)
        ),
        label="b_prismatic",
    )
    builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
    builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)
    j_fixed_pri = builder.add_joint_fixed(
        parent=-1,
        child=a_pri,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j_prismatic = builder.add_joint_prismatic(
        parent=a_pri,
        child=b_pri,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
        limit_lower=-0.3,
        limit_upper=0.3,
    )
    builder.add_articulation([j_fixed_pri, j_prismatic])

    # Ball arm.
    y = rows[2]
    radius = 0.3
    z_offset = -1.0
    a_ball = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity())
    )
    b_ball = builder.add_link(
        xform=wp.transform(
            p=wp.vec3(0.0, y, drop_z + radius + z_offset), q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.1)
        ),
        label="b_ball",
    )
    rigid_cfg = newton.ModelBuilder.ShapeConfig()
    rigid_cfg.density = 0.0
    builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
    builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)
    j_fixed_ball = builder.add_joint_fixed(
        parent=-1,
        child=a_ball,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j_ball = builder.add_joint_ball(
        parent=a_ball,
        child=b_ball,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
    )
    builder.add_articulation([j_fixed_ball, j_ball])
    builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)

    builder.color()
    model = builder.finalize()

    body_indices = {
        "b_rev": model.body_label.index("b_rev"),
        "b_prismatic": model.body_label.index("b_prismatic"),
        "b_ball": model.body_label.index("b_ball"),
    }
    return model, body_indices


def _make_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=8,
        velocity_iterations=1,
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX joint-limit tests run on CUDA only (graph-capture path).",
)
class TestPhoenXBasicJointsPrismaticLimit(unittest.TestCase):
    """Regression: ``example_basic_joints``'s prismatic slider must
    respect its ``[-0.3, 0.3] m`` window under PhoenX."""

    def test_prismatic_limit_holds(self) -> None:
        model, body_indices = _build_basic_joints_model()
        solver = _make_solver(model)

        s0 = model.state()
        s1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        b_pri = body_indices["b_prismatic"]

        # b_prismatic body's z at init (also the slide=0 reference for
        # this scene). With ``parent_xform.z = -upper_hz`` and
        # ``child_xform.z = +cuboid_hz`` and the joint axis +Z, the
        # slide coordinate is ``body_q[b_pri].z - z_init``. The window
        # ``[-0.3, +0.3]`` therefore bounds the body's z to
        # ``[z_init - 0.3, z_init + 0.3]``.
        z_init = float(s0.body_q.numpy()[b_pri][2])

        # 1 s of simulation: 100 frames at 100 fps with 10 inner
        # substeps each. Long enough for the slider to fully sample
        # the bottom of the limit window under gravity (free-fall
        # from the initial joint coord = 0 reaches the lower stop in
        # ~250 ms at v ~ 2.4 m/s).
        fps = 100
        sim_substeps = 10
        sim_dt = 1.0 / fps / sim_substeps

        # Tolerance: PhoenX's rigid Box2D limit converges in a few
        # substeps but still has Baumgarte bias proportional to the
        # in-bound velocity. ~2 cm overshoot at v ~ 2.4 m/s is within
        # the soft-constraint slack; > 5 cm would mean the limit row
        # is acting soft (the bug this test guards against -- soft PD
        # gains let the slider overshoot by 0.2 m).
        slack = 0.05
        step = make_solver_graph_stepper(solver, s0, s1, control, contacts, model, sim_dt)
        max_slide_seen = 0.0
        for _frame in range(100):
            s0, s1 = step(sim_substeps)
            slide = float(s0.body_q.numpy()[b_pri][2]) - z_init
            max_slide_seen = max(max_slide_seen, abs(slide))
            self.assertGreaterEqual(
                slide,
                -0.3 - slack,
                f"prismatic slider undershot lower limit: slide={slide:+.4f} m (window [-0.3, +0.3], slack {slack} m)",
            )
            self.assertLessEqual(
                slide,
                0.3 + slack,
                f"prismatic slider overshot upper limit: slide={slide:+.4f} m (window [-0.3, +0.3], slack {slack} m)",
            )

        # Sanity: the slider must actually reach the lower stop --
        # otherwise the test could pass trivially with the body
        # frozen mid-window.
        self.assertGreater(
            max_slide_seen,
            0.25,
            f"slider never approached the limit window (max |slide| = {max_slide_seen:.4f} m); "
            "test setup may have stopped exercising the limit row.",
        )

    def test_prismatic_settles_at_rest(self) -> None:
        """After a few seconds, the slider must come to rest *inside*
        the limit window. Catches the soft-limit oscillation mode where
        the slider bounces in and out of the window indefinitely.
        """
        model, body_indices = _build_basic_joints_model()
        solver = _make_solver(model)

        s0 = model.state()
        s1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        b_pri = body_indices["b_prismatic"]
        z_init = float(s0.body_q.numpy()[b_pri][2])

        fps = 100
        sim_substeps = 10
        sim_dt = 1.0 / fps / sim_substeps
        # 4 s -- well past any transient bouncing for the chosen
        # solver budget.
        step = make_solver_graph_stepper(solver, s0, s1, control, contacts, model, sim_dt)
        s0, s1 = step(400 * sim_substeps)

        body_qd = s0.body_qd.numpy()[b_pri]
        # Linear velocity must be near zero -- the slider has settled.
        # body_qd is a spatial vector ``[w, v]``; the linear part is
        # the last 3 components.
        v_lin = float(np.linalg.norm(body_qd[3:6]))
        self.assertLess(
            v_lin,
            0.05,
            f"slider still moving after 4 s: |v_lin|={v_lin:.4f} m/s -- "
            "limit row is acting soft and the body bounces in/out of the window.",
        )

        slide_final = float(s0.body_q.numpy()[b_pri][2]) - z_init
        self.assertGreaterEqual(
            slide_final,
            -0.3 - 0.01,
            f"slider rest position {slide_final:+.4f} m below lower limit -0.3",
        )
        self.assertLessEqual(
            slide_final,
            0.3 + 0.01,
            f"slider rest position {slide_final:+.4f} m above upper limit +0.3",
        )


if __name__ == "__main__":
    unittest.main()
