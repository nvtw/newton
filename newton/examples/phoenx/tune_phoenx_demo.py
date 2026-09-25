# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small two-world scene factory for the PhoenX autotuner."""

import warp as wp

import newton
from newton.examples.phoenx.tune_phoenx import TuningScene


def make_scene():
    world = newton.ModelBuilder()
    world.add_ground_plane()
    body = world.add_body(xform=wp.transform((0.0, 0.0, 0.65), wp.quat_identity()))
    world.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
    hinge_pose = wp.transform((0.8, 0.0, 1.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3))
    pendulum = world.add_link(xform=hinge_pose)
    world.add_shape_box(pendulum, hx=0.12, hy=0.12, hz=0.32, xform=wp.transform((0.0, 0.0, -0.32), wp.quat_identity()))
    world.add_joint_revolute(
        -1,
        pendulum,
        parent_xform=hinge_pose,
        child_xform=wp.transform_identity(),
        axis=newton.Axis.Y,
    )
    builder = newton.ModelBuilder()
    builder.add_world(world)
    builder.add_world(world)
    return TuningScene(
        model=builder.finalize(),
        frame_dt=1.0 / 60.0,
        solver_options={"substeps": 4, "solver_iterations": 4, "velocity_iterations": 1},
    )
