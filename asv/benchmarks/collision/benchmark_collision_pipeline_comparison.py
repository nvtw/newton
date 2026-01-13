# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ASV benchmarks comparing CollisionPipeline (standard) vs CollisionPipelineUnified.

This benchmark creates a scene similar to example_basic_shapes3.py:
- A pyramid of stacked cubes
- A wrecking ball (heavy sphere) on a ramp
- A ground plane
- Additional spheres, boxes, and capsules

Only shapes supported by BOTH pipelines are used:
- Sphere, Box, Capsule, Plane

Excluded shapes (only supported by CollisionPipelineUnified):
- Cylinder, Cone (limited or no support in standard pipeline)
"""

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import newton


def build_wrecking_ball_scene(pipeline_type: str, pyramid_size: int = 10):
    """Build a scene with a pyramid of cubes, a wrecking ball, and a ramp.

    Args:
        pipeline_type: "standard" or "unified"
        pyramid_size: Number of cubes at the base of the pyramid.

    Returns:
        Tuple of (model, collision_pipeline, state, solver).
    """
    builder = newton.ModelBuilder()

    # Add a ground plane at z=0
    builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    drop_z = 2.0

    # Add some individual shapes (sphere, capsule, box)
    # SPHERE
    body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()))
    builder.add_shape_sphere(body_sphere, radius=0.5)
    joint = builder.add_joint_free(body_sphere)
    builder.add_articulation([joint])

    # CAPSULE
    body_capsule = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
    builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)
    joint = builder.add_joint_free(body_capsule)
    builder.add_articulation([joint])

    # BOX
    body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()))
    builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)
    joint = builder.add_joint_free(body_box)
    builder.add_articulation([joint])

    # Build pyramid of cubes
    cube_h = 0.4
    gap = 0.02
    y_stack = 6.0
    cube_spacing = 2.1 * cube_h

    for level in range(pyramid_size):
        num_cubes_in_row = pyramid_size - level
        row_width = (num_cubes_in_row - 1) * cube_spacing

        for i in range(num_cubes_in_row):
            x_pos = -row_width / 2 + i * cube_spacing
            z_pos = level * cube_spacing + cube_h
            y_pos = y_stack

            body = builder.add_body(xform=wp.transform(p=wp.vec3(x_pos, y_pos, z_pos), q=wp.quat_identity()))
            builder.add_shape_box(body, hx=cube_h, hy=cube_h, hz=cube_h)
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])

    # WRECKING BALL - Heavy sphere that rolls down a ramp towards the pyramid
    pyramid_height = pyramid_size * cube_spacing
    wrecking_ball_radius = 2.0
    wrecking_ball_mass_multiplier = 10.0

    ramp_length = 20.0
    ramp_height = pyramid_height / 2
    ramp_angle = float(np.arctan2(ramp_height, ramp_length))

    ball_x = 0.0
    ball_y = y_stack + ramp_length * 0.9
    ball_z = ramp_height + wrecking_ball_radius + 0.1

    # Create the wrecking ball
    body_ball = builder.add_body(xform=wp.transform(p=wp.vec3(ball_x, ball_y, ball_z), q=wp.quat_identity()))
    ball_shape_cfg = newton.ModelBuilder.ShapeConfig()
    ball_shape_cfg.density = builder.default_shape_cfg.density * wrecking_ball_mass_multiplier
    builder.add_shape_sphere(body_ball, radius=wrecking_ball_radius, cfg=ball_shape_cfg)
    joint = builder.add_joint_free(body_ball)
    builder.add_articulation([joint])

    # Create a tilted ramp (static box)
    ramp_width = 5.0
    ramp_thickness = 0.5
    ramp_center_y = y_stack + ramp_length / 2
    ramp_center_z = ramp_height / 2

    ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(ramp_angle))

    builder.add_shape_box(
        body=-1,  # Static body (attached to world)
        xform=wp.transform(p=wp.vec3(ball_x, ramp_center_y, ramp_center_z), q=ramp_quat),
        hx=ramp_width / 2,
        hy=ramp_length / 2,
        hz=ramp_thickness / 2,
    )

    # Finalize model (shape_contact_pairs are built automatically)
    model = builder.finalize()

    # Create collision pipeline based on type
    if pipeline_type == "standard":
        collision_pipeline = newton.CollisionPipeline.from_model(
            model,
            rigid_contact_max_per_pair=10,
        )
    else:
        collision_pipeline = newton.CollisionPipelineUnified.from_model(
            model,
            rigid_contact_max_per_pair=10,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
        )

    # Create solver and state
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return model, collision_pipeline, state, solver


class CollisionPipelineComparisonCollide:
    """Benchmark collision detection (model.collide) comparing Standard vs Unified pipelines."""

    repeat = 5
    number = 1
    timeout = 300
    params = [
        [10, 15, 20],  # pyramid_size (55, 120, 210 cubes in pyramid + extras)
        ["standard", "unified"],  # pipeline_type
    ]
    param_names = ["pyramid_size", "pipeline_type"]

    def setup(self, pyramid_size, pipeline_type):
        self.model, self.collision_pipeline, self.state, self.solver = build_wrecking_ball_scene(
            pipeline_type, pyramid_size
        )
        self.control = self.model.control()

        # Warm up
        self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, pyramid_size, pipeline_type):
        """Time the collision detection phase only."""
        self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_contact_count(self, pyramid_size, pipeline_type):
        """Track the number of contacts generated."""
        self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()
        return self.contacts.rigid_contact_count.numpy()[0]

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_shape_count(self, pyramid_size, pipeline_type):
        """Track the number of shapes in the scene."""
        return self.model.shape_count


class CollisionPipelineComparisonStep:
    """Benchmark full simulation step comparing Standard vs Unified pipelines."""

    repeat = 5
    number = 1
    timeout = 300
    params = [
        [10, 15, 20],  # pyramid_size
        ["standard", "unified"],  # pipeline_type
    ]
    param_names = ["pyramid_size", "pipeline_type"]

    def setup(self, pyramid_size, pipeline_type):
        self.model, self.collision_pipeline, self.state_0, self.solver = build_wrecking_ball_scene(
            pipeline_type, pyramid_size
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_dt = 1.0 / 600.0  # 10 substeps at 60fps

        # Warm up
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, pyramid_size, pipeline_type):
        """Time a single simulation substep (collide + solve)."""
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()


class CollisionPipelineComparisonFrame:
    """Benchmark a full frame (multiple substeps) comparing Standard vs Unified pipelines."""

    repeat = 3
    number = 1
    timeout = 600
    params = [
        [10, 15],  # pyramid_size - smaller for frame benchmark
        ["standard", "unified"],  # pipeline_type
    ]
    param_names = ["pyramid_size", "pipeline_type"]

    def setup(self, pyramid_size, pipeline_type):
        self.model, self.collision_pipeline, self.state_0, self.solver = build_wrecking_ball_scene(
            pipeline_type, pyramid_size
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.fps = 60
        self.substeps = 10
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps

        # Warm up with one frame
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_frame(self, pyramid_size, pipeline_type):
        """Time a full frame (10 substeps at 60fps)."""
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "CollisionPipelineComparisonCollide": CollisionPipelineComparisonCollide,
        "CollisionPipelineComparisonStep": CollisionPipelineComparisonStep,
        "CollisionPipelineComparisonFrame": CollisionPipelineComparisonFrame,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)

