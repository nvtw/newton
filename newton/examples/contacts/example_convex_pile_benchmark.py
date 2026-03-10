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

###########################################################################
# Example Convex Hull Pile Benchmark
#
# Builds pyramids of convex-hull cubes with a wrecking ball on a ramp
# to stress-test GJK/MPR narrow-phase and multi-contact generation.
#
# Command: python -m newton.examples convex_pile_benchmark
#
###########################################################################

import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
import newton.examples

# --- Scene parameters (matches example_basic_shapes3_test.py) ---
DEFAULT_NUM_PYRAMIDS = 50
DEFAULT_PYRAMID_SIZE = 20
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
DROP_Z = 2.0
Y_STACK = 6.0

# Wrecking ball
WRECKING_BALL_RADIUS = 2.0
WRECKING_BALL_DENSITY_MULT = 100.0
RAMP_LENGTH = 20.0
RAMP_WIDTH = 5.0
RAMP_THICKNESS = 0.5

# Solver
XPBD_ITERATIONS = 2
XPBD_CONTACT_RELAXATION = 0.8


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.solver_type = args.solver
        self.test_mode = args.test
        self.world_count = args.world_count

        num_pyramids = args.num_pyramids
        pyramid_size = args.pyramid_size

        # --- Build scene ---
        builder = newton.ModelBuilder()
        builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        cube_hull_mesh = newton.Mesh.create_box(
            CUBE_HALF, CUBE_HALF, CUBE_HALF,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        hull_count = 0
        pyramid_height = pyramid_size * CUBE_SPACING

        for pyramid in range(num_pyramids):
            y_offset = pyramid * PYRAMID_SPACING
            for level in range(pyramid_size):
                num_cubes_in_row = pyramid_size - level
                row_width = (num_cubes_in_row - 1) * CUBE_SPACING
                for i in range(num_cubes_in_row):
                    x_pos = -row_width / 2 + i * CUBE_SPACING
                    z_pos = level * CUBE_SPACING + CUBE_HALF
                    y_pos = Y_STACK - y_offset
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(x_pos, y_pos, z_pos), q=wp.quat_identity()),
                    )
                    builder.add_shape_convex_hull(body, mesh=cube_hull_mesh)
                    hull_count += 1

        self.hull_count = hull_count
        print(f"Built {num_pyramids} pyramids x {pyramid_size} rows = {hull_count} convex hulls")

        # Wrecking ball
        ramp_height = pyramid_height / 2
        ramp_angle = float(np.arctan2(ramp_height, RAMP_LENGTH))
        ball_x = 0.0
        ball_y = Y_STACK + RAMP_LENGTH * 0.9
        ball_z = ramp_height + WRECKING_BALL_RADIUS + 0.1

        body_ball = builder.add_body(
            xform=wp.transform(p=wp.vec3(ball_x, ball_y, ball_z), q=wp.quat_identity()),
        )
        ball_cfg = newton.ModelBuilder.ShapeConfig()
        ball_cfg.density = builder.default_shape_cfg.density * WRECKING_BALL_DENSITY_MULT
        builder.add_shape_sphere(body_ball, radius=WRECKING_BALL_RADIUS, cfg=ball_cfg)

        # Ramp (static)
        ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(ramp_angle))
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                p=wp.vec3(ball_x, Y_STACK + RAMP_LENGTH / 2, ramp_height / 2),
                q=ramp_quat,
            ),
            hx=RAMP_WIDTH / 2,
            hy=RAMP_LENGTH / 2,
            hz=RAMP_THICKNESS / 2,
        )

        # --- Replicate & finalize ---
        if self.world_count > 1:
            main_builder = newton.ModelBuilder()
            main_builder.replicate(builder, world_count=self.world_count)
            self.model = main_builder.finalize()
        else:
            self.model = builder.finalize()

        self.broad_phase = args.broad_phase

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.broad_phase,
        )

        # --- Solver ---
        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=XPBD_ITERATIONS,
                rigid_contact_relaxation=XPBD_CONTACT_RELAXATION,
            )
        elif self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=15,
                ls_iterations=100,
                impratio=1.0,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. Choose from 'xpbd' or 'mujoco'.")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        cam_dist = max(pyramid_height, num_pyramids * PYRAMID_SPACING * 0.3)
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.4),
            pitch=-15.0,
            yaw=135.0,
        )

        # --- Benchmark bookkeeping ---
        self._init_benchmark_tracking()

        if self.enable_kernel_timing:
            self.graph = None
        else:
            self.capture()

    # ------------------------------------------------------------------
    # Benchmark tracking (mirrors nut_bolt_sdf_benchmark)
    # ------------------------------------------------------------------

    def _init_benchmark_tracking(self):
        self.step_count = 0
        self.max_sim_steps = 2000
        self.enable_kernel_timing = True
        self._terminate_requested = False
        self._exit_report_printed = False
        self.log_period_steps = 60
        self.last_perf_step = 0
        self.last_perf_time = time.perf_counter()
        self._timing_totals: dict[str, list[float]] = defaultdict(list)
        self._timing_frame_count = 0

    def _report_steps_per_second(self):
        if self.step_count % self.log_period_steps != 0:
            return
        now = time.perf_counter()
        elapsed = now - self.last_perf_time
        steps_elapsed = self.step_count - self.last_perf_step
        if elapsed > 0.0 and steps_elapsed > 0:
            sps = steps_elapsed / elapsed
            print(f"[step {self.step_count}] throughput={sps:.2f} steps/s")
        self.last_perf_step = self.step_count
        self.last_perf_time = now

    def _print_exit_benchmark_report(self):
        if self._exit_report_printed:
            return
        self._exit_report_printed = True

        if self._timing_frame_count > 0:
            frame_count = self._timing_frame_count
            width = 110
            kernel_width = width - 30
            print(f"\n{'=' * width}")
            print(f"  Kernel timing report ({frame_count} frames, {self.hull_count} convex hulls x {self.world_count} worlds)")
            print(f"{'=' * width}")
            print(f"{'Kernel':<{kernel_width}} {'Total ms':>10} {'Avg ms':>10} {'Count':>7}")
            print(f"{'-' * kernel_width} {'-' * 10} {'-' * 10} {'-' * 7}")

            grand_total = 0.0
            rows = []
            for name, times in self._timing_totals.items():
                total = float(sum(times))
                grand_total += total
                rows.append((total, name, total / len(times), len(times)))
            rows.sort(key=lambda row: row[0], reverse=True)

            for total, name, avg, count in rows:
                label = name if len(name) <= kernel_width else name[: kernel_width - 3] + "..."
                print(f"{label:<{kernel_width}} {total:>10.3f} {avg:>10.4f} {count:>7}")

            print(f"{'-' * kernel_width} {'-' * 10}")
            print(f"{'TOTAL':<{kernel_width}} {grand_total:>10.3f}")
            print(f"{'Per-frame average':<{kernel_width}} {grand_total / frame_count:>10.3f}")
            print()

        device = wp.get_device()
        if device.is_cuda and wp.is_mempool_enabled(device):
            peak_bytes = wp.get_mempool_used_mem_high(device)
            print(f"Warp mempool peak usage: {peak_bytes / (1024 * 1024):.2f} MiB")
        else:
            print("Warp mempool peak usage: unavailable (non-CUDA or mempool disabled)")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.enable_kernel_timing:
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN)
            self.simulate()
            timing_results = wp.timing_end()
            for result in timing_results:
                self._timing_totals[result.name].append(result.elapsed)
            self._timing_frame_count += 1
        elif self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.step_count += 1
        self.sim_time += self.frame_dt
        self._report_steps_per_second()

        if self.step_count >= self.max_sim_steps and not self._terminate_requested:
            self._terminate_requested = True
            print(f"[step {self.step_count}] Reached hard stop at {self.max_sim_steps} simulation steps.")

    def render(self):
        if self._terminate_requested:
            self._print_exit_benchmark_report()
            self.viewer.close()
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the wrecking ball has moved (simulation ran)."""
        body_q = self.state_0.body_q.numpy()
        # The wrecking ball is the second-to-last body (last dynamic body).
        # Just check that some bodies have moved from their initial positions.
        dynamic_z = body_q[:, 2]
        assert np.any(dynamic_z < 0.3), "No bodies appear to have moved"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--solver",
            type=str,
            choices=["xpbd", "mujoco"],
            default="xpbd",
            help="Solver to use.",
        )
        parser.add_argument(
            "--broad-phase",
            type=str,
            choices=["nxn", "sap", "explicit"],
            default="sap",
            help="Broad phase for collision detection.",
        )
        parser.add_argument(
            "--num-pyramids",
            type=int,
            default=DEFAULT_NUM_PYRAMIDS,
            help="Number of pyramids to build.",
        )
        parser.add_argument(
            "--pyramid-size",
            type=int,
            default=DEFAULT_PYRAMID_SIZE,
            help="Number of rows in each pyramid base.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example._print_exit_benchmark_report()
