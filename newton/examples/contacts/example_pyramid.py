# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Mesh Cube Pyramid
#
# Builds pyramids of mesh cubes (triangle meshes) with a wrecking ball
# on a ramp to stress-test narrow-phase contact generation with mesh
# collision shapes.
#
# Command: python -m newton.examples pyramid
#
###########################################################################

import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_PYRAMIDS = 3
DEFAULT_PYRAMID_SIZE = 20
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
Y_STACK = 15.0

WRECKING_BALL_RADIUS = 2.0
WRECKING_BALL_DENSITY_MULT = 100.0
RAMP_LENGTH = 20.0
RAMP_WIDTH = 5.0
RAMP_THICKNESS = 0.5

USE_MESH_CUBES = True
USE_SDF = True
SDF_MAX_RESOLUTION = 256

MAX_STEPS = 2000
PRINT_INTERVAL = 100
ENABLE_KERNEL_TIMING = True

XPBD_ITERATIONS = 2
XPBD_CONTACT_RELAXATION = 0.8


def _create_box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    """Create a cube triangle mesh with duplicated vertices for flat shading.

    Each face has its own 4 vertices so normals are not shared across edges,
    giving crisp flat-shaded faces. Each face is split into two right
    triangles of equal area.
    """
    # fmt: off
    vertices = np.array(
        [
            # -Z face (0-3)
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            # +Z face (4-7)
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
            # -X face (8-11)
            [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy,  hz], [-hx,  hy, -hz],
            # +X face (12-15)
            [ hx, -hy, -hz], [ hx,  hy, -hz], [ hx,  hy,  hz], [ hx, -hy,  hz],
            # -Y face (16-19)
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx, -hy,  hz], [-hx, -hy,  hz],
            # +Y face (20-23)
            [-hx,  hy, -hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
             0,  2,  1,  0,  3,  2,  # -Z
             4,  5,  6,  4,  6,  7,  # +Z
             8,  9, 10,  8, 10, 11,  # -X
            12, 13, 14, 12, 14, 15,  # +X
            16, 17, 18, 16, 18, 19,  # -Y
            20, 21, 22, 20, 22, 23,  # +Y
        ],
        dtype=np.int32,
    )
    # fmt: on
    mesh = newton.Mesh(vertices, indices)
    if USE_SDF:
        mesh.build_sdf(max_resolution=SDF_MAX_RESOLUTION)
    return mesh


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.test_mode = args.test
        self.world_count = args.world_count

        num_pyramids = args.num_pyramids
        pyramid_size = args.pyramid_size

        builder = newton.ModelBuilder()
        builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        cube_mesh = _create_box_mesh(CUBE_HALF, CUBE_HALF, CUBE_HALF) if USE_MESH_CUBES else None

        box_count = 0
        top_body_indices = []
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
                    if USE_MESH_CUBES:
                        builder.add_shape_mesh(body, mesh=cube_mesh)
                    else:
                        builder.add_shape_box(body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF)
                    if level == pyramid_size - 1:
                        top_body_indices.append(body)
                    box_count += 1

        self.box_count = box_count
        self.cube_body_indices = list(range(box_count))
        self.top_body_indices = top_body_indices
        shape_label = "mesh cubes" if USE_MESH_CUBES else "boxes"
        print(f"Built {num_pyramids} pyramids x {pyramid_size} rows = {box_count} {shape_label}")

        if not self.test_mode:
            # Wrecking ball
            ramp_height = 8.4
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

        if self.world_count > 1:
            main_builder = newton.ModelBuilder()
            main_builder.replicate(builder, world_count=self.world_count)
            self.model = main_builder.finalize()
        else:
            self.model = builder.finalize()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
        )

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=XPBD_ITERATIONS,
            rigid_contact_relaxation=XPBD_CONTACT_RELAXATION,
            rigid_contact_max_depenetration_velocity=5.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.top_initial_positions = self.state_0.body_q.numpy()[:, :3].copy()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        cam_dist = max(pyramid_height, num_pyramids * PYRAMID_SPACING * 0.3)
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.4),
            pitch=-15.0,
            yaw=135.0,
        )

        self.step_count = 0
        self.body_count = self.model.body_count
        self._timing_totals: dict[str, list[float]] = defaultdict(list)
        self._timing_frame_count = 0
        self._exit_report_printed = False
        self._terminate_requested = False
        self.last_perf_step = 0
        self.last_perf_time = time.perf_counter()
        self._spike_count = 0  # steps where max_lin > 2 m/s (after step 1000)
        self._settled_max_lin = 0.0  # worst max_lin in last 500 steps

        if ENABLE_KERNEL_TIMING:
            self.graph = None
        else:
            self.capture()

    def _cube_velocity_stats(self) -> tuple[float, float, float, float]:
        """Return max and mean linear/angular speeds over pyramid cubes only."""
        body_qd = self.state_0.body_qd.numpy()[self.cube_body_indices]
        linear_speeds = np.linalg.norm(body_qd[:, 3:6], axis=1)
        angular_speeds = np.linalg.norm(body_qd[:, 0:3], axis=1)
        return (
            float(np.max(linear_speeds)),
            float(np.max(angular_speeds)),
            float(np.mean(linear_speeds)),
            float(np.mean(angular_speeds)),
        )

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
        if ENABLE_KERNEL_TIMING:
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN)
            self.simulate()
            for result in wp.timing_end():
                self._timing_totals[result.name].append(result.elapsed)
            self._timing_frame_count += 1
        elif self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.step_count += 1

        if self.step_count % PRINT_INTERVAL == 0:
            max_lin, max_ang, mean_lin, mean_ang = self._cube_velocity_stats()
            now = time.perf_counter()
            elapsed = now - self.last_perf_time
            steps_elapsed = self.step_count - self.last_perf_step
            sps = steps_elapsed / elapsed if elapsed > 0.0 else 0.0
            print(
                f"Step {self.step_count}: "
                f"max lin = {max_lin:.4f}, max ang = {max_ang:.4f}, "
                f"mean lin = {mean_lin:.4f}, mean ang = {mean_ang:.4f} "
                f"[m/s, rad/s], {sps:.2f} steps/s"
            )
            self.last_perf_step = self.step_count
            self.last_perf_time = now

        # Track jitter metrics (after wrecking ball settles)
        if self.step_count > 1000 and self.step_count % PRINT_INTERVAL == 0:
            max_lin, _, _, _ = self._cube_velocity_stats()
            if max_lin > 2.0:
                self._spike_count += 1
            if self.step_count > MAX_STEPS - 500:
                self._settled_max_lin = max(self._settled_max_lin, max_lin)

        if self.step_count >= MAX_STEPS and not self._terminate_requested:
            self._terminate_requested = True
            print(f"Reached {MAX_STEPS} steps, stopping.")
            print(
                f"Jitter: {self._spike_count} spikes (>2 m/s after step 1000), "
                f"settled max = {self._settled_max_lin:.4f} m/s"
            )

    def _print_kernel_timing_report(self):
        if self._exit_report_printed:
            return
        self._exit_report_printed = True

        if self._timing_frame_count > 0:
            frame_count = self._timing_frame_count
            width = 110
            kernel_width = width - 30
            print(f"\n{'=' * width}")
            print(f"  Kernel timing report ({frame_count} frames, {self.box_count} cubes x {self.world_count} worlds)")
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

    def render(self):
        if self._terminate_requested:
            self._print_kernel_timing_report()
            self.viewer.close()
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify pyramid top cubes remain near their initial positions.

        In test mode the wrecking ball is omitted so the pyramids should
        settle under gravity without toppling.  Each top cube must stay
        within ``max_displacement`` of its initial position.
        """
        body_q = self.state_0.body_q.numpy()
        max_displacement = 0.5  # [m]
        for idx in self.top_body_indices:
            current_pos = body_q[idx, :3]
            initial_pos = self.top_initial_positions[idx]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_displacement, (
                f"Top cube body {idx}: displaced {displacement:.4f} m (max allowed {max_displacement:.4f} m)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap")
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
    example._print_kernel_timing_report()
