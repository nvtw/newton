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
# Example SDF Mesh Collision
#
# Demonstrates mesh-mesh collision using SDF (Signed Distance Field).
# Supports two scenes "nut_bolt" and "gears":
#
# Command: python -m newton.examples sdf --scene nut_bolt
#          python -m newton.examples sdf --scene gears
#
###########################################################################

import time
from collections import defaultdict

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples

# Assembly type for the nut and bolt
ASSEMBLY_STR = "m20_loose"

# Gear mesh files available (filename -> key)
GEAR_FILES = [
    ("factory_gear_base_loose_space_5e-4_subdiv_4x.obj", "gear_base"),
    ("factory_gear_large_space_5e-4.obj", "gear_large"),
    ("factory_gear_medium_space_5e-4.obj", "gear_medium"),
    ("factory_gear_small_space_5e-4.obj", "gear_small"),
]
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
ISAACGYM_GEARS_FOLDER = "assets/factory/mesh/factory_gears"

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,  # Contact stiffness for MuJoCo solver
    kd=1e4,  # Contact damping
    gap=0.005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)
MESH_SDF_MAX_RESOLUTION = 512
MESH_SDF_NARROW_BAND_RANGE = (-0.005, 0.005)


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    label: str | None = None,
    center_vec: wp.vec3 | None = None,
    scale: float = 1.0,
) -> int:
    if center_vec is not None:
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    if label == "gear_base":
        body = -1
        builder.add_shape_mesh(
            body, mesh=mesh, scale=(scale, scale, scale), xform=transform, cfg=shape_cfg, label=label
        )
    else:
        body = builder.add_body(label=label, xform=transform)
        builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


def load_mesh_with_sdf(
    mesh_file: str,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    center_origin: bool = True,
) -> tuple[newton.Mesh, wp.vec3]:
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)
    center_vec = wp.vec3(0.0, 0.0, 0.0)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2
        vertices = vertices - center
        center_vec = wp.vec3(center)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=shape_cfg.gap if shape_cfg and shape_cfg.gap is not None else 0.05,
    )
    return mesh, center_vec


class Example:
    def __init__(self, viewer, world_count=1, num_per_world=1, scene="nut_bolt", solver="xpbd", test_mode=False):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Use more substeps for gears scene to improve stability
        self.sim_substeps = 50 if scene == "gears" else 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = world_count
        self.viewer = viewer
        self.scene = scene
        self.solver_type = solver
        self.test_mode = test_mode

        # XPBD contact correction (0.0 = no correction, 1.0 = full correction)
        self.xpbd_contact_relaxation = 0.8

        # Scene scaling factor (1.0 = original size)
        self.scene_scale = 5.0

        # Ground plane offset (negative = below origin)
        self.ground_plane_offset = -0.01

        # Grid dimensions for nut/bolt scene (number of assemblies in X and Y)
        self.num_per_world = num_per_world
        self.grid_x = int(np.ceil(np.sqrt(num_per_world)))
        self.grid_y = int(np.ceil(num_per_world / self.grid_x))

        # Maximum number of rigid contacts to allocate (limits memory usage).
        # Use a per-world budget so default world_count=100 scales appropriately.
        self.rigid_contact_max = 500 * self.world_count

        # Broad phase mode: NXN (O(N²)), SAP (O(N log N)), EXPLICIT (precomputed pairs)
        self.broad_phase = "sap"

        if scene == "nut_bolt":
            world_builder = self._build_nut_bolt_scene()
        elif scene == "gears":
            world_builder = self._build_gears_scene()
        else:
            raise ValueError(f"Unknown scene: {scene}")

        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.gap = 0.01
        # Add ground plane with offset (plane equation: z = offset)
        # For plane equation n·x + d = 0, with n=(0,0,1): z + d = 0, so z = -d
        # Therefore, to get plane at z = offset, we need d = -offset
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_plane_offset),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=self.world_count)

        self.model = main_scene.finalize()

        # Keep model and pipeline contact capacities aligned.
        self.model.rigid_contact_max = self.rigid_contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase=self.broad_phase,
        )

        # Create solver based on user choice
        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=10,
                rigid_contact_relaxation=self.xpbd_contact_relaxation,
            )
        elif self.solver_type == "mujoco":
            num_per_world = self.collision_pipeline.rigid_contact_max // self.world_count
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=num_per_world,
                nconmax=num_per_world,
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

        if self.scene == "gears":
            joint_child = self.model.joint_child.numpy()
            joint_qd_start = self.model.joint_qd_start.numpy()
            joint_f = self.control.joint_f.numpy()
            for body_idx, lbl in enumerate(self.model.body_label):
                if lbl.endswith("/gear_large") or lbl == "gear_large":
                    for j in range(self.model.joint_count):
                        if joint_child[j] == body_idx:
                            qd_start = int(joint_qd_start[j])
                            joint_f[qd_start + 5] = 2.0  # z-axis torque (N·m)
                            break
            self.control.joint_f.assign(joint_f)

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        if scene == "nut_bolt":
            offset = 0.15 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.12 * self.scene_scale), pitch=-15.0, yaw=135.0)
        else:  # gears
            offset = 0.25 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.2 * self.scene_scale), pitch=-25.0, yaw=135.0)

        # Initialize test tracking data (only in test mode for nut_bolt scene)
        self._init_test_tracking()
        self._init_nut_progress_tracking()

        # CUDA graph launches do not provide per-kernel timings.
        # Keep direct simulation path enabled so timing_begin/timing_end captures kernels.
        if self.enable_kernel_timing:
            self.graph = None
        else:
            self.capture()

    @staticmethod
    def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
        """Return world-frame yaw angle [rad] from quaternion [x, y, z, w]."""
        x, y, z, w = quat_xyzw
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _find_nut_body_indices_and_labels(self) -> tuple[list[int], list[str]]:
        """Collect body indices and labels for all nuts across all worlds."""
        nut_indices = []
        nut_labels = []
        for body_idx, label in enumerate(self.model.body_label):
            if label == "nut_0_0" or label.startswith("nut_") or "/nut_" in label:
                nut_indices.append(body_idx)
                nut_labels.append(label)
        return nut_indices, nut_labels

    def _init_nut_progress_tracking(self):
        """Initialize periodic nut motion consistency checks."""
        self.step_count = 0
        self.max_sim_steps = 2000
        self.enable_kernel_timing = True
        self._terminate_requested = False
        self._exit_report_printed = False
        self.nut_log_period_steps = 60
        self.nut_motion_check_period_dt = self.nut_log_period_steps * self.frame_dt
        self.last_perf_step = 0
        self.last_perf_time = time.perf_counter()
        self._timing_totals: dict[str, list[float]] = defaultdict(list)
        self._timing_frame_count = 0

        # Tolerances for consistency checks every 60 steps:
        # deviation tolerance = absolute + relative * |expected|
        self.nut_rot_speed_abs_tol_deg_s = 30.0
        self.nut_rot_speed_rel_tol = 0.35
        self.nut_z_speed_abs_tol_m_s = 0.10
        self.nut_z_speed_rel_tol = 0.40

        if self.scene != "nut_bolt":
            self.nut_log_indices = []
            self.nut_log_labels = []
            self.nut_prev_yaw = np.array([], dtype=np.float64)
            self.nut_cumulative_yaw = np.array([], dtype=np.float64)
            self.nut_current_z = np.array([], dtype=np.float64)
            self.nut_last_check_angle = np.array([], dtype=np.float64)
            self.nut_last_check_z = np.array([], dtype=np.float64)
            return

        self.nut_log_indices, self.nut_log_labels = self._find_nut_body_indices_and_labels()

        if not self.nut_log_indices:
            self.nut_prev_yaw = np.array([], dtype=np.float64)
            self.nut_cumulative_yaw = np.array([], dtype=np.float64)
            self.nut_current_z = np.array([], dtype=np.float64)
            self.nut_last_check_angle = np.array([], dtype=np.float64)
            self.nut_last_check_z = np.array([], dtype=np.float64)
            return

        body_q = self.state_0.body_q.numpy()
        initial_yaw = []
        initial_z = []
        for idx in self.nut_log_indices:
            pose = body_q[idx]
            initial_yaw.append(self._yaw_from_quat_xyzw(pose[3:7]))
            initial_z.append(float(pose[2]))

        self.nut_prev_yaw = np.array(initial_yaw, dtype=np.float64)
        self.nut_cumulative_yaw = np.zeros_like(self.nut_prev_yaw)
        self.nut_current_z = np.array(initial_z, dtype=np.float64)
        self.nut_last_check_angle = self.nut_cumulative_yaw.copy()
        self.nut_last_check_z = self.nut_current_z.copy()

    def _update_nut_motion_tracking(self):
        """Accumulate per-nut yaw and z tracking at each simulation step."""
        if self.scene != "nut_bolt" or len(self.nut_log_indices) == 0:
            return

        body_q = self.state_0.body_q.numpy()
        for i, idx in enumerate(self.nut_log_indices):
            pose = body_q[idx]
            yaw = self._yaw_from_quat_xyzw(pose[3:7])

            # Unwrap yaw increment so accumulated angle can exceed +/-180 deg.
            delta = np.arctan2(np.sin(yaw - self.nut_prev_yaw[i]), np.cos(yaw - self.nut_prev_yaw[i]))
            self.nut_cumulative_yaw[i] += delta
            self.nut_prev_yaw[i] = yaw
            self.nut_current_z[i] = float(pose[2])

    def _check_nut_motion_progress(self):
        """Check nut rotation/downward speeds and print only outliers."""
        if self.scene != "nut_bolt" or self.step_count % self.nut_log_period_steps != 0:
            return

        if len(self.nut_log_indices) == 0:
            return

        if self.nut_motion_check_period_dt <= 0.0:
            return

        rot_speed_deg_s = np.degrees((self.nut_cumulative_yaw - self.nut_last_check_angle) / self.nut_motion_check_period_dt)
        z_speed_m_s = (self.nut_current_z - self.nut_last_check_z) / self.nut_motion_check_period_dt

        expected_rot_speed = float(np.median(rot_speed_deg_s))
        expected_z_speed = float(np.median(z_speed_m_s))

        rot_tol = self.nut_rot_speed_abs_tol_deg_s + self.nut_rot_speed_rel_tol * abs(expected_rot_speed)
        z_tol = self.nut_z_speed_abs_tol_m_s + self.nut_z_speed_rel_tol * abs(expected_z_speed)

        deviating_nuts = []
        for i, label in enumerate(self.nut_log_labels):
            rot_dev = float(rot_speed_deg_s[i] - expected_rot_speed)
            z_dev = float(z_speed_m_s[i] - expected_z_speed)
            if abs(rot_dev) > rot_tol or abs(z_dev) > z_tol:
                deviating_nuts.append((label, rot_speed_deg_s[i], rot_dev, z_speed_m_s[i], z_dev))

        if deviating_nuts:
            lines = [
                (
                    f"[step {self.step_count}] Nut motion deviation: {len(deviating_nuts)} / "
                    f"{len(self.nut_log_labels)} nuts outside tolerance"
                ),
                (
                    "  expected: "
                    f"rot={expected_rot_speed:.2f} deg/s +/- {rot_tol:.2f}, "
                    f"z={expected_z_speed:.5f} m/s +/- {z_tol:.5f}"
                ),
            ]
            for label, rot_speed, rot_dev, z_speed, z_dev in deviating_nuts:
                lines.append(
                    "  "
                    + (
                        f"{label}: rot={float(rot_speed):.2f} deg/s (dev={rot_dev:+.2f}), "
                        f"z={float(z_speed):.5f} m/s (dev={z_dev:+.5f})"
                    )
                )
            print("\n".join(lines))

        self.nut_last_check_angle = self.nut_cumulative_yaw.copy()
        self.nut_last_check_z = self.nut_current_z.copy()

    def _report_steps_per_second(self):
        """Report simulation throughput in steps per second every check period."""
        if self.step_count % self.nut_log_period_steps != 0:
            return

        now = time.perf_counter()
        elapsed = now - self.last_perf_time
        steps_elapsed = self.step_count - self.last_perf_step
        if elapsed > 0.0 and steps_elapsed > 0:
            steps_per_second = steps_elapsed / elapsed
            print(f"[step {self.step_count}] throughput={steps_per_second:.2f} steps/s")

        self.last_perf_step = self.step_count
        self.last_perf_time = now

    def _print_exit_benchmark_report(self):
        """Print kernel timing summary and peak memory usage before exit."""
        if self._exit_report_printed:
            return
        self._exit_report_printed = True

        if self._timing_frame_count > 0:
            frame_count = self._timing_frame_count
            width = 110
            kernel_width = width - 30
            print(f"\n{'=' * width}")
            print(f"  Kernel timing report ({frame_count} frames)")
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

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.gap = 0.01 * self.scene_scale

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        bolt_mesh, bolt_center = load_mesh_with_sdf(bolt_file, shape_cfg=SHAPE_CFG, center_origin=True)
        nut_mesh, nut_center = load_mesh_with_sdf(nut_file, shape_cfg=SHAPE_CFG, center_origin=True)

        # Spacing between assemblies in the grid
        spacing = 0.1 * self.scene_scale

        # Create grid of nut/bolt assemblies
        count = 0
        for i in range(self.grid_x):
            if count >= self.num_per_world:
                break
            for j in range(self.grid_y):
                if count >= self.num_per_world:
                    break
                # Center the grid around origin
                x_offset = (i - (self.grid_x - 1) / 2.0) * spacing
                y_offset = (j - (self.grid_y - 1) / 2.0) * spacing

                # Add bolt at grid position
                bolt_xform = wp.transform(wp.vec3(x_offset, y_offset, 0.0 * self.scene_scale), wp.quat_identity())
                add_mesh_object(
                    world_builder,
                    bolt_mesh,
                    bolt_xform,
                    SHAPE_CFG,
                    label=f"bolt_{i}_{j}",
                    center_vec=bolt_center * self.scene_scale,
                    scale=self.scene_scale,
                )

                # Add nut above bolt at grid position
                nut_xform = wp.transform(
                    wp.vec3(x_offset, y_offset, 0.041 * self.scene_scale),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
                )
                add_mesh_object(
                    world_builder,
                    nut_mesh,
                    nut_xform,
                    SHAPE_CFG,
                    label=f"nut_{i}_{j}",
                    center_vec=nut_center * self.scene_scale,
                    scale=self.scene_scale,
                )
                count += 1

        return world_builder

    def _build_gears_scene(self) -> newton.ModelBuilder:
        print("Downloading gear assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_GEARS_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.gap = 0.003 * self.scene_scale

        for _, (gear_filename, gear_key) in enumerate(GEAR_FILES):
            gear_file = str(asset_path / gear_filename)
            gear_mesh, gear_center = load_mesh_with_sdf(gear_file, shape_cfg=SHAPE_CFG, center_origin=True)
            gear_xform = wp.transform(wp.vec3(0.0, 0.0, 0.01) * self.scene_scale, wp.quat_identity())
            add_mesh_object(
                world_builder,
                gear_mesh,
                gear_xform,
                SHAPE_CFG,
                label=gear_key,
                center_vec=gear_center * self.scene_scale,
                scale=self.scene_scale,
            )

        return world_builder

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            # self.collision_pipeline.collide(self.state_0, self.contacts)
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

        # Track transforms for test validation
        self._track_test_data()
        self._update_nut_motion_tracking()
        self._check_nut_motion_progress()
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

    def _init_test_tracking(self):
        """Initialize tracking data for test validation (nut_bolt scene only)."""
        if not self.test_mode or self.scene != "nut_bolt":
            self.bolt_body_indices = None
            self.nut_body_indices = None
            return

        # Find bolt and nut body indices by key
        self.bolt_body_indices = []
        self.nut_body_indices = []

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                bolt_key = f"bolt_{i}_{j}"
                nut_key = f"nut_{i}_{j}"

                if bolt_key in self.model.body_label:
                    self.bolt_body_indices.append(self.model.body_label.index(bolt_key))
                if nut_key in self.model.body_label:
                    self.nut_body_indices.append(self.model.body_label.index(nut_key))

        # Store initial transforms
        body_q = self.state_0.body_q.numpy()
        self.bolt_initial_transforms = [body_q[idx].copy() for idx in self.bolt_body_indices]
        self.nut_initial_transforms = [body_q[idx].copy() for idx in self.nut_body_indices]

        # Track maximum rotation change and z displacement for nuts
        self.nut_max_rotation_change = [0.0] * len(self.nut_body_indices)
        self.nut_min_z = [body_q[idx][2] for idx in self.nut_body_indices]

    def _track_test_data(self):
        """Track transforms for test validation (called each step in test mode)."""
        if not self.test_mode or self.scene != "nut_bolt":
            return

        body_q = self.state_0.body_q.numpy()

        # Track nut rotation and z position
        for i, nut_idx in enumerate(self.nut_body_indices):
            current_q = body_q[nut_idx]
            initial_q = self.nut_initial_transforms[i]

            # Compute rotation change using quaternion dot product
            # |q1 · q2| = cos(theta/2), where theta is the angle between orientations
            q_current = current_q[3:7]  # quaternion part (x, y, z, w)
            q_initial = initial_q[3:7]
            dot = abs(np.dot(q_current, q_initial))
            dot = min(dot, 1.0)  # Clamp for numerical stability
            rotation_angle = 2.0 * np.arccos(dot)
            self.nut_max_rotation_change[i] = max(self.nut_max_rotation_change[i], rotation_angle)

            # Track minimum z (nuts should move down)
            self.nut_min_z[i] = min(self.nut_min_z[i], current_q[2])

    def test_final(self):
        """Verify simulation state after example completes.

        For nut_bolt scene:
        - Bolts should stay approximately in place (limited displacement)
        - Nuts should rotate (thread engagement) and move slightly downward
        """
        if self.scene != "nut_bolt":
            # For gears scene, just verify simulation ran without error
            return

        body_q = self.state_0.body_q.numpy()

        # Check bolts stayed in place
        max_bolt_displacement = 0.01 * self.scene_scale  # 1cm scaled
        for i, bolt_idx in enumerate(self.bolt_body_indices):
            current_pos = body_q[bolt_idx][:3]
            initial_pos = self.bolt_initial_transforms[i][:3]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_bolt_displacement, (
                f"Bolt {i}: displaced too much. "
                f"Displacement={displacement:.4f} (max allowed={max_bolt_displacement:.4f})"
            )

        # Check nuts rotated and moved down
        min_rotation_threshold = 0.1  # At least ~5.7 degrees of rotation
        for i in range(len(self.nut_body_indices)):
            # Check rotation occurred
            max_rotation = self.nut_max_rotation_change[i]
            assert max_rotation > min_rotation_threshold, (
                f"Nut {i}: did not rotate enough. "
                f"Max rotation={np.degrees(max_rotation):.2f} degrees "
                f"(expected > {np.degrees(min_rotation_threshold):.2f} degrees)"
            )

            # Check nut moved downward (min_z should be less than initial z)
            initial_z = self.nut_initial_transforms[i][2]
            min_z = self.nut_min_z[i]
            assert min_z < initial_z, (
                f"Nut {i}: did not move downward. Initial z={initial_z:.4f}, min z reached={min_z:.4f}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--world-count",
        type=int,
        default=100,
        help="Total number of simulated worlds.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=["nut_bolt", "gears"],
        default="nut_bolt",
        help="Scene to run: 'nut_bolt' or 'gears'.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["xpbd", "mujoco"],
        default="mujoco",
        help="Solver to use: 'xpbd' (Extended Position-Based Dynamics) or 'mujoco' (MuJoCo constraint solver).",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        world_count=args.world_count,
        scene=args.scene,
        solver=args.solver,
        test_mode=args.test,
    )

    newton.examples.run(example, args)
    example._print_exit_benchmark_report()
