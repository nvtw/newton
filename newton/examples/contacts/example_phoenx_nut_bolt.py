# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example PhoenX Nut and Bolt
#
# Demonstrates mesh-mesh collision using the PhoenX PGS solver.
# A nut is placed on a bolt and allowed to screw down under gravity.
#
# Command: python -m newton.examples phoenx_nut_bolt
#
###########################################################################

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples

# Assembly type for the nut and bolt
ASSEMBLY_STR = "m20_loose"

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,
    kd=1e4,
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
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.viewer = viewer
        self.test_mode = args.test

        self.num_per_world = args.num_per_world
        self.grid_x = int(np.ceil(np.sqrt(self.num_per_world)))
        self.grid_y = int(np.ceil(self.num_per_world / self.grid_x))

        self.scene_scale = 1.0
        self.ground_plane_offset = -0.01

        world_builder = self._build_nut_bolt_scene()

        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.gap = 0.001 * self.scene_scale
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_plane_offset),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=self.world_count)

        self.model = main_scene.finalize()

        # Mesh contacts can be numerous — use generous capacity
        max_contacts = max(1000 * self.world_count, 4096)
        self.model.rigid_contact_max = max_contacts

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            num_iterations=12,
            num_velocity_iterations=0,
            default_friction=0.01,
            max_contacts=max_contacts,
            num_substeps=8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        offset = 0.15 * self.scene_scale
        self.viewer.set_world_offsets((offset, offset, 0.0))
        camera_offset = np.sqrt(self.world_count) * offset
        self.viewer.set_camera(pos=wp.vec3(camera_offset, -camera_offset, 0.5 * camera_offset), pitch=-15.0, yaw=135.0)

        self._init_test_tracking()

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.gap = 0.001 * self.scene_scale

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        bolt_mesh, bolt_center = load_mesh_with_sdf(bolt_file, shape_cfg=SHAPE_CFG, center_origin=True)
        nut_mesh, nut_center = load_mesh_with_sdf(nut_file, shape_cfg=SHAPE_CFG, center_origin=True)

        spacing = 0.1 * self.scene_scale

        count = 0
        for i in range(self.grid_x):
            if count >= self.num_per_world:
                break
            for j in range(self.grid_y):
                if count >= self.num_per_world:
                    break
                x_offset = (i - (self.grid_x - 1) / 2.0) * spacing
                y_offset = (j - (self.grid_y - 1) / 2.0) * spacing

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

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self._track_test_data()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _init_test_tracking(self):
        if not self.test_mode:
            self.bolt_body_indices = None
            self.nut_body_indices = None
            return

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

        body_q = self.state_0.body_q.numpy()
        self.bolt_initial_transforms = [body_q[idx].copy() for idx in self.bolt_body_indices]
        self.nut_initial_transforms = [body_q[idx].copy() for idx in self.nut_body_indices]

        self.nut_max_rotation_change = [0.0] * len(self.nut_body_indices)
        self.nut_min_z = [body_q[idx][2] for idx in self.nut_body_indices]

    def _track_test_data(self):
        if not self.test_mode:
            return

        body_q = self.state_0.body_q.numpy()

        for i, nut_idx in enumerate(self.nut_body_indices):
            current_q = body_q[nut_idx]
            initial_q = self.nut_initial_transforms[i]

            q_current = current_q[3:7]
            q_initial = initial_q[3:7]
            dot = abs(np.dot(q_current, q_initial))
            dot = min(dot, 1.0)
            rotation_angle = 2.0 * np.arccos(dot)
            self.nut_max_rotation_change[i] = max(self.nut_max_rotation_change[i], rotation_angle)

            self.nut_min_z[i] = min(self.nut_min_z[i], current_q[2])

    def test_final(self):
        """Verify simulation state after example completes.

        - Bolts should stay approximately in place (limited displacement)
        - Nuts should rotate (thread engagement) and move slightly downward
        """
        body_q = self.state_0.body_q.numpy()

        max_bolt_displacement = 0.02
        for i, bolt_idx in enumerate(self.bolt_body_indices):
            current_pos = body_q[bolt_idx][:3]
            initial_pos = self.bolt_initial_transforms[i][:3]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_bolt_displacement, (
                f"Bolt {i}: displaced too much. "
                f"Displacement={displacement:.4f} (max allowed={max_bolt_displacement:.4f})"
            )

        min_rotation_threshold = 0.1
        for i in range(len(self.nut_body_indices)):
            max_rotation = self.nut_max_rotation_change[i]
            assert max_rotation > min_rotation_threshold, (
                f"Nut {i}: did not rotate enough. "
                f"Max rotation={np.degrees(max_rotation):.2f} degrees "
                f"(expected > {np.degrees(min_rotation_threshold):.2f} degrees)"
            )

            initial_z = self.nut_initial_transforms[i][2]
            min_z = self.nut_min_z[i]
            assert min_z < initial_z, (
                f"Nut {i}: did not move downward. Initial z={initial_z:.4f}, min z reached={min_z:.4f}"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument("--num-per-world", type=int, default=1, help="Number of assemblies per world.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
