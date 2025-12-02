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
# Example Nut and Bolt
#
# Demonstrates mesh-mesh collision between a nut and bolt from
# the Isaac Gym Factory environment assets.
# Both the nut and bolt are free bodies that fall onto a ground plane.
#
# Command: python -m newton.examples nut_bolt
#
###########################################################################

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton._src.utils.download_assets import download_git_folder

# Assembly type for the nut and bolt
ASSEMBLY_STR = "m20_loose"


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh_file: str,
    xform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    key: str | None = None,
    center_origin: bool = True,
) -> int:
    """
    Load a mesh from file and add it as a free body to the model builder.

    Args:
        builder: The ModelBuilder to add the mesh body to.
        mesh_file: Path to the mesh file (OBJ, STL, etc.).
        xform: Transform specifying position and orientation of the body.
        shape_cfg: Optional ShapeConfig for the mesh shape.
        key: Optional key/name for the body.
        center_origin: If True, center the mesh vertices at origin before adding.
    Returns:
        The body index of the created body.
    """
    # Load mesh using trimesh
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2
        vertices = vertices - center

    mesh = newton.Mesh(vertices, indices)

    body = builder.add_body(xform=xform, key=key)
    builder.add_shape_mesh(body, mesh=mesh, cfg=shape_cfg)

    print(f"Loaded mesh '{key or mesh_file}': {len(vertices)} vertices, {len(indices) // 3} triangles")

    return body


class Example:
    def __init__(self, viewer, num_worlds=1):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer

        repo_url = "https://github.com/isaac-sim/IsaacGymEnvs.git"
        print(f"Downloading assets from {repo_url}...")
        asset_path = download_git_folder(repo_url, "assets/factory/mesh/factory_nut_bolt")
        print(f"Assets downloaded to: {asset_path}")

        # Create world template with bolt and nut
        world_builder = newton.ModelBuilder()
        world_builder.rigid_contact_margin = 0.001

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            thickness=0.0, mu=0.01, sdf_max_dims=512, density=8000.0, torsional_friction=0.0, rolling_friction=0.0
        )

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        bolt_xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
        add_mesh_object(world_builder, bolt_file, bolt_xform, shape_cfg, key="bolt", center_origin=True)

        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        nut_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.07),
            wp.quat_identity(),
        )
        add_mesh_object(world_builder, nut_file, nut_xform, shape_cfg, key="nut", center_origin=True)

        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.enable_mesh_sdf_collision = True
        scene.replicate(world_builder, num_worlds=self.num_worlds)

        self.model = scene.finalize()

        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            reduce_contacts=True,
        )

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10, rigid_contact_relaxation=0.9)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.15, 0.15, 0.0))

        self.viewer.set_camera(pos=wp.vec3(0.15, -0.15, 0.12), pitch=-15.0, yaw=135.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help="Total number of simulated worlds.",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds)

    newton.examples.run(example, args)
