# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulate the USD-derived bicycle transmission with PhoenX.

Command: python -m newton.examples phoenx_bike_transmission

Geometry, normals, joint frames and drives come from BikeTransmission.usd.
The runtime uses the extracted OBJ assets and SI scene descriptors, without USD.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.phoenx.bike_transmission_scene import SCENE
from newton.solvers import SolverPhoenX

ASSETS = Path(newton.examples.get_asset_directory()) / "bike_transmission"


def _transform(values):
    return wp.transform(values[:3], values[3:])


def _load_mesh(path, roughness):
    # Exported OBJ vertices, normals, and UVs share indices, including sharp edges.
    vertices, normals, uvs, faces = [], [], [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(x) for x in fields[1:4]])
        elif fields[0] == "vt":
            uvs.append([float(x) for x in fields[1:3]])
        elif fields[0] == "f":
            faces.extend(int(x.split("/")[0]) - 1 for x in fields[1:])
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32) if normals else None,
        uvs=np.asarray(uvs, dtype=np.float32) if uvs else None,
        roughness=roughness,
    )


def build_scene(*, sdf_resolution=0, motor_off=False):
    """Build the authored closed mechanism with SDF mesh collisions in SI units."""
    scene = SCENE
    builder = newton.ModelBuilder(gravity=tuple(scene["gravity"]))
    for body in scene["bodies"]:
        builder.add_link(xform=_transform(body["pose"]), label=body["label"], is_kinematic=body["kinematic"])
    meshes = {}
    for shape in scene["shapes"]:
        key = (shape["mesh"], shape["collision"], sdf_resolution or shape["sdf_resolution"], shape["roughness"])
        if key not in meshes:
            mesh = _load_mesh(ASSETS / shape["mesh"], shape["roughness"])
            if shape["collision"]:
                mesh.build_sdf(
                    max_resolution=key[2],
                    margin=0.001,
                    cache_dir=str(Path(tempfile.gettempdir()) / "newton_bike_transmission_sdf"),
                )
            meshes[key] = mesh
        cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0 if shape["collision"] else 0.0,
            mu=0.5,
            margin=0.0,
            gap=0.0005,
            has_shape_collision=shape["collision"],
            is_visible=shape["visible"],
        )
        builder.add_shape_mesh(
            shape["body"],
            mesh=meshes[key],
            xform=wp.transform(shape["center"], wp.quat_identity()),
            cfg=cfg,
            color=shape["color"],
            label=shape["label"],
        )

    def add_source_joint(joint):
        stiffness, damping = joint["stiffness"], joint["damping"]
        if motor_off and joint["label"].endswith("/FrontGears/RevoluteJoint"):
            stiffness = damping = 0.0
        mode = newton.JointTargetMode.POSITION if stiffness else newton.JointTargetMode.VELOCITY
        parent, child = joint["parent"], joint["child"]
        parent_frame, child_frame = joint["frames"]
        return builder.add_joint_revolute(
            parent,
            child,
            parent_xform=_transform(parent_frame),
            child_xform=_transform(child_frame),
            axis={"X": newton.Axis.X, "Y": newton.Axis.Y, "Z": newton.Axis.Z}[joint["axis"]],
            target_pos=joint["target"],
            target_vel=joint["velocity"],
            target_ke=stiffness,
            target_kd=damping,
            actuator_mode=mode,
            effort_limit=float("inf"),
            velocity_limit=float("inf"),
            limit_lower=joint["lower"],
            limit_upper=joint["upper"],
            collision_filter_parent=True,
            label=joint["label"],
        )

    for joint in scene["joints"]:
        add_source_joint(joint)
    return builder


class Example:
    """Drive the chain and sprockets using mesh contacts and authored revolute joints."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.solver_stats = getattr(args, "solver_stats", False)
        self.frame_dt = 1.0 / 60.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("substeps and iterations must be positive")
        if args.contact_chunk_size < 0:
            raise ValueError("contact chunk size must be nonnegative")
        self.model = build_scene(sdf_resolution=args.sdf_resolution, motor_off=args.motor_off).finalize(
            skip_validation_joints=True
        )
        self.state = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            rigid_contact_max=32768,
            speculative_contact_gap_max=0.002,
            speculative_contact_velocity_filter=False,
            contact_reduction_voxel_depth=args.sdf_voxel_depth_contacts,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            articulation_mode="maximal",
            joint_solver="direct",
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
            velocity_iterations=1,
            parallel_contact_prepare=self.model.device.is_cuda,
            contact_chunk_size=args.contact_chunk_size,
            mass_splitting=True,
            mass_splitting_color_group_size=3,
            mass_splitting_batch_size=2,
            max_colored_partitions=8,
        )
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(0.60, -0.62, 0.43), pitch=-19.1, yaw=120.8)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(2):
            dt = self.frame_dt / 2
            self.pipeline.collide(self.state, self.contacts, dt=dt)
            self.state.clear_forces()
            self.viewer.apply_forces(self.state)
            self.solver.step(self.state, self.state, self.control, self.contacts, dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if self.solver_stats and round(self.sim_time / self.frame_dt) % 60 == 0:
            report = self.solver.step_report()
            print(
                f"PhoenX: {report.num_joints} joints, {report.num_contact_columns} contact columns, "
                f"{report.num_colors} colors; sizes={report.color_sizes}; "
                f"group sizes={report.color_group_sizes}; overflow={report.overflow_size}"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check attachment and hinge errors twice per simulated second."""
        if round(self.sim_time / self.frame_dt) % 30 == 0:
            self.test_final()

    def test_final(self):
        """Reject nonfinite state or a broken chain attachment."""
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all() and np.isfinite(self.state.body_qd.numpy()).all()
        parent, child = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        xp, xc = self.model.joint_X_p.numpy(), self.model.joint_X_c.numpy()
        worst = 0.0
        worst_angle = 0.0
        for i in range(self.model.joint_count):
            a, b = _transform(xp[i]), _transform(xc[i])
            if parent[i] >= 0:
                a = _transform(poses[parent[i]]) * a
            if child[i] >= 0:
                b = _transform(poses[child[i]]) * b
            worst = max(worst, float(wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b))))
            axis = wp.vec3(*{"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[SCENE["joints"][i]["axis"]])
            u, v = wp.transform_vector(a, axis), wp.transform_vector(b, axis)
            angle = np.arctan2(float(wp.length(wp.cross(u, v))), float(wp.dot(u, v)))
            worst_angle = max(worst_angle, angle)
        print(f"BikeTransmission maximum joint attachment error: {worst:.6g} m")
        print(f"BikeTransmission maximum hinge-axis error: {np.rad2deg(worst_angle):.6g} degrees")
        assert worst_angle < 0.02, f"Misaligned transmission hinge: {worst_angle:.6g} rad"
        assert worst < 0.002, f"Broken transmission joint: {worst:.6g} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix")
        parser.add_argument(
            "--solver-stats", action="store_true", help="Print coloring statistics once per simulated second."
        )
        parser.add_argument(
            "--contact-chunk-size",
            type=int,
            default=64,
            help="Maximum contact rows per solver column (0 keeps whole shape pairs).",
        )
        parser.add_argument("--iterations", type=int, default=2, help="Solver iterations per physics substep.")
        parser.add_argument("--substeps", type=int, default=16, help="Physics substeps per 120 Hz contact refresh.")
        parser.add_argument(
            "--sdf-voxel-depth-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Retain deepest-per-voxel SDF contacts during contact reduction.",
        )
        parser.add_argument(
            "--sdf-resolution", type=int, default=0, help="Override source SDF resolutions (0 uses authored values)."
        )
        parser.add_argument("--motor-off", action="store_true", help="Disable the crank drive for passive diagnostics.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
