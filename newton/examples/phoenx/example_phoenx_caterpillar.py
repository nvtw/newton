# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulate the USD-derived full-scale Caterpillar excavator with PhoenX.

Command: python -m newton.examples phoenx_caterpillar

Runtime geometry, materials, body poses, and joints come from an SI descriptor
and extracted OBJ assets; the source USD is not read at runtime.
"""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.phoenx.caterpillar_scene import SCENE
from newton.solvers import SolverPhoenX

ASSETS = Path(newton.examples.get_asset_directory()) / "caterpillar"

STEEL_DENSITY = 7850.0
# The collision meshes close over hollow frame and machinery volumes. This
# effective density gives the full model the documented 390F L operating mass
# while retaining solid-steel mass and inertia for the track-link assemblies.
STRUCTURE_EFFECTIVE_DENSITY = 1470.0
TRACK_PIN_FRICTION = 100.0


def _is_track_link(label):
    return "/left_track_chain/" in label or "/right_track_chain/" in label


def _body_density(label):
    """Return solid or effective SI density for the imported collision volumes."""
    return STEEL_DENSITY if _is_track_link(label) else STRUCTURE_EFFECTIVE_DENSITY


def _transform(values):
    return wp.transform(values[:3], values[3:])


def _shape_appearance(shape):
    """Resolve source material classes into Newton's portable mesh appearance."""
    if shape.get("material") == "glass":
        return (0.18, 0.28, 0.36), 0.03, 0.0, 0.28
    if shape.get("material") == "chrome":
        return (0.78, 0.8, 0.82), 0.08, 1.0, 1.0
    return shape["color"], shape["roughness"], 0.0, 1.0


def _load_mesh(path, roughness, metallic=0.0, opacity=1.0):
    """Load the compact OBJ representation while retaining split normals and UVs."""
    vertices, normals, uvs, faces = [], [], [], []
    with gzip.open(path, "rt") as stream:
        for line in stream:
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
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32) if uvs else None,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
    )


@wp.kernel
def _contact_separation(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    separation: wp.array[wp.float32],
):
    i = wp.tid()
    body0 = shape_body[shape0[i]]
    body1 = shape_body[shape1[i]]
    position0 = point0[i]
    position1 = point1[i]
    if body0 >= 0:
        position0 = wp.transform_point(body_q[body0], position0)
    if body1 >= 0:
        position1 = wp.transform_point(body_q[body1], position1)
    separation[i] = wp.dot(position1 - position0, normal[i]) - margin0[i] - margin1[i]


def build_scene(*, sdf_resolution=128, motor_off=False, track_pin_friction=TRACK_PIN_FRICTION):
    """Build the excavator, both closed track loops, hydraulics, and ground in SI units."""
    builder = newton.ModelBuilder(gravity=tuple(SCENE["gravity"]))
    for body in SCENE["bodies"]:
        builder.add_link(xform=_transform(body["pose"]), label=body["label"], is_kinematic=body["kinematic"])

    meshes = {}
    for shape in SCENE["shapes"]:
        resolution = sdf_resolution or shape["sdf_resolution"]
        color, roughness, metallic, opacity = _shape_appearance(shape)
        key = (shape["mesh"], shape["collision"], shape["approximation"], resolution, roughness, metallic, opacity)
        if key not in meshes:
            mesh = _load_mesh(ASSETS / shape["mesh"], roughness, metallic, opacity)
            if shape["collision"] and shape["approximation"] == "sdf":
                mesh.build_sdf(
                    max_resolution=resolution,
                    margin=0.005,
                    cache_dir=str(Path(tempfile.gettempdir()) / "newton_caterpillar_sdf"),
                )
            meshes[key] = mesh
        cfg = newton.ModelBuilder.ShapeConfig(
            density=_body_density(SCENE["bodies"][shape["body"]]["label"]) if shape["collision"] else 0.0,
            mu=shape["friction"],
            margin=0.0,
            gap=0.002,
            has_shape_collision=shape["collision"],
            is_visible=shape["visible"],
        )
        add_shape = builder.add_shape_convex_hull if shape["approximation"] == "convexHull" else builder.add_shape_mesh
        add_shape(
            shape["body"],
            mesh=meshes[key],
            xform=wp.transform(shape["center"], wp.quat_identity()),
            cfg=cfg,
            color=color,
            label=shape["label"],
        )

    axes = {"X": newton.Axis.X, "Y": newton.Axis.Y, "Z": newton.Axis.Z}
    for joint in SCENE["joints"]:
        common = {
            "parent": joint["parent"],
            "child": joint["child"],
            "parent_xform": _transform(joint["frames"][0]),
            "child_xform": _transform(joint["frames"][1]),
            "collision_filter_parent": not joint["collision_enabled"],
            "label": joint["label"],
        }
        if joint["type"] == "fixed":
            builder.add_joint_fixed(**common)
            continue
        stiffness = 0.0 if motor_off else joint["stiffness"]
        damping = 0.0 if motor_off else joint["damping"]
        mode = newton.JointTargetMode.POSITION if stiffness else newton.JointTargetMode.VELOCITY
        dof = {
            **common,
            "axis": axes[joint["axis"]],
            "target_pos": joint["target"],
            "target_vel": joint["velocity"],
            "target_ke": stiffness,
            "target_kd": damping,
            "actuator_mode": mode,
            "effort_limit": float("inf"),
            "velocity_limit": float("inf"),
            "limit_lower": joint["lower"],
            "limit_upper": joint["upper"],
            "friction": track_pin_friction if _is_track_link(joint["label"]) else 0.0,
        }
        if joint["type"] == "revolute":
            builder.add_joint_revolute(**dof)
        elif joint["type"] == "prismatic":
            builder.add_joint_prismatic(**dof)
        else:
            raise ValueError(f"Unsupported extracted joint type: {joint['type']}")
    builder.add_ground_plane(height=SCENE["ground_height"], cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    return builder


class Example:
    """Run the full-scale excavator and its two articulated track loops."""

    color_group_size = 2
    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("substeps and iterations must be positive")
        if args.contact_chunk_size < 0:
            raise ValueError("contact chunk size must be nonnegative")
        if args.track_pin_friction < 0.0:
            raise ValueError("track pin friction must be nonnegative")
        self.model = build_scene(
            sdf_resolution=args.sdf_resolution,
            motor_off=args.motor_off,
            track_pin_friction=args.track_pin_friction,
        ).finalize(skip_validation_joints=True)
        self.state = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            rigid_contact_max=65536,
            speculative_contact_gap_max=0.01,
            speculative_contact_velocity_filter=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            joint_mode="maximal_direct",
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
            velocity_iterations=1,
            parallel_contact_prepare=self.model.device.is_cuda,
            contact_chunk_size=args.contact_chunk_size,
            mass_splitting=True,
            mass_splitting_color_group_size=self.color_group_size,
            mass_splitting_batch_size=2,
            max_colored_partitions=8,
        )
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(13.6, -14.7, 9.1), pitch=-19.0, yaw=128.3)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        overlap = self.overlap_simulation_render and self.viewer.supports_simulation_render_overlap
        self._render_states = (self.model.state(), self.model.state()) if overlap else None
        self._render_state_done = tuple(
            wp.Event(self.model.device) if self.model.device.is_cuda else None for _ in range(2)
        )
        self._render_contacts = [None, None]
        self._render_contact_snapshot = None
        self._render_state_index = 0
        self._render_state_prepared = False
        self._render_time = self.sim_time

    def prepare_render_state(self):
        """Copy a device-resident pose snapshot before asynchronous physics."""
        self._render_state_index = 1 - self._render_state_index
        done = self._render_state_done[self._render_state_index]
        if done is not None:
            wp.wait_event(done)
        wp.copy(self._render_states[self._render_state_index].body_q, self.state.body_q)
        self._render_contact_snapshot = None
        if self.viewer.show_contacts:
            contacts = self._render_contacts[self._render_state_index]
            if contacts is None:
                contacts = newton.Contacts(
                    self.contacts.rigid_contact_max,
                    0,
                    device=self.contacts.device,
                    requested_attributes={"force"} if self.contacts.force is not None else None,
                )
                self._render_contacts[self._render_state_index] = contacts
            for name in (
                "rigid_contact_count",
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_offset0",
                "rigid_contact_normal",
                "force",
            ):
                source = getattr(self.contacts, name)
                if source is not None:
                    wp.copy(getattr(contacts, name), source)
            self._render_contact_snapshot = contacts
        self._render_time = self.sim_time
        self._render_state_prepared = True

    def simulate(self):
        for _ in range(2):
            dt = self.frame_dt / 2.0
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

    def render(self):
        state = self._render_states[self._render_state_index] if self._render_state_prepared else self.state
        self.viewer.begin_frame(self._render_time if self._render_state_prepared else self.sim_time)
        self.viewer.log_state(state)
        contacts = self._render_contact_snapshot if self._render_state_prepared else self.contacts
        if contacts is not None or not self.viewer.show_contacts:
            self.viewer.log_contacts(contacts if contacts is not None else self.contacts, state)
        if self._render_state_prepared:
            done = self._render_state_done[self._render_state_index]
            if done is not None:
                wp.record_event(done)
        self.viewer.end_frame()
        self._render_state_prepared = False

    def test_post_step(self):
        """Check mechanism integrity twice per simulated second."""
        if round(self.sim_time / self.frame_dt) % 30 == 0:
            self.test_final()

    def test_final(self):
        """Reject nonfinite state, broken joint constraints, or through contacts."""
        poses = self.state.body_q.numpy()
        velocities = self.state.body_qd.numpy()
        assert np.isfinite(poses).all() and np.isfinite(velocities).all()
        parent, child = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        xp, xc = self.model.joint_X_p.numpy(), self.model.joint_X_c.numpy()
        source_by_label = {joint["label"]: joint for joint in SCENE["joints"]}
        worst_linear = 0.0
        worst_angular = 0.0
        axes = {"X": wp.vec3(1.0, 0.0, 0.0), "Y": wp.vec3(0.0, 1.0, 0.0), "Z": wp.vec3(0.0, 0.0, 1.0)}
        for i, label in enumerate(self.model.joint_label):
            source = source_by_label[label]
            a, b = _transform(xp[i]), _transform(xc[i])
            if parent[i] >= 0:
                a = _transform(poses[parent[i]]) * a
            if child[i] >= 0:
                b = _transform(poses[child[i]]) * b
            delta = wp.transform_get_translation(b) - wp.transform_get_translation(a)
            axis = axes[source["axis"]]
            if source["type"] == "prismatic":
                world_axis = wp.transform_vector(a, axis)
                delta -= world_axis * wp.dot(delta, world_axis)
            worst_linear = max(worst_linear, float(wp.length(delta)))
            if source["type"] == "revolute":
                u, v = wp.transform_vector(a, axis), wp.transform_vector(b, axis)
                angle = np.arctan2(float(wp.length(wp.cross(u, v))), float(wp.dot(u, v)))
            else:
                relative = wp.mul(
                    wp.quat_inverse(wp.transform_get_rotation(a)),
                    wp.transform_get_rotation(b),
                )
                angle = 2.0 * np.arctan2(
                    float(wp.length(wp.vec3(relative[0], relative[1], relative[2]))),
                    abs(float(relative[3])),
                )
            worst_angular = max(worst_angular, angle)

        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        separation = wp.zeros(contact_count, dtype=float, device=self.model.device)
        if contact_count:
            wp.launch(
                _contact_separation,
                dim=contact_count,
                inputs=[
                    self.state.body_q,
                    self.model.shape_body,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.rigid_contact_point0,
                    self.contacts.rigid_contact_point1,
                    self.contacts.rigid_contact_normal,
                    self.contacts.rigid_contact_margin0,
                    self.contacts.rigid_contact_margin1,
                    separation,
                ],
            )
        penetration = max(0.0, -float(separation.numpy().min())) if contact_count else 0.0
        print(f"Caterpillar maximum joint translation error: {worst_linear:.6g} m")
        print(f"Caterpillar maximum joint angular error: {np.rad2deg(worst_angular):.6g} degrees")
        print(f"Caterpillar maximum contact penetration: {penetration:.6g} m")
        assert worst_linear < 0.002, f"Broken excavator joint translation: {worst_linear:.6g} m"
        assert worst_angular < 0.002, f"Broken excavator joint rotation: {worst_angular:.6g} rad"
        assert penetration < 0.15, f"Excavator contact passed through a part: {penetration:.6g} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix")
        parser.add_argument("--contact-chunk-size", type=int, default=64)
        parser.add_argument("--iterations", type=int, default=2, help="Solver iterations per physics substep.")
        parser.add_argument("--substeps", type=int, default=5, help="Physics substeps per 120 Hz contact refresh.")
        parser.add_argument(
            "--sdf-resolution", type=int, default=128, help="SDF resolution (0 uses the authored values)."
        )
        parser.add_argument("--motor-off", action="store_true", help="Disable all authored hydraulic and motor drives.")
        parser.add_argument(
            "--track-pin-friction",
            type=float,
            default=TRACK_PIN_FRICTION,
            help="Coulomb friction torque at each sealed track pin in N m.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
