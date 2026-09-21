# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulate the USD-derived tourbillon clock with PhoenX.

Command: python -m newton.examples phoenx_tourbillon_clock

Geometry, normals, joint frames and drives come from TourbillonClock.usd.
The runtime uses the extracted OBJ assets and SI scene descriptors, without USD.
"""

import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.phoenx.tourbillon_clock_scene import SCENE
from newton.solvers import SolverPhoenX

ASSETS = Path(newton.examples.get_asset_directory()) / "tourbillon_clock"
REDUNDANT_BEARING_JOINT = "/World/Clock/SmallConnectorWatchGear/RevoluteJoint_01"
BEARING_FRICTION = 1.0e-2
OSCILLATOR_FRICTION_SCALE = 0.05


@wp.kernel
def _apply_angular_drag(
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    inertia: wp.array[wp.mat33],
    damping: wp.array[float],
    forces: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    rotation = wp.transform_get_rotation(poses[body])
    omega = wp.quat_rotate_inv(rotation, wp.spatial_bottom(velocities[body]))
    torque = wp.quat_rotate(rotation, -damping[body] * (inertia[body] * omega))
    forces[body] += wp.spatial_vector(wp.vec3(0.0), torque)


def _transform(values):
    return wp.transform(values[:3], values[3:])


def _active_joints():
    """Drop the second coaxial bearing authored between the same body pair."""
    return [joint for joint in SCENE["joints"] if joint["label"] != REDUNDANT_BEARING_JOINT]


def _load_mesh(path, roughness):
    # Exported OBJ vertices, normals, and UVs share indices, including sharp edges.
    vertices, normals, uvs, faces = [], [], [], []
    text = path.read_text()
    for line in text.splitlines():
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


def build_scene(*, sdf_resolution=0, motor_off=False, bearing_friction=BEARING_FRICTION):
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
            if shape["collision"] and shape["approximation"] == "sdf":
                mesh.build_sdf(
                    max_resolution=key[2],
                    margin=0.001,
                    cache_dir=str(Path(tempfile.gettempdir()) / "newton_tourbillon_clock_sdf"),
                )
            meshes[key] = mesh
        cfg = newton.ModelBuilder.ShapeConfig(
            density=shape["density"] if shape["collision"] else 0.0,
            mu=shape["friction"],
            margin=0.0,
            gap=0.0005,
            has_shape_collision=shape["collision"],
            is_visible=shape["visible"],
        )
        add_shape = builder.add_shape_convex_hull if shape["approximation"] == "convexHull" else builder.add_shape_mesh
        add_shape(
            shape["body"],
            mesh=meshes[key],
            xform=wp.transform(shape["center"], wp.quat_identity()),
            cfg=cfg,
            color=shape["color"],
            label=shape["label"],
        )
    for body_index, body in enumerate(scene["bodies"]):
        target_mass = body["mass"]
        if target_mass is None or body["kinematic"]:
            continue
        source_mass = builder.body_mass[body_index]
        if source_mass <= 0.0:
            raise ValueError(f"Cannot apply mass override to massless body {body['label']}")
        inertia_scale = target_mass / source_mass
        builder.body_mass[body_index] = target_mass
        builder.body_inv_mass[body_index] = 1.0 / target_mass
        builder.body_inertia[body_index] *= inertia_scale
        builder.body_inv_inertia[body_index] = wp.inverse(builder.body_inertia[body_index])
    for joint in _active_joints():
        stiffness, damping = joint["stiffness"], joint["damping"]
        friction = bearing_friction
        if "/TourbillonOscillator/" in joint["label"]:
            friction *= OSCILLATOR_FRICTION_SCALE
        if not joint["drive_enabled"]:
            stiffness = damping = 0.0
        if motor_off and joint["velocity"] != 0.0:
            stiffness = damping = 0.0
        mode = newton.JointTargetMode.POSITION if stiffness else newton.JointTargetMode.VELOCITY
        builder.add_joint_revolute(
            joint["parent"],
            joint["child"],
            parent_xform=_transform(joint["frames"][0]),
            child_xform=_transform(joint["frames"][1]),
            axis={"X": newton.Axis.X, "Y": newton.Axis.Y, "Z": newton.Axis.Z}[joint["axis"]],
            target_pos=joint["target"],
            target_vel=joint["velocity"],
            target_ke=stiffness,
            target_kd=damping,
            actuator_mode=mode,
            effort_limit=float("inf"),
            velocity_limit=float("inf"),
            friction=friction,
            limit_lower=joint["lower"],
            limit_upper=joint["upper"],
            collision_filter_parent=not joint["collision_enabled"],
            label=joint["label"],
        )
    return builder


class Example:
    """Drive the clock train and nested tourbillon through authored gear contacts."""

    color_group_size = 2
    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("substeps and iterations must be positive")
        if not 1 <= args.direct_joint_projection_passes <= args.iterations:
            raise ValueError("direct joint projection passes must be between 1 and iterations")
        if args.contact_chunk_size < 0:
            raise ValueError("contact chunk size must be nonnegative")
        if args.bearing_friction < 0.0:
            raise ValueError("bearing friction must be nonnegative")
        self.model = build_scene(
            sdf_resolution=args.sdf_resolution,
            motor_off=args.motor_off,
            bearing_friction=args.bearing_friction,
        ).finalize(skip_validation_joints=True)
        self.joints = _active_joints()
        self.angular_damping = wp.array(
            [body["angular_damping"] for body in SCENE["bodies"]], dtype=float, device=self.model.device
        )
        self.state = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            rigid_contact_max=32768,
            speculative_contact_gap_max=0.002,
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
            direct_joint_projection_passes=args.direct_joint_projection_passes,
            parallel_contact_prepare=False,
            enable_body_pair_grouping=True,
            contact_chunk_size=args.contact_chunk_size,
            mass_splitting=True,
            mass_splitting_color_group_size=self.color_group_size,
            mass_splitting_batch_size=2,
            max_colored_partitions=8,
        )
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(0.80, -0.90, 0.44), pitch=-15.0, yaw=132.0)
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
        """Snapshot device-resident poses and optional contacts for asynchronous rendering."""
        if self._render_states is None:
            return
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
            dt = self.frame_dt / 2
            self.pipeline.collide(self.state, self.contacts, dt=dt)
            self.state.clear_forces()
            wp.launch(
                _apply_angular_drag,
                dim=self.model.body_count,
                inputs=[self.state.body_q, self.state.body_qd, self.model.body_inertia, self.angular_damping],
                outputs=[self.state.body_f],
                device=self.model.device,
            )
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
        """Check attachment and hinge errors twice per simulated second."""
        if round(self.sim_time / self.frame_dt) % 30 == 0:
            self.test_final()

    def test_final(self):
        """Reject nonfinite state or broken tourbillon and gear-train attachments."""
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all() and np.isfinite(self.state.body_qd.numpy()).all()
        parent, child = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        xp, xc = self.model.joint_X_p.numpy(), self.model.joint_X_c.numpy()
        worst = 0.0
        worst_angle = 0.0
        worst_joint = worst_hinge = ""
        for i in range(self.model.joint_count):
            a, b = _transform(xp[i]), _transform(xc[i])
            if parent[i] >= 0:
                a = _transform(poses[parent[i]]) * a
            if child[i] >= 0:
                b = _transform(poses[child[i]]) * b
            gap = float(wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b)))
            if gap > worst:
                worst, worst_joint = gap, self.joints[i]["label"]
            axis = wp.vec3(*{"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[self.joints[i]["axis"]])
            u, v = wp.transform_vector(a, axis), wp.transform_vector(b, axis)
            angle = np.arctan2(float(wp.length(wp.cross(u, v))), float(wp.dot(u, v)))
            if angle > worst_angle:
                worst_angle, worst_hinge = angle, self.joints[i]["label"]
        print(f"TourbillonClock maximum joint attachment error: {worst:.6g} m ({worst_joint})")
        print(f"TourbillonClock maximum hinge-axis error: {np.rad2deg(worst_angle):.6g} degrees ({worst_hinge})")
        assert worst_angle < 0.005, f"Misaligned clock hinge: {worst_angle:.6g} rad"
        assert worst < 0.001, f"Broken clock joint: {worst:.6g} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix")
        parser.add_argument(
            "--contact-chunk-size",
            type=int,
            default=64,
            help="Maximum contact rows per solver column (0 keeps whole shape pairs).",
        )
        parser.add_argument("--iterations", type=int, default=4, help="Mixed solver iterations per physics substep.")
        parser.add_argument(
            "--direct-joint-projection-passes",
            type=int,
            default=2,
            help="Exact D6 projections distributed across each contact solve.",
        )
        parser.add_argument("--substeps", type=int, default=8, help="Physics substeps per 120 Hz contact refresh.")
        parser.add_argument(
            "--sdf-resolution",
            type=int,
            default=128,
            help="Collision SDF resolution override (0 uses authored values).",
        )
        parser.add_argument(
            "--bearing-friction",
            type=float,
            default=BEARING_FRICTION,
            help="Coulomb friction torque at clock bearings in N m.",
        )
        parser.add_argument(
            "--motor-off",
            action="store_true",
            help="Disable the escapement drive while retaining the authored springs and damping.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
