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

STEEL_DENSITY = 7850.0
ALUMINUM_DENSITY = 2700.0
DEFAULT_DENSITY = 1000.0
# Calibrated at the default rear load. A tension-dependent pin-loss model can
# replace this constant once joint reaction magnitudes are exposed to examples.
CHAIN_JOINT_FRICTION = 5.0e-4
DRIVETRAIN_CONTACT_FRICTION = 0.1
DEFAULT_CADENCE_RPM = 60.0
DEFAULT_REAR_LOAD_DAMPING = 0.020053523
DEFAULT_STARTUP_RAMP_TIME = 2.0
# The 120 Hz collision cadence advances a front tooth by about 5 mm at
# 60 rpm. Keep one update of travel in the velocity-derived search envelope.
SPECULATIVE_CONTACT_GAP_MAX = 0.006
DERAILLEUR_PRELOAD_SCALE = 3.0
DERAILLEUR_DAMPING_SCALE = 2.0
DERAILLEUR_SPRING_LABELS = frozenset(
    {
        "/World/Xform/Changer/RD_R9250_BASE/ShiftUpper/Swivel/RevoluteJoint",
        "/World/Xform/Changer/RD_R9250_CAGE/SmallGearFrame/RevoluteJoint",
    }
)


@wp.kernel
def _approach_joint_velocity(targets: wp.array[wp.float32], dof: int, target: float, max_delta: float):
    error = target - targets[dof]
    targets[dof] += wp.clamp(error, -max_delta, max_delta)


def _simulation_schedule(cadence_rpm: float, contact_updates_per_frame: int = 0, substeps: int = 0) -> tuple[int, int]:
    """Resolve contact and integration rates from sprocket travel."""
    if cadence_rpm <= 0.0:
        raise ValueError("cadence must be positive")
    if contact_updates_per_frame < 0 or substeps < 0:
        raise ValueError("contact updates and substeps must be nonnegative")
    contact_updates = contact_updates_per_frame or max(2, int(np.ceil(cadence_rpm / 30.0)))
    resolved_substeps = substeps or (8 if contact_updates <= 2 else 6)
    return contact_updates, resolved_substeps


def _transform(values):
    return wp.transform(values[:3], values[3:])


def _body_density(label):
    """Return the SI density for parts with a known source material."""
    if "/Chain/" in label:
        return STEEL_DENSITY
    if label.endswith(("/FrontGears", "/BackGears")) or "SmallGear" in label:
        return ALUMINUM_DENSITY
    return DEFAULT_DENSITY


def _joint_drive_parameters(
    joint,
    derailleur_preload_scale,
    derailleur_damping_scale,
    cadence_rpm=DEFAULT_CADENCE_RPM,
    rear_load_damping=DEFAULT_REAR_LOAD_DAMPING,
):
    """Return SI drive parameters, including calibrated derailleur preload."""
    target = joint["target"]
    damping = joint["damping"]
    velocity = joint["velocity"]
    if joint["label"] in DERAILLEUR_SPRING_LABELS:
        target *= derailleur_preload_scale
        damping *= derailleur_damping_scale
    elif joint["label"].endswith("/FrontGears/RevoluteJoint"):
        velocity = -cadence_rpm * 2.0 * np.pi / 60.0
    elif joint["label"].endswith("/BackGears/LoadRevoluteJoint"):
        damping = rear_load_damping
    return joint["stiffness"], damping, target, velocity


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


def build_scene(
    *,
    sdf_resolution=0,
    motor_off=False,
    chain_joint_friction=CHAIN_JOINT_FRICTION,
    derailleur_preload_scale=DERAILLEUR_PRELOAD_SCALE,
    derailleur_damping_scale=DERAILLEUR_DAMPING_SCALE,
    cadence_rpm=DEFAULT_CADENCE_RPM,
    rear_load_damping=DEFAULT_REAR_LOAD_DAMPING,
):
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
        body_label = scene["bodies"][shape["body"]]["label"]
        cfg = newton.ModelBuilder.ShapeConfig(
            density=_body_density(body_label) if shape["collision"] else 0.0,
            # A bicycle drivetrain is lubricated; dry-contact friction makes
            # tooth engagement stick laterally instead of transmitting load normally.
            mu=DRIVETRAIN_CONTACT_FRICTION,
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
        stiffness, damping, target, velocity = _joint_drive_parameters(
            joint,
            derailleur_preload_scale,
            derailleur_damping_scale,
            cadence_rpm,
            rear_load_damping,
        )
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
            target_pos=target,
            target_vel=velocity,
            target_ke=stiffness,
            target_kd=damping,
            actuator_mode=mode,
            effort_limit=float("inf"),
            velocity_limit=float("inf"),
            friction=chain_joint_friction if "/Chain/" in joint["label"] else 0.0,
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

    color_group_size = 2

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.solver_stats = getattr(args, "solver_stats", False)
        self.frame_dt = 1.0 / 60.0
        self.cadence_rpm = getattr(args, "cadence_rpm", DEFAULT_CADENCE_RPM)
        self.rear_load_damping = getattr(args, "rear_load_damping", DEFAULT_REAR_LOAD_DAMPING)
        self.speculative_contact_gap_max = getattr(args, "speculative_contact_gap_max", SPECULATIVE_CONTACT_GAP_MAX)
        self.startup_ramp_time = getattr(args, "startup_ramp_time", DEFAULT_STARTUP_RAMP_TIME)
        self.contact_updates_per_frame, self.substeps = _simulation_schedule(
            self.cadence_rpm,
            getattr(args, "contact_updates_per_frame", 0),
            args.substeps,
        )
        if args.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 1 <= args.direct_joint_projection_passes <= args.iterations:
            raise ValueError("direct joint projection passes must be between 1 and iterations")
        if (
            args.contact_chunk_size < 0
            or args.chain_joint_friction < 0.0
            or args.derailleur_preload_scale <= 0.0
            or args.derailleur_damping_scale <= 0.0
            or self.cadence_rpm <= 0.0
            or self.rear_load_damping < 0.0
            or self.speculative_contact_gap_max < 0.0
            or self.startup_ramp_time < 0.0
        ):
            raise ValueError(
                "contact chunk size, chain joint friction, rear load damping, speculative contact gap, and startup "
                "ramp time must be nonnegative; "
                "cadence and derailleur scales must be positive"
            )
        self.model = build_scene(
            sdf_resolution=args.sdf_resolution,
            motor_off=args.motor_off,
            chain_joint_friction=args.chain_joint_friction,
            derailleur_preload_scale=args.derailleur_preload_scale,
            derailleur_damping_scale=args.derailleur_damping_scale,
            cadence_rpm=self.cadence_rpm,
            rear_load_damping=self.rear_load_damping,
        ).finalize(skip_validation_joints=True)
        self.state = self.model.state()
        self.control = self.model.control()
        labels = list(self.model.joint_label)
        self._front_joint = labels.index("/World/Xform/FrontGears/RevoluteJoint")
        self._rear_joint = labels.index("/World/Xform/BackGears/LoadRevoluteJoint")
        self._front_dof = int(self.model.joint_qd_start.numpy()[self._front_joint])
        self._drive_target_velocity = 0.0 if args.motor_off else -self.cadence_rpm * 2.0 * np.pi / 60.0
        if self.startup_ramp_time > 0.0 and not args.motor_off:
            targets = self.control.joint_target_qd.numpy()
            targets[self._front_dof] = 0.0
            self.control.joint_target_qd.assign(targets)
        self.chain_bodies = np.asarray(
            [i for i, label in enumerate(self.model.body_label) if "/Chain/" in label], dtype=np.int32
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            rigid_contact_max=32768,
            speculative_contact_gap_max=self.speculative_contact_gap_max,
            speculative_contact_velocity_filter=True,
            contact_reduction_voxel_depth=args.sdf_voxel_depth_contacts,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            articulation_mode="maximal",
            joint_solver="direct",
            step_layout="single_world",
            substeps=self.substeps,
            solver_iterations=args.iterations,
            velocity_iterations=1,
            direct_joint_projection_passes=args.direct_joint_projection_passes,
            parallel_contact_prepare=False,
            enable_body_pair_grouping=False,
            contact_chunk_size=args.contact_chunk_size,
            mass_splitting=True,
            mass_splitting_color_group_size=self.color_group_size,
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
        if self.startup_ramp_time > 0.0 and self._drive_target_velocity != 0.0:
            wp.launch(
                _approach_joint_velocity,
                dim=1,
                inputs=[
                    self.control.joint_target_qd,
                    self._front_dof,
                    self._drive_target_velocity,
                    abs(self._drive_target_velocity) * self.frame_dt / self.startup_ramp_time,
                ],
                device=self.model.device,
            )
        for _ in range(self.contact_updates_per_frame):
            dt = self.frame_dt / self.contact_updates_per_frame
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
            drivetrain = self.drivetrain_metrics()
            print(
                f"PhoenX: {report.num_joints} joints, {report.num_contact_columns} contact columns, "
                f"{report.num_colors} colors; sizes={report.color_sizes}; "
                f"group sizes={report.color_group_sizes}; overflow={report.overflow_size}; "
                f"crank={drivetrain['crank_rpm']:.1f} rpm, rear={drivetrain['rear_rpm']:.1f} rpm, "
                f"rear load={drivetrain['rear_load_power_w']:.2f} W"
            )

    def drivetrain_metrics(self):
        """Return host-side drivetrain speed and rear dynamometer load diagnostics."""
        poses = self.state.body_q.numpy()
        velocities = self.state.body_qd.numpy()
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        parent_frames = self.model.joint_X_p.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        axes = self.model.joint_axis.numpy()

        def angular_speed(joint):
            frame = _transform(parent_frames[joint])
            parent = int(parents[joint])
            if parent >= 0:
                frame = _transform(poses[parent]) * frame
            axis = axes[int(dof_starts[joint])]
            world_axis = np.asarray(wp.transform_vector(frame, wp.vec3(*axis)))
            relative_omega = velocities[int(children[joint]), 3:]
            if parent >= 0:
                relative_omega = relative_omega - velocities[parent, 3:]
            return float(np.dot(world_axis, relative_omega))

        front_speed = angular_speed(self._front_joint)
        rear_speed = angular_speed(self._rear_joint)
        speed_ratio = abs(rear_speed / front_speed) if abs(front_speed) > 1.0e-6 else 0.0
        return {
            "crank_rpm": abs(front_speed) * 60.0 / (2.0 * np.pi),
            "rear_rpm": abs(rear_speed) * 60.0 / (2.0 * np.pi),
            "speed_ratio": speed_ratio,
            "rear_load_torque_nm": abs(rear_speed) * self.rear_load_damping,
            "rear_load_power_w": rear_speed * rear_speed * self.rear_load_damping,
        }

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
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per physics substep.")
        parser.add_argument(
            "--direct-joint-projection-passes",
            type=int,
            default=2,
            help="Exact D6 projections distributed across each contact solve (default: 2).",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=0,
            help="Physics substeps per contact refresh (0 selects 8 at 120 Hz and 6 above it).",
        )
        parser.add_argument(
            "--contact-updates-per-frame",
            type=int,
            default=0,
            help="Collision/contact refreshes per 60 Hz frame (0 scales with cadence, from 120 Hz).",
        )
        parser.add_argument(
            "--sdf-voxel-depth-contacts",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Retain deepest-per-voxel SDF contacts during contact reduction (disabled by default).",
        )
        parser.add_argument(
            "--sdf-resolution", type=int, default=0, help="Override source SDF resolutions (0 uses authored values)."
        )
        parser.add_argument("--motor-off", action="store_true", help="Disable the crank drive for passive diagnostics.")
        parser.add_argument(
            "--cadence-rpm", type=float, default=DEFAULT_CADENCE_RPM, help="Crank velocity-drive target in rpm."
        )
        parser.add_argument(
            "--startup-ramp-time",
            type=float,
            default=DEFAULT_STARTUP_RAMP_TIME,
            help="Seconds used to ramp the crank drive from rest (0 applies the target immediately).",
        )
        parser.add_argument(
            "--rear-load-damping",
            type=float,
            default=DEFAULT_REAR_LOAD_DAMPING,
            help="Viscous rear dynamometer load in N m s/rad.",
        )
        parser.add_argument(
            "--speculative-contact-gap-max",
            type=float,
            default=SPECULATIVE_CONTACT_GAP_MAX,
            help="Maximum velocity-derived collision search extension in metres.",
        )
        parser.add_argument(
            "--chain-joint-friction",
            type=float,
            default=CHAIN_JOINT_FRICTION,
            help="Coulomb friction torque at each chain pin in N m.",
        )
        parser.add_argument(
            "--derailleur-preload-scale",
            type=float,
            default=DERAILLEUR_PRELOAD_SCALE,
            help="Scale both rear-derailleur spring preload angles (default: 3).",
        )
        parser.add_argument(
            "--derailleur-damping-scale",
            type=float,
            default=DERAILLEUR_DAMPING_SCALE,
            help="Scale both rear-derailleur spring damping values (default: 2).",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
