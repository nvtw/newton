# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the Colibri mechanism with the experimental PhoenX rigid solver.

The shared Python scene describes all bodies, joints, drives, cylinders, and
meter-scaled OBJ meshes; no USD file is loaded at runtime. The source gravity is
1 m/s². Bodies start at rest without damping. Contacts refresh at 120 Hz, with
24 internal physics steps per refresh and neutral SOR (1.0). Sequential groups
of four constraint colors share mass copies for the jointed mechanism.

The experimental temporal solver keeps material friction anchors between
contact refreshes and applies paired impulses at common world points. It uses
one biased solve per internal step, followed by a final velocity relaxation.
The assembly is free to move. Flower and slider geometry belong to the base
body; the counterweight cylinder uses its authored density. The base/frame axle
has no position spring or damper.
Joint and fresh-contact checks screen instability and overlap. Powered runs
check sustained crank tracking. With --motor-off, the crank is passive and
support stationarity is checked after an initial two-second settling window.

The default viewer is OptiX, with four independent worlds in a 2x2 grid.
Use --num-worlds N to change the count, or --num-worlds 1 for one mechanism.

Command: python -m newton.examples phoenx_colibri
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.kamino.example_kamino_colibri import BODY_ORDER, build_scene
from newton.examples.kamino.example_kamino_colibri import Example as ColibriChecks
from newton.solvers import SolverPhoenX

# Contact offsets [m] from the composed source scene and ovphysx 0.5.11 SDK import
# at 120 Hz. Rest offsets are zero. This includes authored 1-2 mm offsets and
# automatic offsets (dependent on geometry, gravity, and timestep).
CONTACT_OFFSETS = {
    "CamFollower/CamFollowerTail": 0.0001388888888888889,
    "CamFollower/Cylinder_32": 0.001,
    "CamFollowerBody/Cam_Follower_Body": 0.0001388888888888889,
    "CamFollowerBody/Cylinder_31": 0.001,
    "CamFollowerHead/Cam_Follower_Head": 0.0001388888888888889,
    "CamWheelBottom/Cam_Wheel_Bottom": 0.00015000000000000004,
    "CamWheelBottom/Cylinder_38": 0.001,
    "CamWheelBottom/Cylinder_39": 0.001,
    "CamWheelBottom/Cylinder_40": 0.001,
    "CamWheelBottom/Cylinder_41": 0.001,
    "CamWheelHead/Cam_Wheel_Body": 0.00015000000000000004,
    "CamWheelHead/Cam_Wheel_Head": 0.0001388888888888889,
    "CamWheelHead/Cylinder_30": 0.001,
    "CamWheelHead/Cylinder_31": 0.001,
    "CamWheelHead/Cylinder_32": 0.001,
    "CamWheelHead/Cylinder_33": 0.001,
    "CamWheelTail/Cam_Wheel_Tail": 0.00015000000000000004,
    "CamWheelTail/Cylinder_34": 0.001,
    "CamWheelTail/Cylinder_35": 0.001,
    "CamWheelTail/Cylinder_36": 0.001,
    "CamWheelTail/Cylinder_37": 0.001,
    "Crank/Crank": 0.0001388888888888889,
    "Crank/Gear_Small_Lower": 0.0001388888888888889,
    "Frame/Cylinder": 0.0007500000670552256,
    "Frame/Cylinders/Cylinder": 0.001,
    "Frame/Cylinders/Cylinder_01": 0.001,
    "Frame/Cylinders/Cylinder_02": 0.001,
    "Frame/Cylinders/Cylinder_03": 0.001,
    "Frame/Cylinders/Cylinder_04": 0.001,
    "Frame/Cylinders/Cylinder_05": 0.001,
    "Frame/Cylinders/Cylinder_06": 0.001,
    "Frame/Cylinders/Cylinder_07": 0.001,
    "Frame/Cylinders/Cylinder_08": 0.001,
    "Frame/Cylinders/Cylinder_09": 0.001,
    "Frame/Cylinders/Cylinder_10": 0.001,
    "Frame/Cylinders/Cylinder_11": 0.001,
    "Frame/Cylinders/Cylinder_12": 0.001,
    "Frame/Cylinders/Cylinder_13": 0.001,
    "Frame/Cylinders/Cylinder_14": 0.001,
    "Frame/Cylinders/Cylinder_15": 0.001,
    "Frame/Cylinders/Cylinder_16": 0.001,
    "Frame/Cylinders/Cylinder_17": 0.001,
    "Frame/Cylinders/Cylinder_18": 0.001,
    "Frame/Cylinders/Cylinder_19": 0.001,
    "Frame/Cylinders/Cylinder_20": 0.001,
    "Frame/Cylinders/Cylinder_21": 0.001,
    "Frame/Cylinders/Cylinder_22": 0.001,
    "Frame/Cylinders/Cylinder_23": 0.001,
    "Frame/Cylinders/Cylinder_24": 0.001,
    "Frame/Cylinders/Cylinder_25": 0.001,
    "Frame/Cylinders/Cylinder_26": 0.001,
    "Frame/Cylinders/Cylinder_27": 0.001,
    "Frame/Cylinders/Cylinder_28": 0.001,
    "Frame/Cylinders/Cylinder_29": 0.001,
    "Frame/FrameMesh": 0.0001388888888888889,
    "Frame/Support_Frame": 0.0001388888888888889,
    "FrameGround/Base": 0.0001388888888888889,
    "FrameGround/Flower/Flower_Stem": 0.0001388888888888889,
    "FrameGround/Flower/Slider/Cube": 0.0001388888888888889,
    "Gear_Large__3x_00/Gear_Large__3x_0": 0.0001388888888888889,
    "Gear_Large__3x_01/Gear_Large__3x_01": 0.0001388888888888889,
    "Gear_Large__3x_02/Gear_Large__3x_02": 0.0001388888888888889,
    "GearedSpinner/Gear_Small_Upper": 0.0001388888888888889,
    "GearedSpinner/Spinner_Link_Thick": 0.00015000000000000004,
    "GearedSpinner/Spinner_Link_Thin": 0.0001388888888888889,
    "HummerBody/Cylinder_30": 0.001,
    "HummerBody/Hummer_Body_Left": 0.0001388888888888889,
    "HummerBody/Hummer_Body_Right": 0.0001388888888888889,
    "HummerBody/Tail_Fan_Gear": 0.0001388888888888889,
    "HummerHead/Cylinder_33": 0.001,
    "HummerHead/Hummer_Head_Left": 0.0001388888888888889,
    "HummerHead/Hummer_Head_Right": 0.0001388888888888889,
    "Hypocycloid_Gear__3x_0/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "Hypocycloid_Gear__3x_01/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh": 0.002,
    "ShoulderLeft/Shoulder_Pivot__2x__mirrored": 0.00029848079681396507,
    "ShoulderRight/Shoulder_Pivot__2x_": 0.00029848079681396507,
    "Tail/Tail_Feather_A__2x_": 0.0003500466585159303,
    "Tail/Tail_Feather_A__2x__mirrored": 0.0003500466585159303,
    "Tail/Tail_Feather_B__2x_": 0.00035004658699035657,
    "Tail/Tail_Feather_B__2x__mirrored": 0.00035004658699035657,
    "Tail/Tail_Feather_C": 0.00035004658699035657,
    "TailMount/Cylinder_32": 0.001,
    "TailMount/Cylinder_33": 0.001,
    "TailMount/Cylinder_34": 0.001,
    "TailMount/Tail_Mount_Left": 0.0001388888888888889,
    "TailMount/Tail_Mount_Left_Standoff_A": 0.0001388888888888889,
    "TailMount/Tail_Mount_Left_Standoff_B": 0.0001388888888888889,
    "TailMount/Tail_Mount_Right": 0.0001388888888888889,
    "TailPinion/Tail_Pinion_Thick": 0.0001388888888888889,
    "TailPinion/Tail_Pinion_Thin": 0.0001388888888888889,
    "TailRack/Cylinder_34": 0.001,
    "TailRack/Tail_Rack": 0.0001388888888888889,
    "WingLeft/Shoulder_Thin__2x__mirrored": 0.0004975449800491338,
    "WingLeftConnector/Cylinder_35": 0.001,
    "WingLinkArcLeft/Wing_Link_Arc__2x__1": 0.0001388888888888889,
    "WingLinkArcRight/Wing_Link_Arc__2x_": 0.0001388888888888889,
    "WingLinkStraightLeft/Wing_Link_Straight__2x__1": 0.0001388888888888889,
    "WingLinkStraightRight/Wing_Link_Straight__2x_": 0.0001388888888888889,
    "WingRight/Shoulder_Thin__2x_": 0.0004975449800491336,
    "WingRightConnector/Cylinder_34": 0.001,
    "ground_plane": 0.02,
}


@wp.kernel(enable_backward=False)
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
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    p0 = point0[i]
    p1 = point1[i]
    if b0 >= 0:
        p0 = wp.transform_point(body_q[b0], p0)
    if b1 >= 0:
        p1 = wp.transform_point(body_q[b1], p1)
    separation[i] = wp.dot(p1 - p0, normal[i]) - margin0[i] - margin1[i]


class Example(ColibriChecks):
    """Simulate the source assembly using ordinary rigid joint/contact blocks."""

    color_group_size = 2

    def __init__(self, viewer, args):
        if args.num_worlds < 1:
            raise ValueError("Number of worlds must be positive")
        self.num_worlds = args.num_worlds
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        if args.contact_updates_per_frame < 1:
            raise ValueError("Contact updates per frame must be positive")
        self.contact_updates = args.contact_updates_per_frame
        self.viewer = viewer
        self.overlap_simulation_render = args.render_overlap
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.fix_base = args.fix_base
        self.motor_enabled = not args.motor_off
        self._support_test_enabled = not args.fix_base and args.body_count == len(BODY_ORDER)
        self._support_reference = None
        builder = build_scene(
            body_count=args.body_count,
            fix_base=self.fix_base,
            contact_gap=0.001,
            source_contact_offsets=True,
            mesh_cylinders=True,
            sdf_resolution=0,
            counterweight_density_scale=args.counterweight_density_scale,
            attach_flower_to_base=True,
            enable_frame_drive=False,
            enable_crank_drive=self.motor_enabled,
        )
        for index, label in enumerate(builder.shape_label):
            if label in CONTACT_OFFSETS:
                builder.shape_gap[index] = CONTACT_OFFSETS[label]
        if self.num_worlds > 1:
            scene = newton.ModelBuilder()
            scene.replicate(builder, self.num_worlds)
            builder = scene
        # The authored closed loops have small initial attachment residuals.
        self.model = builder.finalize(skip_validation_joints=True)
        newton.eval_ik(self.model, self.model, self.model.joint_q, self.model.joint_qd)
        # Retain the authored body poses instead of projecting them through a tree.
        self.state_0 = self.model.state()
        self.initial_q = self.state_0.body_q.numpy().copy()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            # TGS retains friction anchors separately. Reusing normal geometry
            # across refreshes can make curved gear/pin contacts inconsistent.
            contact_matching="latest",
            rigid_contact_max=8192 * self.num_worlds,
            speculative_contact_gap_max=0.005,
            speculative_contact_velocity_filter=not getattr(args, "geometric_candidates", True),
        )
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            joint_mode="maximal_pgs",
            solver_scheme="tgs",
            # Use the global color-group scheduler across independent worlds;
            # the per-world fast-tail scheduler does not support temporal mass copies.
            step_layout="single_world",
            parallel_contact_prepare=True,
            contact_chunk_size=0,
            mass_splitting=True,
            mass_splitting_color_group_size=self.color_group_size,
            mass_splitting_batch_size=2,
            max_colored_partitions=8,
            substeps=args.substeps,
            solver_iterations=args.iterations,
            velocity_relaxation="final_substep",
            prepare_refresh_stride=1,
            sor_boost=1.0,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.65, 0.65, 0.0))
        extent = math.ceil(math.sqrt(self.num_worlds))
        self.viewer.set_camera(wp.vec3(0.4 * extent, -0.7 * extent, 0.2 + 0.15 * extent), pitch=-10, yaw=120)
        self._drive_test_time = None
        self._drive_test_duration = 0.0
        self._drive_test_integral = 0.0
        self._audit_pipeline = None
        self._penetration_log = args.penetration_log
        if self._penetration_log is not None:
            with self._penetration_log.open("w", newline="") as stream:
                csv.writer(stream).writerow(
                    ["time_s", "penetration_m", "shape0", "shape1", "gear_cylinder_penetration_m"]
                )
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self._render_states = (
            (self.model.state(), self.model.state())
            if self.overlap_simulation_render and self.viewer.supports_simulation_render_overlap
            else None
        )
        self._render_state_done = tuple(
            wp.Event(self.model.device) if self.model.device.is_cuda else None for _ in range(2)
        )
        self._render_contacts = [None, None]
        self._render_contact_snapshot = None
        self._render_state_index = 0
        self._render_state_prepared = False
        self._render_time = self.sim_time

    def prepare_render_state(self):
        """Snapshot state before physics advances on the separate stream."""
        self._render_state_index = 1 - self._render_state_index
        done = self._render_state_done[self._render_state_index]
        if done is not None:
            wp.wait_event(done)
        self._render_states[self._render_state_index].assign(self.state_0)
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
            # Copy only the fields consumed by Viewer.log_contacts. Use the
            # state buffer's reuse event for both snapshots, before physics
            # overwrites its live contact data on this stream.
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

    def render(self):
        state = self._render_states[self._render_state_index] if self._render_state_prepared else self.state_0
        self.viewer.begin_frame(self._render_time if self._render_state_prepared else self.sim_time)
        self.viewer.log_state(state)
        contacts = self._render_contact_snapshot if self._render_state_prepared else self.contacts
        if contacts is not None or not self.viewer.show_contacts:
            # The disabled call hides old glyphs without reading live arrays.
            # If the GUI enables contacts after snapshotting, show them on the
            # next frame rather than reading contacts from advancing physics.
            self.viewer.log_contacts(contacts if contacts is not None else self.contacts, state)
        if self._render_state_prepared:
            done = self._render_state_done[self._render_state_index]
            if done is not None:
                wp.record_event(done)
        self.viewer.end_frame()
        self._render_state_prepared = False

    def simulate(self):
        for _ in range(self.contact_updates):
            dt = self.frame_dt / self.contact_updates
            self.collision_pipeline.collide(self.state_0, self.contacts, dt=dt)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, dt)

    def _test_support_stationarity(self):
        """Check unpowered support motion after the initial settling window."""
        if self.motor_enabled or not self._support_test_enabled or self.sim_time < 2.0:
            return
        bases = [i for i, label in enumerate(self.model.body_label) if label == "FrameGround"]
        pose = self.state_0.body_q.numpy()[bases].astype(np.float64)
        if self._support_reference is None:
            self._support_reference = pose.copy()
        reference = self._support_reference
        drift = np.max(np.linalg.norm(pose[:, :2] - reference[:, :2], axis=1))
        alignment = np.abs(np.sum(pose[:, 3:] * reference[:, 3:], axis=1)) / (
            np.linalg.norm(pose[:, 3:], axis=1) * np.linalg.norm(reference[:, 3:], axis=1)
        )
        rotation = np.max(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0)))
        assert drift < 0.00005, f"Support creep: {drift:.6f} m since settling"
        assert rotation < 0.0005, f"Support rotation: {rotation:.6f} rad since settling"

    def _test_drive_tracking(self):
        """Reject stalled or poorly tracking crank motion after settling."""
        if not self.motor_enabled or not self._support_test_enabled or self.sim_time < 2.0:
            return
        joints = [i for i, label in enumerate(self.model.joint_label) if label == "Frame/Crank"]
        dofs = self.model.joint_qd_start.numpy()[joints].astype(int)
        target = self.control.joint_target_qd.numpy()[dofs]
        speed = self.state_0.joint_qd.numpy()[dofs]
        if self._drive_test_time is not None:
            dt = self.sim_time - self._drive_test_time
            self._drive_test_duration += dt
            self._drive_test_integral += dt * speed
        self._drive_test_time = self.sim_time
        if self._drive_test_duration >= 1.0:
            mean = self._drive_test_integral / self._drive_test_duration
            assert np.all(np.abs(mean - target) < 0.05 * np.abs(target)), (
                f"Crank tracking: mean {mean} rad/s, target {target} rad/s"
            )

    def test_post_step(self):
        """Check joint attachment and fresh, independently generated contacts."""
        super().test_post_step()
        self._test_support_stationarity()
        self._test_drive_tracking()
        depth, labels, _gear_depth = self._measure_contact_penetration()
        # This screens substantial overlap; it does not prove tooth engagement.
        assert depth < 0.001, f"Fresh penetration {depth:.6f} m between {labels}"

    def step(self):
        super().step()
        if self._penetration_log is not None:
            depth, labels, gear_depth = self._measure_contact_penetration()
            with self._penetration_log.open("a", newline="") as stream:
                csv.writer(stream).writerow([self.sim_time, depth, *labels, gear_depth])

    def _measure_contact_penetration(self):
        """Measure fresh contact depths without changing simulation history."""
        if self._audit_pipeline is None:
            # Separate matching and contact buffers leave simulation history intact.
            self._audit_capacity = 16384 * self.num_worlds
            self._audit_pipeline = newton.CollisionPipeline(
                self.model, rigid_contact_max=self._audit_capacity, contact_matching="disabled"
            )
            self._audit_contacts = self._audit_pipeline.contacts()
            self._audit_separation = wp.zeros(self._audit_capacity, dtype=wp.float32, device=self.model.device)
            self._audit_filtered_pairs = set(self.model.shape_collision_filter_pairs)
        self._audit_pipeline.collide(self.state_0, self._audit_contacts)
        contacts = self._audit_contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert count < self._audit_capacity, f"Fresh contact capacity reached: {count}/{self._audit_capacity}"
        if count == 0:
            return 0.0, ("", ""), 0.0
        wp.launch(
            _contact_separation,
            dim=count,
            inputs=[
                self.state_0.body_q,
                self.model.shape_body,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
            ],
            outputs=[self._audit_separation],
            device=self.model.device,
        )
        gaps = self._audit_separation.numpy()[:count]
        assert np.isfinite(gaps).all(), "Non-finite fresh contact separation"
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        for a, b in zip(shape0, shape1, strict=True):
            assert tuple(sorted((a, b))) not in self._audit_filtered_pairs, f"Filtered contact pair: {a}, {b}"
        worst = int(np.argmin(gaps))
        labels = (self.model.shape_label[shape0[worst]], self.model.shape_label[shape1[worst]])
        gear_depth = 0.0
        for index in np.flatnonzero(gaps < 0.0):
            pair = (self.model.shape_label[shape0[index]], self.model.shape_label[shape1[index]])
            if any("Hypocycloid" in label for label in pair) and any("Cylinder" in label for label in pair):
                gear_depth = max(gear_depth, -float(gaps[index]))
        return max(0.0, -float(gaps[worst])), labels, gear_depth

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix")
        parser.add_argument(
            "--num-worlds", "--world-count", type=int, default=4, help="Number of independent Colibris (default: 4)."
        )
        parser.add_argument(
            "--penetration-log",
            type=Path,
            help="Write fresh contact depth and worst shape pair per frame to CSV (adds diagnostic overhead).",
        )
        parser.add_argument(
            "--body-count",
            type=int,
            default=len(BODY_ORDER),
            help="Build the first N mechanism bodies, starting at FrameGround.",
        )
        parser.add_argument(
            "--counterweight-density-scale",
            type=float,
            default=1.0,
            help="Density multiplier for Frame/Cylinder; scales its mass and inertia before body assembly (1.0 = authored).",
        )
        parser.add_argument(
            "--render-overlap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Overlap rendering of a state snapshot with the next physics frame when supported by the viewer.",
        )
        parser.add_argument(
            "--motor-off",
            action="store_true",
            help="Disable crank actuation, including servo damping, for unpowered creep diagnostics.",
        )
        parser.add_argument("--fix-base", action="store_true", help="Anchor the base for diagnostics.")
        admission = parser.add_mutually_exclusive_group()
        admission.add_argument(
            "--geometric-candidates",
            action="store_true",
            default=True,
            help="Retain nearby candidates inside the motion envelope when initial velocities underpredict approach.",
        )
        admission.add_argument(
            "--velocity-filtered-candidates",
            dest="geometric_candidates",
            action="store_false",
            help="Diagnostic legacy admission: discard candidates receding at contact generation time.",
        )
        parser.add_argument(
            "--contact-updates-per-frame",
            type=int,
            default=2,
            help="Diagnostic contact refreshes per 60 Hz frame; substeps apply to each refresh.",
        )
        parser.add_argument("--substeps", type=int, default=24, help="Physics steps per contact refresh.")
        parser.add_argument(
            "--iterations", type=int, default=1, help="Joint/contact sweeps per substep (temporal mode requires 1)."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
