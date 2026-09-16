# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the Colibri mechanism with the experimental PhoenX rigid solver.

The shared Python scene describes all bodies, joints, drives, cylinders, and
meter-scaled OBJ meshes; no USD file is loaded at runtime. The source gravity is
1 m/s². Bodies start at rest without damping. Contacts refresh at 120 Hz, with
30 internal physics steps per refresh and neutral SOR (1.0). Sequential groups
of four constraint colors share mass copies for the jointed mechanism.

This example is an experimental solver stress test. Its default configuration
has passed a 300-second headless run with joint and fresh-contact checks.
The free assembly still slides and yaws; its escape check measures motion
relative to FrameGround. These checks screen instability and overlap; they
do not establish correct support friction or gear engagement.

Command: python -m newton.examples phoenx_colibri
"""

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

    def __init__(self, viewer, args):
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.fix_base = args.fix_base
        builder = build_scene(
            body_count=args.body_count,
            fix_base=self.fix_base,
            contact_gap=0.001,
            source_contact_offsets=True,
            mesh_cylinders=True,
            sdf_resolution=0,
        )
        for index, label in enumerate(builder.shape_label):
            if label in CONTACT_OFFSETS:
                builder.shape_gap[index] = CONTACT_OFFSETS[label]
        # The authored closed loops have small initial attachment residuals.
        self.model = builder.finalize(skip_validation_joints=True)
        newton.eval_ik(self.model, self.model, self.model.joint_q, self.model.joint_qd)
        # Retain the authored body poses instead of projecting them through a tree.
        self.state_0 = self.model.state()
        self.initial_q = self.state_0.body_q.numpy().copy()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            rigid_contact_max=8192,
            speculative_contact_gap_max=0.005,
            speculative_contact_velocity_filter=not getattr(args, "geometric_candidates", True),
        )
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            articulation_mode="maximal",
            joint_solver="block_pgs",
            step_layout="single_world",
            parallel_contact_prepare=True,
            contact_chunk_size=6,
            mass_splitting=True,
            mass_splitting_color_group_size=4,
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
        self.viewer.set_camera(wp.vec3(0.4, -0.7, 0.35), pitch=-10, yaw=120)
        self._audit_pipeline = None
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(2):
            dt = self.frame_dt / 2
            self.collision_pipeline.collide(self.state_0, self.contacts, dt=dt)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, dt)

    def test_post_step(self):
        """Check joint attachment and fresh, independently generated contacts."""
        super().test_post_step()
        if self._audit_pipeline is None:
            # Separate matching and contact buffers leave simulation history intact.
            self._audit_capacity = 16384
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
            return
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
        # This screens substantial overlap; it does not prove tooth engagement.
        assert gaps[worst] > -0.001, f"Fresh penetration {-gaps[worst]:.6f} m between {labels}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--body-count",
            type=int,
            default=len(BODY_ORDER),
            help="Build the first N mechanism bodies, starting at FrameGround.",
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
        parser.add_argument("--substeps", type=int, default=30, help="Physics steps per 120 Hz contact refresh.")
        parser.add_argument("--iterations", type=int, default=4, help="Joint/contact sweeps per physics step.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
