# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot G1
#
# Shows how to set up a simulation of a G1 robot articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_g1 --world-count 16
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

# Set to True for PhoenX reduced coordinates; False uses full-coordinate joint mechanisms.
PHOENX_USE_REDUCED_COORDINATES = True


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

        self.viewer = viewer

        # Pick the solver backend selected by the parser.
        solver_name = getattr(args, "solver", "mujoco")

        g1 = newton.ModelBuilder()
        if solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        g1.default_shape_cfg.ke = 1.0e3
        g1.default_shape_cfg.kd = 2.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")

        g1.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0, 0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 500.0
            g1.joint_target_kd[i] = 10.0
            g1.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = args.use_mujoco_contacts if args else False

        if solver_name == "phoenx":
            # Let PhoenX own the temporal schedule. Joint equalities solve
            # directly; two biased and two bias-free inequality sweeps keep
            # impact and resting friction converged at five temporal steps.
            self.sim_substeps = 1
            self.sim_dt = self.frame_dt
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=5,
                solver_iterations=2,
                velocity_iterations=2,
                articulation_mode="reduced" if PHOENX_USE_REDUCED_COORDINATES else "maximal",
            )
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=150,
                cone="elliptic",
                impratio=100,
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=use_mujoco_contacts,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.use_mujoco_contacts = use_mujoco_contacts and solver_name == "mujoco"
        if self.use_mujoco_contacts:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            contact_matching = "sticky" if solver_name == "phoenx" else "disabled"
            self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching=contact_matching)
            self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self._graph_input_state = self.state_0
        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        if not self.use_mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

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

    def test_final(self):
        """Verify the robot remains finite, settled, and graph-stateful."""
        if self.state_0 is not self._graph_input_state:
            raise ValueError("G1 CUDA graph did not preserve its input state buffer")
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        # fmt: off
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are small",
            lambda q, qd: max(abs(qd))
            < 0.015,  # Relaxed from 0.005 - G1 has higher residual velocities with collision pipeline
        )
        # fmt: on

    def test_post_step(self):
        """Reject persistent horizontal drift after the initial impact."""
        frame = getattr(self, "_test_frame", 0) + 1
        self._test_frame = frame
        if frame not in (120, 500):
            return
        pelvis = self.state_0.body_q.numpy()[0]
        xy = (float(pelvis[0]), float(pelvis[1]))
        if frame == 120:
            self._test_settled_xy = xy
            return
        dx = xy[0] - self._test_settled_xy[0]
        dy = xy[1] - self._test_settled_xy[1]
        drift = math.hypot(dx, dy)
        if drift > 0.05:
            raise ValueError(f"G1 drifted {drift:.3f} m horizontally after impact")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument(
            "--solver",
            choices=["mujoco", "phoenx"],
            default="phoenx",
            help="Rigid-body solver backend. 'phoenx' (default) uses SolverPhoenX; 'mujoco' uses the MuJoCo/Warp solver.",
        )
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
