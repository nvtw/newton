# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg


class Example:
    def __init__(self, viewer, args=None):

        solver = getattr(args, "solver", "kamino")
        msg.info(f"Using solver = {solver}")
        self._solver_name = solver

        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.001))
        self.sim_dt = self.frame_dt / self.sim_substeps
        msg.info(
            f"Using sim_dt = {self.sim_dt} ({self.sim_substeps} substeps per frame)"
        )
        self.sim_time = 0.0
        self.viewer = viewer

        # Load the USD and add it to the builder
        model_name = "hoberman_sphere"

        filename = f"{model_name}_articulation.usda"
        model_dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "models"
        )
        model_path = os.path.join(model_dir_path, filename)

        # Create a single-robot model builder and register the solver-specific custom attributes
        msg.notif("Creating and configuring the model builder...")
        robot_builder = newton.ModelBuilder()
        if solver == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        elif solver == "xpbd":
            newton.solvers.SolverXPBD.register_custom_attributes(robot_builder)
        elif solver == "vbd":
            newton.solvers.SolverVBD.register_custom_attributes(robot_builder)
        elif solver == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
        elif solver == "phoenx":
            newton.solvers.SolverPhoenX.register_custom_attributes(robot_builder)

        # Load the basic four-bar USD and add it to the builder
        msg.notif("Loading USD asset and adding it to the model builder...")
        robot_builder.add_usd(
            model_path,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            collapse_fixed_joints=False,
            apply_up_axis_from_stage=True,
        )

        # robot_builder.add_ground_plane(height=-2.0)

        # Create the model from the builder
        msg.notif("Creating the model from the builder...")
        robot_builder.color()
        self.model = robot_builder.finalize(skip_validation_joints=True)

        self.model.set_gravity([0.0, 0.0, 0.0])

        # Set initial velocities
        lin_vel_global = np.array([0.0, 0.0, 0.0])
        ang_vel_global = np.array([0.0, 0.0, 1.0]) * 0.1
        body_q_np = self.model.body_q.numpy()
        body_qd = []
        for body_id in range(self.model.body_count):
            lin_vel = lin_vel_global + np.cross(ang_vel_global, body_q_np[body_id, 0:3])
            body_qd.append(np.concatenate((lin_vel, ang_vel_global)))
        self.model.body_qd.assign(body_qd)

        # Create the solver for the given model
        msg.notif("Creating the solver for the given model...")
        if solver == "kamino":
            # Create and configure settings for SolverKamino and the collision detector
            solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
            solver_config.sparse_dynamics = True
            solver_config.sparse_jacobian = True
            solver_config.use_collision_detector = False
            solver_config.collision_detector.pipeline = "unified"
            solver_config.collision_detector.max_contacts = 1
            solver_config.collision_detector.max_contacts_per_pair = 1
            solver_config.collision_detector.max_contacts_per_world = 1
            solver_config.collision_detector.max_triangle_pairs = 1
            solver_config.dynamics.preconditioning = True
            solver_config.dynamics.linear_solver_type = "CR"
            solver_config.dynamics.linear_solver_kwargs = {"maxiter": 9}
            solver_config.constraints.alpha = 0.01
            solver_config.padmm.primal_tolerance = 1e-4
            solver_config.padmm.dual_tolerance = 1e-4
            solver_config.padmm.compl_tolerance = 1e-4
            solver_config.padmm.max_iterations = 50
            solver_config.padmm.rho_0 = 0.02
            solver_config.padmm.eta = 1e-5
            solver_config.padmm.use_acceleration = True
            solver_config.padmm.warmstart_mode = "containers"
            solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"
            solver_config.padmm.penalty_update_method = "fixed"
            solver_config.integrator = "euler"

            self.solver = newton.solvers.SolverKamino(
                model=self.model, config=solver_config
            )
        elif solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=10,
                joint_angular_relaxation=0.1,
            )
        elif solver == "vbd":
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=10,
                rigid_avbd_alpha=1.0,
            )
        elif solver == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                model=self.model,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                impratio=100.0,
                ls_iterations=50,
                use_mujoco_contacts=False,
            )
        elif solver == "phoenx":
            # PhoenX advances ``substeps`` PGS substeps internally per
            # ``solver.step`` call, so we hand it the full frame_dt
            # and skip the per-substep Python loop in ``simulate``.
            # The integrator dt is therefore ``frame_dt / substeps``,
            # matching the other solvers' ``sim_dt``.
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=self.sim_substeps,
                solver_iterations=8,
                velocity_iterations=1,
            )

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if solver == "mujoco":
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            # self.contacts = self.model.contacts()
            # PhoenX accepts ``contacts=None`` and skips its contact
            # ingest pipeline, which is what we want here -- the
            # struts overlap by design and any narrow-phase result
            # would explode the sphere on frame 1.
            self.contacts = None

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Run warm-up
        self.graph = None
        self.step()

        # Reset states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_time = 0.0

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        if self._solver_name == "phoenx":
            # PhoenX owns its own substep loop -- one ``solver.step``
            # call advances ``substeps`` PGS substeps at
            # ``frame_dt / substeps`` each. Skipping the Python-side
            # substep loop also cuts capture-time kernel emission.
            #
            # State buffers: a per-substep ``state_0 <-> state_1``
            # swap inside a graph-captured region aliases the buffer
            # the kernels read from to the buffer they wrote into,
            # so each replay would re-read the never-updated input
            # (see the diagnostic in
            # ``example_robot_dr_legs_phoenx.py``). We instead copy
            # the just-stepped pose back into ``state_0`` -- the
            # copy is captured alongside the solver kernels and
            # refreshes the input buffer on every replay.
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.frame_dt
            )
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
            return

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # Since rendering is called after stepping the simulation, the previous and next
        # states correspond to self.state_1 and self.state_0 due to the reference swaps,
        # so contacts are rendered with self.state_1 to match the body positions at the
        # time of contact generation.
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="phoenx",
        choices=("kamino", "xpbd", "vbd", "mujoco", "phoenx"),
        help="Solver backend to use.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    example.viewer._paused = True  # Start paused to inspect the initial configuration

    msg.notif("Starting the simulation...")
    newton.examples.run(example, args)
