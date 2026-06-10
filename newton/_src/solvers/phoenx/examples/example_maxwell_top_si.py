# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import warp as wp

import newton
import newton.examples

USD_PATH = Path(r"C:\Users\twidmer\Downloads\MaxwellTopSI\MaxwellTopSI2.usda")
SDF_CACHE_DIR = USD_PATH.parent / ".sdf_cache"
FPS = 300
SUBSTEPS = 5
SOLVER_ITERATIONS = 5
CONTACT_GAP = 0.0
CONTACT_MARGIN = 0.0

# World-frame initial angular velocity for the spinning body [deg/s].
# Main2 carries the authored spin in the USD; Spiral is the tip (zero spin).
TOP_BODY_PATH = "/World/Main2/Main2_obj0"
TOP_ANGULAR_VELOCITY_DEG_S = (300.0, 0.0, 10000.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.solver_type = getattr(args, "solver", "phoenx")
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SUBSTEPS
        self.sim_time = 0.0
        self._printed_contact_count = False

        builder = newton.ModelBuilder()
        builder.sdf_cache_dir = SDF_CACHE_DIR
        result = builder.add_usd(
            str(USD_PATH),
            schema_resolvers=[
                newton.usd.SchemaResolverNewton(),
                newton.usd.SchemaResolverPhysx(),
            ],
        )
        builder.shape_gap[:] = [CONTACT_GAP] * len(builder.shape_gap)
        builder.shape_margin[:] = [CONTACT_MARGIN] * len(builder.shape_margin)
        builder.shape_collision_filter_pairs.clear()
        print(
            "Loaded MaxwellTopSI2.usda: "
            f"{len(result['path_body_map'])} bodies, "
            f"{len(result['path_joint_map'])} joints, "
            f"{len(result['path_shape_map'])} shapes"
        )
        print(f"SDF texture cache: {SDF_CACHE_DIR}")

        self.model = builder.finalize(skip_validation_joints=True)
        print(
            "Collision setup: "
            f"{self.model.shape_contact_pair_count} contact pairs, "
            f"SDF indices {self.model.shape_sdf_index.numpy().tolist()}"
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._set_top_angular_velocity(result["path_body_map"])
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.collision_pipeline.contacts()

        if self.solver_type == "phoenx":
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=SUBSTEPS,
                solver_iterations=SOLVER_ITERATIONS,
            )
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=SOLVER_ITERATIONS)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
        print(f"Using {self.solver_type} solver")

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.0, -3.0, 2.0), pitch=-25.0, yaw=135.0)
        self.capture()

    @staticmethod
    def _apply_body_angular_velocity(
        state: newton.State,
        model: newton.Model,
        path_body_map: dict[str, int],
        body_path: str,
        angular_velocity_deg_s: tuple[float, float, float],
    ) -> None:
        """Set world-frame angular velocity [rad/s] on ``state.body_qd``."""
        body_idx = path_body_map[body_path]
        angular_velocity = wp.vec3(
            math.radians(angular_velocity_deg_s[0]),
            math.radians(angular_velocity_deg_s[1]),
            math.radians(angular_velocity_deg_s[2]),
        )
        body_qd_np = state.body_qd.numpy()
        body_qd_np[body_idx, 3:6] = [angular_velocity[0], angular_velocity[1], angular_velocity[2]]
        state.body_qd.assign(body_qd_np)

        model_body_qd_np = model.body_qd.numpy()
        model_body_qd_np[body_idx, 3:6] = body_qd_np[body_idx, 3:6]
        model.body_qd.assign(model_body_qd_np)
        print(f"Set {body_path} angular velocity to {angular_velocity_deg_s} deg/s")

    def _set_top_angular_velocity(self, path_body_map: dict[str, int]) -> None:
        self._apply_body_angular_velocity(
            self.state_0,
            self.model,
            path_body_map,
            TOP_BODY_PATH,
            TOP_ANGULAR_VELOCITY_DEG_S,
        )

    def capture(self):
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.model.collide(
            self.state_0,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        if self.solver_type == "phoenx":
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
            self.state_0.assign(self.state_1)
        else:
            for _ in range(SUBSTEPS):
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0.assign(self.state_1)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        if not self._printed_contact_count:
            print(f"Initial rigid contacts: {int(self.contacts.rigid_contact_count.numpy()[0])}")
            self._printed_contact_count = True
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            type=str,
            default="phoenx",
            choices=["phoenx", "xpbd"],
            help="Rigid-body solver backend.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
