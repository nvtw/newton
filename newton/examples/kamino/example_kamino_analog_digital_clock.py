# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Analog Digital Clock
#
# Simulates a closed-loop mechanical clock imported from USD with Kamino's
# full-coordinate DVI solver and Newton SDF contacts.
#
# Command: python -m newton.examples kamino_analog_digital_clock
#
###########################################################################

import sys
from pathlib import Path

# Direct IDE launches otherwise resolve whichever Newton checkout happens to be
# on PYTHONPATH instead of the checkout containing this example.
source_root = None
if __package__ in (None, ""):
    source_root = Path(__file__).resolve().parents[3]
    source_root_str = str(source_root)
    while source_root_str in sys.path:
        sys.path.remove(source_root_str)
    sys.path.insert(0, source_root_str)

import numpy as np
import warp as wp

import newton
import newton.examples

if source_root is not None and Path(newton.__file__).resolve().parents[1] != source_root:
    raise RuntimeError(
        f"This example belongs to {source_root}, but Python imported Newton from {newton.__file__}. "
        "Restart Python with the matching checkout's environment."
    )


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion_vector = quaternion[:3]
    return vector + 2.0 * np.cross(
        quaternion_vector,
        np.cross(quaternion_vector, vector) + quaternion[3] * vector,
    )


def _transform_point(body_q: np.ndarray, body: int, point: np.ndarray) -> np.ndarray:
    return body_q[body, :3] + _rotate_vector(body_q[body, 3:7], point)


def _relative_quaternion(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_conjugate = np.array([-parent[0], -parent[1], -parent[2], parent[3]])

    return _quaternion_multiply(parent_conjugate, child)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    vector = (
        left[3] * right[:3]
        + right[3] * left[:3]
        + np.cross(left[:3], right[:3])
    )
    scalar = left[3] * right[3] - np.dot(left[:3], right[:3])
    return np.append(vector, scalar)


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.viewer = viewer
        self.device = wp.get_device()
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, args.substeps) if args else 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.num_rigid_contacts_per_world = 2048
        builder.default_shape_cfg.margin = 5.0e-4
        builder.default_shape_cfg.gap = 0.0
        asset_file = Path(__file__).resolve().parents[1] / "assets" / "analog_digital_clock.usd"
        import_result = builder.add_usd(
            str(asset_file),
            joint_ordering=None,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
            force_position_velocity_actuation=True,
            hide_collision_shapes=False,
            ignore_paths=[r"/World/GroundPlane.*"],
        )
        builder.add_ground_plane()

        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 2048

        config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver="dvi",
            sparse_dynamics=True,
            sparse_jacobian=True,
        )
        config.use_collision_detector = False
        config.use_fk_solver = False
        config.integrator = "moreau"
        config.constraints.alpha = 0.1
        config.constraints.beta = 0.011
        config.constraints.gamma = 0.015
        config.dynamics.preconditioning = False
        config.dynamics.linear_solver_type = "CR"
        config.dynamics.linear_solver_kwargs = {"maxiter": 12}
        config.dvi.bilateral_solver_type = "LLTBRCM"
        config.dvi.bilateral_solver_kwargs = {"parallel_factorization": True}
        config.dvi.tolerance = 1.0e-4
        config.dvi.regularization = 1.0e-5
        config.dvi.max_alternating_iterations = 6
        config.dvi.inequality_sweeps_per_iteration = 4
        config.dvi.bilateral_solve_interval = 1
        config.dvi.contact_warmstart_method = "key_and_position_with_net_force_backup"
        self.solver = newton.solvers.SolverKamino(self.model, config=config)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            reduce_contacts=True,
            rigid_contact_max=2048,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.joint_parent = self.model.joint_parent.numpy()
        self.joint_child = self.model.joint_child.numpy()
        self.joint_X_p = self.model.joint_X_p.numpy()
        self.joint_X_c = self.model.joint_X_c.numpy()
        self.joint_axis = self.model.joint_axis.numpy()
        self.joint_qd_start = self.model.joint_qd_start.numpy()
        self.imported_body_count = len(import_result["path_body_map"])
        self.imported_joint_count = len(import_result["path_joint_map"])
        self.frame_body = import_result["path_body_map"]["/World/Clock/Frame"]
        self.motor_body = import_result["path_body_map"]["/World/Clock/MotorGear"]
        self.motor_shape = import_result["path_shape_map"]["/World/Clock/MotorGear/mesh"]
        self.middle_shape = import_result["path_shape_map"]["/World/Clock/MiddleGear/mesh"]
        initial_body_q = self.state_0.body_q.numpy()
        self.initial_motor_q = _relative_quaternion(
            initial_body_q[self.frame_body, 3:7],
            initial_body_q[self.motor_body, 3:7],
        )

        self.viewer.set_model(self.model)
        self._camera_pending = True
        self._set_camera()

        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _set_camera(self):
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.1, -1.1, 0.65), pitch=0.0, yaw=0.0)
            camera = getattr(self.viewer, "camera", None)
            if camera is not None and hasattr(camera, "look_at"):
                camera.look_at(wp.vec3(0.03, 0.06, 0.42))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
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
        self.viewer.log_contacts(self.contacts, self.state_1)
        if self._camera_pending:
            self._set_camera()
            self._camera_pending = False
        self.viewer.end_frame()

    def test_final(self):
        """Verify body state and closed-loop joint stability."""
        assert self.imported_body_count == 25, f"Expected 25 clock bodies, imported {self.imported_body_count}"
        assert self.imported_joint_count == 31, f"Expected 31 clock joints, imported {self.imported_joint_count}"
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body poses contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf values"
        motor_q = _relative_quaternion(body_q[self.frame_body, 3:7], body_q[self.motor_body, 3:7])
        motor_rotation = 2.0 * np.arccos(np.clip(abs(np.dot(self.initial_motor_q, motor_q)), 0.0, 1.0))
        assert motor_rotation > 1.0e-4, "Clock motor did not rotate"
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
        has_motor_contact = np.any(
            ((shape0 == self.motor_shape) & (shape1 == self.middle_shape))
            | ((shape0 == self.middle_shape) & (shape1 == self.motor_shape))
        )
        assert has_motor_contact, "Motor gear did not contact the middle gear"
        max_joint_gap = 0.0
        max_axis_error = 0.0
        for joint, (parent, child, parent_xform, child_xform) in enumerate(zip(
            self.joint_parent,
            self.joint_child,
            self.joint_X_p,
            self.joint_X_c,
            strict=True,
        )):
            parent_anchor = _transform_point(body_q, parent, parent_xform[:3])
            child_anchor = _transform_point(body_q, child, child_xform[:3])
            max_joint_gap = max(max_joint_gap, float(np.linalg.norm(parent_anchor - child_anchor)))
            parent_rotation = _quaternion_multiply(body_q[parent, 3:7], parent_xform[3:7])
            child_rotation = _quaternion_multiply(body_q[child, 3:7], child_xform[3:7])
            axis = self.joint_axis[self.joint_qd_start[joint]]
            parent_axis = _rotate_vector(parent_rotation, axis)
            child_axis = _rotate_vector(child_rotation, axis)
            max_axis_error = max(max_axis_error, float(1.0 - abs(np.dot(parent_axis, child_axis))))
        assert max_joint_gap < 0.005, f"Clock joint anchors drifted apart: gap={max_joint_gap:.3f} m"
        assert max_axis_error < 5.0e-4, f"Clock joint axes drifted out of alignment: error={max_axis_error:.3g}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per rendered frame.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
