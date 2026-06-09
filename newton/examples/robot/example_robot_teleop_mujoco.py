# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Teleop MuJoCo
#
# FR3/Panda arm teleoperation baseline using IK target control and the
# MuJoCo solver.  The target can be moved with the viewer gizmo, keyboard, or
# an Xbox-style gamepad through pyglet's controller API.
#
# Command: python -m newton.examples robot_teleop_mujoco
#
###########################################################################

from __future__ import annotations

import argparse
import copy
import math
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _vec3_to_np(v: wp.vec3) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)


def _np_to_vec3(v: np.ndarray) -> wp.vec3:
    return wp.vec3(float(v[0]), float(v[1]), float(v[2]))


@wp.kernel
def _write_robot_targets_kernel(
    joint_q_ik: wp.array2d[wp.float32],
    previous_joint_target_q: wp.array[wp.float32],
    gripper_value: wp.float32,
    dt: wp.float32,
    velocity_feedforward: wp.float32,
    max_target_velocity: wp.float32,
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
):
    for i in range(7):
        q = joint_q_ik[0, i]
        qd = velocity_feedforward * (q - previous_joint_target_q[i]) / dt
        joint_target_q[i] = q
        joint_target_qd[i] = wp.clamp(qd, -max_target_velocity, max_target_velocity)
        previous_joint_target_q[i] = q

    for i in range(7, 9):
        q = gripper_value
        qd = velocity_feedforward * (q - previous_joint_target_q[i]) / dt
        joint_target_q[i] = q
        joint_target_qd[i] = wp.clamp(qd, -max_target_velocity, max_target_velocity)
        previous_joint_target_q[i] = q


class WindowStats:
    def __init__(self, maxlen: int = 600):
        self.values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, name: str, value: float) -> None:
        if math.isfinite(value):
            self.values[name].append(float(value))

    def summary(self, name: str) -> tuple[float, float, float] | None:
        values = self.values.get(name)
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
        return sum(values) / len(values), ordered[index], max(values)

    def latest(self, name: str) -> float | None:
        values = self.values.get(name)
        return values[-1] if values else None


@dataclass
class TeleopCommand:
    translation: np.ndarray
    rotation: np.ndarray
    gripper_delta: float = 0.0
    active: bool = False
    source: str = "scripted"


class GamepadInput:
    def __init__(self):
        self.enabled = False
        self.status = "disabled"
        self._controller: Any | None = None

    def open(self, window: Any | None = None) -> bool:
        try:
            import pyglet  # noqa: PLC0415
        except Exception as exc:
            self.status = f"pyglet controller API unavailable ({type(exc).__name__})"
            return False

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*has no controller mappings.*")
                controllers = pyglet.input.get_controllers()
        except Exception as exc:
            self.status = f"gamepad scan failed ({type(exc).__name__})"
            return False

        if not controllers:
            self.status = "gamepad not found"
            return False

        controller = controllers[0]
        try:
            controller.open(window=window)
        except Exception as exc:
            self.status = f"gamepad open failed ({type(exc).__name__})"
            return False

        self._controller = controller
        self.enabled = True
        name = getattr(controller, "name", None) or "gamepad"
        self.status = f"{name} connected"
        return True

    @staticmethod
    def _axis(value: float, deadzone: float = 0.15) -> float:
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) <= deadzone:
            return 0.0
        return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)

    def read(self) -> TeleopCommand:
        if not self.enabled or self._controller is None:
            return TeleopCommand(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), source="gamepad")

        controller = self._controller
        left_x = self._axis(getattr(controller, "leftx", 0.0))
        left_y = self._axis(getattr(controller, "lefty", 0.0))
        right_x = self._axis(getattr(controller, "rightx", 0.0))
        right_y = self._axis(getattr(controller, "righty", 0.0))
        left_trigger = self._axis(getattr(controller, "lefttrigger", 0.0), deadzone=0.05)
        right_trigger = self._axis(getattr(controller, "righttrigger", 0.0), deadzone=0.05)

        translation = np.array(
            [
                -left_y,
                left_x,
                right_trigger - left_trigger,
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [
                float(bool(getattr(controller, "rightshoulder", False)))
                - float(bool(getattr(controller, "leftshoulder", False))),
                -right_y,
                right_x,
            ],
            dtype=np.float32,
        )
        gripper_delta = float(bool(getattr(controller, "b", False))) - float(bool(getattr(controller, "a", False)))

        active = bool(np.linalg.norm(translation) > 1.0e-4 or np.linalg.norm(rotation) > 1.0e-4 or gripper_delta)
        return TeleopCommand(translation, rotation, gripper_delta, active=active, source="gamepad")

    def close(self) -> None:
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.sim_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0

        self.viewer = viewer
        self.device = wp.get_device()
        self.test_mode = args.test
        self.input_mode = "scripted" if args.test or args.viewer == "null" else args.input
        self.linear_speed = args.linear_speed
        self.angular_speed = args.angular_speed
        self.gripper_speed = args.gripper_speed
        self.ik_iterations = max(1, int(args.ik_iterations))
        self.velocity_feedforward = float(args.velocity_feedforward)
        self.max_target_velocity = float(args.max_target_velocity)
        self.arm_stiffness = float(args.arm_stiffness)
        self.arm_damping = float(args.arm_damping)
        self.arm_effort_limit = float(args.arm_effort_limit)
        self.gripper_stiffness = float(args.gripper_stiffness)
        self.gripper_damping = float(args.gripper_damping)
        self.gripper_effort_limit = float(args.gripper_effort_limit)
        self.metrics_interval = max(0, args.metrics_interval)
        self.metrics_warmup_frames = max(0, int(args.metrics_warmup_frames))
        self.sync_latency = args.sync_latency
        self.use_mujoco_cpu = args.mujoco_backend == "cpu"
        self.render_shadows = bool(args.render_shadows)
        self.print_metrics_on_close = bool(args.print_metrics) or args.benchmark is not False
        self._printed_metrics = False

        self.stats = WindowStats(maxlen=max(60, int(args.stats_window)))
        self.settle_samples_ms: list[float] = []
        self._pending_settle_started: float | None = None
        self._pending_settle_target: np.ndarray | None = None

        self.workspace_min = np.array([0.20, -0.45, 0.12], dtype=np.float32)
        self.workspace_max = np.array([0.75, 0.45, 0.90], dtype=np.float32)

        robot = self._build_robot()
        self.model_ik = copy.deepcopy(robot).finalize()

        scene = newton.ModelBuilder()
        scene.add_builder(robot)
        self._add_manipulation_scene(scene)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.state_ik = self.model_ik.state()
        newton.eval_fk(self.model_ik, self.model_ik.joint_q, self.model_ik.joint_qd, self.state_ik)

        self.ee_index = self._find_body(self.model_ik, "fr3_hand_tcp", fallback=10)
        body_q_np = self.state_ik.body_q.numpy()
        self.target_tf = wp.transform(*body_q_np[self.ee_index])
        self.target_tf = self._clamp_transform(self.target_tf)
        self.target_base = wp.transform(
            wp.transform_get_translation(self.target_tf),
            wp.transform_get_rotation(self.target_tf),
        )

        self.gripper_open = 0.04
        self.gripper_closed = 0.0
        self.gripper_target = self.gripper_open
        self.previous_joint_target_q = wp.empty(9, dtype=wp.float32, device=self.device)

        self._setup_ik()
        wp.copy(self.control.joint_target_q[:9], self.model.joint_q[:9])
        wp.copy(self.previous_joint_target_q, self.control.joint_target_q[:9])
        self._write_robot_targets()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco_cpu,
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            iterations=args.solver_iterations,
            ls_iterations=args.solver_ls_iterations,
            njmax=200,
            nconmax=100,
        )

        self.gamepad = GamepadInput()
        self.active_input_source = self.input_mode
        if self.input_mode in ("auto", "gamepad"):
            window = getattr(getattr(self.viewer, "renderer", None), "window", None)
            opened = self.gamepad.open(window=window)
            if opened:
                self.active_input_source = "gamepad"
            elif self.input_mode == "gamepad":
                self.active_input_source = "keyboard"
            else:
                self.active_input_source = "keyboard"

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self._configure_viewer_performance()
        self._set_initial_camera()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.capture_ik()
        self.capture()

    def _build_robot(self) -> newton.ModelBuilder:
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
        )

        init_q = [
            -0.0036802115,
            0.023901723,
            0.003680411,
            -2.3683236,
            -0.00012918962,
            2.3922248,
            0.785492,
            0.04,
            0.04,
        ]
        robot.joint_q[:9] = init_q
        robot.joint_target_q[:9] = init_q

        for i in range(7):
            robot.joint_target_ke[i] = self.arm_stiffness
            robot.joint_target_kd[i] = self.arm_damping
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
            robot.joint_armature[i] = 0.05

        for i in range(7, 9):
            robot.joint_target_ke[i] = self.gripper_stiffness
            robot.joint_target_kd[i] = self.gripper_damping
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
            robot.joint_armature[i] = 0.2

        robot.joint_effort_limit[:7] = [self.arm_effort_limit] * 7
        robot.joint_effort_limit[7:9] = [self.gripper_effort_limit] * 2
        return robot

    def _set_initial_camera(self) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return

        camera_pos = wp.vec3(1.0, -1.25, 0.75)
        look_at = wp.vec3(0.45, 0.0, 0.35)
        self.viewer.set_camera(pos=camera_pos, pitch=-18.0, yaw=115.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "look_at"):
            camera.look_at(look_at)

    def _configure_viewer_performance(self) -> None:
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "draw_shadows"):
            renderer.draw_shadows = self.render_shadows

    def _add_manipulation_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, kd=50.0)
        cube_cfg = newton.ModelBuilder.ShapeConfig(mu=1.2, kd=80.0, density=600.0)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.50, 0.0, 0.02), wp.quat_identity()),
            hx=0.28,
            hy=0.24,
            hz=0.02,
            cfg=table_cfg,
            color=(0.35, 0.35, 0.38),
            label="teleop_table",
        )
        cube_size = 0.045
        cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.50, 0.0, 0.04 + 0.5 * cube_size), wp.quat_identity()),
            label="teleop_cube",
        )
        builder.add_shape_box(
            body=cube_body,
            hx=0.5 * cube_size,
            hy=0.5 * cube_size,
            hz=0.5 * cube_size,
            cfg=cube_cfg,
            color=(0.2, 0.55, 0.85),
            label="teleop_cube_shape",
        )
        builder.add_ground_plane()

    def _find_body(self, model: newton.Model, name: str, fallback: int) -> int:
        for i, label in enumerate(model.body_label):
            if label.endswith(f"/{name}") or label == name:
                return i
        return fallback

    def _setup_ik(self) -> None:
        pos = wp.transform_get_translation(self.target_tf)
        rot = wp.transform_get_rotation(self.target_tf)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([pos], dtype=wp.vec3, device=self.device),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(rot)], dtype=wp.vec4, device=self.device),
        )
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_ik.joint_limit_lower,
            joint_limit_upper=self.model_ik.joint_limit_upper,
            weight=10.0,
        )
        self.joint_q_ik = wp.array(self.model_ik.joint_q, shape=(1, self.model_ik.joint_coord_count))
        self.ik_iters = self.ik_iterations
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def capture_ik(self) -> None:
        self.graph_ik = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def capture(self) -> None:
        self.graph = None
        if self.device.is_cuda and not self.use_mujoco_cpu:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _keyboard_command(self) -> TeleopCommand:
        if not hasattr(self.viewer, "is_key_down"):
            return TeleopCommand(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), source="keyboard")

        def key(name: str) -> float:
            return 1.0 if self.viewer.is_key_down(name) else 0.0

        translation = np.array(
            [
                key("l") - key("j"),
                key("i") - key("k"),
                key("u") - key("o"),
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [
                key("x") - key("z"),
                key("v") - key("c"),
                key("m") - key("n"),
            ],
            dtype=np.float32,
        )
        gripper_delta = key("h") - key("g")
        active = bool(np.linalg.norm(translation) > 0.0 or np.linalg.norm(rotation) > 0.0 or gripper_delta)
        return TeleopCommand(translation, rotation, gripper_delta, active=active, source="keyboard")

    def _scripted_command(self) -> TeleopCommand:
        phase = 2.0 * math.pi * 0.25 * self.sim_time
        base_pos = _vec3_to_np(wp.transform_get_translation(self.target_base))
        target_pos = base_pos + np.array(
            [0.08 * math.sin(phase), 0.06 * math.sin(0.7 * phase), 0.04 * math.sin(1.3 * phase)],
            dtype=np.float32,
        )
        target_pos = np.minimum(np.maximum(target_pos, self.workspace_min), self.workspace_max)
        current_pos = _vec3_to_np(wp.transform_get_translation(self.target_tf))
        delta = (target_pos - current_pos) / max(self.frame_dt * self.linear_speed, 1.0e-6)
        delta = np.minimum(np.maximum(delta, -1.0), 1.0)

        if self.frame_index >= self.metrics_warmup_frames and self.frame_index % max(1, int(0.5 / self.frame_dt)) == 0:
            self._pending_settle_started = time.perf_counter()
            self._pending_settle_target = target_pos.copy()

        return TeleopCommand(delta.astype(np.float32), np.zeros(3, dtype=np.float32), active=True, source="scripted")

    def _read_command(self) -> TeleopCommand:
        if self.input_mode == "scripted":
            return self._scripted_command()
        if self.active_input_source == "gamepad" and self.gamepad.enabled:
            command = self.gamepad.read()
            if command.active:
                return command
        return self._keyboard_command()

    def _clamp_transform(self, transform: wp.transform) -> wp.transform:
        pos = _vec3_to_np(wp.transform_get_translation(transform))
        pos = np.minimum(np.maximum(pos, self.workspace_min), self.workspace_max)
        return wp.transform(_np_to_vec3(pos), wp.transform_get_rotation(transform))

    def _apply_command(self, command: TeleopCommand) -> None:
        if command.source == "scripted":
            self._apply_pose_delta(command.translation, command.rotation)
        elif command.active:
            self._apply_pose_delta(command.translation, command.rotation)

        if command.gripper_delta:
            self.gripper_target += float(command.gripper_delta) * self.frame_dt * self.gripper_speed
            self.gripper_target = min(self.gripper_open, max(self.gripper_closed, self.gripper_target))

    def _apply_pose_delta(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        pos = _vec3_to_np(wp.transform_get_translation(self.target_tf))
        pos += translation * (self.linear_speed * self.frame_dt)
        pos = np.minimum(np.maximum(pos, self.workspace_min), self.workspace_max)

        q = wp.transform_get_rotation(self.target_tf)
        if abs(float(rotation[0])) > 1.0e-6:
            q = (
                wp.quat_from_axis_angle(
                    wp.vec3(1.0, 0.0, 0.0),
                    float(rotation[0]) * self.angular_speed * self.frame_dt,
                )
                * q
            )
        if abs(float(rotation[1])) > 1.0e-6:
            q = (
                wp.quat_from_axis_angle(
                    wp.vec3(0.0, 1.0, 0.0),
                    float(rotation[1]) * self.angular_speed * self.frame_dt,
                )
                * q
            )
        if abs(float(rotation[2])) > 1.0e-6:
            q = (
                wp.quat_from_axis_angle(
                    wp.vec3(0.0, 0.0, 1.0),
                    float(rotation[2]) * self.angular_speed * self.frame_dt,
                )
                * q
            )
        q = wp.normalize(q)

        self.target_tf = wp.transform(_np_to_vec3(pos), q)

    def _push_ik_targets(self) -> None:
        self.target_tf = self._clamp_transform(self.target_tf)
        pos = wp.transform_get_translation(self.target_tf)
        rot = wp.transform_get_rotation(self.target_tf)
        self.pos_obj.set_target_position(0, pos)
        self.rot_obj.set_target_rotation(0, _quat_to_vec4(rot))

    def _write_robot_targets(self) -> None:
        wp.launch(
            _write_robot_targets_kernel,
            dim=1,
            inputs=[
                self.joint_q_ik,
                self.previous_joint_target_q,
                float(self.gripper_target),
                float(self.frame_dt),
                self.velocity_feedforward,
                self.max_target_velocity,
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
            device=self.device,
        )

    def _time_section(self, name: str, fn) -> None:
        start = time.perf_counter()
        fn()
        if self.sync_latency:
            wp.synchronize()
        if self.frame_index >= self.metrics_warmup_frames:
            self.stats.add(f"{name}_ms", (time.perf_counter() - start) * 1000.0)

    def solve_ik(self) -> None:
        if self.graph_ik:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _update_tracking_metrics(self) -> None:
        if (
            self.metrics_interval == 0
            or self.frame_index < self.metrics_warmup_frames
            or self.frame_index % self.metrics_interval != 0
        ):
            return

        body_q = self.state_0.body_q.numpy()
        ee_pos = np.asarray(body_q[self.ee_index][:3], dtype=np.float32)
        target_pos = _vec3_to_np(wp.transform_get_translation(self.target_tf))
        pos_error = float(np.linalg.norm(target_pos - ee_pos))
        self.stats.add("target_error_m", pos_error)

        if self._pending_settle_started is not None and self._pending_settle_target is not None:
            pending_error = float(np.linalg.norm(self._pending_settle_target - ee_pos))
            if pending_error < 0.025:
                latency_ms = (time.perf_counter() - self._pending_settle_started) * 1000.0
                self.settle_samples_ms.append(latency_ms)
                self.stats.add("settle_latency_ms", latency_ms)
                self._pending_settle_started = None
                self._pending_settle_target = None

    def step(self) -> None:
        frame_start = time.perf_counter()

        def input_step():
            command = self._read_command()
            self._apply_command(command)

        self._time_section("input", input_step)
        self._push_ik_targets()
        self._time_section("ik", self.solve_ik)
        self._time_section("target_write", self._write_robot_targets)
        self._time_section("sim", (lambda: wp.capture_launch(self.graph)) if self.graph else self.simulate)
        if self.frame_index >= self.metrics_warmup_frames:
            self.stats.add("local_loop_ms", (time.perf_counter() - frame_start) * 1000.0)

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._update_tracking_metrics()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        target_pos = wp.transform_get_translation(self.target_tf)
        target_points = wp.array([target_pos], dtype=wp.vec3, device=self.device)
        self.viewer.log_points(
            "teleop_target",
            target_points,
            radii=0.025,
            colors=(0.95, 0.25, 0.15),
        )

        if hasattr(self.viewer, "log_gizmo") and self.input_mode != "scripted":
            self.viewer.log_gizmo("target_tcp", self.target_tf, snap_to=self.target_tf)

        self.viewer.end_frame()

    def render_ui(self, imgui) -> None:
        imgui.text(f"Input: {self.active_input_source}")
        if self.input_mode in ("auto", "gamepad"):
            imgui.text(self.gamepad.status)
        for label, key in (
            ("Loop", "local_loop_ms"),
            ("Input", "input_ms"),
            ("IK", "ik_ms"),
            ("Sim", "sim_ms"),
            ("Target error", "target_error_m"),
            ("Settle", "settle_latency_ms"),
        ):
            summary = self.stats.summary(key)
            if summary is None:
                continue
            mean, p95, _ = summary
            unit = "m" if key.endswith("_m") else "ms"
            imgui.text(f"{label}: mean {mean:.3f} {unit}, p95 {p95:.3f} {unit}")

    def test_final(self) -> None:
        summary = self.stats.summary("target_error_m")
        if summary is None:
            raise AssertionError("No target tracking samples were collected")
        mean_error, _, max_error = summary
        if not math.isfinite(mean_error) or mean_error > 0.20:
            raise AssertionError(f"Mean target tracking error too large: {mean_error:.4f} m")
        if not math.isfinite(max_error) or max_error > 0.45:
            raise AssertionError(f"Max target tracking error too large: {max_error:.4f} m")
        self.print_latency_summary()

    def print_latency_summary(self) -> None:
        if self._printed_metrics:
            return
        self._printed_metrics = True
        print("Teleop MuJoCo latency summary:")
        for label, key in (
            ("local loop", "local_loop_ms"),
            ("input", "input_ms"),
            ("IK", "ik_ms"),
            ("target write", "target_write_ms"),
            ("simulation", "sim_ms"),
            ("target error", "target_error_m"),
            ("settle latency", "settle_latency_ms"),
        ):
            summary = self.stats.summary(key)
            if summary is None:
                continue
            mean, p95, max_value = summary
            unit = "m" if key.endswith("_m") else "ms"
            print(f"  {label}: mean={mean:.3f} {unit}, p95={p95:.3f} {unit}, max={max_value:.3f} {unit}")
        if self.device.is_cuda and not self.sync_latency:
            print("  timing note: GPU timings are host enqueue timings; pass --sync-latency for completion timing.")

    def close(self) -> None:
        if self.print_metrics_on_close and self.stats.summary("local_loop_ms") is not None:
            self.print_latency_summary()
        self.gamepad.close()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        parser.add_argument(
            "--input",
            choices=("auto", "keyboard", "gamepad", "scripted"),
            default="auto",
            help="Target pose input source. Gamepad uses Xbox-style controls.",
        )
        parser.add_argument(
            "--mujoco-backend",
            choices=("warp", "cpu"),
            default="warp",
            help="MuJoCo backend to use for the rigid solve.",
        )
        parser.add_argument("--sim-substeps", type=int, default=2, help="Simulation substeps per rendered frame.")
        parser.add_argument("--linear-speed", type=float, default=0.50, help="Target translation speed [m/s].")
        parser.add_argument("--angular-speed", type=float, default=1.6, help="Target angular speed [rad/s].")
        parser.add_argument("--gripper-speed", type=float, default=0.12, help="Gripper open/close speed [m/s].")
        parser.add_argument("--ik-iterations", type=int, default=16, help="IK solver iterations per frame.")
        parser.add_argument(
            "--velocity-feedforward",
            type=float,
            default=1.5,
            help="Scale applied to joint target velocity feed-forward.",
        )
        parser.add_argument(
            "--max-target-velocity",
            type=float,
            default=20.0,
            help="Clamp for joint target velocity feed-forward [rad/s or m/s].",
        )
        parser.add_argument("--solver-iterations", type=int, default=20, help="MuJoCo solver iterations.")
        parser.add_argument("--solver-ls-iterations", type=int, default=8, help="MuJoCo line-search iterations.")
        parser.add_argument("--arm-stiffness", type=float, default=3500.0, help="Arm joint target stiffness.")
        parser.add_argument("--arm-damping", type=float, default=220.0, help="Arm joint target damping.")
        parser.add_argument("--arm-effort-limit", type=float, default=500.0, help="Arm joint effort limit [N*m].")
        parser.add_argument("--gripper-stiffness", type=float, default=900.0, help="Gripper joint target stiffness.")
        parser.add_argument("--gripper-damping", type=float, default=80.0, help="Gripper joint target damping.")
        parser.add_argument(
            "--gripper-effort-limit",
            type=float,
            default=60.0,
            help="Gripper joint effort limit [N].",
        )
        parser.add_argument(
            "--metrics-interval",
            type=int,
            default=15,
            help="Frames between tracking-error samples; set 0 to disable tracking readback for lowest latency.",
        )
        parser.add_argument(
            "--metrics-warmup-frames",
            type=int,
            default=3,
            help="Initial frames to exclude from latency and tracking summaries.",
        )
        parser.add_argument("--stats-window", type=int, default=600, help="Number of recent samples in UI summaries.")
        parser.add_argument(
            "--print-metrics",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Print the latency summary when the example exits.",
        )
        parser.add_argument(
            "--sync-latency",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Synchronize after timed sections so GPU timings include completion latency.",
        )
        parser.add_argument(
            "--render-shadows",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render GL shadows. Disabled by default for higher interactive FPS.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
