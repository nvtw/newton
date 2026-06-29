# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a scripted FR3 teleoperation control loop.

The measured loop covers deterministic six-DoF command generation, IK,
joint-target writes, and two completed MuJoCo physics substeps. Rendering,
physical input devices, transport, perception, and display latency are outside
the benchmark scope.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
import newton.ik as ik
import newton.utils
from newton import JointTargetMode


@wp.kernel
def _write_robot_targets(
    joint_q_ik: wp.array2d[wp.float32],
    previous_joint_target_q: wp.array[wp.float32],
    gripper_value: wp.float32,
    dt: wp.float32,
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
):
    for i in range(7):
        q = joint_q_ik[0, i]
        qd = 1.5 * (q - previous_joint_target_q[i]) / dt
        joint_target_q[i] = q
        joint_target_qd[i] = wp.clamp(qd, -20.0, 20.0)
        previous_joint_target_q[i] = q

    for i in range(7, 9):
        q = gripper_value
        qd = 1.5 * (q - previous_joint_target_q[i]) / dt
        joint_target_q[i] = q
        joint_target_qd[i] = wp.clamp(qd, -20.0, 20.0)
        previous_joint_target_q[i] = q


@dataclass(frozen=True)
class _TeleopMode:
    device: str
    mujoco_backend: str
    requires_cuda: bool = False
    requires_cuda_graph: bool = False


_TELEOP_MODES = {
    "mjwarp_cuda_graph": _TeleopMode(
        device="cuda:0",
        mujoco_backend="warp",
        requires_cuda=True,
        requires_cuda_graph=True,
    ),
    "mjwarp_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="warp"),
    "mujoco_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="cpu"),
}


class _WindowStats:
    def __init__(self, maxlen: int):
        self.values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, name: str, value: float) -> None:
        if math.isfinite(value):
            self.values[name].append(float(value))

    def summary(self, name: str) -> tuple[float, float, float]:
        values = self.values.get(name)
        if not values:
            raise RuntimeError(f"No teleop samples collected for {name!r}")
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return sum(values) / len(values), ordered[p95_index], ordered[-1]

    def clear(self) -> None:
        self.values.clear()


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _quat_to_np(q: wp.quat) -> np.ndarray:
    return np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float32)


def _vec3_to_np(v: wp.vec3) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)


class _TeleopLoop:
    fps = 60
    sim_substeps = 2
    linear_speed = 0.5
    angular_speed = 1.6
    gripper_speed = 0.12

    def __init__(self, mode: _TeleopMode, stats_window: int):
        self.device = wp.get_device(mode.device)
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.stats = _WindowStats(stats_window)

        robot = self._build_robot()
        self.model_ik = copy.deepcopy(robot).finalize()

        scene = newton.ModelBuilder()
        scene.add_builder(robot)
        self._add_scene(scene)
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        state_ik = self.model_ik.state()
        newton.eval_fk(self.model_ik, self.model_ik.joint_q, self.model_ik.joint_qd, state_ik)
        self.ee_index = self._find_body(self.model_ik, "fr3_hand_tcp")
        body_q = state_ik.body_q.numpy()
        self.target_tf = wp.transform(*body_q[self.ee_index])
        self.target_base = wp.transform(
            wp.transform_get_translation(self.target_tf),
            wp.transform_get_rotation(self.target_tf),
        )

        self.gripper_target = 0.04
        self.previous_joint_target_q = wp.empty(9, dtype=wp.float32, device=self.device)
        self._setup_ik()
        wp.copy(self.control.joint_target_q[:9], self.model.joint_q[:9])
        wp.copy(self.previous_joint_target_q, self.control.joint_target_q[:9])
        self._write_targets()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=mode.mujoco_backend == "cpu",
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            iterations=20,
            ls_iterations=8,
            njmax=200,
            nconmax=100,
        )

        self.graph_ik = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=16)
            self.graph_ik = capture.graph

        self.graph_sim = None
        if self.device.is_cuda and mode.mujoco_backend == "warp":
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph_sim = capture.graph

    def _build_robot(self) -> newton.ModelBuilder:
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
        )

        initial_q = [
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
        robot.joint_q[:9] = initial_q
        robot.joint_target_q[:9] = initial_q

        for i in range(7):
            robot.joint_target_ke[i] = 3500.0
            robot.joint_target_kd[i] = 220.0
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
            robot.joint_armature[i] = 0.05
        for i in range(7, 9):
            robot.joint_target_ke[i] = 900.0
            robot.joint_target_kd[i] = 80.0
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)
            robot.joint_armature[i] = 0.2

        robot.joint_effort_limit[:7] = [500.0] * 7
        robot.joint_effort_limit[7:9] = [60.0] * 2
        return robot

    @staticmethod
    def _add_scene(builder: newton.ModelBuilder) -> None:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.50, 0.0, 0.02), wp.quat_identity()),
            hx=0.28,
            hy=0.24,
            hz=0.02,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, kd=50.0),
        )
        cube_size = 0.045
        cube_body = builder.add_body(xform=wp.transform(wp.vec3(0.50, 0.0, 0.04 + 0.5 * cube_size), wp.quat_identity()))
        builder.add_shape_box(
            body=cube_body,
            hx=0.5 * cube_size,
            hy=0.5 * cube_size,
            hz=0.5 * cube_size,
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.2, kd=80.0, density=600.0),
        )
        builder.add_ground_plane()

    @staticmethod
    def _find_body(model: newton.Model, name: str) -> int:
        for index, label in enumerate(model.body_label):
            if label.endswith(f"/{name}") or label == name:
                return index
        raise RuntimeError(f"Body {name!r} was not found in the teleop model")

    def _setup_ik(self) -> None:
        target_pos = wp.transform_get_translation(self.target_tf)
        target_rot = wp.transform_get_rotation(self.target_tf)
        self.pos_objective = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([target_pos], dtype=wp.vec3, device=self.device),
        )
        self.rot_objective = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(target_rot)], dtype=wp.vec4, device=self.device),
        )
        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_ik.joint_limit_lower,
            joint_limit_upper=self.model_ik.joint_limit_upper,
            weight=10.0,
        )
        self.joint_q_ik = wp.array(self.model_ik.joint_q, shape=(1, self.model_ik.joint_coord_count))
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[self.pos_objective, self.rot_objective, joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _update_command(self) -> None:
        phase = 2.0 * math.pi * 0.25 * self.sim_time
        base_pos = _vec3_to_np(wp.transform_get_translation(self.target_base))
        desired_pos = base_pos + np.array(
            [0.08 * math.sin(phase), 0.06 * math.sin(0.7 * phase), 0.04 * math.sin(1.3 * phase)],
            dtype=np.float32,
        )
        current_pos = _vec3_to_np(wp.transform_get_translation(self.target_tf))
        delta = (desired_pos - current_pos) / (self.frame_dt * self.linear_speed)
        current_pos += np.clip(delta, -1.0, 1.0) * (self.linear_speed * self.frame_dt)

        rotation = (
            0.30 * math.sin(0.6 * phase),
            0.25 * math.sin(0.9 * phase),
            0.20 * math.sin(1.1 * phase),
        )
        target_rot = wp.transform_get_rotation(self.target_tf)
        axes = (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))
        for axis, value in zip(axes, rotation, strict=True):
            target_rot = wp.quat_from_axis_angle(axis, value * self.angular_speed * self.frame_dt) * target_rot

        self.target_tf = wp.transform(wp.vec3(*current_pos), wp.normalize(target_rot))
        gripper_delta = -1.0 if math.sin(0.5 * phase) >= 0.0 else 1.0
        self.gripper_target = float(
            np.clip(self.gripper_target + gripper_delta * self.gripper_speed * self.frame_dt, 0.0, 0.04)
        )

    def _write_targets(self) -> None:
        wp.launch(
            _write_robot_targets,
            dim=1,
            inputs=[self.joint_q_ik, self.previous_joint_target_q, self.gripper_target, self.frame_dt],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
            device=self.device,
        )

    def _simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _record_tracking_error(self) -> None:
        body_q = self.state_0.body_q.numpy()
        end_effector = body_q[self.ee_index]
        target_pos = _vec3_to_np(wp.transform_get_translation(self.target_tf))
        self.stats.add("target_error_m", float(np.linalg.norm(target_pos - end_effector[:3])))

        target_rot = _quat_to_np(wp.transform_get_rotation(self.target_tf))
        quat_dot = min(1.0, abs(float(np.dot(end_effector[3:7], target_rot))))
        self.stats.add("target_rotation_error_rad", 2.0 * math.acos(quat_dot))

    def step(self, *, collect_tracking: bool = False) -> None:
        frame_start = time.perf_counter()
        self._update_command()
        self.pos_objective.set_target_position(0, wp.transform_get_translation(self.target_tf))
        self.rot_objective.set_target_rotation(0, _quat_to_vec4(wp.transform_get_rotation(self.target_tf)))
        if self.graph_ik is None:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=16)
        else:
            wp.capture_launch(self.graph_ik)
        self._write_targets()
        if self.graph_sim is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph_sim)

        if collect_tracking:
            self._record_tracking_error()
        else:
            wp.synchronize_device(self.device)
            self.stats.add("local_loop_ms", (time.perf_counter() - frame_start) * 1000.0)

        self.sim_time += self.frame_dt
        self.frame_index += 1

    def clear_metrics(self) -> None:
        self.stats.clear()


def _skip_unavailable_mode(mode: _TeleopMode) -> None:
    if mode.requires_cuda and wp.get_cuda_device_count() == 0:
        raise SkipNotImplemented
    if mode.requires_cuda_graph:
        with wp.ScopedDevice(mode.device):
            if not wp.is_mempool_enabled(wp.get_device()):
                raise SkipNotImplemented


class _TeleopMuJoCoBenchmark:
    """Measure the scripted synchronous teleop control-loop core."""

    params = (tuple(_TELEOP_MODES.keys()),)
    param_names = ["mode"]
    repeat = 3
    number = 1
    rounds = 2
    timeout = 600
    num_frames = 300
    warmup_frames = 60

    def setup(self, mode: str) -> None:
        self.mode = _TELEOP_MODES[mode]
        _skip_unavailable_mode(self.mode)
        previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        try:
            with wp.ScopedDevice(self.mode.device):
                self.loop = _TeleopLoop(self.mode, self.num_frames)
        finally:
            newton.use_coord_layout_targets = previous_target_layout

        self._step_frames(self.warmup_frames)
        self.loop.clear_metrics()

    def time_teleop_loop(self, mode: str) -> None:
        self._step_frames(self.num_frames)

    def track_mean_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[0]

    track_mean_loop_ms.unit = "ms/frame"

    def track_p95_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self.loop.stats.summary("local_loop_ms")[1]

    track_p95_loop_ms.unit = "ms/frame"

    def track_frame_overrun_pct(self, mode: str) -> float:
        self._measure_frames()
        values = self.loop.stats.values["local_loop_ms"]
        overruns = sum(value > self.loop.frame_dt * 1000.0 for value in values)
        return 100.0 * overruns / len(values)

    track_frame_overrun_pct.unit = "%"

    def track_mean_target_error_m(self, mode: str) -> float:
        self._measure_frames(collect_tracking=True)
        return self.loop.stats.summary("target_error_m")[0]

    track_mean_target_error_m.unit = "m"

    def track_mean_target_rotation_error_rad(self, mode: str) -> float:
        self._measure_frames(collect_tracking=True)
        return self.loop.stats.summary("target_rotation_error_rad")[0]

    track_mean_target_rotation_error_rad.unit = "rad"

    def _step_frames(self, frame_count: int, *, collect_tracking: bool = False) -> None:
        with wp.ScopedDevice(self.mode.device):
            for _ in range(frame_count):
                self.loop.step(collect_tracking=collect_tracking)

    def _measure_frames(self, *, collect_tracking: bool = False) -> None:
        self.loop.clear_metrics()
        self._step_frames(self.num_frames, collect_tracking=collect_tracking)


class FastTeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Pull-request smoke benchmark for the CUDA graph teleop path."""

    params = (("mjwarp_cuda_graph",),)
    repeat = 2
    num_frames = 120
    warmup_frames = 30


class TeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Nightly teleop benchmark covering GPU and CPU solver backends."""


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastTeleopMuJoCo": FastTeleopMuJoCo,
        "TeleopMuJoCo": TeleopMuJoCo,
    }
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--bench", action="append", choices=benchmark_list.keys())
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
