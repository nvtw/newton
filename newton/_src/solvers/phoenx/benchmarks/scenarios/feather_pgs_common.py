# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes copied from Dylan Turpin's FeatherPGS nightly benchmark.

The source is newton/tools/solver_benchmark.py at commit af928ab3
on dylanturpin/newton-collab:07--fpgs--nightly--slurm-path. Keep the
model construction here in sync with that source so measurements can be
compared with the FeatherPGS report made on the same GPU.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.utils
from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle, _gpu_used_bytes

_TABLETOP_OBJECTS = (
    ("sphere", (0.6, -0.1, 1.06), None, 0.05),
    ("sphere", (0.675, -0.025, 1.06), None, 0.05),
    ("sphere", (0.75, 0.05, 1.06), None, 0.05),
    ("sphere", (0.825, 0.125, 1.06), None, 0.05),
    ("capsule", (0.6, -0.15, 1.16), (1.0, 0.5, 0.0, 0.0), 0.04),
    ("capsule", (0.8, -0.15, 1.16), (1.0, 0.5, 0.0, 0.0), 0.04),
    ("sphere", (0.7, -0.15, 1.41), None, 0.04),
    ("sphere", (0.7, -0.05, 1.41), None, 0.04),
    ("sphere", (0.7, 0.05, 1.41), None, 0.04),
    ("sphere", (0.7, 0.15, 1.41), None, 0.04),
    ("capsule", (0.55, 0.0, 1.2), (0.0, 0.5, 0.0, 1.0), 0.04),
    ("capsule", (0.85, 0.0, 1.2), (0.0, 0.5, 0.0, 1.0), 0.04),
    ("box", (0.65, 0.0, 1.19), None, 0.03),
    ("box", (0.65, 0.0, 1.25), None, 0.03),
    ("box", (0.65, 0.0, 1.31), None, 0.03),
    ("box", (0.75, 0.0, 1.19), None, 0.03),
    ("box", (0.75, 0.0, 1.25), None, 0.03),
    ("box", (0.75, 0.0, 1.31), None, 0.03),
)


def _build_robot(robot: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    if robot == "g1":
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=0.0, limit_kd=0.0, friction=0.0)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        asset_path = newton.utils.download_asset("unitree_g1")
        builder.add_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        for i in range(6, builder.joint_dof_count):
            builder.joint_target_ke[i] = 1000.0
            builder.joint_target_kd[i] = 5.0
    elif robot == "h1":
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        asset_path = newton.utils.download_asset("unitree_h1")
        builder.add_usd(
            str(asset_path / "usd" / "h1_minimal.usda"),
            ignore_paths=["/GroundPlane"],
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150.0
            builder.joint_target_kd[i] = 5.0
    else:
        raise ValueError(f"unknown robot {robot!r}")

    builder.approximate_meshes("bounding_box")
    return builder


def _normalized_quat(values: tuple[float, float, float, float] | None) -> wp.quat:
    if values is None:
        return wp.quat_identity()
    length = math.sqrt(sum(value * value for value in values))
    return wp.quat(*(value / length for value in values))


def _add_tabletop_object(
    builder: newton.ModelBuilder,
    kind: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float] | None,
    size: float,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
) -> None:
    body = builder.add_body(xform=wp.transform(wp.vec3(*position), _normalized_quat(rotation)))
    if kind == "sphere":
        builder.add_shape_sphere(body, radius=size, cfg=shape_cfg)
    elif kind == "capsule":
        builder.add_shape_capsule(body, radius=size, half_height=size, cfg=shape_cfg)
    elif kind == "box":
        builder.add_shape_box(body, hx=size, hy=size, hz=size, cfg=shape_cfg)
    else:
        raise ValueError(f"unknown tabletop object {kind!r}")


def _add_static_box(
    builder: newton.ModelBuilder,
    position: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    shape_cfg: newton.ModelBuilder.ShapeConfig,
) -> None:
    builder.add_shape_box(
        -1,
        xform=wp.transform(position, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=shape_cfg,
    )


def _build_model(scenario: str, num_worlds: int) -> newton.Model:
    if scenario == "feather_pgs_g1_flat":
        robot = _build_robot("g1")
        builder = newton.ModelBuilder()
        builder.replicate(robot, num_worlds)
        builder.add_ground_plane()
    elif scenario == "feather_pgs_h1_tabletop":
        robot = _build_robot("h1")
        builder = newton.ModelBuilder()
        object_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        for _ in range(num_worlds):
            builder.begin_world()
            builder.add_builder(robot)
            for obj in _TABLETOP_OBJECTS:
                _add_tabletop_object(builder, *obj, object_cfg)
            builder.end_world()

        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        for position, half_extents in (
            ((0.8, 0.0, 0.75), (0.5, 1.0, 0.01)),
            ((0.9, 0.0, 0.86), (0.01, 0.21, 0.1)),
            ((0.5, 0.0, 0.86), (0.01, 0.21, 0.1)),
            ((0.7, -0.2, 0.86), (0.21, 0.01, 0.1)),
            ((0.7, 0.2, 0.86), (0.21, 0.01, 0.1)),
        ):
            _add_static_box(builder, position, half_extents, table_cfg)
        builder.add_ground_plane()
    else:
        raise ValueError(f"unknown FeatherPGS benchmark scenario {scenario!r}")

    model = builder.finalize()
    model.shape_margin.fill_(0.001)
    return model


def _validate_simulation_state(
    state: newton.State,
    initial_body_q: np.ndarray,
    contacts: newton.Contacts,
) -> dict[str, float | int]:
    """Validate that a measured simulation remained physical and advanced."""
    arrays = {
        "joint_q": state.joint_q.numpy(),
        "joint_qd": state.joint_qd.numpy(),
        "body_q": state.body_q.numpy(),
        "body_qd": state.body_qd.numpy(),
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"simulation produced non-finite values in state.{name}")

    body_q = arrays["body_q"].reshape(-1, 7)
    initial_body_q = initial_body_q.reshape(-1, 7)
    quaternion_error = float(np.abs(np.linalg.norm(body_q[:, 3:7], axis=1) - 1.0).max(initial=0.0))
    if quaternion_error > 1.0e-3:
        raise RuntimeError(f"maximum body quaternion norm error is {quaternion_error:.3g}")

    max_translation = float(np.linalg.norm(body_q[:, :3] - initial_body_q[:, :3], axis=1).max(initial=0.0))
    if max_translation <= 1.0e-8:
        raise RuntimeError("simulation did not move any rigid body")

    body_qd = arrays["body_qd"].reshape(-1, 6)
    max_linear_speed = float(np.linalg.norm(body_qd[:, :3], axis=1).max(initial=0.0))
    max_angular_speed = float(np.linalg.norm(body_qd[:, 3:], axis=1).max(initial=0.0))
    if max_linear_speed > 100.0:
        raise RuntimeError(f"maximum body linear speed is implausibly high ({max_linear_speed:.3g} m/s)")
    if max_angular_speed > 500.0:
        raise RuntimeError(f"maximum body angular speed is implausibly high ({max_angular_speed:.3g} rad/s)")

    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    if contact_count <= 0:
        raise RuntimeError("simulation produced no rigid contacts")

    return {
        "validation_max_translation_m": max_translation,
        "validation_max_linear_speed_m_s": max_linear_speed,
        "validation_max_angular_speed_rad_s": max_angular_speed,
        "validation_quaternion_error": quaternion_error,
        "validation_rigid_contact_count": contact_count,
    }


def build(
    scenario: str,
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    *,
    articulation_mode: str = "maximal",
) -> SceneHandle:
    """Build one copied FeatherPGS scene for PhoenX or MuJoCo Warp."""
    mem_before = _gpu_used_bytes()
    model = _build_model(scenario, num_worlds)

    if solver_name == "phoenx":
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=solver_iterations,
            articulation_mode=articulation_mode,
            step_layout="multi_world",
        )
        measured_iterations = solver_iterations
    elif solver_name == "mujoco":
        if scenario == "feather_pgs_g1_flat":
            njmax, nconmax = 210, 35
        else:
            njmax, nconmax = 512, 128
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=njmax,
            nconmax=nconmax,
            cone="pyramidal",
            iterations=100,
            ls_iterations=50,
        )
        measured_iterations = 100
    else:
        raise ValueError(f"unknown solver {solver_name!r}")

    state_0 = model.state()
    initial_body_q = state_0.body_q.numpy()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)
    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / substeps
    states = {"current": state_0, "next": state_1}

    def simulate_one_frame() -> None:
        for _ in range(substeps):
            model.collide(states["current"], contacts)
            states["current"].clear_forces()
            solver.step(states["current"], states["next"], control, contacts, sim_dt)
            states["current"], states["next"] = states["next"], states["current"]

    def validate() -> dict[str, float | int]:
        return _validate_simulation_state(states["current"], initial_body_q, contacts)

    wp.synchronize_device()
    setup_bytes = max(0, _gpu_used_bytes() - mem_before)
    return SceneHandle(
        name=scenario,
        solver_name=solver_name,
        num_worlds=num_worlds,
        validate=validate,
        substeps=substeps,
        solver_iterations=measured_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
