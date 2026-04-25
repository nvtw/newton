# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tower-of-Jitter contact-dense benchmark scenario.

Replicates the ``example_tower.Example`` scene (40 layers x 32 planks
stacked on a ring) across ``num_worlds`` worlds. Every plank rests on
3-4 neighbours so per-world colour counts run in the hundreds --
a complementary stress test to the sparse humanoid scenarios where
each world has <50 constraints. Useful when tuning kernel block
dimensions that can overfit to sparse colours.
"""

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.runner import (
    SceneHandle,
    _gpu_used_bytes,
)


# Tower scale trimmed from the 40x32 example default so the benchmark
# stays representative without blowing the 90 s CI budget at larger
# world counts. The per-world contact-density profile is preserved --
# just with fewer planks per ring.
_TOWER_HEIGHT_LAYERS = 8
_BOXES_PER_RING = 16
_PLANK_HX = 1.5
_PLANK_HY = 0.1
_PLANK_HZ = 0.5
_RING_RADIUS = 4.0
_PLANK_DENSITY = 1000.0


def _build_tower_builder() -> newton.ModelBuilder:
    """One tower's worth of planks arranged on a ring, no ground."""
    tower = newton.ModelBuilder()
    tower.default_shape_cfg.ke = 1.0e4
    tower.default_shape_cfg.kd = 5.0e2
    tower.default_shape_cfg.kf = 5.0e3
    tower.default_shape_cfg.mu = 0.75

    half_rot = 2.0 * math.pi / (2 * _BOXES_PER_RING)
    full_rot = 2.0 * half_rot
    orientation_rad = 0.0
    for layer in range(_TOWER_HEIGHT_LAYERS):
        orientation_rad += half_rot
        for _ in range(_BOXES_PER_RING):
            cos_o = math.cos(orientation_rad)
            sin_o = math.sin(orientation_rad)
            world_x = -sin_o * _RING_RADIUS
            world_y = cos_o * _RING_RADIUS
            world_z = 0.5 + layer * (2.0 * _PLANK_HZ + 0.01)
            quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), orientation_rad)
            body = tower.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(world_x), float(world_y), float(world_z)), q=quat
                ),
            )
            tower.add_shape_box(
                body,
                hx=_PLANK_HX,
                hy=_PLANK_HY,
                hz=_PLANK_HZ,
                cfg=newton.ModelBuilder.ShapeConfig(density=_PLANK_DENSITY),
            )
            orientation_rad += full_rot
    return tower


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
    *,
    step_layout: str = "multi_world",
) -> SceneHandle:
    """Build a tower-per-world scene with ``num_worlds`` replicated
    towers driven by ``solver_name`` ('phoenx' or 'mujoco').

    ``step_layout`` is forwarded to :class:`SolverPhoenX`; pass
    ``"single_world"`` together with ``num_worlds=1`` (and a much
    larger per-tower scale) to bench the capture-while path on the
    contact-dense regime it targets.
    """
    device = wp.get_device()
    mem_before = _gpu_used_bytes()

    tower = _build_tower_builder()
    if solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(tower)

    builder = newton.ModelBuilder()
    builder.replicate(tower, num_worlds)
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.add_ground_plane()

    model = builder.finalize()

    fps = 60
    frame_dt = 1.0 / fps

    if solver_name == "phoenx":
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
            step_layout=step_layout,
        )
        outer_steps = 1
        call_dt = frame_dt
    elif solver_name == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=solver_iterations,
            ls_iterations=50,
            njmax=800,
            nconmax=400,
        )
        outer_steps = substeps
        call_dt = frame_dt / substeps
    else:
        raise ValueError(f"unknown solver '{solver_name}'")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.contacts()

    box = {"state_0": state_0, "state_1": state_1}

    def simulate_one_frame() -> None:
        model.collide(box["state_0"], contacts)
        for _ in range(outer_steps):
            box["state_0"].clear_forces()
            solver.step(box["state_0"], box["state_1"], control, contacts, call_dt)
            box["state_0"], box["state_1"] = box["state_1"], box["state_0"]

    wp.synchronize_device()
    setup_bytes = max(0, _gpu_used_bytes() - mem_before)

    return SceneHandle(
        name="tower",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
