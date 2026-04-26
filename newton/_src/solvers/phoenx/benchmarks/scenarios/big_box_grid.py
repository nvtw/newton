# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-big-world contact-dense benchmark scenario.

Stacks a NxN grid of unit cubes M layers deep, all in a single
PhoenX world. Designed to exercise the ``step_layout="single_world"``
path -- thousands of bodies + a colour graph that fans out across
the whole device. ``num_worlds`` is fixed to 1 by construction; the
existing ``num_worlds`` argument controls how many replica scenes
the host benches in parallel via the runner harness, but each
SolverPhoenX instance still sees one big world.

Useful for measuring the crossover where ``"single_world"`` starts
to beat ``"multi_world"`` (``"multi_world"`` ties one warp per
world and stalls when a single world swallows tens of colours
each with thousands of cids).
"""

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.runner import (
    SceneHandle,
    _gpu_used_bytes,
)

# ~3600 boxes / world by default. Big enough for single_world to
# pull ahead but small enough to fit on a 24 GB GPU without
# touching the rigid_contact_max safety factor.
_GRID_SIDE = 30
_LAYERS = 4
_BOX_HE = 0.5
_GAP = 0.05
_DENSITY = 1000.0


def _build_grid_builder() -> newton.ModelBuilder:
    """One world of ``_GRID_SIDE^2 * _LAYERS`` boxes plus a ground plane."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke = 1.0e4
    b.default_shape_cfg.kd = 5.0e2
    b.default_shape_cfg.kf = 5.0e3
    b.default_shape_cfg.mu = 0.5
    spacing = 2.0 * _BOX_HE + _GAP
    half = (_GRID_SIDE - 1) * 0.5 * spacing
    for L in range(_LAYERS):
        z = _BOX_HE + L * (2.0 * _BOX_HE + 0.01)
        for ix in range(_GRID_SIDE):
            for iy in range(_GRID_SIDE):
                x = ix * spacing - half
                y = iy * spacing - half
                body = b.add_body(xform=wp.transform(p=wp.vec3(x, y, z)))
                b.add_shape_box(
                    body,
                    hx=_BOX_HE,
                    hy=_BOX_HE,
                    hz=_BOX_HE,
                    cfg=newton.ModelBuilder.ShapeConfig(density=_DENSITY),
                )
    return b


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
    *,
    step_layout: str = "single_world",
) -> SceneHandle:
    """Build a NxN grid stack as ``num_worlds`` replica worlds.

    Set ``step_layout`` to compare layouts on the same scene (default
    is ``"single_world"`` -- the regime this scenario was chosen to
    showcase).
    """
    device = wp.get_device()
    mem_before = _gpu_used_bytes()

    grid = _build_grid_builder()
    if solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(grid)

    builder = newton.ModelBuilder()
    builder.replicate(grid, num_worlds)
    builder.add_ground_plane()

    # ``skip_shape_contact_pairs=True``: the precomputed shape-pair
    # list is only used by the ``"explicit"`` broad phase. Building
    # it is an O(N^2) Python loop -- avoid it under SAP.
    model = builder.finalize(skip_shape_contact_pairs=True)

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
            njmax=10000,
            nconmax=5000,
        )
        outer_steps = substeps
        call_dt = frame_dt / substeps
    else:
        raise ValueError(f"unknown solver '{solver_name}'")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    # SAP broad phase: NXN allocates the worst-case ``N*(N-1)/2``
    # candidate-pair buffer per world; for 3600 boxes that's already
    # ~6.5 M pairs of slack we can avoid by letting SAP report only
    # actual overlaps.
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        contact_matching="sticky",
    )
    model._collision_pipeline = pipeline
    contacts = pipeline.contacts()

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
        name=f"big_box_grid_{step_layout}",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
