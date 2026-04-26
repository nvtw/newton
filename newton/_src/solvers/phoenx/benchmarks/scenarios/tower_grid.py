# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-tower contact-dense benchmark scenario.

Tiles ``GRID_SIDE x GRID_SIDE`` copies of the
:mod:`newton._src.solvers.phoenx.examples.example_tower` 40-layer
plank stack on a 2D grid in a single PhoenX world. Designed as the
big-scene complement to the small ``big_box_grid`` scenario:
thousands of bodies, very dense contact graph, hundreds of colours
per substep -- exactly the regime where ``step_layout="single_world"``
optimisations need to hold up.

A 6x6 grid of 40-layer towers is ~46 k bodies and the colour graph
fans out across the entire device.

``num_worlds`` is fixed to 1 by construction (single-world layout).
The runner harness's ``num_worlds`` argument controls how many
replica scenes the host benches in parallel; each
SolverPhoenX instance still sees one big multi-tower world.
"""

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.runner import (
    SceneHandle,
    _gpu_used_bytes,
)

# Defaults tuned to land at "interesting" colour counts without
# needing every test machine to have 40 GB of headroom. 6x6 towers x
# 40 layers x 32 planks = 30720 bodies; on top of the box +
# ground-plane primitives that lands at 31 k contact columns.
_GRID_SIDE = 6
_TOWER_LAYERS = 40
_BOXES_PER_RING = 32
_PLANK_HX = 1.5
_PLANK_HY = 0.1
_PLANK_HZ = 0.5
_RING_RADIUS = 19.5
_HALF_ROTATION_STEP = 2.0 * math.pi / 64.0
_FULL_ROTATION_STEP = 2.0 * _HALF_ROTATION_STEP
_PLANK_DENSITY = 1000.0

# Per-tower footprint: 2 * (RING_RADIUS + PLANK_HX) plus a generous
# inter-tower gap so towers settle independently without sharing
# contact bodies (we want each tower's stack stiffness measured, not
# their cross-coupling).
_TOWER_FOOTPRINT = 2.0 * (_RING_RADIUS + _PLANK_HX)
_TOWER_GAP = 5.0
_TILE_SPACING = _TOWER_FOOTPRINT + _TOWER_GAP


def _add_one_tower(builder: newton.ModelBuilder, *, center_x: float, center_y: float, layers: int) -> None:
    """Append one circular plank tower at ``(center_x, center_y)``.

    Mirrors :func:`example_tower._build_scene`'s plank loop: alternating
    half-step + full-step orientation walk per layer, +Z up, plank
    half-extents matching the example.
    """
    orientation_rad = 0.0
    for layer in range(layers):
        orientation_rad += _HALF_ROTATION_STEP
        for _ in range(_BOXES_PER_RING):
            cos_o = math.cos(orientation_rad)
            sin_o = math.sin(orientation_rad)
            local_x = 0.0
            local_y = _RING_RADIUS
            local_z = 0.5 + layer
            world_x = cos_o * local_x - sin_o * local_y + center_x
            world_y = sin_o * local_x + cos_o * local_y + center_y
            world_z = local_z
            quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), orientation_rad)
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(world_x), float(world_y), float(world_z)),
                    q=quat,
                ),
            )
            builder.add_shape_box(
                body,
                hx=_PLANK_HX,
                hy=_PLANK_HY,
                hz=_PLANK_HZ,
                cfg=newton.ModelBuilder.ShapeConfig(density=_PLANK_DENSITY),
            )
            orientation_rad += _FULL_ROTATION_STEP


def _build_grid_of_towers(grid_side: int = _GRID_SIDE, layers: int = _TOWER_LAYERS) -> newton.ModelBuilder:
    """One world: ``grid_side x grid_side`` towers of ``layers`` rings."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke = 1.0e4
    b.default_shape_cfg.kd = 5.0e2
    b.default_shape_cfg.kf = 5.0e3
    b.default_shape_cfg.mu = 0.5
    half = (grid_side - 1) * 0.5 * _TILE_SPACING
    for ix in range(grid_side):
        for iy in range(grid_side):
            cx = ix * _TILE_SPACING - half
            cy = iy * _TILE_SPACING - half
            _add_one_tower(b, center_x=cx, center_y=cy, layers=layers)
    return b


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
    *,
    step_layout: str = "single_world",
    grid_side: int = _GRID_SIDE,
    layers: int = _TOWER_LAYERS,
) -> SceneHandle:
    """Build a ``grid_side x grid_side`` array of plank towers.

    Defaults to ``step_layout="single_world"`` -- the regime this
    scenario exercises. Pass ``step_layout="multi_world"`` if you
    want to compare layouts on the same scene.
    """
    device = wp.get_device()
    mem_before = _gpu_used_bytes()

    grid = _build_grid_of_towers(grid_side=grid_side, layers=layers)
    if solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(grid)

    builder = newton.ModelBuilder()
    builder.replicate(grid, num_worlds)
    builder.add_ground_plane()

    # ``skip_shape_contact_pairs=True``: the precomputed shape-pair
    # list is only consumed by the ``"explicit"`` broad phase mode.
    # Building it is an O(N^2) Python loop -- on a 6x6 tower grid
    # (~30 k shapes) that's ~450 M iterations and several GB of
    # contiguous wp.vec2i. Skipping it is mandatory once the scene
    # outgrows a few thousand shapes; safe under SAP / NXN.
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
            njmax=200000,
            nconmax=100000,
        )
        outer_steps = substeps
        call_dt = frame_dt / substeps
    else:
        raise ValueError(f"unknown solver '{solver_name}'")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    # SAP broad phase + explicit caps:
    # * ``broad_phase="sap"`` so candidate pairs are reported
    #   incrementally rather than allocated as the worst-case
    #   ``N*(N-1)/2`` slab.
    # * ``shape_pairs_max`` / ``rigid_contact_max`` follow the
    #   kapla-tower convention: an explicit budget proportional to
    #   the body count, with ~25% headroom over what the scene
    #   actually emits during settling.
    n_bricks = grid_side * grid_side * layers * _BOXES_PER_RING
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        contact_matching="sticky",
        # ~75 contact-candidate pairs per brick (4-6 neighbours x 12 corners)
        shape_pairs_max=max(50_000, 75 * n_bricks),
        # ~32 actual contacts per brick at peak (8 neighbours x 4 corners
        # within reach during compression).
        rigid_contact_max=max(50_000, 32 * n_bricks),
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
        name=f"tower_grid_{grid_side}x{grid_side}_{step_layout}",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
