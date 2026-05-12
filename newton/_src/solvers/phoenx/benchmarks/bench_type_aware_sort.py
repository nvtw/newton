# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bench: type-aware locality sort for the PGS iterate kernel.

Hypothesis: PhoenX's per-color sort currently orders constraints
within a colour by ``body_min`` (cache locality) but mixes constraint
types arbitrarily within a warp. The iterate kernel's runtime
dispatch ladder (contacts -> cloth-tri -> soft-tet -> cloth-bending
-> joint) is then divergent on mixed-type warps: every branch is
masked through, every read taken.

Promoting the constraint type to the high bits of the locality sort
key clusters same-type cids within each colour, so each warp should
mostly execute a single branch's code. Body-min sub-order is
preserved (24 bits) so cache locality within a type is intact, and
``eid`` in the lowest 24 bits is the deterministic tie-break that
folds the existing pass-1 sort into the same single radix pass.

The sort runs **once per coloring rebuild** (per step, not per
substep iteration), so any added cost is amortized over ~30-100
iterate launches per step. The win is concentrated in mixed
cloth + soft + rigid + contact scenes -- pure rigid or pure cloth
scenes see no change (one type dominates the whole CSR).

This bench builds a representative mixed scene and times the inner
solve (prepare + iterate + relax kernels only, no coloring /
integrate / contact ingest) under graph capture. Runs in two modes:

* ``--mode baseline``: current 2-pass (eid then ``color, body_min``)
* ``--mode type-aware``: single-pass packed key ``(color << 57) |
  (type << 53) | (body_min << 29) | eid_low``

Pick the mode via the ``PHOENX_TYPE_AWARE_LOCALITY_SORT`` env var so
the same script captures both numbers in one run for direct
comparison.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_type_aware_sort \\
        [--frames 1000] [--warmup 30]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_tower_scene(device: wp.Device) -> tuple[PhoenXWorld, newton.Model, "object", "object", "object"]:
    """Rigid-only tower stress: 32-cube vertical stack on ground.

    Pure rigid + contact scene -- a regression check for the
    type-aware sort. Since every cid in this scene is either a
    rigid-rigid contact or a rigid-plane contact (both ``ctype =
    CONTACT_SENTINEL``), the type field in the packed key is
    constant within a colour and the within-colour ordering is
    determined purely by ``body_min`` + ``eid_low`` -- exactly what
    the baseline produces minus the saved pass-1 eid sort. We expect
    a near-neutral or slight-positive delta here.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    half = 0.5
    for i in range(32):
        b = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 + i * 0.999), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(b, hx=half, hy=half, hz=half)
    model = builder.finalize(device=device)
    num_bodies_phoenx = int(model.body_count) + 1
    bodies = body_container_zeros(num_bodies_phoenx, device=device)
    bodies.orientation.assign(
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        )
    )
    state_init = model.state()
    wp.launch(
        init_phoenx_bodies_kernel,
        dim=int(model.body_count),
        inputs=[
            model.body_q, state_init.body_qd, model.body_com,
            model.body_inv_mass, model.body_inv_inertia,
        ],
        outputs=[
            bodies.position, bodies.orientation, bodies.velocity,
            bodies.angular_velocity, bodies.inverse_mass, bodies.inverse_inertia,
            bodies.inverse_inertia_world, bodies.motion_type, bodies.body_com,
        ],
        device=device,
    )
    constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
    world = PhoenXWorld(
        bodies=bodies, constraints=constraints, num_joints=0,
        num_particles=0, num_cloth_triangles=0, num_soft_tetrahedra=0,
        num_worlds=1, substeps=4, solver_iterations=16, rigid_contact_max=8192,
        step_layout="single_world", mass_splitting=True,
        max_colored_partitions=12, device=device,
    )
    world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", contact_matching="sticky")
    state = model.state()
    contacts = pipeline.contacts()
    return world, model, state, contacts, pipeline


def _build_mixed_scene(device: wp.Device) -> tuple[PhoenXWorld, newton.Model, "object", "object"]:
    """Mixed scene: rigid pyramids + cloth grid + soft-tet block + ground.

    All three constraint types (contact, cloth-tri, soft-tet) get
    populated, plus the trivially-many contacts (cloth-particle vs
    ground, soft-particle vs ground, rigid-vs-ground, rigid-vs-rigid)
    that exercise the per-color warp coherence -- a single colour at
    rest in this scene typically holds a mix of all of the above, so
    sorting by type really does change which branch each warp lane
    enters.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)

    # Pyramid stack of rigid cubes (12 bodies, lots of contacts).
    half = 0.1
    spacing = 2.0 * half + 0.002
    for layer in range(3):
        side = 3 - layer
        base = -0.5 * (side - 1) * spacing
        base_z = (layer + 0.5) * spacing
        for row in range(side):
            for col in range(side):
                b = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(base + col * spacing, base + row * spacing, base_z),
                        q=wp.quat_identity(),
                    ),
                    mass=1.0,
                )
                builder.add_shape_box(b, hx=half, hy=half, hz=half)

    # Cloth grid (16x16 -> 512 triangles, 289 particles). Pinned on
    # two opposite edges so it stays in place; the bottom of the
    # cloth touches the pyramid for cloth-rigid contacts.
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e7, 0.3)
    cloth_dim = 16
    cloth_cell = 0.08
    cloth_z = 1.6
    builder.add_cloth_grid(
        pos=wp.vec3(-cloth_dim * cloth_cell * 0.5, -cloth_dim * cloth_cell * 0.5, cloth_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=cloth_dim,
        dim_y=cloth_dim,
        cell_x=cloth_cell,
        cell_y=cloth_cell,
        mass=0.02,
        fix_left=True,
        fix_right=True,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        particle_radius=0.04,
    )

    # Soft-tet block (4x4x4 -> 320 tets, 125 particles).
    k_mu, k_lambda = soft_tet_lame_from_youngs_poisson(5.0e7, 0.3)
    soft_res = 4
    soft_size = 0.4
    soft_cell = soft_size / soft_res
    builder.add_soft_grid(
        pos=wp.vec3(-soft_size * 0.5, -soft_size * 0.5, 2.4),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=soft_res,
        dim_y=soft_res,
        dim_z=soft_res,
        cell_x=soft_cell,
        cell_y=soft_cell,
        cell_z=soft_cell,
        density=500.0,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=0.0,
        add_surface_mesh_edges=False,
    )

    model = builder.finalize(device=device)
    num_bodies_phoenx = int(model.body_count) + 1
    bodies = body_container_zeros(num_bodies_phoenx, device=device)
    bodies.orientation.assign(
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        )
    )
    state_init = model.state()
    wp.launch(
        init_phoenx_bodies_kernel,
        dim=int(model.body_count),
        inputs=[
            model.body_q,
            state_init.body_qd,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
        ],
        outputs=[
            bodies.position,
            bodies.orientation,
            bodies.velocity,
            bodies.angular_velocity,
            bodies.inverse_mass,
            bodies.inverse_inertia,
            bodies.inverse_inertia_world,
            bodies.motion_type,
            bodies.body_com,
        ],
        device=device,
    )

    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0,
        num_cloth_triangles=int(model.tri_count),
        num_soft_tetrahedra=int(model.tet_count),
        device=device,
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=int(model.tri_count),
        num_soft_tetrahedra=int(model.tet_count),
        num_worlds=1,
        substeps=4,
        solver_iterations=16,
        rigid_contact_max=16384,
        step_layout="single_world",
        mass_splitting=True,
        max_colored_partitions=12,
        device=device,
    )
    world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
    world.populate_cloth_triangles_from_model(model)
    world.populate_soft_tetrahedra_from_model(model, beta_mu=5.0)
    pipeline = world.setup_cloth_collision_pipeline(
        model,
        cloth_thickness=0.005,
        cloth_gap=0.010,
        soft_body_thickness=0.005,
        soft_body_gap=0.010,
        rigid_contact_max=16384,
    )
    state = model.state()
    contacts = pipeline.contacts()
    return world, model, state, contacts


def _step(world: PhoenXWorld, model, state, contacts, *, collision_pipeline=None) -> None:
    if collision_pipeline is not None:
        # Rigid-only scene: standard CollisionPipeline + model.collide.
        model.collide(state, contacts=contacts, collision_pipeline=collision_pipeline)
        world.step(1.0 / 60.0, contacts=contacts, shape_body=model.shape_body)
    else:
        # Mixed-deformable scene: cloth-aware pipeline lives on the world.
        world.collide(state, contacts)
        world.step(1.0 / 60.0, contacts=contacts)


def _measure(world: PhoenXWorld, model, state, contacts, *, frames: int, warmup: int, collision_pipeline=None) -> dict[str, float]:
    device = wp.get_device()

    # Warm-up out of capture -- populates contacts, runs coloring, gets
    # the iterate kernel cached.
    for _ in range(warmup):
        _step(world, model, state, contacts, collision_pipeline=collision_pipeline)
    wp.synchronize_device(device)

    # Capture one step into a graph; replay it ``frames`` times.
    with wp.ScopedCapture(device=device) as cap:
        _step(world, model, state, contacts, collision_pipeline=collision_pipeline)
    graph = cap.graph

    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(frames):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    ms_per_step = 1000.0 * (t1 - t0) / max(frames, 1)

    # Diagnostic: how many active constraints, how many colours.
    n_active = int(world._num_active_constraints.numpy()[0])
    n_colors = int(world._partitioner.num_colors.numpy()[0])
    eids = world._partitioner.element_ids_by_color.numpy()[:n_active]
    color_starts = world._partitioner.color_starts.numpy()
    n_joint_like = world.num_joints + world.num_cloth_triangles + world.num_cloth_bending + world.num_soft_tetrahedra

    # Within-colour same-type-pair fraction: how often consecutive
    # eids in a colour have the same constraint type. High = warps
    # see one type at a time.
    same_pairs = 0
    total_pairs = 0
    for c in range(n_colors):
        a = int(color_starts[c])
        b = int(color_starts[c + 1])
        if b - a < 2:
            continue
        slice_eids = eids[a:b]
        kinds = np.where(slice_eids >= n_joint_like, 1, 0)
        # Crude: contact vs non-contact. Sub-categorising non-contact
        # by reading constraint_get_type from host requires a
        # numpy copy of constraints.data which is heavy; the
        # contact-vs-the-rest split is the dominant divergence axis.
        same_pairs += int((kinds[:-1] == kinds[1:]).sum())
        total_pairs += b - a - 1
    same_type_frac = same_pairs / max(total_pairs, 1)

    return {
        "ms_per_step": ms_per_step,
        "n_active": n_active,
        "n_colors": n_colors,
        "same_type_pair_frac": same_type_frac,
    }


_ORIGINAL_SET_TYPE_SOURCE = None


def _run_one_mode(*, type_aware: bool, frames: int, warmup: int, scene: str) -> dict[str, float]:
    """Build the scene and time it under one sort configuration.

    When ``type_aware=False``, monkey-patches
    :meth:`IncrementalContactPartitioner.set_type_source` to a no-op
    so :class:`PhoenXWorld` builds without wiring the type-aware
    sort. The world otherwise always calls ``set_type_source`` in
    ``__init__``; this hook is the cleanest way to compare the two
    paths inside one Python process without mutating module state.
    """
    from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import IncrementalContactPartitioner

    global _ORIGINAL_SET_TYPE_SOURCE
    if _ORIGINAL_SET_TYPE_SOURCE is None:
        _ORIGINAL_SET_TYPE_SOURCE = IncrementalContactPartitioner.set_type_source
    if type_aware:
        IncrementalContactPartitioner.set_type_source = _ORIGINAL_SET_TYPE_SOURCE
    else:
        IncrementalContactPartitioner.set_type_source = lambda self, *a, **k: None

    device = wp.get_device()
    collision_pipeline = None
    if scene == "mixed":
        world, model, state, contacts = _build_mixed_scene(device)
    elif scene == "tower":
        world, model, state, contacts, collision_pipeline = _build_tower_scene(device)
    else:
        raise ValueError(f"unknown scene {scene!r}")
    print(
        f"  scene: bodies={int(model.body_count)} particles={int(model.particle_count)} "
        f"tris={int(model.tri_count)} tets={int(model.tet_count)}"
    )

    results = []
    for trial in range(3):
        r = _measure(
            world, model, state, contacts,
            frames=frames, warmup=warmup if trial == 0 else 1,
            collision_pipeline=collision_pipeline,
        )
        results.append(r)
        print(
            f"  trial {trial}: ms/step={r['ms_per_step']:.3f}  "
            f"n_active={r['n_active']}  n_colors={r['n_colors']}  "
            f"same-contact-pair-fraction={r['same_type_pair_frac']:.3f}"
        )
    return {
        "median_ms": statistics.median(r["ms_per_step"] for r in results),
        "n_active": results[-1]["n_active"],
        "same_type_pair_frac": results[-1]["same_type_pair_frac"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=500, help="captured-step replays for timing")
    parser.add_argument("--warmup", type=int, default=30, help="frames stepped before capture (contacts converge)")
    args = parser.parse_args()

    device = wp.get_device()
    if not device.is_cuda:
        raise SystemExit("This bench requires CUDA (graph capture).")

    for scene in ("mixed", "tower"):
        print(f"\n###### scene = {scene!r} ######")
        print("=== baseline (type-aware locality sort OFF) ===")
        baseline = _run_one_mode(type_aware=False, frames=args.frames, warmup=args.warmup, scene=scene)
        print(f"baseline median ms/step: {baseline['median_ms']:.3f}\n")

        print("=== type-aware locality sort ON ===")
        type_aware = _run_one_mode(type_aware=True, frames=args.frames, warmup=args.warmup, scene=scene)
        print(f"type-aware median ms/step: {type_aware['median_ms']:.3f}\n")

        speedup = baseline["median_ms"] / max(type_aware["median_ms"], 1.0e-6)
        delta_pct = (type_aware["median_ms"] - baseline["median_ms"]) / baseline["median_ms"] * 100.0
        print(f"[{scene}] speedup: {speedup:.3f}x  ({delta_pct:+.1f}% change in ms/step)")
        print(f"[{scene}] same-contact-pair-fraction: baseline={baseline['same_type_pair_frac']:.3f} "
              f"type-aware={type_aware['same_type_pair_frac']:.3f}")


if __name__ == "__main__":
    wp.init()
    main()
