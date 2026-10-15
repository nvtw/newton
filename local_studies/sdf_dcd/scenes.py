# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SDF-heavy PhoenX scenes for comparing edge sampling and cached sampling.

Every factory returns a ``TuningScene`` (see ``newton/examples/phoenx/tune_phoenx.py``).
All colliding shapes are meshes with texture SDFs, including the ground slab, so
every pair exercises the SDF path.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx.tune_phoenx import TuningScene

SOLVER_OPTIONS = {
    "step_layout": "single_world",
    "substeps": 4,
    "solver_iterations": 8,
    "velocity_iterations": 1,
}


def _box_mesh(hx, hy, hz, res=64):
    m = newton.Mesh.create_box(hx, hy, hz, duplicate_vertices=False, compute_normals=False, compute_uvs=False)
    m.build_sdf(max_resolution=res, narrow_band_range=(-0.02, 0.02), margin=0.02)
    return m


def _add_slab(b, half=1.5, thick=0.1):
    slab = _box_mesh(half, half, thick, res=128)
    b.add_shape_mesh(-1, mesh=slab, xform=wp.transform((0.0, 0.0, -thick), wp.quat_identity()))


def _drift_metrics(model, start_q):
    """Max displacement of any body from its pose at the first sample [m]."""
    ref = {}

    def metrics(state, _contacts):
        q = state.body_q.numpy()[:, :3]
        if "q" not in ref:
            ref["q"] = q.copy()
        d = np.linalg.norm(q - ref["q"], axis=1)
        v = state.body_qd.numpy()
        return {"drift_m": float(d.max()), "speed_m_s": float(np.linalg.norm(v[:, :3], axis=1).max())}

    return metrics


def make_box_pyramid(levels=6, size=0.05):
    """Pyramid of mesh cubes, placed touching. Ideal result: nothing moves."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.002
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.6
    _add_slab(b)
    cube = _box_mesh(size, size, size)
    for lvl in range(levels):
        n = levels - lvl
        for i in range(n):
            x = (i - 0.5 * (n - 1)) * 2.0 * size * 1.02
            body = b.add_body(xform=wp.transform((x, 0.0, size + 2.0 * size * lvl), wp.quat_identity()))
            b.add_shape_mesh(body, mesh=cube)
    model = b.finalize()
    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        solver_options=dict(SOLVER_OPTIONS),
        extra_metrics=_drift_metrics(model, None),
    )


def make_box_tower(height=10, size=0.05):
    """Column of cubes with alternating yaw; probes stacking stability."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.002
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.6
    _add_slab(b)
    cube = _box_mesh(size, size, size)
    for lvl in range(height):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3 * (lvl % 2))
        body = b.add_body(xform=wp.transform((0.0, 0.0, size + 2.0 * size * lvl), q))
        b.add_shape_mesh(body, mesh=cube)
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), extra_metrics=_drift_metrics(model, None)
    )


def _bunny_mesh():
    from pxr import Usd

    import newton.examples  # noqa: PLC0415
    import newton.usd as newton_usd  # noqa: PLC0415

    stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    src = newton_usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))
    mesh = newton.Mesh(src.vertices, src.indices)
    mesh.build_sdf(max_resolution=128, narrow_band_range=(-0.02, 0.02), margin=0.02)
    return mesh


def _kinetic_energy(model):
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()

    def observe(state):
        qd = state.body_qd.numpy()
        q = state.body_q.numpy()
        e = 0.0
        for i in range(len(mass)):
            v = qd[i, :3]
            w = qd[i, 3:]
            # world angular velocity -> body frame for the body-frame inertia
            x, y, z, s = q[i, 3:]
            r = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * s), 2 * (x * z + y * s)],
                    [2 * (x * y + z * s), 1 - 2 * (x * x + z * z), 2 * (y * z - x * s)],
                    [2 * (x * z - y * s), 2 * (y * z + x * s), 1 - 2 * (x * x + y * y)],
                ]
            )
            wb = r.T @ w
            e += 0.5 * mass[i] * float(v @ v) + 0.5 * float(wb @ inertia[i] @ wb)
        return {"kinetic_energy_j": e}

    return observe


def make_bunny_pile(count=27, seed=0):
    """Bunnies dropped into a pile on an SDF slab (paper Fig. 1/7 style)."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.004
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.5
    _add_slab(b)
    mesh = _bunny_mesh()
    scale = 0.15
    rng = np.random.default_rng(seed)
    side = int(math.ceil(count ** (1.0 / 3.0)))
    for i in range(count):
        lvl, row, col = i // (side * side), (i // side) % side, i % side
        p = ((col - 0.5 * (side - 1)) * 0.3, (row - 0.5 * (side - 1)) * 0.3, 0.2 + 0.3 * lvl)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q = wp.quat_from_axis_angle(wp.vec3(*axis), float(rng.uniform(-np.pi, np.pi)))
        body = b.add_body(xform=wp.transform(p, q))
        b.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale))
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), observables=_kinetic_energy(model)
    )


def _gear_mesh(teeth=16, r_out=0.1, r_in=0.08, half_thick=0.02, bore=0.02):
    """Extruded spur-gear outline with a square tooth profile."""
    pts = []
    n = teeth * 4
    for k in range(n):
        a = 2.0 * math.pi * k / n
        r = r_out if (k % 4) in (1, 2) else r_in
        pts.append((r * math.cos(a), r * math.sin(a)))
    pts = np.asarray(pts)
    m = len(pts)
    verts = np.concatenate(
        [
            np.column_stack([pts, np.full(m, -half_thick)]),
            np.column_stack([pts, np.full(m, half_thick)]),
            [[0.0, 0.0, -half_thick], [0.0, 0.0, half_thick]],
        ]
    ).astype(np.float32)
    cb, ct = 2 * m, 2 * m + 1
    tris = []
    for k in range(m):
        k1 = (k + 1) % m
        tris += [[cb, k1, k], [ct, m + k, m + k1], [k, k1, m + k1], [k, m + k1, m + k]]
    mesh = newton.Mesh(verts, np.asarray(tris, dtype=np.int32).flatten())
    mesh.build_sdf(max_resolution=128, narrow_band_range=(-0.01, 0.01), margin=0.01)
    return mesh


def make_gear_stack(height=6):
    """Spur gears stacked flat with random yaw (paper Fig. 5 right)."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.002
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.6
    _add_slab(b)
    gear = _gear_mesh()
    rng = np.random.default_rng(1)
    for lvl in range(height):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0, math.pi)))
        body = b.add_body(xform=wp.transform((0.01 * (lvl % 2), 0.0, 0.0205 + 0.041 * lvl), q))
        b.add_shape_mesh(body, mesh=gear)
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), extra_metrics=_drift_metrics(model, None)
    )


def make_nut_bolt(scale=1.0):
    """IsaacGym M20 nut threading down a static bolt under gravity."""
    import newton.examples  # noqa: PLC0415
    from newton.examples.contacts.example_nut_bolt_sdf import (  # noqa: PLC0415
        ASSEMBLY_STR,
        ISAACGYM_ENVS_REPO_URL,
        ISAACGYM_NUT_BOLT_FOLDER,
        SHAPE_CFG,
        load_mesh_with_sdf,
    )

    path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
    bolt, bolt_c = load_mesh_with_sdf(str(path / f"factory_bolt_{ASSEMBLY_STR}.obj"), SHAPE_CFG)
    nut, nut_c = load_mesh_with_sdf(str(path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj"), SHAPE_CFG)
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, mu=0.01, gap=0.0005, density=8000.0)
    b.add_shape_mesh(-1, mesh=bolt, xform=wp.transform(bolt_c, wp.quat_identity()), cfg=cfg)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 8)
    nut_body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.041) + wp.quat_rotate(q, nut_c), q))
    b.add_shape_mesh(nut_body, mesh=nut, cfg=cfg)
    model = b.finalize()

    def observe(state):
        q = state.body_q.numpy()[0]
        qd = state.body_qd.numpy()[0]
        return {"nut_z_m": float(q[2]), "nut_spin_rad_s": float(qd[5])}

    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        solver_options={**SOLVER_OPTIONS, "substeps": 5, "solver_iterations": 8},
        observables=observe,
    )


def make_bunny_single():
    """One bunny dropped on the slab; must come to rest."""
    return make_bunny_pile(count=1)


def make_bunny_drops(count=16, seed=3, only=None):
    """Independent bunny drops in random orientations (paper Sec. 4.3)."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.004
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.5
    _add_slab(b)
    mesh = _bunny_mesh()
    rng = np.random.default_rng(seed)
    side = int(math.ceil(math.sqrt(count)))
    for i in range(count):
        p = ((i % side - 0.5 * (side - 1)) * 0.6, (i // side - 0.5 * (side - 1)) * 0.6, 0.25)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q = wp.quat_from_axis_angle(wp.vec3(*axis), float(rng.uniform(-np.pi, np.pi)))
        if only is not None and i != only:
            continue
        body = b.add_body(xform=wp.transform(p, q))
        b.add_shape_mesh(body, mesh=mesh, scale=(0.15, 0.15, 0.15))
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), observables=_kinetic_energy(model)
    )


def _drops(mesh, count=16, seed=3, scale=1.0, gap=0.004):
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = gap
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.5
    _add_slab(b)
    rng = np.random.default_rng(seed)
    side = int(math.ceil(math.sqrt(count)))
    for i in range(count):
        p = ((i % side - 0.5 * (side - 1)) * 0.6, (i // side - 0.5 * (side - 1)) * 0.6, 0.25)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q = wp.quat_from_axis_angle(wp.vec3(*axis), float(rng.uniform(-np.pi, np.pi)))
        body = b.add_body(xform=wp.transform(p, q))
        b.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale))
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), observables=_kinetic_energy(model)
    )


def make_cube_drops():
    return _drops(_box_mesh(0.05, 0.05, 0.05))


def make_sphere_drops():
    m = newton.Mesh.create_sphere(0.06, num_latitudes=24, num_longitudes=24, compute_normals=False, compute_uvs=False)
    m.build_sdf(max_resolution=64, narrow_band_range=(-0.02, 0.02), margin=0.02)
    return _drops(m)


def make_bunny_pile_large():
    """343 bunnies dropped into a pile."""
    scene = make_bunny_pile(count=343)
    return scene


def make_gear_stacks_large(rows=10, height=6):
    """rows x rows independent gear stacks."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.gap = 0.002
    b.default_shape_cfg.margin = 0.0
    b.default_shape_cfg.mu = 0.6
    _add_slab(b, half=0.25 * rows + 0.5)
    gear = _gear_mesh()
    rng = np.random.default_rng(1)
    for r in range(rows):
        for c in range(rows):
            for lvl in range(height):
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0, math.pi)))
                p = (
                    (r - 0.5 * (rows - 1)) * 0.5 + 0.01 * (lvl % 2),
                    (c - 0.5 * (rows - 1)) * 0.5,
                    0.0205 + 0.041 * lvl,
                )
                body = b.add_body(xform=wp.transform(p, q))
                b.add_shape_mesh(body, mesh=gear)
    model = b.finalize()
    return TuningScene(
        model, frame_dt=1.0 / 60.0, solver_options=dict(SOLVER_OPTIONS), extra_metrics=_drift_metrics(model, None)
    )
