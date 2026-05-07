# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tri-on-cube rest test: rigid first-class triangle vs cloth triangle.

Reference scene: a 1 m^3 cube resting on the ground plane with a single
triangle placed just above the cube top. The triangle should remain
essentially motionless under gravity for both variants:

* :class:`TestRigidTriRestOnCube` -- rigid first-class
  ``GeoType.TRIANGLE`` shape (static, ``body=-1``).
* :class:`TestClothTriRestOnCube` -- cloth (3 free particles + XPBD
  elasticity row, hanging-cloth material parameters).

The rigid variant is a regression check: the rigid first-class triangle
contact path is known to produce stable resting collisions. The cloth
variant verifies that the unified narrow-phase path (cloth tris stamped
as canonical ``GeoType.TRIANGLE``) plus the access-mode-driven contact
position writeback give cloth particles equally stable resting contacts.

CUDA-only; the high-level :class:`~newton.solvers.SolverPhoenX` wraps
:class:`~newton._src.solvers.phoenx.solver_phoenx.PhoenXWorld` and uses
CUDA graph capture internally on every step replay.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 8
SOLVER_ITERATIONS = 8
NUM_FRAMES = 120  # 2 s

# 1 m^3 cube resting on ground (cube body at z = CUBE_HE).
CUBE_HE = 0.5
CUBE_TOP = 2.0 * CUBE_HE  # 1.0 m

# Particle / material parameters lifted from the hanging-cloth example.
PARTICLE_MASS = 0.05
CLOTH_MARGIN = 0.01
YOUNGS_MODULUS = 5.0e8
POISSON_RATIO = 0.3

# Triangle shape: 0.5 m equilateral, centred above cube top at a small
# initial gap. Vertices share the same world coordinates between the
# rigid and cloth variants so the two scenes are directly comparable.
TRI_SIDE = 0.5
TRI_H = TRI_SIDE * np.sqrt(3.0) / 2.0
TRI_VERTS_2D = np.array(
    [
        (-0.5 * TRI_SIDE, -TRI_H / 3.0),
        (+0.5 * TRI_SIDE, -TRI_H / 3.0),
        (0.0, +2.0 * TRI_H / 3.0),
    ],
    dtype=np.float32,
)
TRI_REST_Z = CUBE_TOP + 0.005  # 5 mm above cube top (within speculative gap)

# Pass tolerance: 3 mm peak displacement over the 2 s run. Tight enough
# that any regression in the cloth contact path (e.g. velocity-only
# writes that XPBD's substep-exit recovery wipes out, or a missing
# access-mode flip after the velocity update) is caught -- such bugs
# raise the drift well above this threshold.
PASS_DISP_MM = 3.0


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion ``(x, y, z, w)``."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
        w = (R[2, 1] - R[1, 2]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
        w = (R[0, 2] - R[2, 0]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        w = (R[1, 0] - R[0, 1]) / s
    return np.array([x, y, z, w], dtype=np.float32)


def _add_cube(builder: newton.ModelBuilder) -> int:
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, CUBE_HE), q=wp.quat_identity()),
    )
    builder.add_shape_box(
        cube_body,
        hx=CUBE_HE,
        hy=CUBE_HE,
        hz=CUBE_HE,
        cfg=newton.ModelBuilder.ShapeConfig(density=60.0, mu=0.6),
    )
    return cube_body


def _add_rigid_static_triangle(builder: newton.ModelBuilder) -> None:
    """Place a face-up canonical ``GeoType.TRIANGLE`` static shape so its
    world vertices match the cloth-tri vertex layout."""
    a = np.array([TRI_VERTS_2D[0, 0], TRI_VERTS_2D[0, 1], 0.0], dtype=np.float32)
    b = np.array([TRI_VERTS_2D[1, 0], TRI_VERTS_2D[1, 1], 0.0], dtype=np.float32)
    c = np.array([TRI_VERTS_2D[2, 0], TRI_VERTS_2D[2, 1], 0.0], dtype=np.float32)
    edge_ab = b - a
    edge_ac = c - a
    ab_len = float(np.linalg.norm(edge_ab))
    axis_z = edge_ab / ab_len
    c_z = float(np.dot(edge_ac, axis_z))
    ac_perp = edge_ac - c_z * axis_z
    c_y = float(np.linalg.norm(ac_perp))
    axis_y = ac_perp / c_y
    axis_x = np.cross(axis_y, axis_z)
    R = np.column_stack([axis_x, axis_y, axis_z])
    q = _mat_to_quat(R)
    builder.add_shape_triangle(
        body=-1,
        xform=wp.transform(
            wp.vec3(float(a[0]), float(a[1]), TRI_REST_Z),
            wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        ),
        edge_ab=ab_len,
        point_c=(c_y, c_z),
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6),
    )


def _add_cloth_triangle(builder: newton.ModelBuilder) -> None:
    """Three free particles + one elastic triangle row at ``TRI_REST_Z``."""
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(
        YOUNGS_MODULUS, POISSON_RATIO
    )
    p_idx = []
    for i in range(3):
        p_idx.append(
            builder.add_particle(
                pos=wp.vec3(
                    float(TRI_VERTS_2D[i, 0]),
                    float(TRI_VERTS_2D[i, 1]),
                    TRI_REST_Z,
                ),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=PARTICLE_MASS,
                radius=0.5 * TRI_SIDE,
            )
        )
    builder.add_triangle(
        i=p_idx[0],
        j=p_idx[1],
        k=p_idx[2],
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
    )


def _build_model(add_triangle_fn) -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    _add_cube(builder)
    add_triangle_fn(builder)
    builder.gravity = -GRAVITY
    return builder.finalize()


def _run(add_triangle_fn) -> dict:
    """Step the scene for ``NUM_FRAMES`` and return rest-state metrics.

    For rigid scenes (no particles), only cube drift is reported. For
    cloth scenes, particle peak displacement / velocity / final z are
    sampled on every frame from the solver-internal particle state
    (``solver.world.particles.position``); the Newton ``State.particle_q``
    is not written back by the solver wrapper.
    """
    model = _build_model(add_triangle_fn)
    solver = newton.solvers.SolverPhoenX(
        model, substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    contacts = model.contacts(collision_pipeline=pipeline)

    has_particles = int(model.particle_count) > 0
    initial_cube = state_0.body_q.numpy()[0, :3].copy()
    initial_p = (
        solver.world.particles.position.numpy().copy() if has_particles else None
    )

    cube_max_xy = 0.0
    cube_max_dz = 0.0
    p_disp_max = 0.0
    p_vel_max = 0.0
    dt = 1.0 / FPS
    for _ in range(NUM_FRAMES):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        cube_pos = state_0.body_q.numpy()[0, :3]
        cube_max_xy = max(cube_max_xy, float(np.linalg.norm(cube_pos[:2] - initial_cube[:2])))
        cube_max_dz = max(cube_max_dz, abs(float(cube_pos[2] - initial_cube[2])))
        if has_particles:
            tp = solver.world.particles.position.numpy()
            tv = solver.world.particles.velocity.numpy()
            p_disp_max = max(p_disp_max, float(np.linalg.norm(tp - initial_p, axis=1).max()))
            p_vel_max = max(p_vel_max, float(np.linalg.norm(tv, axis=1).max()))

    return {
        "cube_max_xy_drift": cube_max_xy,
        "cube_max_z_drift": cube_max_dz,
        "particle_max_disp": p_disp_max,
        "particle_max_vel": p_vel_max,
    }


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX rest tests run on CUDA only")
class TestRigidTriRestOnCube(unittest.TestCase):
    """Rigid first-class triangle on top of a 1 m^3 cube must not move."""

    def test_static_rigid_triangle_does_not_move_cube(self) -> None:
        r = _run(_add_rigid_static_triangle)
        self.assertLess(
            r["cube_max_xy_drift"] * 1000.0,
            PASS_DISP_MM,
            f"cube acquired lateral drift > {PASS_DISP_MM} mm: {r['cube_max_xy_drift']*1000:.3f} mm",
        )
        self.assertLess(
            r["cube_max_z_drift"] * 1000.0,
            PASS_DISP_MM,
            f"cube acquired vertical drift > {PASS_DISP_MM} mm: {r['cube_max_z_drift']*1000:.3f} mm",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX rest tests run on CUDA only")
class TestClothTriRestOnCube(unittest.TestCase):
    """Cloth triangle on top of a 1 m^3 cube must not appreciably move.

    Uses the hanging-cloth example's material parameters
    (``YOUNGS_MODULUS = 5e8 Pa``, ``POISSON_RATIO = 0.3``,
    ``CLOTH_MARGIN = 0.01 m``, ``PARTICLE_MASS = 0.05 kg``) to keep the
    test scenario consistent with PhoenX's most-exercised cloth setup.
    """

    def test_cloth_triangle_rests_on_cube(self) -> None:
        r = _run(_add_cloth_triangle)
        self.assertLess(
            r["particle_max_disp"] * 1000.0,
            PASS_DISP_MM,
            f"cloth particle drifted > {PASS_DISP_MM} mm: {r['particle_max_disp']*1000:.3f} mm",
        )
        self.assertLess(
            r["cube_max_xy_drift"] * 1000.0,
            PASS_DISP_MM,
            f"cube acquired lateral drift from cloth: {r['cube_max_xy_drift']*1000:.3f} mm",
        )
        self.assertLess(
            r["cube_max_z_drift"] * 1000.0,
            PASS_DISP_MM,
            f"cube acquired vertical drift from cloth: {r['cube_max_z_drift']*1000:.3f} mm",
        )


if __name__ == "__main__":
    unittest.main()
