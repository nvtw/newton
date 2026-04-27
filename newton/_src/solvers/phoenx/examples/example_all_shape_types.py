# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# All-shape-types stress test for PhoenX.
#
# A procedural 6x6 terrain mesh with an 8x8x15 = 960-body grid of
# mixed primitives (sphere / box / capsule / cylinder / cone /
# icosahedron-convex-hull) raining down on it. Exercises every
# shape type that PhoenX + Newton's narrow phase support in a
# single scene -- handy as a smoke / correctness check and as a
# perf regression guard for mixed-primitive contact generation.
#
# Scene parameters mirror ``newton/examples/basic/example_basic_shapes6.py``
# so the two demos can be compared side-by-side; the only difference
# is the solver (PhoenX here, XPBD / MuJoCo there).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_all_shape_types
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.geometry import create_mesh_terrain
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    default_cone_half_extents,
    default_cylinder_half_extents,
    default_mesh_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

# --- Scene constants ---------------------------------------------------

#: Procedural terrain grid (6x6 blocks of 5x5 m each).
TERRAIN_GRID_SIZE = (6, 6)
TERRAIN_BLOCK_SIZE = (5.0, 5.0)
TERRAIN_TYPES = ["flat", "wave", "random_grid", "pyramid_stairs"]
TERRAIN_PARAMS = {
    "pyramid_stairs": {"step_width": 0.4, "step_height": 0.05, "platform_width": 0.8},
    "random_grid": {"grid_width": 0.4, "grid_height_range": (0, 0.1)},
    "wave": {"wave_amplitude": 0.2, "wave_frequency": 1.5},
}
TERRAIN_SEED = 42
TERRAIN_OFFSET_P = (-15.0, -15.0, -0.5)

#: Body grid configuration -- size, spacing, jitter, starting corner.
GRID_SIZE_X = 8
GRID_SIZE_Y = 8
GRID_SIZE_Z = 15
GRID_SPACING = 1.5
GRID_OFFSET = (-10.0, -10.0, 2.0)
POSITION_RANDOMNESS = 0.2
RNG_SEED = 42

#: Per-shape dimensions.
SPHERE_R = 0.3
BOX_HE = 0.3
CAPSULE_R = 0.2
CAPSULE_HH = 0.4
CYLINDER_R = 0.25
CYLINDER_HH = 0.35
CONE_R = 0.3
CONE_HH = 0.4
ICO_RADIUS = 0.35

SHAPE_TYPES = ("sphere", "box", "capsule", "cylinder", "cone", "icosahedron")


def _build_icosahedron_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat-shaded icosahedron mesh.

    Returns ``(vertices, normals, indices)`` with one duplicated set
    of vertices per triangular face so each face gets its own normal
    (flat shading). Geometry-wise it's a standard 12-vertex / 20-face
    icosahedron projected onto the sphere of radius :data:`ICO_RADIUS`.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    base = base / np.linalg.norm(base, axis=1, keepdims=True) * ICO_RADIUS
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    verts: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    indices: list[int] = []
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = base[face[0]], base[face[1]], base[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n)
        verts.extend([v0, v1, v2])
        normals.extend([n, n, n])
        b = face_idx * 3
        indices.extend([b, b + 1, b + 2])
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
    )


class Example(PortedExample):
    """All-shape-types pile on a procedural terrain, run on PhoenX."""

    fps = 60
    sim_substeps = 8
    solver_iterations = 8
    # 960 dynamic bodies + a heavy terrain mesh: NXN broad-phase tests
    # ~460k pairs every frame, SAP collapses that to near-linear.
    broad_phase = "sap"
    # Single big world: PhoenX's ``single_world`` layout drives the
    # global Jones-Plassmann colouring with per-colour persistent grid
    # launches via ``wp.capture_while`` -- the regime that wins for
    # one large world (vs. ``multi_world``'s per-world fast-tail
    # kernels tuned for thousands of small worlds). Same trio of
    # perf knobs as ``example_b2d_double_domino``.
    step_layout = "single_world"
    # Skip contact arrows so ``viewer.log_state`` stays on ViewerGL's
    # CUDA-OpenGL interop path (no per-frame host sync). With 960
    # dynamic bodies in mixed-primitive contact the arrows would be
    # unreadable anyway.
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        # ---- Static terrain mesh ------------------------------------
        terrain_vertices, terrain_indices = create_mesh_terrain(
            grid_size=TERRAIN_GRID_SIZE,
            block_size=TERRAIN_BLOCK_SIZE,
            terrain_types=TERRAIN_TYPES,
            terrain_params=TERRAIN_PARAMS,
            seed=TERRAIN_SEED,
        )
        terrain_mesh = newton.Mesh(terrain_vertices, terrain_indices)
        builder.add_shape_mesh(
            body=-1,
            mesh=terrain_mesh,
            xform=wp.transform(p=wp.vec3(*TERRAIN_OFFSET_P), q=wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        # ---- Icosahedron convex-hull source mesh --------------------
        ico_vertices, ico_normals, ico_indices = _build_icosahedron_mesh()
        ico_mesh = newton.Mesh(ico_vertices, ico_indices, normals=ico_normals)
        # Vertex set used for the convex-hull picking OBB. Flat-shaded
        # duplicates give the same AABB as the dedup'd base set, so
        # passing the duplicated array directly is fine and saves the
        # separate base-vertex bookkeeping.
        ico_base_for_obb = ico_vertices.reshape(-1, 3)

        # ---- Body grid ----------------------------------------------
        rng = np.random.default_rng(RNG_SEED)
        extents: list = []
        shape_index = 0
        for ix in range(GRID_SIZE_X):
            for iy in range(GRID_SIZE_Y):
                for iz in range(GRID_SIZE_Z):
                    bx = GRID_OFFSET[0] + ix * GRID_SPACING
                    by = GRID_OFFSET[1] + iy * GRID_SPACING
                    bz = GRID_OFFSET[2] + iz * GRID_SPACING
                    rx = (rng.random() - 0.5) * 2.0 * POSITION_RANDOMNESS
                    ry = (rng.random() - 0.5) * 2.0 * POSITION_RANDOMNESS
                    rz = (rng.random() - 0.5) * 2.0 * POSITION_RANDOMNESS
                    pos = wp.vec3(bx + rx, by + ry, bz + rz)

                    shape_type = SHAPE_TYPES[shape_index % len(SHAPE_TYPES)]
                    shape_index += 1

                    body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))

                    if shape_type == "sphere":
                        builder.add_shape_sphere(body, radius=SPHERE_R)
                        extents.append(default_sphere_half_extents(SPHERE_R))
                    elif shape_type == "box":
                        builder.add_shape_box(body, hx=BOX_HE, hy=BOX_HE, hz=BOX_HE)
                        extents.append(default_box_half_extents(BOX_HE, BOX_HE, BOX_HE))
                    elif shape_type == "capsule":
                        builder.add_shape_capsule(body, radius=CAPSULE_R, half_height=CAPSULE_HH)
                        extents.append(default_capsule_half_extents(CAPSULE_R, CAPSULE_HH))
                    elif shape_type == "cylinder":
                        builder.add_shape_cylinder(body, radius=CYLINDER_R, half_height=CYLINDER_HH)
                        extents.append(default_cylinder_half_extents(CYLINDER_R, CYLINDER_HH))
                    elif shape_type == "cone":
                        builder.add_shape_cone(body, radius=CONE_R, half_height=CONE_HH)
                        extents.append(default_cone_half_extents(CONE_R, CONE_HH))
                    else:  # icosahedron
                        builder.add_shape_convex_hull(body, mesh=ico_mesh)
                        extents.append(default_mesh_half_extents(ico_base_for_obb))

        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -25.0, 18.0), pitch=-25.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
