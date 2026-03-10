# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Extensive tests for MPR-with-inflation: accuracy vs pure MPR, GJK, and analytical primitives.

Tests validate signed distance, contact point, and normal accuracy across:
- All shape type combinations (box, sphere, capsule, ellipsoid, cylinder, cone)
- Parametric sweeps over gap/overlap, orientation, shape size, and margin
- Comparison against analytical primitive contact functions
- Large inflation (margin >> body size)
- Detection range boundary (margin_a + margin_b + gap_a + gap_b)
- Hard edge cases: identical positions, touching, tiny shapes, extreme aspect ratios
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.collision_primitive import (
    collide_sphere_box,
    collide_sphere_sphere,
)
from newton._src.geometry.gjk_margin import create_solve_mpr_margin
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    support_map,
)
from newton._src.geometry.types import GeoType

# Create solver instances
solve_mpr_margin = create_solve_mpr_margin(support_map)
solve_full_gjk = create_solve_closest_distance(support_map)
solve_mpr = create_solve_mpr(support_map)


# ═══════════════════════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════════════════════


@wp.kernel
def overlap_comparison_kernel(
    shape_types_a: wp.array(dtype=int),
    scales_a: wp.array(dtype=wp.vec3),
    shape_types_b: wp.array(dtype=int),
    scales_b: wp.array(dtype=wp.vec3),
    positions_a: wp.array(dtype=wp.vec3),
    positions_b: wp.array(dtype=wp.vec3),
    orientations_a: wp.array(dtype=wp.quat),
    orientations_b: wp.array(dtype=wp.quat),
    margins: wp.array(dtype=float),
    mi_collision: wp.array(dtype=int),
    mi_distance: wp.array(dtype=float),
    mi_point: wp.array(dtype=wp.vec3),
    mi_normal: wp.array(dtype=wp.vec3),
    mpr_collision: wp.array(dtype=int),
    mpr_distance: wp.array(dtype=float),
    mpr_point: wp.array(dtype=wp.vec3),
    mpr_normal: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    geom_a = GenericShapeData()
    geom_a.shape_type = shape_types_a[tid]
    geom_a.scale = scales_a[tid]
    geom_b = GenericShapeData()
    geom_b.shape_type = shape_types_b[tid]
    geom_b.scale = scales_b[tid]
    data_provider = SupportMapDataProvider()

    mc, md, mp, mn = wp.static(solve_mpr_margin)(
        geom_a,
        geom_b,
        orientations_a[tid],
        orientations_b[tid],
        positions_a[tid],
        positions_b[tid],
        margins[tid],
        data_provider,
    )
    mi_collision[tid] = int(mc)
    mi_distance[tid] = md
    mi_point[tid] = mp
    mi_normal[tid] = mn

    xc, xd, xp, xn = wp.static(solve_mpr)(
        geom_a,
        geom_b,
        orientations_a[tid],
        orientations_b[tid],
        positions_a[tid],
        positions_b[tid],
        0.0,
        data_provider,
    )
    mpr_collision[tid] = int(xc)
    mpr_distance[tid] = xd
    mpr_point[tid] = xp
    mpr_normal[tid] = xn


@wp.kernel
def separated_comparison_kernel(
    shape_types_a: wp.array(dtype=int),
    scales_a: wp.array(dtype=wp.vec3),
    shape_types_b: wp.array(dtype=int),
    scales_b: wp.array(dtype=wp.vec3),
    positions_a: wp.array(dtype=wp.vec3),
    positions_b: wp.array(dtype=wp.vec3),
    orientations_a: wp.array(dtype=wp.quat),
    orientations_b: wp.array(dtype=wp.quat),
    margins: wp.array(dtype=float),
    mi_collision: wp.array(dtype=int),
    mi_distance: wp.array(dtype=float),
    mi_point: wp.array(dtype=wp.vec3),
    mi_normal: wp.array(dtype=wp.vec3),
    gjk_collision: wp.array(dtype=int),
    gjk_distance: wp.array(dtype=float),
    gjk_point: wp.array(dtype=wp.vec3),
    gjk_normal: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    geom_a = GenericShapeData()
    geom_a.shape_type = shape_types_a[tid]
    geom_a.scale = scales_a[tid]
    geom_b = GenericShapeData()
    geom_b.shape_type = shape_types_b[tid]
    geom_b.scale = scales_b[tid]
    data_provider = SupportMapDataProvider()

    mc, md, mp, mn = wp.static(solve_mpr_margin)(
        geom_a,
        geom_b,
        orientations_a[tid],
        orientations_b[tid],
        positions_a[tid],
        positions_b[tid],
        margins[tid],
        data_provider,
    )
    mi_collision[tid] = int(mc)
    mi_distance[tid] = md
    mi_point[tid] = mp
    mi_normal[tid] = mn

    rc, rd, rp, rn = wp.static(solve_full_gjk)(
        geom_a,
        geom_b,
        orientations_a[tid],
        orientations_b[tid],
        positions_a[tid],
        positions_b[tid],
        0.0,
        data_provider,
    )
    gjk_collision[tid] = int(rc)
    gjk_distance[tid] = rd
    gjk_point[tid] = rp
    gjk_normal[tid] = rn


@wp.kernel
def analytical_comparison_kernel(
    # Sphere-sphere cases
    ss_pos_a: wp.array(dtype=wp.vec3),
    ss_radius_a: wp.array(dtype=float),
    ss_pos_b: wp.array(dtype=wp.vec3),
    ss_radius_b: wp.array(dtype=float),
    ss_margins: wp.array(dtype=float),
    ss_n: int,
    # Results: MPR-inflation
    ss_mi_collision: wp.array(dtype=int),
    ss_mi_distance: wp.array(dtype=float),
    ss_mi_point: wp.array(dtype=wp.vec3),
    ss_mi_normal: wp.array(dtype=wp.vec3),
    # Results: analytical
    ss_ref_distance: wp.array(dtype=float),
    ss_ref_point: wp.array(dtype=wp.vec3),
    ss_ref_normal: wp.array(dtype=wp.vec3),
):
    """Compare MPR-inflation against analytical sphere-sphere."""
    tid = wp.tid()
    if tid >= ss_n:
        return

    pos_a = ss_pos_a[tid]
    r_a = ss_radius_a[tid]
    pos_b = ss_pos_b[tid]
    r_b = ss_radius_b[tid]
    margin = ss_margins[tid]

    # Analytical reference
    ref_dist, ref_pt, ref_n = collide_sphere_sphere(pos_a, r_a, pos_b, r_b)
    ss_ref_distance[tid] = ref_dist
    ss_ref_point[tid] = ref_pt
    ss_ref_normal[tid] = ref_n

    # MPR-inflation (spheres as-is, not shrunken like in the pipeline)
    geom_a = GenericShapeData()
    geom_a.shape_type = GeoType.SPHERE
    geom_a.scale = wp.vec3(r_a, 0.0, 0.0)
    geom_b = GenericShapeData()
    geom_b.shape_type = GeoType.SPHERE
    geom_b.scale = wp.vec3(r_b, 0.0, 0.0)
    data_provider = SupportMapDataProvider()
    ori = wp.quat(0.0, 0.0, 0.0, 1.0)

    mc, md, mp, mn = wp.static(solve_mpr_margin)(
        geom_a,
        geom_b,
        ori,
        ori,
        pos_a,
        pos_b,
        margin,
        data_provider,
    )
    ss_mi_collision[tid] = int(mc)
    ss_mi_distance[tid] = md
    ss_mi_point[tid] = mp
    ss_mi_normal[tid] = mn


@wp.kernel
def analytical_sphere_box_kernel(
    sphere_pos: wp.array(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=float),
    box_pos: wp.array(dtype=wp.vec3),
    box_quat: wp.array(dtype=wp.quat),
    box_size: wp.array(dtype=wp.vec3),
    margins: wp.array(dtype=float),
    n: int,
    mi_collision: wp.array(dtype=int),
    mi_distance: wp.array(dtype=float),
    mi_point: wp.array(dtype=wp.vec3),
    mi_normal: wp.array(dtype=wp.vec3),
    ref_distance: wp.array(dtype=float),
    ref_point: wp.array(dtype=wp.vec3),
    ref_normal: wp.array(dtype=wp.vec3),
):
    """Compare MPR-inflation against analytical sphere-box."""
    tid = wp.tid()
    if tid >= n:
        return

    s_pos = sphere_pos[tid]
    s_r = sphere_radius[tid]
    b_pos = box_pos[tid]
    b_quat = box_quat[tid]
    b_size = box_size[tid]
    margin = margins[tid]

    # Analytical reference
    box_rot = wp.quat_to_matrix(b_quat)
    r_dist, r_pt, r_n = collide_sphere_box(s_pos, s_r, b_pos, box_rot, b_size)
    ref_distance[tid] = r_dist
    ref_point[tid] = r_pt
    ref_normal[tid] = r_n

    # MPR-inflation
    geom_a = GenericShapeData()
    geom_a.shape_type = GeoType.SPHERE
    geom_a.scale = wp.vec3(s_r, 0.0, 0.0)
    geom_b = GenericShapeData()
    geom_b.shape_type = GeoType.BOX
    geom_b.scale = b_size
    data_provider = SupportMapDataProvider()
    ori_a = wp.quat(0.0, 0.0, 0.0, 1.0)

    mc, md, mp, mn = wp.static(solve_mpr_margin)(
        geom_a,
        geom_b,
        ori_a,
        b_quat,
        s_pos,
        b_pos,
        margin,
        data_provider,
    )
    mi_collision[tid] = int(mc)
    mi_distance[tid] = md
    mi_point[tid] = mp
    mi_normal[tid] = mn


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _quat_from_axis_angle(axis, angle_rad):
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    ax = [a * s for a in axis]
    return (ax[0], ax[1], ax[2], c)


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


class _RunnerMixin:
    """Shared launcher for overlap and separated kernels."""

    def _make_arrays(self, cases):
        n = len(cases)
        return dict(
            shape_types_a=wp.array([c["type_a"] for c in cases], dtype=int, device=DEVICE),
            scales_a=wp.array([c["scale_a"] for c in cases], dtype=wp.vec3, device=DEVICE),
            shape_types_b=wp.array([c["type_b"] for c in cases], dtype=int, device=DEVICE),
            scales_b=wp.array([c["scale_b"] for c in cases], dtype=wp.vec3, device=DEVICE),
            positions_a=wp.array([c["pos_a"] for c in cases], dtype=wp.vec3, device=DEVICE),
            positions_b=wp.array([c["pos_b"] for c in cases], dtype=wp.vec3, device=DEVICE),
            orientations_a=wp.array([c.get("ori_a", IDENTITY_QUAT) for c in cases], dtype=wp.quat, device=DEVICE),
            orientations_b=wp.array([c.get("ori_b", IDENTITY_QUAT) for c in cases], dtype=wp.quat, device=DEVICE),
            margins=wp.array([c["margin"] for c in cases], dtype=float, device=DEVICE),
        )

    def _run_overlap(self, cases):
        n = len(cases)
        inputs = self._make_arrays(cases)
        outputs = {}
        for prefix in ["mi", "mpr"]:
            outputs[f"{prefix}_collision"] = wp.zeros(n, dtype=int, device=DEVICE)
            outputs[f"{prefix}_distance"] = wp.zeros(n, dtype=float, device=DEVICE)
            outputs[f"{prefix}_point"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
            outputs[f"{prefix}_normal"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
        wp.launch(
            overlap_comparison_kernel,
            dim=n,
            inputs=[
                inputs["shape_types_a"],
                inputs["scales_a"],
                inputs["shape_types_b"],
                inputs["scales_b"],
                inputs["positions_a"],
                inputs["positions_b"],
                inputs["orientations_a"],
                inputs["orientations_b"],
                inputs["margins"],
            ],
            outputs=[
                outputs["mi_collision"],
                outputs["mi_distance"],
                outputs["mi_point"],
                outputs["mi_normal"],
                outputs["mpr_collision"],
                outputs["mpr_distance"],
                outputs["mpr_point"],
                outputs["mpr_normal"],
            ],
            device=DEVICE,
        )
        return {k: v.numpy() for k, v in outputs.items()}

    def _run_separated(self, cases):
        n = len(cases)
        inputs = self._make_arrays(cases)
        outputs = {}
        for prefix in ["mi", "gjk"]:
            outputs[f"{prefix}_collision"] = wp.zeros(n, dtype=int, device=DEVICE)
            outputs[f"{prefix}_distance"] = wp.zeros(n, dtype=float, device=DEVICE)
            outputs[f"{prefix}_point"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
            outputs[f"{prefix}_normal"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
        wp.launch(
            separated_comparison_kernel,
            dim=n,
            inputs=[
                inputs["shape_types_a"],
                inputs["scales_a"],
                inputs["shape_types_b"],
                inputs["scales_b"],
                inputs["positions_a"],
                inputs["positions_b"],
                inputs["orientations_a"],
                inputs["orientations_b"],
                inputs["margins"],
            ],
            outputs=[
                outputs["mi_collision"],
                outputs["mi_distance"],
                outputs["mi_point"],
                outputs["mi_normal"],
                outputs["gjk_collision"],
                outputs["gjk_distance"],
                outputs["gjk_point"],
                outputs["gjk_normal"],
            ],
            device=DEVICE,
        )
        return {k: v.numpy() for k, v in outputs.items()}

    @staticmethod
    def _case(type_a, scale_a, type_b, scale_b, pos_a, pos_b, margin=0.5, **kw):
        c = {
            "type_a": int(type_a),
            "scale_a": tuple(scale_a),
            "type_b": int(type_b),
            "scale_b": tuple(scale_b),
            "pos_a": tuple(pos_a),
            "pos_b": tuple(pos_b),
            "margin": margin,
        }
        c.update(kw)
        return c


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Overlap: MPR-inflation vs pure MPR — parametric sweeps
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverlapVsPureMPR(unittest.TestCase, _RunnerMixin):
    """Overlapping shapes: MPR-inflation should match pure MPR and analytical values."""

    def test_box_box_sweep_overlap_depths(self):
        """Sweep box-box overlap from shallow to deep along X."""
        cases = []
        seps = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0]
        for sep in seps:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [sep, 0.0, 0.0],
                )
            )
        r = self._run_overlap(cases)
        for i, sep in enumerate(seps):
            expected = -(1.0 - sep)
            with self.subTest(sep=sep):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], expected, places=2)
                if r["mpr_collision"][i] and abs(r["mpr_distance"][i]) > 1e-4:
                    self.assertAlmostEqual(r["mi_distance"][i], r["mpr_distance"][i], places=2)

    def test_box_box_sweep_all_axes(self):
        """Box-box overlap along each axis."""
        cases = []
        for axis_idx, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            pos_b = [0.8 * a for a in axis]
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    pos_b,
                )
            )
        r = self._run_overlap(cases)
        for i in range(3):
            self.assertAlmostEqual(r["mi_distance"][i], -0.2, places=2)

    def test_box_box_sweep_rotations(self):
        """Box-box overlap at various rotations around Z axis."""
        cases = []
        angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        for angle_deg in angles:
            q = _quat_from_axis_angle([0, 0, 1], math.radians(angle_deg))
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    ori_b=q,
                )
            )
        r = self._run_overlap(cases)
        for i, angle in enumerate(angles):
            with self.subTest(angle=angle):
                self.assertTrue(r["mi_collision"][i], f"angle={angle}")
                self.assertLess(r["mi_distance"][i], 0.0, f"angle={angle}")

    def test_sphere_sphere_sweep_depths(self):
        """Sphere-sphere sweep from shallow to deep overlap."""
        cases = []
        seps = [0.75, 0.6, 0.4, 0.2, 0.0]
        for sep in seps:
            cases.append(
                self._case(
                    GeoType.SPHERE,
                    [0.5, 0.0, 0.0],
                    GeoType.SPHERE,
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [sep, 0.0, 0.0],
                )
            )
        r = self._run_overlap(cases)
        for i, sep in enumerate(seps):
            expected = sep - 0.8
            with self.subTest(sep=sep):
                self.assertAlmostEqual(r["mi_distance"][i], expected, places=2)

    def test_mixed_shape_overlaps(self):
        """Various shape type combinations overlapping."""
        cases = [
            # Sphere-box
            self._case(GeoType.SPHERE, [0.5, 0.0, 0.0], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]),
            # Ellipsoid-box
            self._case(
                GeoType.ELLIPSOID, [1.0, 0.5, 0.3], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]
            ),
            # Capsule-capsule
            self._case(
                GeoType.CAPSULE, [0.3, 0.5, 0.0], GeoType.CAPSULE, [0.3, 0.5, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]
            ),
            # Capsule-box
            self._case(
                GeoType.CAPSULE, [0.3, 0.5, 0.0], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]
            ),
            # Ellipsoid-ellipsoid
            self._case(
                GeoType.ELLIPSOID, [1.0, 0.5, 0.3], GeoType.ELLIPSOID, [0.8, 0.4, 0.3], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]
            ),
            # Cylinder-box
            self._case(
                GeoType.CYLINDER, [0.5, 0.5, 0.0], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]
            ),
            # Cone-box
            self._case(GeoType.CONE, [0.5, 0.5, 0.0], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
        ]
        expected_overlap = [-0.2, -0.3, -0.2, -0.2, -0.3, None, None]
        r = self._run_overlap(cases)
        for i in range(len(cases)):
            with self.subTest(i=i):
                self.assertTrue(r["mi_collision"][i], f"case {i}")
                self.assertLess(r["mi_distance"][i], 0.0)
                if expected_overlap[i] is not None:
                    self.assertAlmostEqual(r["mi_distance"][i], expected_overlap[i], places=2)

    def test_overlap_normals_unit_length_and_direction(self):
        """Normals should be unit length and point from A to B."""
        cases = []
        dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        for d in dirs:
            pos_b = [0.8 * x for x in d]
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    pos_b,
                )
            )
        r = self._run_overlap(cases)
        for i, d in enumerate(dirs):
            mi_n = r["mi_normal"][i]
            self.assertAlmostEqual(np.linalg.norm(mi_n), 1.0, places=2)
            self.assertGreater(np.dot(mi_n, d), 0.9)

    def test_overlap_different_margins_same_depth(self):
        """Overlap depth should not depend on margin."""
        cases = []
        for margin in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_overlap(cases)
        for i, margin in enumerate([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]):
            with self.subTest(margin=margin):
                self.assertAlmostEqual(r["mi_distance"][i], -0.2, places=2)

    def test_deep_penetration(self):
        """One shape fully inside another."""
        cases = [
            # Small box inside large box
            self._case(GeoType.BOX, [2.0, 2.0, 2.0], GeoType.BOX, [0.3, 0.3, 0.3], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]),
            # Sphere inside sphere
            self._case(
                GeoType.SPHERE, [1.0, 0.0, 0.0], GeoType.SPHERE, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]
            ),
        ]
        r = self._run_overlap(cases)
        for i in range(len(cases)):
            self.assertTrue(r["mi_collision"][i])
            self.assertLess(r["mi_distance"][i], -0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Separated: MPR-inflation vs GJK — parametric sweeps
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeparatedVsGJK(unittest.TestCase, _RunnerMixin):
    """Separated shapes: MPR-inflation should match GJK reference."""

    def test_box_box_sweep_gaps(self):
        """Sweep box-box gaps from tiny to near-margin along X."""
        cases = []
        gaps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]
        for gap in gaps:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=0.5,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate(gaps):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i], f"gap={gap}")
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)
                self.assertAlmostEqual(r["mi_distance"][i], r["gjk_distance"][i], places=2)

    def test_box_box_sweep_all_axes(self):
        """Box-box separated along each axis."""
        cases = []
        for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
            pos_b = [1.2 * a for a in axis]
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    pos_b,
                    margin=0.5,
                )
            )
        r = self._run_separated(cases)
        for i in range(6):
            self.assertAlmostEqual(r["mi_distance"][i], 0.2, places=2)
            self.assertAlmostEqual(r["mi_distance"][i], r["gjk_distance"][i], places=2)

    def test_box_box_sweep_rotations(self):
        """Box-box separated at various rotations."""
        cases = []
        angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        for angle_deg in angles:
            q = _quat_from_axis_angle([0, 0, 1], math.radians(angle_deg))
            # Use margin=1.5 to ensure gap < margin for all rotations
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    ori_b=q,
                    margin=1.5,
                )
            )
        r = self._run_separated(cases)
        for i, angle in enumerate(angles):
            with self.subTest(angle=angle):
                self.assertTrue(r["mi_collision"][i], f"angle={angle}")
                self.assertAlmostEqual(r["mi_distance"][i], r["gjk_distance"][i], places=2)

    def test_sphere_sphere_sweep_gaps(self):
        """Sphere-sphere separated at various gaps."""
        cases = []
        gaps = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49]
        for gap in gaps:
            cases.append(
                self._case(
                    GeoType.SPHERE,
                    [0.5, 0.0, 0.0],
                    GeoType.SPHERE,
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.8 + gap, 0.0, 0.0],
                    margin=0.5,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate(gaps):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)

    def test_mixed_shapes_separated(self):
        """Various shape combinations separated."""
        cases = [
            # Sphere-box face
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                margin=0.5,
            ),
            # Ellipsoid-box
            self._case(
                GeoType.ELLIPSOID,
                [1.0, 0.5, 0.3],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.7, 0.0, 0.0],
                margin=0.5,
            ),
            # Capsule-capsule parallel
            self._case(
                GeoType.CAPSULE,
                [0.3, 0.5, 0.0],
                GeoType.CAPSULE,
                [0.3, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                margin=0.5,
            ),
            # Capsule-box
            self._case(
                GeoType.CAPSULE,
                [0.3, 0.5, 0.0],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                margin=0.5,
            ),
            # Cylinder-box
            self._case(
                GeoType.CYLINDER,
                [0.5, 0.5, 0.0],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                margin=0.5,
            ),
            # Cone-box
            self._case(
                GeoType.CONE,
                [0.5, 0.5, 0.0],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                margin=0.5,
            ),
            # Ellipsoid-ellipsoid
            self._case(
                GeoType.ELLIPSOID,
                [1.0, 0.5, 0.3],
                GeoType.ELLIPSOID,
                [0.8, 0.4, 0.3],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        expected_gaps = [0.2, 0.2, 0.2, 0.2, 0.2, None, 0.2]
        r = self._run_separated(cases)
        for i in range(len(cases)):
            with self.subTest(i=i):
                self.assertTrue(r["mi_collision"][i], f"case {i}")
                self.assertGreater(r["mi_distance"][i], 0.0)
                self.assertAlmostEqual(r["mi_distance"][i], r["gjk_distance"][i], places=2)
                if expected_gaps[i] is not None:
                    self.assertAlmostEqual(r["mi_distance"][i], expected_gaps[i], places=2)

    def test_separated_normals_match_gjk(self):
        """Normals from MPR-inflation should agree with GJK for separated shapes."""
        cases = []
        dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for d in dirs:
            pos_b = [1.2 * x for x in d]
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    pos_b,
                    margin=0.5,
                )
            )
            # Spheres
            pos_b_s = [(0.8 + 0.2) * x for x in d]
            cases.append(
                self._case(
                    GeoType.SPHERE,
                    [0.5, 0.0, 0.0],
                    GeoType.SPHERE,
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    pos_b_s,
                    margin=0.5,
                )
            )
        r = self._run_separated(cases)
        for i in range(len(cases)):
            mi_n = r["mi_normal"][i]
            gjk_n = r["gjk_normal"][i]
            self.assertAlmostEqual(np.linalg.norm(mi_n), 1.0, places=2)
            self.assertGreater(np.dot(mi_n, gjk_n), 0.9, f"i={i} mi={mi_n} gjk={gjk_n}")

    def test_separated_contact_points_close_to_gjk(self):
        """Contact points should be close to GJK reference."""
        cases = [
            self._case(
                GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0], margin=0.5
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                margin=0.5,
            ),
            self._case(
                GeoType.ELLIPSOID,
                [1.0, 0.5, 0.3],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.7, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        for i in range(len(cases)):
            mi_pt = r["mi_point"][i]
            gjk_pt = r["gjk_point"][i]
            dist = np.linalg.norm(mi_pt - gjk_pt)
            self.assertLess(dist, 0.15, f"i={i} mi={mi_pt} gjk={gjk_pt}")

    def test_beyond_margin_not_detected(self):
        """Shapes beyond margin should return collision=False."""
        cases = [
            self._case(
                GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], margin=0.5
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                margin=0.5,
            ),
            self._case(
                GeoType.ELLIPSOID,
                [1.0, 0.5, 0.3],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        for i in range(len(cases)):
            self.assertFalse(r["mi_collision"][i], f"case {i}")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Large inflation — margin >> body size
# ═══════════════════════════════════════════════════════════════════════════════


class TestLargeInflation(unittest.TestCase, _RunnerMixin):
    """Test with inflation as large as or larger than the shapes."""

    def test_margin_equal_to_body_size(self):
        """Margin equal to box half-extent (doubles effective size)."""
        # Box half-extent 0.5, margin=0.5 → effective half-extent ~1.0
        cases = []
        for gap in [0.01, 0.1, 0.3, 0.49]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=0.5,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate([0.01, 0.1, 0.3, 0.49]):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)

    def test_margin_2x_body_size(self):
        """Margin = 2x body size."""
        cases = []
        for gap in [0.01, 0.1, 0.5, 0.9]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=1.0,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate([0.01, 0.1, 0.5, 0.9]):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)
                self.assertAlmostEqual(r["mi_distance"][i], r["gjk_distance"][i], places=2)

    def test_margin_5x_body_size(self):
        """Margin = 5x body size (extreme inflation)."""
        cases = []
        for gap in [0.01, 0.1, 0.5, 1.0, 2.0, 2.49]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=2.5,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate([0.01, 0.1, 0.5, 1.0, 2.0, 2.49]):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)

    def test_margin_10x_body_size(self):
        """Margin = 10x body size (very extreme inflation)."""
        cases = []
        for gap in [0.01, 0.5, 2.0, 4.9]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=5.0,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate([0.01, 0.5, 2.0, 4.9]):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=1)

    def test_large_margin_sphere_sphere(self):
        """Large margin on spheres."""
        cases = []
        for margin in [0.5, 1.0, 2.0, 5.0]:
            gap = margin * 0.9  # Just inside margin
            cases.append(
                self._case(
                    GeoType.SPHERE,
                    [0.5, 0.0, 0.0],
                    GeoType.SPHERE,
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.8 + gap, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_separated(cases)
        for i, margin in enumerate([0.5, 1.0, 2.0, 5.0]):
            gap = margin * 0.9
            with self.subTest(margin=margin):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=1)

    def test_large_margin_overlap_unchanged(self):
        """Large margins should not affect overlap depth."""
        cases = []
        for margin in [0.1, 1.0, 5.0, 20.0]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_overlap(cases)
        for i, margin in enumerate([0.1, 1.0, 5.0, 20.0]):
            with self.subTest(margin=margin):
                self.assertAlmostEqual(r["mi_distance"][i], -0.2, places=2)

    def test_tiny_shapes_large_margin(self):
        """Very small shapes with large margin."""
        cases = []
        for gap in [0.01, 0.1, 0.5]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.05, 0.05, 0.05],
                    GeoType.BOX,
                    [0.05, 0.05, 0.05],
                    [0.0, 0.0, 0.0],
                    [0.1 + gap, 0.0, 0.0],
                    margin=1.0,
                )
            )
        r = self._run_separated(cases)
        for i, gap in enumerate([0.01, 0.1, 0.5]):
            with self.subTest(gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Comparison against analytical primitive contact functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestVsAnalyticalPrimitives(unittest.TestCase):
    """Compare MPR-inflation against analytical sphere-sphere and sphere-box."""

    def test_sphere_sphere_sweep_vs_analytical(self):
        """Sweep sphere-sphere from overlap through separated, compare to analytical."""
        positions_b = []
        radii_a = []
        radii_b = []
        margins_list = []

        # Various separations: overlap, touching, separated
        seps = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        r_a, r_b = 0.5, 0.3
        margin = 0.5
        n = len(seps)

        for sep in seps:
            positions_b.append((sep, 0.0, 0.0))
            radii_a.append(r_a)
            radii_b.append(r_b)
            margins_list.append(margin)

        outputs = {}
        for prefix in ["ss_mi", "ss_ref"]:
            if "mi" in prefix:
                outputs[f"{prefix}_collision"] = wp.zeros(n, dtype=int, device=DEVICE)
            outputs[f"{prefix}_distance"] = wp.zeros(n, dtype=float, device=DEVICE)
            outputs[f"{prefix}_point"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
            outputs[f"{prefix}_normal"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)

        wp.launch(
            analytical_comparison_kernel,
            dim=n,
            inputs=[
                wp.array([(0.0, 0.0, 0.0)] * n, dtype=wp.vec3, device=DEVICE),
                wp.array(radii_a, dtype=float, device=DEVICE),
                wp.array(positions_b, dtype=wp.vec3, device=DEVICE),
                wp.array(radii_b, dtype=float, device=DEVICE),
                wp.array(margins_list, dtype=float, device=DEVICE),
                n,
            ],
            outputs=[
                outputs["ss_mi_collision"],
                outputs["ss_mi_distance"],
                outputs["ss_mi_point"],
                outputs["ss_mi_normal"],
                outputs["ss_ref_distance"],
                outputs["ss_ref_point"],
                outputs["ss_ref_normal"],
            ],
            device=DEVICE,
        )

        mi_dist = outputs["ss_mi_distance"].numpy()
        mi_coll = outputs["ss_mi_collision"].numpy()
        ref_dist = outputs["ss_ref_distance"].numpy()
        mi_pt = outputs["ss_mi_point"].numpy()
        ref_pt = outputs["ss_ref_point"].numpy()
        mi_n = outputs["ss_mi_normal"].numpy()
        ref_n = outputs["ss_ref_normal"].numpy()

        for i, sep in enumerate(seps):
            expected = sep - (r_a + r_b)
            with self.subTest(sep=sep, expected=expected):
                # Analytical should be exact
                self.assertAlmostEqual(ref_dist[i], expected, places=4)
                # MPR-inflation should detect within margin
                if expected <= margin:
                    self.assertTrue(mi_coll[i], f"sep={sep}")
                    self.assertAlmostEqual(
                        mi_dist[i], expected, places=2, msg=f"mi={mi_dist[i]:.4f} ref={ref_dist[i]:.4f}"
                    )
                    # Normals should agree
                    dot = np.dot(mi_n[i], ref_n[i])
                    self.assertGreater(dot, 0.9, f"mi_n={mi_n[i]} ref_n={ref_n[i]}")
                    # Contact points should be close
                    pt_dist = np.linalg.norm(mi_pt[i] - ref_pt[i])
                    self.assertLess(pt_dist, 0.15)

    def test_sphere_box_sweep_vs_analytical(self):
        """Sweep sphere-box from overlap through separated, compare to analytical."""
        sphere_positions = []
        sphere_radii = []
        box_positions = []
        box_quats = []
        box_sizes = []
        margins_list = []

        seps = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        s_r = 0.5
        b_size = (0.5, 0.5, 0.5)
        margin = 0.5
        n = len(seps)

        for sep in seps:
            sphere_positions.append((0.0, 0.0, 0.0))
            sphere_radii.append(s_r)
            box_positions.append((sep, 0.0, 0.0))
            box_quats.append(IDENTITY_QUAT)
            box_sizes.append(b_size)
            margins_list.append(margin)

        outputs = {}
        for prefix in ["mi", "ref"]:
            if prefix == "mi":
                outputs["mi_collision"] = wp.zeros(n, dtype=int, device=DEVICE)
            outputs[f"{prefix}_distance"] = wp.zeros(n, dtype=float, device=DEVICE)
            outputs[f"{prefix}_point"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
            outputs[f"{prefix}_normal"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)

        wp.launch(
            analytical_sphere_box_kernel,
            dim=n,
            inputs=[
                wp.array(sphere_positions, dtype=wp.vec3, device=DEVICE),
                wp.array(sphere_radii, dtype=float, device=DEVICE),
                wp.array(box_positions, dtype=wp.vec3, device=DEVICE),
                wp.array(box_quats, dtype=wp.quat, device=DEVICE),
                wp.array(box_sizes, dtype=wp.vec3, device=DEVICE),
                wp.array(margins_list, dtype=float, device=DEVICE),
                n,
            ],
            outputs=[
                outputs["mi_collision"],
                outputs["mi_distance"],
                outputs["mi_point"],
                outputs["mi_normal"],
                outputs["ref_distance"],
                outputs["ref_point"],
                outputs["ref_normal"],
            ],
            device=DEVICE,
        )

        mi_dist = outputs["mi_distance"].numpy()
        mi_coll = outputs["mi_collision"].numpy()
        ref_dist = outputs["ref_distance"].numpy()
        mi_n = outputs["mi_normal"].numpy()
        ref_n = outputs["ref_normal"].numpy()

        for i, sep in enumerate(seps):
            # Analytical: gap = sep - s_r - box_half_extent_x = sep - 1.0
            expected = sep - 1.0
            with self.subTest(sep=sep, expected=expected):
                self.assertAlmostEqual(ref_dist[i], expected, places=4)
                if expected <= margin:
                    self.assertTrue(mi_coll[i], f"sep={sep}")
                    self.assertAlmostEqual(
                        mi_dist[i], expected, places=2, msg=f"mi={mi_dist[i]:.4f} ref={ref_dist[i]:.4f}"
                    )
                    dot = np.dot(mi_n[i], ref_n[i])
                    self.assertGreater(dot, 0.9)

    def test_sphere_sphere_various_directions_vs_analytical(self):
        """Sphere-sphere along various directions, compare to analytical."""
        positions_b = []
        radii_a = []
        radii_b = []
        margins_list = []

        r_a, r_b = 0.5, 0.3
        margin = 0.5
        dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
        gap = 0.1
        n = len(dirs)

        for d in dirs:
            norm = math.sqrt(sum(x**2 for x in d))
            direction = [x / norm for x in d]
            dist = r_a + r_b + gap
            positions_b.append(tuple(x * dist for x in direction))
            radii_a.append(r_a)
            radii_b.append(r_b)
            margins_list.append(margin)

        outputs = {}
        for prefix in ["ss_mi", "ss_ref"]:
            if "mi" in prefix:
                outputs[f"{prefix}_collision"] = wp.zeros(n, dtype=int, device=DEVICE)
            outputs[f"{prefix}_distance"] = wp.zeros(n, dtype=float, device=DEVICE)
            outputs[f"{prefix}_point"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
            outputs[f"{prefix}_normal"] = wp.zeros(n, dtype=wp.vec3, device=DEVICE)

        wp.launch(
            analytical_comparison_kernel,
            dim=n,
            inputs=[
                wp.array([(0.0, 0.0, 0.0)] * n, dtype=wp.vec3, device=DEVICE),
                wp.array(radii_a, dtype=float, device=DEVICE),
                wp.array(positions_b, dtype=wp.vec3, device=DEVICE),
                wp.array(radii_b, dtype=float, device=DEVICE),
                wp.array(margins_list, dtype=float, device=DEVICE),
                n,
            ],
            outputs=[
                outputs["ss_mi_collision"],
                outputs["ss_mi_distance"],
                outputs["ss_mi_point"],
                outputs["ss_mi_normal"],
                outputs["ss_ref_distance"],
                outputs["ss_ref_point"],
                outputs["ss_ref_normal"],
            ],
            device=DEVICE,
        )

        mi_dist = outputs["ss_mi_distance"].numpy()
        ref_dist = outputs["ss_ref_distance"].numpy()

        for i, d in enumerate(dirs):
            with self.subTest(direction=d):
                self.assertAlmostEqual(ref_dist[i], gap, places=4)
                self.assertAlmostEqual(mi_dist[i], gap, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Detection range boundary
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectionRange(unittest.TestCase, _RunnerMixin):
    """Verify contacts are detected up to the full margin distance."""

    def test_detection_at_margin_boundary(self):
        """Contacts at exactly the margin distance should be detected."""
        cases = []
        for margin in [0.1, 0.5, 1.0, 2.0]:
            gap = margin - 0.01  # Just inside margin
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_separated(cases)
        for i, margin in enumerate([0.1, 0.5, 1.0, 2.0]):
            with self.subTest(margin=margin):
                self.assertTrue(r["mi_collision"][i], f"margin={margin}")

    def test_no_detection_beyond_margin(self):
        """Contacts just beyond the margin should NOT be detected."""
        cases = []
        for margin in [0.1, 0.5, 1.0, 2.0]:
            gap = margin + 0.1  # Beyond margin
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_separated(cases)
        for i, margin in enumerate([0.1, 0.5, 1.0, 2.0]):
            with self.subTest(margin=margin):
                self.assertFalse(r["mi_collision"][i], f"margin={margin}")

    def test_margin_independence_of_distance(self):
        """Same gap, different margins: distance should be the same."""
        cases = []
        gap = 0.2
        for margin in [0.3, 0.5, 1.0, 5.0]:
            cases.append(
                self._case(
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    GeoType.BOX,
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                    [1.0 + gap, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_separated(cases)
        for i in range(len(cases)):
            self.assertTrue(r["mi_collision"][i])
            self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)

    def test_sphere_detection_range(self):
        """Sphere-sphere detection at various fractions of margin."""
        margin = 1.0
        fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        cases = []
        for frac in fractions:
            gap = margin * frac
            cases.append(
                self._case(
                    GeoType.SPHERE,
                    [0.5, 0.0, 0.0],
                    GeoType.SPHERE,
                    [0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.8 + gap, 0.0, 0.0],
                    margin=margin,
                )
            )
        r = self._run_separated(cases)
        for i, frac in enumerate(fractions):
            gap = margin * frac
            with self.subTest(frac=frac, gap=gap):
                self.assertTrue(r["mi_collision"][i])
                self.assertAlmostEqual(r["mi_distance"][i], gap, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Hard edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase, _RunnerMixin):
    """Boundary conditions, degeneracies, and stress tests."""

    def test_just_touching(self):
        """Shapes exactly touching (signed_distance ≈ 0)."""
        cases = [
            self._case(
                GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], margin=0.5
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        for i in range(len(cases)):
            self.assertAlmostEqual(r["mi_distance"][i], 0.0, places=2)

    def test_barely_separated_and_barely_overlapping(self):
        """Very small gap and overlap (1mm)."""
        eps = 0.001
        cases_sep = [
            self._case(
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0 + eps, 0.0, 0.0],
                margin=0.5,
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.8 + eps, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        cases_ovl = [
            self._case(
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0 - eps, 0.0, 0.0],
                margin=0.5,
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.8 - eps, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r_sep = self._run_separated(cases_sep)
        r_ovl = self._run_overlap(cases_ovl)
        for i in range(2):
            self.assertAlmostEqual(r_sep["mi_distance"][i], eps, places=2)
            self.assertAlmostEqual(r_ovl["mi_distance"][i], -eps, places=2)

    def test_very_small_shapes(self):
        """Shapes much smaller than typical (1mm)."""
        cases = [
            self._case(
                GeoType.BOX,
                [0.001, 0.001, 0.001],
                GeoType.BOX,
                [0.001, 0.001, 0.001],
                [0.0, 0.0, 0.0],
                [0.003, 0.0, 0.0],
                margin=0.01,
            ),
        ]
        r = self._run_separated(cases)
        self.assertTrue(r["mi_collision"][0])
        self.assertAlmostEqual(r["mi_distance"][0], 0.001, places=3)

    def test_very_large_shapes(self):
        """Very large shapes."""
        cases = [
            self._case(
                GeoType.BOX,
                [100.0, 100.0, 100.0],
                GeoType.BOX,
                [100.0, 100.0, 100.0],
                [0.0, 0.0, 0.0],
                [210.0, 0.0, 0.0],
                margin=20.0,
            ),
        ]
        r = self._run_separated(cases)
        self.assertTrue(r["mi_collision"][0])
        self.assertAlmostEqual(r["mi_distance"][0], 10.0, places=0)

    def test_extreme_aspect_ratios(self):
        """Flat/thin boxes."""
        cases = [
            # Very flat box (pancake)
            self._case(
                GeoType.BOX,
                [5.0, 5.0, 0.01],
                GeoType.BOX,
                [5.0, 5.0, 0.01],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.12],
                margin=0.5,
            ),
            # Very thin box (needle)
            self._case(
                GeoType.BOX,
                [0.01, 0.01, 5.0],
                GeoType.BOX,
                [0.01, 0.01, 5.0],
                [0.0, 0.0, 0.0],
                [0.12, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        for i in range(2):
            self.assertTrue(r["mi_collision"][i])
            self.assertAlmostEqual(r["mi_distance"][i], 0.1, places=1)

    def test_shapes_at_non_origin(self):
        """Both shapes far from origin."""
        cases = [
            self._case(
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [100.0, 200.0, 300.0],
                [101.2, 200.0, 300.0],
                margin=0.5,
            ),
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [-50.0, -50.0, -50.0],
                [-49.0, -50.0, -50.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        self.assertAlmostEqual(r["mi_distance"][0], 0.2, places=2)
        self.assertAlmostEqual(r["mi_distance"][1], 0.2, places=2)

    def test_shapes_at_non_origin_overlap(self):
        """Overlapping shapes far from origin."""
        cases = [
            self._case(
                GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [100.0, 200.0, 300.0], [100.8, 200.0, 300.0]
            ),
        ]
        r = self._run_overlap(cases)
        self.assertAlmostEqual(r["mi_distance"][0], -0.2, places=2)

    def test_rotated_at_non_origin(self):
        """Both shapes rotated and far from origin."""
        q30 = _quat_from_axis_angle([0, 0, 1], math.pi / 6.0)
        q45 = _quat_from_axis_angle([1, 0, 0], math.pi / 4.0)
        cases = [
            self._case(
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                GeoType.BOX,
                [0.5, 0.5, 0.5],
                [10.0, 20.0, 30.0],
                [11.5, 20.0, 30.0],
                ori_a=q30,
                ori_b=q45,
                margin=1.0,
            ),
        ]
        r = self._run_separated(cases)
        self.assertTrue(r["mi_collision"][0])
        self.assertAlmostEqual(r["mi_distance"][0], r["gjk_distance"][0], places=2)

    def test_large_size_difference(self):
        """Very different shape sizes."""
        cases = [
            self._case(
                GeoType.BOX,
                [5.0, 5.0, 5.0],
                GeoType.BOX,
                [0.01, 0.01, 0.01],
                [0.0, 0.0, 0.0],
                [5.02, 0.0, 0.0],
                margin=0.5,
            ),
        ]
        r = self._run_separated(cases)
        self.assertTrue(r["mi_collision"][0])
        self.assertAlmostEqual(r["mi_distance"][0], 0.01, places=2)

    def test_identical_positions_deep_overlap(self):
        """Shapes at the exact same position (maximum overlap)."""
        cases = [
            self._case(GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            self._case(
                GeoType.SPHERE, [0.5, 0.0, 0.0], GeoType.SPHERE, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            ),
        ]
        r = self._run_overlap(cases)
        for i in range(2):
            self.assertTrue(r["mi_collision"][i])
            self.assertLess(r["mi_distance"][i], -0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Continuity across the overlap/separation boundary
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinuity(unittest.TestCase, _RunnerMixin):
    """Signed distance should be continuous as shapes transition overlap ↔ separation."""

    def test_box_box_continuity_sweep(self):
        """Sweep box-box from separated through touching to overlapping."""
        offsets = [1.4, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 1.0, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7]
        cases = [
            self._case(
                GeoType.BOX, [0.5, 0.5, 0.5], GeoType.BOX, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [o, 0.0, 0.0], margin=0.5
            )
            for o in offsets
        ]
        r_sep = self._run_separated(cases)
        r_ovl = self._run_overlap(cases)

        distances = []
        for i, offset in enumerate(offsets):
            expected = offset - 1.0
            dist = r_sep["mi_distance"][i] if expected >= 0 else r_ovl["mi_distance"][i]
            distances.append(dist)
            if abs(expected) <= 0.49:  # Within margin
                self.assertAlmostEqual(dist, expected, places=2, msg=f"offset={offset}")

        # Check monotonicity
        for i in range(1, len(distances)):
            self.assertLessEqual(distances[i], distances[i - 1] + 0.02)

    def test_sphere_sphere_continuity_sweep(self):
        """Sweep sphere-sphere across boundary."""
        offsets = [1.1, 1.0, 0.9, 0.85, 0.82, 0.81, 0.805, 0.8, 0.795, 0.79, 0.78, 0.75, 0.7, 0.6]
        cases = [
            self._case(
                GeoType.SPHERE,
                [0.5, 0.0, 0.0],
                GeoType.SPHERE,
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [o, 0.0, 0.0],
                margin=0.5,
            )
            for o in offsets
        ]
        r_sep = self._run_separated(cases)
        r_ovl = self._run_overlap(cases)

        distances = []
        for i, offset in enumerate(offsets):
            expected = offset - 0.8
            dist = r_sep["mi_distance"][i] if expected >= 0 else r_ovl["mi_distance"][i]
            distances.append(dist)
            if abs(expected) <= 0.49:
                self.assertAlmostEqual(dist, expected, places=2)

        for i in range(1, len(distances)):
            self.assertLessEqual(distances[i], distances[i - 1] + 0.02)


if __name__ == "__main__":
    wp.init()
    unittest.main()
