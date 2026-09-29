# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check capsule-cylinder point and line contacts and general-query fallbacks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.geometry.collision_primitive import _collide_capsule_cylinder_barrel
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query_barrel(
    positions: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    handled: wp.array[int],
    distances: wp.array[float],
    centers: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    ok, distance, center, normal = _collide_capsule_cylinder_barrel(
        positions[i], axes[i], 0.25, 0.5, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 3.75, 1.5
    )
    handled[i] = int(ok)
    distances[i] = distance
    centers[i] = center
    normals[i] = normal


def test_barrel_guards(test, device):
    """Preserve parallel, cap/rim and core-intersection fallback; handle endpoints."""
    positions = [
        (3.95, 0.0, 0.3),  # interior barrel witness
        (4.45, 0.0, 0.3),  # core endpoint is the closest witness
        (4.1, 0.0, 0.3),  # separated but analytically handled
        (3.95, 0.0, 0.3),  # parallel: preserve the general manifold
        (3.95, 0.0, 1.5),  # exact rim: fallback
        (3.95, 0.0, 1.6),  # outside barrel: fallback
        (3.7, 0.0, 0.3),  # core intersects cylinder: fallback
    ]
    axes = [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    n = len(positions)
    with wp.ScopedDevice(device):
        handled = wp.zeros(n, dtype=int)
        distances = wp.zeros(n, dtype=float)
        centers = wp.zeros(n, dtype=wp.vec3)
        normals = wp.zeros(n, dtype=wp.vec3)
        wp.launch(
            query_barrel,
            n,
            [wp.array(positions, dtype=wp.vec3), wp.array(axes, dtype=wp.vec3), handled, distances, centers, normals],
        )
    np.testing.assert_array_equal(handled.numpy(), [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(distances.numpy()[:3], [-0.05, -0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(normals.numpy()[:3], [[-1.0, 0.0, 0.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(centers.numpy()[:3], [[3.725, 0.0, 0.3], [3.725, 0.0, 0.3], [3.8, 0.0, 0.3]], atol=1e-6)
    test.assertTrue(np.all(distances.numpy()[3:] >= MAXVAL * 0.99))


def test_pipeline_dispatch(test, device):
    """Emit two line contacts, one point contact, or a general-query fallback."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in (
                "barrel",
                "parallel",
                "parallel_clipped",
                "near_parallel",
                "cap_face",
                "cap_clipped",
                "cap_clipped_negative",
                "cap_bridge",
                "cap_outside",
                "cap_tilted",
                "cap_penetrating",
                "cap_deep",
                "cap_bottom",
                "cap_axial",
                "rim",
                "core",
                "rounded",
            ):
                with test.subTest(reversed_order=reversed_order, kind=kind):
                    builder = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.00025)
                    cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    cylinder_kwargs = {"body": cylinder_body, "radius": 0.00375, "half_height": 0.0015, "cfg": cfg}
                    if kind == "rounded":
                        cylinder_kwargs["barrel_radius"] = 0.02
                    capsule_kwargs = {
                        "body": capsule_body,
                        "radius": 0.00025,
                        "half_height": 0.005 if kind == "cap_bridge" else 0.0005,
                        "cfg": cfg,
                    }
                    if reversed_order:
                        builder.add_shape_capsule(**capsule_kwargs)
                        builder.add_shape_cylinder(**cylinder_kwargs)
                    else:
                        builder.add_shape_cylinder(**cylinder_kwargs)
                        builder.add_shape_capsule(**capsule_kwargs)
                    model = builder.finalize()
                    state = model.state()
                    poses = model.body_q.numpy()
                    poses[capsule_body, :3] = (0.00395, 0.0, 0.0003)
                    axis = wp.vec3(0.0, 1.0, 0.0)
                    if kind in ("parallel", "parallel_clipped", "cap_axial"):
                        axis = wp.vec3(0.0, 0.0, 1.0)
                    elif kind == "near_parallel":
                        axis = wp.normalize(wp.vec3(0.05, 0.0, 1.0))
                    elif kind == "cap_tilted":
                        axis = wp.normalize(wp.vec3(1.0, 0.0, 0.01))
                    elif kind in (
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_outside",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_bottom",
                    ):
                        axis = wp.vec3(1.0, 0.0, 0.0)
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    if kind in (
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_outside",
                        "cap_tilted",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_axial",
                    ):
                        poses[capsule_body, :3] = (0.0, 0.0, 0.0017)
                    if kind == "cap_clipped":
                        poses[capsule_body, 0] = 0.0037
                    elif kind == "cap_clipped_negative":
                        poses[capsule_body, 0] = -0.0037
                    elif kind == "cap_outside":
                        poses[capsule_body, 0] = 0.0045
                    elif kind == "cap_penetrating":
                        poses[capsule_body, 2] = 0.00145
                    elif kind == "cap_deep":
                        poses[capsule_body, 2] = 0.001
                    elif kind == "cap_bottom":
                        poses[capsule_body, :3] = (0.0, 0.0, -0.0017)
                    elif kind == "parallel_clipped":
                        poses[capsule_body, 2] = 0.0013
                    elif kind == "rim":
                        poses[capsule_body, 2] = 0.0016
                    elif kind == "core":
                        poses[capsule_body, 0] = 0.0037
                    state.body_q.assign(poses)
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
                    analytic_kinds = (
                        "barrel",
                        "parallel",
                        "parallel_clipped",
                        "near_parallel",
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_tilted",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_bottom",
                    )
                    test.assertEqual(count, 0 if kind in analytic_kinds else 1)
                    if kind in ("cap_axial", "cap_outside", "rim", "core"):
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    if kind in analytic_kinds and kind != "near_parallel":
                        expected_count = 1 if kind == "barrel" else 2
                        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), expected_count)
                        bodies = model.shape_body.numpy()
                        core_points = []
                        for i in range(expected_count):
                            a = int(contacts.rigid_contact_shape0.numpy()[i])
                            b = int(contacts.rigid_contact_shape1.numpy()[i])
                            x0 = np.asarray(
                                wp.transform_point(
                                    wp.transform(*poses[bodies[a]]), wp.vec3(*contacts.rigid_contact_point0.numpy()[i])
                                )
                            )
                            x1 = np.asarray(
                                wp.transform_point(
                                    wp.transform(*poses[bodies[b]]), wp.vec3(*contacts.rigid_contact_point1.numpy()[i])
                                )
                            )
                            normal = contacts.rigid_contact_normal.numpy()[i]
                            gap = float(
                                (x1 - x0) @ normal
                                - contacts.rigid_contact_margin0.numpy()[i]
                                - contacts.rigid_contact_margin1.numpy()[i]
                            )
                            expected_gap = -0.00005
                            if kind == "cap_tilted":
                                capsule_core = x0 if bodies[a] == capsule_body else x1
                                expected_gap = capsule_core[2] - 0.0015 - 0.00025
                            elif kind == "cap_penetrating":
                                expected_gap = -0.0003
                            elif kind == "cap_deep":
                                expected_gap = -0.00075
                            test.assertAlmostEqual(gap, expected_gap, delta=1e-8)
                            if kind == "cap_bottom":
                                expected = np.array([0.0, 0.0, -1.0])
                            elif kind in (
                                "cap_face",
                                "cap_clipped",
                                "cap_clipped_negative",
                                "cap_bridge",
                                "cap_tilted",
                                "cap_penetrating",
                                "cap_deep",
                            ):
                                expected = np.array([0.0, 0.0, 1.0])
                            else:
                                expected = np.array([1.0, 0.0, 0.0])
                            if bodies[a] == capsule_body:
                                expected = -expected
                            np.testing.assert_allclose(normal, expected, atol=1e-6)
                            core_points.append(x0 if bodies[a] == capsule_body else x1)
                        if expected_count == 2:
                            expected_length = 0.001
                            if kind == "parallel_clipped":
                                expected_length = 0.0007
                            elif kind in ("cap_clipped", "cap_clipped_negative"):
                                expected_length = 0.00055
                            elif kind == "cap_bridge":
                                expected_length = 0.0075
                            test.assertAlmostEqual(
                                np.linalg.norm(core_points[1] - core_points[0]), expected_length, delta=1e-8
                            )
                            expected_cap_x = {
                                "cap_face": [-0.0005, 0.0005],
                                "cap_clipped": [0.0032, 0.00375],
                                "cap_clipped_negative": [-0.00375, -0.0032],
                                "cap_bridge": [-0.00375, 0.00375],
                            }
                            if kind in expected_cap_x:
                                np.testing.assert_allclose(
                                    sorted(point[0] for point in core_points), expected_cap_x[kind], atol=1e-8
                                )


class TestCapsuleCylinderBarrel(unittest.TestCase):
    """Check point and line contacts and preserve general convex fallbacks."""


add_function_test(TestCapsuleCylinderBarrel, "test_barrel_guards", test_barrel_guards, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_pipeline_dispatch", test_pipeline_dispatch, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
