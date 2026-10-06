# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check specialized capsule-cylinder point, line, and rim contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.geometry.collision_primitive import (
    _capsule_cylinder_rim_normal,
    collide_capsule_cylinder,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query_capsule_cylinder(
    positions: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    cylinder_axes: wp.array[wp.vec3],
    dimensions: wp.array[wp.vec4],
    distances: wp.array[wp.vec2],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    dims = dimensions[i]
    d0, p0, d1, _p1, normal = collide_capsule_cylinder(
        positions[i], axes[i], dims[0], dims[1], wp.vec3(0.0), cylinder_axes[i], dims[2], dims[3]
    )
    distances[i] = wp.vec2(d0, d1)
    points[i] = p0
    normals[i] = normal


def test_support_witnesses(test, device):
    """Certify separated and penetrating witnesses across poses and scales."""
    rng = np.random.default_rng(42)
    count = 288
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    positions = rng.uniform(-2.0, 2.0, size=(count, 3))
    dimensions = rng.uniform(0.1, 1.5, size=(count, 4))
    # Include full core intersections, zero-length capsules, and all scales.
    positions[:24] *= 0.1
    positions[:6] = 0.0
    axes[:6] = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1e-5, 0, 1], [1, 0, 1e-5]]
    axes[:6] /= np.linalg.norm(axes[:6], axis=1)[:, None]
    dimensions[:6] = [0.25, 3.0, 1.0, 1.0]
    dimensions[24:32, 1] = 0.0
    # Cap normals require an endpoint inside the disk even for tiny tilts.
    # Otherwise the selected rim feature must retain its interior witness.
    for index, tilt in enumerate((0.0, -1e-7, 1e-7, -1e-6, 1e-6, -1e-5, 1e-5, -1e-3, 1e-3), start=count - 9):
        positions[index] = [-0.203244, 0.504936, -0.719]
        axes[index] = [0.340771, -0.940146, tilt]
        axes[index] /= np.linalg.norm(axes[index])
        dimensions[index] = [0.01, 0.8536813536163314, 0.3, 0.73]
    # Almost horizontal overhangs and almost circular projected rims also need
    # valid surface witnesses, not just an accurate scalar gap.
    index = 32
    for c in (1e-9, 1e-8, 1e-7, 1e-5, 0.001, 0.01, 1.0 - 1e-5):
        sine = np.sqrt(1.0 - c * c)
        for a in (-0.1, -1e-4, 1e-4, 0.1):
            for b in (0.3, 1.0):
                axes[index] = [sine, 0.0, c]
                positions[index] = (a + sine) * np.array([-c, 0.0, sine]) + [0.0, b, 0.0]
                dimensions[index] = [0.1, 2.0, 1.0, 1.0]
                index += 1
    scales = np.tile([0.001, 1.0, 1000.0], count // 3)
    rotations, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    # Preserve sub-epsilon tilts instead of losing them in frame roundoff.
    rotations[32:56] = np.eye(3)
    with wp.ScopedDevice(device):
        distances = wp.zeros(count, dtype=wp.vec2)
        points = wp.zeros(count, dtype=wp.vec3)
        normals = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            query_capsule_cylinder,
            count,
            [
                wp.array(np.einsum("nij,nj->ni", rotations, positions) * scales[:, None], dtype=wp.vec3),
                wp.array(np.einsum("nij,nj->ni", rotations, axes), dtype=wp.vec3),
                wp.array(rotations[:, :, 2], dtype=wp.vec3),
                wp.array(dimensions * scales[:, None], dtype=wp.vec4),
                distances,
                points,
                normals,
            ],
        )
    gaps = distances.numpy()[:, 0] / scales
    manifolds = distances.numpy()[:, 1] < MAXVAL
    centers = np.einsum("nji,nj->ni", rotations, points.numpy()) / scales[:, None]
    outward = np.einsum("nji,nj->ni", rotations, -normals.numpy())
    # An independent collection of separating planes detects missed features.
    sampled = rng.normal(size=(16384, 3))
    sampled /= np.linalg.norm(sampled, axis=1)[:, None]
    sampled = np.concatenate((sampled, np.eye(3), -np.eye(3)))
    for i in range(count):
        with test.subTest(case=i, scale=scales[i]):
            radius, length, cylinder_radius, height = dimensions[i]
            # The deliberately approximate near-parallel manifold bounds core
            # drift by .01*R, so its tangent-plane witness error is O(.0001*R).
            witness_tolerance = 5e-5 * cylinder_radius + 2e-6 if manifolds[i] else 2e-5
            normal = outward[i]
            np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=2e-6)
            plane_gaps = (
                sampled @ positions[i]
                - length * np.abs(sampled @ axes[i])
                - height * np.abs(sampled[:, 2])
                - cylinder_radius * np.linalg.norm(sampled[:, :2], axis=1)
                - radius
            )
            test.assertGreaterEqual(gaps[i] + 2e-5, np.max(plane_gaps))
            capsule_surface = centers[i] + 0.5 * gaps[i] * normal
            cylinder_surface = centers[i] - 0.5 * gaps[i] * normal
            along = np.clip((capsule_surface - positions[i]) @ axes[i], -length, length)
            core = positions[i] + along * axes[i]
            test.assertAlmostEqual(np.linalg.norm(capsule_surface - core), radius, delta=witness_tolerance)
            radial = np.linalg.norm(cylinder_surface[:2])
            axial = abs(cylinder_surface[2])
            test.assertLessEqual(radial, cylinder_radius + witness_tolerance)
            test.assertLessEqual(axial, height + witness_tolerance)
            test.assertLessEqual(min(abs(radial - cylinder_radius), abs(axial - height)), witness_tolerance)


@wp.kernel
def query_rim_normals(parameters: wp.array[wp.vec3], represented: wp.array[wp.vec3], normals: wp.array[wp.vec3]):
    i = wp.tid()
    c = parameters[i][0]
    sine = wp.sqrt(1.0 - c * c)
    axis = wp.vec3(sine, 0.0, c)
    e0 = wp.vec3(-c, 0.0, sine)
    relative = (parameters[i][1] + sine) * e0 + wp.vec3(0.0, parameters[i][2], 0.0)
    represented[i] = wp.vec3(c, wp.dot(relative, e0) - sine, parameters[i][2])
    normals[i] = _capsule_cylinder_rim_normal(relative, axis, 1.0, 1.0, 1.0)


def test_rim_conditioning(test, device):
    """Check thin ellipses, nearly circular rims, and repeated-root limits."""
    parameters = []
    for c in (0.0, 1e-9, 1e-8, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-5):
        for a in (-2.0, -0.1, -1e-4, 0.0, 1e-4, 0.1, 2.0):
            for b in (0.0, 1e-4, 0.01, 0.3, 0.9, 1.0, 1.1, 2.0, c * abs(a)):
                parameters.append((c, a, b))
    parameters = np.asarray(parameters, dtype=np.float32)
    with wp.ScopedDevice(device):
        normals = wp.zeros(len(parameters), dtype=wp.vec3)
        represented = wp.zeros(len(parameters), dtype=wp.vec3)
        wp.launch(query_rim_normals, len(parameters), [wp.array(parameters, dtype=wp.vec3), represented, normals])
    angles = np.linspace(0.0, 0.5 * np.pi, 16385)
    cs, sn = np.cos(angles), np.sin(angles)
    for (c, a, b), normal in zip(represented.numpy().astype(np.float64), normals.numpy(), strict=True):
        with test.subTest(c=c, a=a, b=b):
            x = normal @ np.array([-c, 0.0, np.sqrt(1.0 - c * c)])
            y = normal[1]
            radius = np.sqrt(c * c * x * x + y * y)
            gap = a * x + b * y - radius
            sampled = a * cs + b * sn - np.sqrt(c * c * cs * cs + sn * sn)
            test.assertGreaterEqual(gap + 2e-5, sampled.max())
            if x > 1e-5 and y > 1e-5:
                derivative = -a * y + b * x - (1.0 - c * c) * x * y / radius
                test.assertLessEqual(abs(derivative), 2e-5 * max(1.0, abs(a), b))


def test_pipeline_dispatch(test, device):
    """Handle sharp cylinders without GJK/MPR and preserve rounded-cylinder dispatch."""
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
                "cap_axial_bottom",
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
                    if kind in ("parallel", "parallel_clipped", "cap_axial", "cap_axial_bottom"):
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
                        "cap_axial_bottom",
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
                    elif kind == "cap_axial_bottom":
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
                    manifold_kinds = (
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
                        "cap_axial",
                        "cap_axial_bottom",
                    )
                    test.assertEqual(count, 1 if kind == "rounded" else 0)
                    if kind in ("cap_outside", "rim", "core"):
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    if kind in manifold_kinds and kind != "near_parallel":
                        expected_count = 1 if kind in ("barrel", "cap_axial", "cap_axial_bottom") else 2
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
                            elif kind in ("cap_axial", "cap_axial_bottom"):
                                expected_gap = -0.00055
                            test.assertAlmostEqual(gap, expected_gap, delta=1e-8)
                            if kind in ("cap_bottom", "cap_axial_bottom"):
                                expected = np.array([0.0, 0.0, -1.0])
                            elif kind in (
                                "cap_face",
                                "cap_clipped",
                                "cap_clipped_negative",
                                "cap_bridge",
                                "cap_tilted",
                                "cap_penetrating",
                                "cap_deep",
                                "cap_axial",
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


def test_core_crosses_both_caps(test, device):
    """Resolve a core crossing both caps with the whole capsule's support gap."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.01)
        cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
        capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
        builder.add_shape_cylinder(body=cylinder_body, radius=10.0, half_height=1.0, cfg=cfg)
        builder.add_shape_capsule(body=capsule_body, radius=0.25, half_height=3.0, cfg=cfg)
        model = builder.finalize()
        state = model.state()
        poses = model.body_q.numpy()
        poses[capsule_body, :3] = (0.0, 0.0, 0.1)
        axis = wp.normalize(wp.vec3(1.0, 0.0, 1.0))
        poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
        state.body_q.assign(poses)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 0)
        bodies = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()
        point0 = contacts.rigid_contact_point0.numpy()
        point1 = contacts.rigid_contact_point1.numpy()
        for i in range(count):
            if bodies[shape0[i]] == capsule_body:
                capsule_local, cylinder_local = point0[i], point1[i]
            else:
                capsule_local, cylinder_local = point1[i], point0[i]
            capsule_world = np.asarray(wp.transform_point(wp.transform(*poses[capsule_body]), wp.vec3(*capsule_local)))
            cylinder_world = np.asarray(
                wp.transform_point(wp.transform(*poses[cylinder_body]), wp.vec3(*cylinder_local))
            )
            normal = contacts.rigid_contact_normal.numpy()[i]
            if bodies[shape0[i]] == capsule_body:
                np.testing.assert_allclose(normal, [0.0, 0.0, -1.0], atol=1e-6)
            else:
                np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
            gap = (
                capsule_world[2]
                - cylinder_world[2]
                - contacts.rigid_contact_margin0.numpy()[i]
                - contacts.rigid_contact_margin1.numpy()[i]
            )
            # Moving upward must lift the bottommost core endpoint past the
            # top cap. Clipping the core to z=0 understates this penetration.
            test.assertAlmostEqual(gap, 0.1 - 3.0 / np.sqrt(2.0) - 1.0 - 0.25, delta=2e-6)


def test_rim_overhang_and_near_parallel(test, device):
    """Preserve rim gaps and two barrel witnesses when the capsule tilts."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in ("rim", "barrel"):
                angles = (30.0, 60.0, 80.0) if kind == "rim" else (0.0, 0.001, -0.001)
                for angle in angles:
                    for direction in (-1.0, 1.0):
                        for drop in (0.0, 0.05):
                            with test.subTest(order=reversed_order, kind=kind, angle=angle, cap=direction, drop=drop):
                                builder = newton.ModelBuilder()
                                cfg = newton.ModelBuilder.ShapeConfig(gap=0.2)
                                body = builder.add_body()
                                cylinder = {"body": -1, "radius": 1.0, "half_height": 1.0, "cfg": cfg}
                                capsule = {
                                    "body": body,
                                    "radius": 0.1,
                                    "half_height": 2.0 if kind == "rim" else 0.5,
                                    "cfg": cfg,
                                }
                                if reversed_order:
                                    builder.add_shape_capsule(**capsule)
                                    builder.add_shape_cylinder(**cylinder)
                                else:
                                    builder.add_shape_cylinder(**cylinder)
                                    builder.add_shape_capsule(**capsule)
                                if kind == "rim":
                                    theta = np.radians(angle)
                                    axis = np.array([np.cos(theta), 0.0, -direction * np.sin(theta)])
                                    outward = np.array([np.sin(theta), 0.0, direction * np.cos(theta)])
                                    center = np.array([1.0, 0.0, direction]) + 0.1 * outward - 0.5 * axis
                                    center[2] -= direction * drop
                                else:
                                    axis = direction * np.array([np.sin(angle), 0.0, np.cos(angle)])
                                    center = [1.1 + 0.5 * abs(np.sin(angle)) - drop, 0.0, 0.0]
                                model = builder.finalize()
                                state = model.state()
                                rotation = wp.quat_from_axis_angle(
                                    wp.vec3(0.0, 1.0, 0.0), float(np.arctan2(axis[0], axis[2]))
                                )
                                pose = wp.transform(wp.vec3(*center), rotation)
                                state.body_q.assign([pose])
                                pipeline = newton.CollisionPipeline(model)
                                contacts = pipeline.contacts()
                                pipeline.collide(state, contacts)
                                count = int(contacts.rigid_contact_count.numpy()[0])
                                test.assertGreater(count, 0)
                                test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 0)
                                gaps = []
                                core_points = []
                                for i in range(count):
                                    points = []
                                    for shape, point in (
                                        (
                                            contacts.rigid_contact_shape0.numpy()[i],
                                            contacts.rigid_contact_point0.numpy()[i],
                                        ),
                                        (
                                            contacts.rigid_contact_shape1.numpy()[i],
                                            contacts.rigid_contact_point1.numpy()[i],
                                        ),
                                    ):
                                        world = np.asarray(point)
                                        if model.shape_body.numpy()[shape] == body:
                                            world = np.asarray(wp.transform_point(pose, wp.vec3(*point)))
                                            core_points.append(world)
                                        points.append(world)
                                    normal = contacts.rigid_contact_normal.numpy()[i]
                                    gaps.append(
                                        (points[1] - points[0]) @ normal
                                        - contacts.rigid_contact_margin0.numpy()[i]
                                        - contacts.rigid_contact_margin1.numpy()[i]
                                    )
                                    if kind == "rim":
                                        test.assertGreater(abs(normal[0]), 0.1)
                                if kind == "rim":
                                    test.assertAlmostEqual(min(gaps), -drop * np.cos(theta), delta=5e-4)
                                else:
                                    test.assertEqual(count, 2)
                                    np.testing.assert_allclose(
                                        sorted(p[2] for p in core_points),
                                        [-0.5 * np.cos(angle), 0.5 * np.cos(angle)],
                                        atol=1e-6,
                                    )
                                    np.testing.assert_allclose(
                                        sorted(gaps), [-drop, abs(np.sin(angle)) - drop], atol=1e-6
                                    )


def test_near_horizontal_cap_witnesses(test, device):
    """Keep cap/rim witnesses on both surfaces for nearly horizontal cores."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for direction in (-1.0, 1.0):
                for z, radius in ((0.54, 0.75), (0.719, 0.01), (0.72, 0.01), (0.725, 0.01), (0.73, 0.01)):
                    for tilt in (0.0, -1e-7, 1e-7, -1e-6, 1e-6):
                        with test.subTest(order=reversed_order, cap=direction, z=z, radius=radius, tilt=tilt):
                            builder = newton.ModelBuilder()
                            cfg = newton.ModelBuilder.ShapeConfig(gap=0.01, density=0.0)
                            body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                            cylinder = {"body": -1, "radius": 1.38, "half_height": 0.72, "cfg": cfg}
                            capsule = {"body": body, "radius": radius, "half_height": 0.94, "cfg": cfg}
                            if reversed_order:
                                capsule_id = builder.add_shape_capsule(**capsule)
                                builder.add_shape_cylinder(**cylinder)
                            else:
                                builder.add_shape_cylinder(**cylinder)
                                capsule_id = builder.add_shape_capsule(**capsule)
                            model = builder.finalize()
                            state = model.state()
                            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * wp.pi + tilt)
                            pose = wp.transform(wp.vec3(0.8, 0.85, direction * z), rotation)
                            state.body_q.assign([pose])
                            axis = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)), dtype=float)
                            axis /= np.linalg.norm(axis)
                            pipeline = newton.CollisionPipeline(model)
                            contacts = pipeline.contacts()
                            pipeline.collide(state, contacts)
                            count = int(contacts.rigid_contact_count.numpy()[0])
                            test.assertGreater(count, 0)
                            shape0 = contacts.rigid_contact_shape0.numpy()
                            shape1 = contacts.rigid_contact_shape1.numpy()
                            point0 = contacts.rigid_contact_point0.numpy()
                            point1 = contacts.rigid_contact_point1.numpy()
                            margin0 = contacts.rigid_contact_margin0.numpy()
                            margin1 = contacts.rigid_contact_margin1.numpy()
                            normals = contacts.rigid_contact_normal.numpy()
                            for i in range(count):
                                x0 = point0[i].astype(float)
                                x1 = point1[i].astype(float)
                                if shape0[i] == capsule_id:
                                    x0 = np.asarray(wp.transform_point(pose, wp.vec3(*x0)), dtype=float)
                                if shape1[i] == capsule_id:
                                    x1 = np.asarray(wp.transform_point(pose, wp.vec3(*x1)), dtype=float)
                                normal = normals[i].astype(float)
                                gap = (x1 - x0) @ normal - margin0[i] - margin1[i]
                                test.assertAlmostEqual(gap, z - 0.72 - radius, delta=3e-6)
                                s0 = x0 + margin0[i] * normal
                                s1 = x1 - margin1[i] * normal
                                ca, cy = (s0, s1) if shape0[i] == capsule_id else (s1, s0)
                                relative = ca - np.array([0.8, 0.85, direction * z])
                                along = np.clip(relative @ axis, -0.94, 0.94)
                                test.assertAlmostEqual(np.linalg.norm(relative - along * axis), radius, delta=3e-6)
                                test.assertLessEqual(np.linalg.norm(cy[:2]), 1.38 + 3e-6)
                                test.assertAlmostEqual(cy[2], direction * 0.72, delta=3e-6)


def test_cap_rim_support_witnesses(test, device):
    """Keep cap/rim witnesses on both surfaces for almost horizontal overhangs."""
    rotation = wp.quat(-0.6648025512695312, -0.2409108579158783, 0.0, 0.7071064114570618)
    axis = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)), dtype=float)
    axis /= np.linalg.norm(axis)
    with wp.ScopedDevice(device):
        for scale in (0.001, 1.0, 1000.0):
            for reversed_order in (False, True):
                with test.subTest(scale=scale, order=reversed_order):
                    radius, length, cylinder_radius, height = np.array([0.01, 0.8536813536163314, 0.3, 0.73]) * scale
                    position = np.array([-0.203244, 0.504936, -0.719]) * scale
                    pose = wp.transform(wp.vec3(*position), rotation)
                    builder = newton.ModelBuilder()
                    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.01 * scale)
                    capsule = {"body": body, "radius": radius, "half_height": length, "cfg": cfg}
                    cylinder = {"body": -1, "radius": cylinder_radius, "half_height": height, "cfg": cfg}
                    if reversed_order:
                        capsule_id = builder.add_shape_capsule(**capsule)
                        builder.add_shape_cylinder(**cylinder)
                    else:
                        builder.add_shape_cylinder(**cylinder)
                        capsule_id = builder.add_shape_capsule(**capsule)
                    model = builder.finalize()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(model.state(), contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    shapes = contacts.rigid_contact_shape0.numpy()
                    points0 = contacts.rigid_contact_point0.numpy()
                    points1 = contacts.rigid_contact_point1.numpy()
                    normals = contacts.rigid_contact_normal.numpy()
                    margins0 = contacts.rigid_contact_margin0.numpy()
                    margins1 = contacts.rigid_contact_margin1.numpy()
                    tolerance = 3e-6 * scale
                    for i in range(count):
                        x0, x1 = points0[i].astype(float), points1[i].astype(float)
                        if shapes[i] == capsule_id:
                            x0 = np.asarray(wp.transform_point(pose, wp.vec3(*x0)), dtype=float)
                        else:
                            x1 = np.asarray(wp.transform_point(pose, wp.vec3(*x1)), dtype=float)
                        normal = normals[i].astype(float)
                        s0, s1 = x0 + margins0[i] * normal, x1 - margins1[i] * normal
                        cap, cyl = (s0, s1) if shapes[i] == capsule_id else (s1, s0)
                        along = np.clip((cap - position) @ axis, -length, length)
                        test.assertAlmostEqual(np.linalg.norm(cap - position - along * axis), radius, delta=tolerance)
                        test.assertLessEqual(np.linalg.norm(cyl[:2]), cylinder_radius + tolerance)
                        test.assertAlmostEqual(cyl[2], -height, delta=tolerance)
                        gap = (x1 - x0) @ normal - margins0[i] - margins1[i]
                        test.assertAlmostEqual(gap, -0.021 * scale, delta=tolerance)


def test_conditioned_feature_witnesses(test, device):
    """Preserve valid cap and barrel witnesses at ill-conditioned boundaries."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [-33.37416458129883, -7.06756591796875, -22.036640167236328],
            [-0.018604129552841187, 1.0235445499420166, -167.64295959472656],
            [-5.021750450134277, 55.52897262573242, 24.3689022064209],
            [72.64119720458984, -3.374417781829834, 29.78777313232422],
            [35.210872650146484, -32.48945999145508, -2.9023711681365967],
            [9.716127, 4.9476957, 8.939304],
        ]
    )
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.4947160482406616, 0.2887207865715027, -0.819692850112915],
            [-0.9986806511878967, 0.031890347599983215, -0.04024896398186684],
            [0.7282047271728516, -0.6798352003097534, -0.08684459328651428],
            [0.73636454, 0.29118964, 0.6107175],
        ]
    )
    cylinder_axes = np.array(
        [
            [0.8, 0.17, 0.54],
            [0.8217647671699524, 0.17402389645576477, 0.5426033139228821],
            [0.0, 0.0, 1.0],
            [-0.06646548211574554, 0.9085239768028259, 0.41251251101493835],
            [0.0, 0.0, 1.0],
            [0.7286396026611328, -0.6793518662452698, -0.08697929233312607],
            [0.736281, 0.29129338, 0.61076874],
        ]
    )
    dimensions = np.array(
        [
            [0.1, 0.0, 1.0, 10.0],
            [0.06330857425928116, 0.0, 1.0, 42.1290168762207],
            [0.777827262878418, 0.9754909873008728, 1.0, 91.13072204589844],
            [0.5759442448616028, 0.03788183629512787, 1.0, 61.29697799682617],
            [0.5687828660011292, 73.5628662109375, 1.0, 27.199745178222656],
            [0.08787105977535248, 83.95655822753906, 1.0, 78.32493591308594],
            [0.01377502, 12.711492, 1.0, 11.89699],
        ]
    )
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    cylinder_axes /= np.linalg.norm(cylinder_axes, axis=1)[:, None]
    for scale in (0.001, 1.0, 1000.0):
        with wp.ScopedDevice(device):
            distances = wp.empty(len(positions), dtype=wp.vec2)
            points = wp.empty(len(positions), dtype=wp.vec3)
            normals = wp.empty(len(positions), dtype=wp.vec3)
            wp.launch(
                query_capsule_cylinder,
                len(positions),
                [
                    wp.array(positions * scale, dtype=wp.vec3),
                    wp.array(axes, dtype=wp.vec3),
                    wp.array(cylinder_axes, dtype=wp.vec3),
                    wp.array(dimensions * scale, dtype=wp.vec4),
                    distances,
                    points,
                    normals,
                ],
            )
        gaps = distances.numpy().astype(float) / scale
        centers = points.numpy().astype(float) / scale
        outward = -normals.numpy().astype(float)
        test.assertLess(gaps[-1, 1], MAXVAL)
        for i, (radius, length, cylinder_radius, height) in enumerate(dimensions[:-1]):
            with test.subTest(scale=scale, case=i):
                tolerance = 4e-6 * max(np.max(dimensions[i]), np.linalg.norm(positions[i]))
                normal = outward[i]
                capsule_surface = centers[i] + 0.5 * gaps[i, 0] * normal
                cylinder_surface = centers[i] - 0.5 * gaps[i, 0] * normal
                along = np.clip((capsule_surface - positions[i]) @ axes[i], -length, length)
                test.assertAlmostEqual(
                    np.linalg.norm(capsule_surface - positions[i] - along * axes[i]), radius, delta=tolerance
                )
                axial = cylinder_surface @ cylinder_axes[i]
                radial = np.linalg.norm(cylinder_surface - axial * cylinder_axes[i])
                test.assertLessEqual(radial, cylinder_radius + tolerance)
                test.assertLessEqual(abs(axial), height + tolerance)
                test.assertLessEqual(min(abs(radial - cylinder_radius), abs(abs(axial) - height)), tolerance)
                axial_normal = normal @ cylinder_axes[i]
                support_gap = (
                    positions[i] @ normal
                    - length * abs(axes[i] @ normal)
                    - height * abs(axial_normal)
                    - cylinder_radius * np.linalg.norm(normal - axial_normal * cylinder_axes[i])
                    - radius
                )
                test.assertAlmostEqual(gaps[i, 0], support_gap, delta=tolerance)


def test_gap_admission(test, device):
    """Generate separated barrel and cap contacts inside the combined shape gap."""
    with wp.ScopedDevice(device):
        for kind, positions in (
            ("barrel", (0.0043, 0.0046)),
            ("cap", (0.00205, 0.00235)),
        ):
            for position, expected_count in zip(positions, (1 if kind == "barrel" else 2, 0), strict=True):
                with test.subTest(kind=kind, position=position):
                    builder = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.00025)
                    cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    builder.add_shape_cylinder(body=cylinder_body, radius=0.00375, half_height=0.0015, cfg=cfg)
                    builder.add_shape_capsule(body=capsule_body, radius=0.00025, half_height=0.0005, cfg=cfg)
                    model = builder.finalize()
                    state = model.state()
                    poses = model.body_q.numpy()
                    if kind == "barrel":
                        poses[capsule_body, :3] = (position, 0.0, 0.0)
                        axis = wp.vec3(0.0, 1.0, 0.0)
                    else:
                        poses[capsule_body, :3] = (0.0, 0.0, position)
                        axis = wp.vec3(1.0, 0.0, 0.0)
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    state.body_q.assign(poses)
                    for speculative in (False, True):
                        with test.subTest(speculative=speculative):
                            pipeline = newton.CollisionPipeline(
                                model,
                                speculative_contact_gap_max=0.001 if speculative else None,
                            )
                            contacts = pipeline.contacts()
                            if speculative:
                                pipeline.collide(state, contacts, dt=0.01)
                            else:
                                pipeline.collide(state, contacts)
                            test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), expected_count)


def test_generic_dispatch_bounds(test, device):
    """Handle sharp pairs without GJK and keep live rounded-cylinder routing."""
    with wp.ScopedDevice(device):
        for rounded in (False, True):
            for reversed_order in (False, True):
                builder = newton.ModelBuilder()
                cylinder = {"body": -1, "radius": 1.0, "half_height": 1.0}
                if rounded:
                    cylinder["barrel_radius"] = 2.0
                body = builder.add_body(xform=wp.transform(wp.vec3(1.09, 0.0, 0.0), wp.quat_identity()))
                capsule = {"body": body, "radius": 0.1, "half_height": 0.5}
                if reversed_order:
                    builder.add_shape_capsule(**capsule)
                    builder.add_shape_cylinder(**cylinder)
                else:
                    builder.add_shape_cylinder(**cylinder)
                    builder.add_shape_capsule(**capsule)
                model = builder.finalize()
                state = model.state()
                for mode in ("nxn", "sap", "explicit"):
                    with test.subTest(rounded=rounded, order=reversed_order, broad_phase=mode):
                        pipeline = newton.CollisionPipeline(
                            model,
                            broad_phase=mode,
                            shape_pairs_filtered=wp.array([[0, 1]], dtype=wp.vec2i) if mode == "explicit" else None,
                        )
                        contacts = pipeline.contacts()
                        pipeline.collide(state, contacts)
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                        test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), int(rounded))
                        if not rounded:
                            scale = model.shape_scale.numpy()
                            cylinder_index = int(reversed_order)
                            scale[cylinder_index, 2] = 2.0
                            model.shape_scale.assign(scale)
                            pipeline.collide(state, contacts)
                            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                            test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 1)
                            scale[cylinder_index, 2] = 0.0
                            model.shape_scale.assign(scale)


def test_endpoint_tangent_to_rim(test, device):
    """Return the endpoint gap when rounding rejects every stationary rim feature."""
    angles = np.linspace(0.05, 1.52, 400)
    offsets = np.array([-0.05, -0.01, 0.0, 1e-4, 0.01])
    angle, offset = (grid.ravel() for grid in np.meshgrid(angles, offsets))
    count = len(angle)
    # The core leaves the rim tangentially from an endpoint on its bisector.
    outward = np.stack([np.sin(angle), np.zeros(count), np.cos(angle)], axis=1)
    axes = np.stack([np.cos(angle), np.zeros(count), -np.sin(angle)], axis=1)
    positions = np.array([1.0, 0.0, 0.75]) + (0.2 + offset)[:, None] * outward + 0.6 * axes
    with wp.ScopedDevice(device):
        distances = wp.zeros(count, dtype=wp.vec2)
        points = wp.zeros(count, dtype=wp.vec3)
        normals = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            query_capsule_cylinder,
            count,
            [
                wp.array(positions, dtype=wp.vec3),
                wp.array(axes, dtype=wp.vec3),
                wp.array(np.tile([0.0, 0.0, 1.0], (count, 1)), dtype=wp.vec3),
                wp.array(np.tile([0.2, 0.6, 1.0, 0.75], (count, 1)), dtype=wp.vec4),
                distances,
                points,
                normals,
            ],
        )
    np.testing.assert_allclose(distances.numpy()[:, 0], offset, atol=2e-6)
    np.testing.assert_allclose(normals.numpy(), -outward, atol=2e-3)


def test_barrel_witness_at_cap_plane(test, device):
    """Keep barrel normals whose core witness lies on a cap plane within rounding."""
    angles = np.linspace(-1.4, 1.4, 200)
    sides = np.array([-1.0, 1.0])
    alongs = np.array([-0.4, -0.1, 0.0, 0.25])
    offsets = np.array([-0.01, 0.0, 0.001])
    angle, side, along, offset = (grid.ravel() for grid in np.meshgrid(angles, sides, alongs, offsets))
    count = len(angle)
    # The core is perpendicular to the barrel normal +y, and its closest
    # point to the cylinder axis lies exactly at a cap height.
    axes = np.stack([np.cos(angle), np.zeros(count), np.sin(angle)], axis=1)
    positions = np.stack([-along * axes[:, 0], 0.5 + offset, side * 0.75 - along * axes[:, 2]], axis=1)
    with wp.ScopedDevice(device):
        distances = wp.zeros(count, dtype=wp.vec2)
        points = wp.zeros(count, dtype=wp.vec3)
        normals = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            query_capsule_cylinder,
            count,
            [
                wp.array(positions, dtype=wp.vec3),
                wp.array(axes, dtype=wp.vec3),
                wp.array(np.tile([0.0, 0.0, 1.0], (count, 1)), dtype=wp.vec3),
                wp.array(np.tile([0.05, 0.5, 0.45, 0.75], (count, 1)), dtype=wp.vec4),
                distances,
                points,
                normals,
            ],
        )
    np.testing.assert_allclose(distances.numpy()[:, 0], offset, atol=2e-6)


def _support_gaps(normals, positions, axes, dims):
    """Return float64 support gaps for outward normals in the cylinder frame."""
    radius, length, cylinder_radius, height = (dims[:, i, None] for i in range(4))
    return (
        np.einsum("nkj,nj->nk", normals, positions)
        - length * np.abs(np.einsum("nkj,nj->nk", normals, axes))
        - height * np.abs(normals[..., 2])
        - cylinder_radius * np.hypot(normals[..., 0], normals[..., 1])
        - radius
    )


def _maximize_support_gap(starts, positions, axes, dims, iterations=300):
    """Refine each start by a tangent-plane pattern search on the unit sphere."""
    normal = starts / np.linalg.norm(starts, axis=-1, keepdims=True)
    gap = _support_gaps(normal[:, None], positions, axes, dims)[:, 0]
    step = np.full(len(normal), 0.1)
    for _ in range(iterations):
        reference = np.where(np.abs(normal[:, :1]) > 0.9, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        v = np.cross(normal, u)
        improved = np.zeros(len(normal), dtype=bool)
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            candidate = normal + step[:, None] * (a * u + b * v)
            candidate /= np.linalg.norm(candidate, axis=-1, keepdims=True)
            candidate_gap = _support_gaps(candidate[:, None], positions, axes, dims)[:, 0]
            better = candidate_gap > gap
            normal = np.where(better[:, None], candidate, normal)
            gap = np.where(better, candidate_gap, gap)
            improved |= better
        step = np.minimum(np.where(improved, 1.5 * step, 0.5 * step), 0.3)
    return gap


def _cylinder_sdf(points, cylinder_radius, height):
    radial = np.hypot(points[..., 0], points[..., 1]) - cylinder_radius
    axial = np.abs(points[..., 2]) - height
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    return outside + np.minimum(np.maximum(radial, axial), 0.0)


def _signed_distance(positions, axes, dims, solver_normals):
    """Return a float64 reference for the capsule-cylinder signed distance.

    The signed distance is the maximum support gap over unit normals. Every
    normal gives a lower bound, so refine sampled, feature, and solver
    normals. For an exterior core, also minimize the convex cylinder SDF.
    """
    radius, length, cylinder_radius, height = dims.T
    i = np.arange(1500) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / len(i))
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    sphere = np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=-1)
    sampled = _support_gaps(np.broadcast_to(sphere, (len(positions), *sphere.shape)), positions, axes, dims)
    starts = [sphere[np.argsort(-sampled, axis=1)[:, j]] for j in range(4)]
    starts += [solver_normals + 1e-30, np.tile([0.0, 0.0, 1.0], (len(positions), 1))]
    starts += [-np.tile([0.0, 0.0, 1.0], (len(positions), 1))]
    best = np.max([_maximize_support_gap(start, positions, axes, dims) for start in starts], axis=0)
    low, high = -length, length.copy()
    for _ in range(100):
        a, b = low + (high - low) / 3.0, high - (high - low) / 3.0
        left = _cylinder_sdf(positions + a[:, None] * axes, cylinder_radius, height) < _cylinder_sdf(
            positions + b[:, None] * axes, cylinder_radius, height
        )
        high = np.where(left, b, high)
        low = np.where(left, low, a)
    core = _cylinder_sdf(positions + (0.5 * (low + high))[:, None] * axes, cylinder_radius, height)
    return np.where(core > 0.0, core - radius, best)


def test_signed_distance_reference(test, device):
    """Match random and feature-boundary contacts to a float64 reference."""
    rng = np.random.default_rng(7)
    count = 1500
    dims = np.column_stack(
        [
            rng.uniform(0.01, 0.4, 2 * count),
            rng.uniform(0.0, 1.5, 2 * count),
            rng.uniform(0.1, 1.5, 2 * count),
            rng.uniform(0.1, 1.5, 2 * count),
        ]
    )
    axes = rng.normal(size=(2 * count, 3))
    positions = rng.uniform(-2.0, 2.0, size=(2 * count, 3))
    # Touch a rim, cap edge, or barrel end with an endpoint or interior core point.
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radial = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1)
    side = rng.choice([-1.0, 1.0], count)
    cone = rng.choice([0.0, 1e-7, 0.3, 0.8, 1.2, 0.5 * np.pi - 1e-7, 0.5 * np.pi], count)
    outward = np.sin(cone)[:, None] * radial + (side * np.cos(cone))[:, None] * [0.0, 0.0, 1.0]
    tangent = np.cross(outward, [0.0, 0.0, 1.0])
    tangent[np.linalg.norm(tangent, axis=1) < 1e-9] = np.cross(radial, [0.0, 0.0, 1.0])[
        np.linalg.norm(tangent, axis=1) < 1e-9
    ]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    tangent = np.where(rng.random(count)[:, None] < 0.5, tangent, np.cross(outward, tangent))
    radius, length, cylinder_radius, height = dims[count:].T
    offset = rng.choice([-0.05, -1e-4, 0.0, 1e-4, 0.05], count) * radius
    rim = cylinder_radius[:, None] * radial + (side * height)[:, None] * [0.0, 0.0, 1.0]
    along = np.where(rng.random(count) < 0.5, length, rng.uniform(-0.9, 0.9, count) * length)
    axes[count:] = tangent
    positions[count:] = rim + (radius + offset)[:, None] * outward + along[:, None] * tangent
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    # Pass a rotated pose, then compare in the cylinder frame.
    rotation, _ = np.linalg.qr(rng.normal(size=(len(axes), 3, 3)))
    with wp.ScopedDevice(device):
        distances = wp.zeros(len(axes), dtype=wp.vec2)
        points = wp.zeros(len(axes), dtype=wp.vec3)
        normals = wp.zeros(len(axes), dtype=wp.vec3)
        wp.launch(
            query_capsule_cylinder,
            len(axes),
            [
                wp.array(np.einsum("nij,nj->ni", rotation, positions), dtype=wp.vec3),
                wp.array(np.einsum("nij,nj->ni", rotation, axes), dtype=wp.vec3),
                wp.array(rotation[:, :, 2], dtype=wp.vec3),
                wp.array(dims, dtype=wp.vec4),
                distances,
                points,
                normals,
            ],
        )
    distance = distances.numpy().astype(float)
    point = np.einsum("nji,nj->ni", rotation, points.numpy().astype(float))
    normal = np.einsum("nji,nj->ni", rotation, normals.numpy().astype(float))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    test.assertTrue(np.all(np.isfinite(distance[:, 0])) and np.all(np.isfinite(normal)))
    reference = _signed_distance(positions, axes, dims, -normal)
    scale = np.linalg.norm(positions, axis=1) + dims.sum(axis=1)
    tolerance = 2e-6 * scale
    line = distance[:, 1] < 0.5 * MAXVAL
    # No contact may be deeper than the reference signed distance.
    np.testing.assert_array_less(reference - tolerance, distance[:, 0])
    np.testing.assert_array_less(reference[line] - tolerance[line], distance[line, 1])
    # A single contact's normal must certify its distance as a support gap.
    certified = _support_gaps(-normal[:, None], positions, axes, dims)[:, 0]
    np.testing.assert_array_less(distance[~line, 0] - tolerance[~line], certified[~line])
    np.testing.assert_allclose(np.minimum(distance[line, 0], distance[line, 1]), reference[line], atol=tolerance.max())
    # Both witnesses lie on their surfaces.
    radius, length, cylinder_radius, height = dims.T
    capsule = point - 0.5 * distance[:, :1] * normal
    cylinder = point + 0.5 * distance[:, :1] * normal
    along = np.clip(np.einsum("ni,ni->n", capsule - positions, axes), -length, length)
    capsule_gap = np.linalg.norm(capsule - positions - along[:, None] * axes, axis=1) - radius
    np.testing.assert_array_less(np.abs(capsule_gap), tolerance)
    np.testing.assert_array_less(np.abs(_cylinder_sdf(cylinder, cylinder_radius, height)), tolerance)


class TestCapsuleCylinderBarrel(unittest.TestCase):
    """Check point, line, rim, and penetrating contacts."""


add_function_test(
    TestCapsuleCylinderBarrel, "test_support_witnesses", test_support_witnesses, devices=get_test_devices()
)
add_function_test(TestCapsuleCylinderBarrel, "test_rim_conditioning", test_rim_conditioning, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_generic_dispatch_bounds", test_generic_dispatch_bounds, devices=get_test_devices()
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_core_crosses_both_caps",
    test_core_crosses_both_caps,
    devices=get_test_devices(),
)
add_function_test(TestCapsuleCylinderBarrel, "test_gap_admission", test_gap_admission, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_signed_distance_reference",
    test_signed_distance_reference,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_barrel_witness_at_cap_plane",
    test_barrel_witness_at_cap_plane,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_endpoint_tangent_to_rim",
    test_endpoint_tangent_to_rim,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_conditioned_feature_witnesses",
    test_conditioned_feature_witnesses,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_cap_rim_support_witnesses",
    test_cap_rim_support_witnesses,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_near_horizontal_cap_witnesses",
    test_near_horizontal_cap_witnesses,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_rim_overhang_and_near_parallel",
    test_rim_overhang_and_near_parallel,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel, "test_pipeline_dispatch", test_pipeline_dispatch, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
