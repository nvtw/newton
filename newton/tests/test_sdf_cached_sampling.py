# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for cached, importance-sampled SDF contacts (``sdf_contact_method="cached_sampling"``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import SDFCachedSampling
from newton.tests.unittest_utils import get_cuda_test_devices


def _box(h, res=64):
    mesh = newton.Mesh.create_box(h[0], h[1], h[2], duplicate_vertices=False, compute_normals=False, compute_uvs=False)
    mesh.build_sdf(max_resolution=res, narrow_band_range=(-0.02, 0.02), margin=0.02)
    return mesh


def _sphere(r, res=64):
    mesh = newton.Mesh.create_sphere(r, num_latitudes=32, num_longitudes=32, compute_normals=False, compute_uvs=False)
    mesh.build_sdf(max_resolution=res, narrow_band_range=(-0.02, 0.02), margin=0.02)
    return mesh


def _cube_on_slab(device, gap=0.002, z=0.05, rotation=None):
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=0.0, mu=0.5)
    builder.add_shape_mesh(
        -1, mesh=_box((0.3, 0.3, 0.05)), xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), cfg=cfg
    )
    body = builder.add_body(xform=wp.transform((0.0, 0.0, z), rotation if rotation is not None else wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=_box((0.05, 0.05, 0.05)), cfg=cfg)
    return builder.finalize(device=device)


def _contacts(contacts):
    n = int(contacts.rigid_contact_count.numpy()[0])
    return {
        "count": n,
        "shape0": contacts.rigid_contact_shape0.numpy()[:n],
        "shape1": contacts.rigid_contact_shape1.numpy()[:n],
        "point0": contacts.rigid_contact_point0.numpy()[:n],
        "point1": contacts.rigid_contact_point1.numpy()[:n],
        "normal": contacts.rigid_contact_normal.numpy()[:n],
    }


def _separation(model, state, c):
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()
    out = []
    for i in range(c["count"]):
        p = []
        for shape, point in ((c["shape0"][i], c["point0"][i]), (c["shape1"][i], c["point1"][i])):
            b = shape_body[shape]
            if b >= 0:
                q = body_q[b]
                p.append(
                    np.asarray(wp.transform_point(wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:])), wp.vec3(*point)))
                )
            else:
                p.append(np.asarray(point))
        out.append(float(np.dot(p[1] - p[0], c["normal"][i])))
    return np.asarray(out)


def _captured_collide(pipeline, state, contacts):
    with wp.ScopedCapture() as capture:
        pipeline.collide(state, contacts)
    return capture.graph


class TestSDFCachedSampling(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SDFCachedSampling.Config(cache_size=0).validate()
        with self.assertRaises(ValueError):
            SDFCachedSampling.Config(selection="best").validate()
        with self.assertRaises(ValueError):
            SDFCachedSampling.Config(tolerance=0.0).validate()
        SDFCachedSampling.Config().validate()

    def test_pipeline_rejects_config_for_edge_method(self):
        for device in get_cuda_test_devices():
            model = _cube_on_slab(device)
            with self.assertRaises(ValueError):
                newton.CollisionPipeline(model, sdf_cached_sampling_config=SDFCachedSampling.Config())
            with self.assertRaises(ValueError):
                newton.CollisionPipeline(model, sdf_contact_method="unknown")
            self.assertIsNone(newton.CollisionPipeline(model).narrow_phase.sdf_cached_sampling)

    def test_overlapping_spheres_depth_and_normal(self):
        """Two SDF spheres overlapping along x: depth, normal and position of the deepest point."""
        for device in get_cuda_test_devices():
            r, overlap = 0.05, 0.004
            builder = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig(gap=0.001, margin=0.0)
            sphere = _sphere(r)
            for x in (-r + 0.5 * overlap, r - 0.5 * overlap):
                body = builder.add_body(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
                builder.add_shape_mesh(body, mesh=sphere, cfg=cfg)
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(model, sdf_contact_method="cached_sampling")
            contacts = pipeline.contacts()
            state = model.state()
            graph = _captured_collide(pipeline, state, contacts)
            for _ in range(5):
                wp.capture_launch(graph)
            c = _contacts(contacts)
            self.assertGreater(c["count"], 0)
            sep = _separation(model, state, c)
            deepest = int(np.argmin(sep))
            voxel = 2.0 * r / 64
            self.assertAlmostEqual(sep[deepest], -overlap, delta=voxel)
            self.assertGreater(abs(float(c["normal"][deepest][0])), 0.98)

    def test_resting_box_spans_face_and_persists(self):
        """A cube resting on a slab keeps contacts at its face corners, unchanged across calls."""
        for device in get_cuda_test_devices():
            model = _cube_on_slab(device)
            pipeline = newton.CollisionPipeline(model, sdf_contact_method="cached_sampling")
            contacts = pipeline.contacts()
            state = model.state()
            graph = _captured_collide(pipeline, state, contacts)
            for _ in range(40):
                wp.capture_launch(graph)
            first = _contacts(contacts)
            for _ in range(10):
                wp.capture_launch(graph)
            last = _contacts(contacts)
            self.assertEqual(first["count"], pipeline.narrow_phase.sdf_cached_sampling.config.cache_size)
            # Cube-local contact points (shape 1 is the cube) cover all four quadrants.
            pts = last["point1"][:, :2]
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    self.assertTrue(np.any((pts[:, 0] * sx > 0.02) & (pts[:, 1] * sy > 0.02)), pts)
            np.testing.assert_allclose(np.sort(first["point1"], axis=0), np.sort(last["point1"], axis=0), atol=1e-6)
            np.testing.assert_allclose(last["normal"][:, 2], 1.0, atol=1e-3)

    def test_cache_keyed_by_shape_pair(self):
        """Reordering the broad-phase pairs leaves a static scene's contacts unchanged."""
        for device in get_cuda_test_devices():
            builder = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig(gap=0.002, margin=0.0)
            builder.add_shape_mesh(
                -1, mesh=_box((0.5, 0.5, 0.05)), xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), cfg=cfg
            )
            cube = _box((0.05, 0.05, 0.05))
            for x in (-0.2, 0.2):
                body = builder.add_body(xform=wp.transform((x, 0.0, 0.05), wp.quat_identity()))
                builder.add_shape_mesh(body, mesh=cube, cfg=cfg)
            model = builder.finalize(device=device)
            pairs = wp.array([[0, 1], [0, 2]], dtype=wp.vec2i, device=device)
            pipeline = newton.CollisionPipeline(
                model, sdf_contact_method="cached_sampling", broad_phase="explicit", shape_pairs_filtered=pairs
            )
            contacts = pipeline.contacts()
            state = model.state()
            graph = _captured_collide(pipeline, state, contacts)
            for _ in range(30):
                wp.capture_launch(graph)
            before = _contacts(contacts)
            pairs.assign(np.array([[0, 2], [0, 1]], dtype=np.int32))
            wp.capture_launch(graph)
            after = _contacts(contacts)
            self.assertEqual(before["count"], after["count"])
            for shape in (1, 2):
                a = np.sort(before["point1"][before["shape1"] == shape], axis=0)
                b = np.sort(after["point1"][after["shape1"] == shape], axis=0)
                np.testing.assert_allclose(a, b, atol=1e-6)

    def test_reset_drops_cache(self):
        for device in get_cuda_test_devices():
            model = _cube_on_slab(device)
            pipeline = newton.CollisionPipeline(model, sdf_contact_method="cached_sampling")
            contacts = pipeline.contacts()
            state = model.state()
            for _ in range(5):
                pipeline.collide(state, contacts)
            cs = pipeline.narrow_phase.sdf_cached_sampling
            self.assertEqual(int(cs.prev_pair_count.numpy()[0]), 1)
            pipeline.reset_contact_matching()
            self.assertEqual(int(cs.prev_pair_count.numpy()[0]), 0)

    def test_deterministic(self):
        """Two pipelines produce bit-identical contacts for the same moving scene."""
        for device in get_cuda_test_devices():
            builder = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig(gap=0.004, margin=0.0)
            sphere = _sphere(0.05)
            builder.add_shape_mesh(
                -1, mesh=_box((0.5, 0.5, 0.05)), xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), cfg=cfg
            )
            for i in range(4):
                body = builder.add_body(xform=wp.transform((0.09 * i - 0.13, 0.0, 0.049), wp.quat_identity()))
                builder.add_shape_mesh(body, mesh=sphere, cfg=cfg)
            model = builder.finalize(device=device)
            results = []
            for _ in range(2):
                pipeline = newton.CollisionPipeline(model, sdf_contact_method="cached_sampling", deterministic=True)
                contacts = pipeline.contacts()
                state = model.state()
                q0 = state.body_q.numpy().copy()
                trace = []
                for f in range(20):
                    q = q0.copy()
                    q[:, 0] += 0.002 * f
                    angle = 0.05 * f
                    q[:, 3:] = [0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)]
                    state.body_q.assign(q)
                    pipeline.collide(state, contacts)
                    c = _contacts(contacts)
                    trace.append(np.concatenate([c["point0"].ravel(), c["point1"].ravel(), c["normal"].ravel()]))
                results.append(np.concatenate(trace))
            np.testing.assert_array_equal(results[0], results[1])

    def test_hydroelastic_and_bvh_pairs_unchanged(self):
        """Hydroelastic pairs and mesh pairs without a texture SDF keep their contacts."""
        for device in get_cuda_test_devices():
            builder = newton.ModelBuilder()
            hydro = newton.ModelBuilder.ShapeConfig(gap=0.002, margin=0.0, is_hydroelastic=True, kh=1.0e8)
            hydro.configure_sdf(max_resolution=32)
            for z in (0.0, 0.098):
                body = builder.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
                builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=hydro)
            cfg = newton.ModelBuilder.ShapeConfig(gap=0.002, margin=0.0)
            plain = newton.Mesh.create_box(
                0.05, 0.05, 0.05, duplicate_vertices=False, compute_normals=False, compute_uvs=False
            )
            for z in (0.0, 0.099):
                body = builder.add_body(xform=wp.transform((1.0, 0.0, z), wp.quat_identity()))
                builder.add_shape_mesh(body, mesh=plain, cfg=cfg)
            model = builder.finalize(device=device)
            outputs = []
            for method in ("edge", "cached_sampling"):
                pipeline = newton.CollisionPipeline(
                    model,
                    sdf_contact_method=method,
                    sdf_hydroelastic_config=newton.geometry.HydroelasticSDF.Config(),
                    deterministic=True,
                )
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                outputs.append(_contacts(contacts))
            self.assertGreater(outputs[0]["count"], 0)
            self.assertEqual(outputs[0]["count"], outputs[1]["count"])
            for key in ("shape0", "shape1", "point0", "point1", "normal"):
                np.testing.assert_array_equal(outputs[0][key], outputs[1][key])

    def test_phoenx_cube_settles(self):
        """A cube dropped tilted onto a slab comes to rest on a face under PhoenX (graph-captured)."""
        for device in get_cuda_test_devices():
            tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.6, 0.2)), 0.9)
            model = _cube_on_slab(device, gap=0.004, z=0.15, rotation=tilt)
            pipeline = newton.CollisionPipeline(model, sdf_contact_method="cached_sampling", contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=4, solver_iterations=8)
            contacts = pipeline.contacts()
            control = model.control()
            s0, s1 = model.state(), model.state()

            with wp.ScopedCapture() as capture:
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, control, contacts, 1.0 / 60.0)
                pipeline.collide(s1, contacts)
                solver.step(s1, s0, control, contacts, 1.0 / 60.0)
            for _ in range(90):
                wp.capture_launch(capture.graph)
            qd = s0.body_qd.numpy()[0]
            z = float(s0.body_q.numpy()[0][2])
            self.assertLess(float(np.linalg.norm(qd)), 1.0e-3)
            self.assertAlmostEqual(z, 0.05, delta=0.002)


if __name__ == "__main__":
    unittest.main(verbosity=2)
