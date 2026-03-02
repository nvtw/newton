# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for texture-based SDF construction and sampling.

Validates TextureSDFData construction, sampling accuracy against NanoVDB,
gradient quality, extrapolation, array indexing, and multi-resolution behavior.

Note: These tests require GPU (CUDA) since wp.Texture3D only supports CUDA devices.
"""

import unittest

import numpy as np
import warp as wp

from newton import Mesh
from newton._src.geometry.sdf_contact import sample_sdf_extrapolated, sample_sdf_grad_extrapolated
from newton._src.geometry.sdf_texture import (
    QuantizationMode,
    TextureSDFData,
    create_empty_texture_sdf_data,
    create_texture_sdf_from_mesh,
    texture_sample_sdf,
    texture_sample_sdf_grad,
)
from newton._src.geometry.sdf_utils import SDFData, SDF
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_cuda_available = wp.is_cuda_available()


def _create_box_mesh(half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> Mesh:
    """Create a simple box mesh for testing."""
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,  # Bottom
            4, 5, 6, 4, 6, 7,  # Top
            0, 1, 5, 0, 5, 4,  # Front
            2, 3, 7, 2, 7, 6,  # Back
            0, 4, 7, 0, 7, 3,  # Left
            1, 2, 6, 1, 6, 5,  # Right
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


@wp.kernel
def _sample_texture_sdf_kernel(
    sdf: TextureSDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = texture_sample_sdf(sdf, query_points[tid])


@wp.kernel
def _sample_texture_sdf_grad_kernel(
    sdf: TextureSDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, grad = texture_sample_sdf_grad(sdf, query_points[tid])
    results[tid] = dist
    gradients[tid] = grad


@wp.kernel
def _sample_nanovdb_value_kernel(
    sdf_data: SDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = sample_sdf_extrapolated(sdf_data, query_points[tid])


@wp.kernel
def _sample_nanovdb_grad_kernel(
    sdf_data: SDFData,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, grad = sample_sdf_grad_extrapolated(sdf_data, query_points[tid])
    results[tid] = dist
    gradients[tid] = grad


@wp.kernel
def _sample_texture_sdf_from_array_kernel(
    sdf_table: wp.array(dtype=TextureSDFData),
    sdf_idx: int,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    results[tid] = texture_sample_sdf(sdf_table[sdf_idx], query_points[tid])


def _build_texture_and_nanovdb(mesh, resolution=64, margin=0.05, narrow_band_range=(-0.1, 0.1)):
    """Build both texture SDF and NanoVDB SDF for comparison."""
    device = "cuda:0"
    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    # Build texture SDF
    tex_sdf, coarse_tex, subgrid_tex = create_texture_sdf_from_mesh(
        wp_mesh,
        margin=margin,
        narrow_band_range=narrow_band_range,
        max_resolution=resolution,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )

    # Build NanoVDB SDF
    mesh.build_sdf(
        max_resolution=resolution,
        narrow_band_range=narrow_band_range,
        margin=margin,
    )
    nanovdb_data = mesh.sdf.to_kernel_data()

    return tex_sdf, coarse_tex, subgrid_tex, nanovdb_data, wp_mesh


def _generate_query_points(mesh, num_points=1000, seed=42):
    """Generate random query points near the mesh."""
    rng = np.random.default_rng(seed)
    verts = mesh.vertices
    min_ext = verts.min(axis=0) - 0.05
    max_ext = verts.max(axis=0) + 0.05

    # Mix of near-surface and random points
    num_near = num_points * 7 // 10
    num_random = num_points - num_near

    vert_indices = rng.integers(0, len(verts), size=num_near)
    offsets = rng.normal(0, 0.02, size=(num_near, 3)).astype(np.float32)
    near_points = verts[vert_indices] + offsets

    random_points = rng.uniform(min_ext, max_ext, size=(num_random, 3)).astype(np.float32)

    points = np.concatenate([near_points, random_points], axis=0)
    rng.shuffle(points)
    return points


class TestTextureSDF(unittest.TestCase):
    pass


def test_texture_sdf_construction(test, device):
    """Build TextureSDFData and verify fields are populated."""
    mesh = _create_box_mesh()
    tex_sdf, coarse_tex, subgrid_tex, _, wp_mesh = _build_texture_and_nanovdb(mesh)

    test.assertGreater(tex_sdf.coarse_size_x, 0)
    test.assertGreater(tex_sdf.coarse_size_y, 0)
    test.assertGreater(tex_sdf.coarse_size_z, 0)
    test.assertGreater(tex_sdf.inv_sdf_dx, 0.0)
    test.assertGreater(tex_sdf.subgrid_size, 0)
    test.assertEqual(tex_sdf.subgrid_size_f, float(tex_sdf.subgrid_size))
    test.assertEqual(tex_sdf.subgrid_samples_f, float(tex_sdf.subgrid_size + 1))

    # Verify box bounds contain the mesh
    box_lower = np.array([tex_sdf.sdf_box_lower[0], tex_sdf.sdf_box_lower[1], tex_sdf.sdf_box_lower[2]])
    box_upper = np.array([tex_sdf.sdf_box_upper[0], tex_sdf.sdf_box_upper[1], tex_sdf.sdf_box_upper[2]])
    mesh_min = mesh.vertices.min(axis=0)
    mesh_max = mesh.vertices.max(axis=0)
    test.assertTrue(np.all(box_lower <= mesh_min))
    test.assertTrue(np.all(box_upper >= mesh_max))


def test_texture_sdf_values_match_nanovdb(test, device):
    """Compare texture SDF vs NanoVDB at random points."""
    mesh = _create_box_mesh()
    tex_sdf, coarse_tex, subgrid_tex, nanovdb_data, wp_mesh = _build_texture_and_nanovdb(mesh)

    query_np = _generate_query_points(mesh, num_points=1000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    tex_results = wp.zeros(1000, dtype=float, device=device)
    nano_results = wp.zeros(1000, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=1000, inputs=[tex_sdf, query_points, tex_results], device=device)
    wp.launch(_sample_nanovdb_value_kernel, dim=1000, inputs=[nanovdb_data, query_points, nano_results], device=device)
    wp.synchronize()

    tex_np = tex_results.numpy()
    nano_np = nano_results.numpy()

    # Filter to valid points (both give reasonable values)
    valid = (np.abs(tex_np) < 1e5) & (np.abs(nano_np) < 1e5)
    test.assertTrue(np.sum(valid) > 500, f"Too few valid points: {np.sum(valid)}")

    diff = np.abs(tex_np[valid] - nano_np[valid])
    mean_err = diff.mean()
    test.assertLess(mean_err, 0.02, f"Mean SDF error too large: {mean_err:.6f}")


def test_texture_sdf_gradient_accuracy(test, device):
    """Compare texture analytical gradient vs NanoVDB gradient."""
    mesh = _create_box_mesh()
    tex_sdf, coarse_tex, subgrid_tex, nanovdb_data, wp_mesh = _build_texture_and_nanovdb(mesh)

    query_np = _generate_query_points(mesh, num_points=1000)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    tex_vals = wp.zeros(1000, dtype=float, device=device)
    tex_grads = wp.zeros(1000, dtype=wp.vec3, device=device)
    nano_vals = wp.zeros(1000, dtype=float, device=device)
    nano_grads = wp.zeros(1000, dtype=wp.vec3, device=device)

    wp.launch(
        _sample_texture_sdf_grad_kernel,
        dim=1000, inputs=[tex_sdf, query_points, tex_vals, tex_grads], device=device,
    )
    wp.launch(
        _sample_nanovdb_grad_kernel,
        dim=1000, inputs=[nanovdb_data, query_points, nano_vals, nano_grads], device=device,
    )
    wp.synchronize()

    tg = tex_grads.numpy()
    ng = nano_grads.numpy()
    tv = tex_vals.numpy()
    nv = nano_vals.numpy()

    # Compute gradient angles for valid points
    valid_mask = (np.abs(tv) < 1e5) & (np.abs(nv) < 1e5)
    n1 = np.linalg.norm(tg, axis=1)
    n2 = np.linalg.norm(ng, axis=1)
    grad_valid = valid_mask & (n1 > 1e-8) & (n2 > 1e-8)

    test.assertTrue(np.sum(grad_valid) > 300, f"Too few valid gradient points: {np.sum(grad_valid)}")

    tg_n = tg[grad_valid] / n1[grad_valid, None]
    ng_n = ng[grad_valid] / n2[grad_valid, None]
    dots = np.sum(tg_n * ng_n, axis=1)
    angles = np.arccos(np.clip(dots, -1, 1)) * 180.0 / np.pi

    mean_angle = float(angles.mean())
    test.assertLess(mean_angle, 10.0, f"Mean gradient angle too large: {mean_angle:.2f} deg")


def test_texture_sdf_extrapolation(test, device):
    """Points outside box have correct extrapolated distance."""
    mesh = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
    tex_sdf, coarse_tex, subgrid_tex, _, wp_mesh = _build_texture_and_nanovdb(mesh)

    # Points well outside the box along +X axis
    outside_points = np.array([
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ], dtype=np.float32)
    query_points = wp.array(outside_points, dtype=wp.vec3, device=device)
    results = wp.zeros(4, dtype=float, device=device)

    wp.launch(_sample_texture_sdf_kernel, dim=4, inputs=[tex_sdf, query_points, results], device=device)
    wp.synchronize()

    vals = results.numpy()
    # Points far outside should have positive distance
    for i in range(4):
        test.assertGreater(vals[i], 0.5, f"Point {i} should be far outside, got dist={vals[i]:.4f}")


def test_texture_sdf_array_indexing(test, device):
    """Create wp.array(dtype=TextureSDFData) with 2 entries, sample from kernel via index."""
    mesh1 = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
    mesh2 = _create_box_mesh(half_extents=(0.3, 0.3, 0.3))

    wp_mesh1 = wp.Mesh(
        points=wp.array(mesh1.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh1.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    wp_mesh2 = wp.Mesh(
        points=wp.array(mesh2.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh2.indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    tex_sdf1, coarse1, sub1 = create_texture_sdf_from_mesh(
        wp_mesh1, margin=0.05, narrow_band_range=(-0.1, 0.1), max_resolution=32, device=device,
    )
    tex_sdf2, coarse2, sub2 = create_texture_sdf_from_mesh(
        wp_mesh2, margin=0.05, narrow_band_range=(-0.1, 0.1), max_resolution=32, device=device,
    )

    sdf_array = wp.array([tex_sdf1, tex_sdf2], dtype=TextureSDFData, device=device)

    # Query point at origin (inside both boxes)
    query = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    results0 = wp.zeros(1, dtype=float, device=device)
    results1 = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        _sample_texture_sdf_from_array_kernel,
        dim=1, inputs=[sdf_array, 0, query, results0], device=device,
    )
    wp.launch(
        _sample_texture_sdf_from_array_kernel,
        dim=1, inputs=[sdf_array, 1, query, results1], device=device,
    )
    wp.synchronize()

    val0 = float(results0.numpy()[0])
    val1 = float(results1.numpy()[0])

    # Origin is inside both boxes, so both should be negative
    test.assertLess(val0, 0.0, f"Origin should be inside box1, got {val0:.4f}")
    test.assertLess(val1, 0.0, f"Origin should be inside box2, got {val1:.4f}")
    # Box2 is smaller, so origin should be closer to its surface (less negative)
    test.assertGreater(val1, val0, f"Origin should be closer to surface in smaller box: val0={val0:.4f}, val1={val1:.4f}")


def test_texture_sdf_multi_resolution(test, device):
    """Test at resolutions 32, 64, 128, 256 - higher res should be more accurate."""
    mesh = _create_box_mesh()
    query_np = _generate_query_points(mesh, num_points=500)
    query_points = wp.array(query_np, dtype=wp.vec3, device=device)

    # Build NanoVDB reference at high resolution
    mesh_copy = _create_box_mesh()
    mesh_copy.build_sdf(max_resolution=256, narrow_band_range=(-0.1, 0.1), margin=0.05)
    ref_data = mesh_copy.sdf.to_kernel_data()
    ref_results = wp.zeros(500, dtype=float, device=device)
    wp.launch(_sample_nanovdb_value_kernel, dim=500, inputs=[ref_data, query_points, ref_results], device=device)
    wp.synchronize()
    ref_np = ref_results.numpy()

    prev_mean_err = float("inf")
    for resolution in [32, 64, 128]:
        wp_mesh = wp.Mesh(
            points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        tex_sdf, coarse_tex, subgrid_tex = create_texture_sdf_from_mesh(
            wp_mesh, margin=0.05, narrow_band_range=(-0.1, 0.1), max_resolution=resolution, device=device,
        )
        tex_results = wp.zeros(500, dtype=float, device=device)
        wp.launch(_sample_texture_sdf_kernel, dim=500, inputs=[tex_sdf, query_points, tex_results], device=device)
        wp.synchronize()

        tex_np = tex_results.numpy()
        valid = (np.abs(tex_np) < 1e5) & (np.abs(ref_np) < 1e5)
        if np.sum(valid) > 100:
            mean_err = float(np.abs(tex_np[valid] - ref_np[valid]).mean())
            # Error should decrease (or at least not increase much) with resolution
            test.assertLess(mean_err, prev_mean_err * 2.0,
                            f"Error increased too much at res={resolution}: {mean_err:.6f} vs prev {prev_mean_err:.6f}")
            prev_mean_err = mean_err


def test_texture_sdf_in_model(test, device):
    """Build a scene with 2 mesh shapes with SDFs and verify model.texture_sdf_data."""
    import newton

    builder = newton.ModelBuilder(gravity=0.0)

    for i in range(2):
        body = builder.add_body(xform=wp.transform(wp.vec3(float(i) * 2.0, 0.0, 0.0)))
        mesh = _create_box_mesh(half_extents=(0.5, 0.5, 0.5))
        mesh.build_sdf(max_resolution=8)
        builder.add_shape_mesh(body, mesh=mesh)

    model = builder.finalize(device=device)

    # Both shapes should have SDF indices
    sdf_indices = model.shape_sdf_index.numpy()
    test.assertEqual(sdf_indices[0], 0)
    test.assertEqual(sdf_indices[1], 1)

    # texture_sdf_data should have 2 entries
    test.assertIsNotNone(model.texture_sdf_data)
    test.assertEqual(len(model.texture_sdf_data), 2)

    # Both entries should have non-zero coarse_size_x (not empty)
    tex_np = model.texture_sdf_data.numpy()
    for idx in range(2):
        test.assertGreater(tex_np[idx]["coarse_size_x"], 0, f"texture_sdf_data[{idx}] is empty")

    # Texture references should be kept alive
    test.assertEqual(len(model.texture_sdf_coarse_textures), 2)
    test.assertEqual(len(model.texture_sdf_subgrid_textures), 2)


def test_empty_texture_sdf_data(test, device):
    """Verify create_empty_texture_sdf_data returns a valid empty struct."""
    empty = create_empty_texture_sdf_data()
    test.assertEqual(empty.coarse_size_x, 0)
    test.assertEqual(empty.coarse_size_y, 0)
    test.assertEqual(empty.coarse_size_z, 0)
    test.assertFalse(empty.scale_baked)


# Register tests for CUDA devices
devices = get_cuda_test_devices()
add_function_test(TestTextureSDF, "test_texture_sdf_construction", test_texture_sdf_construction, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_values_match_nanovdb", test_texture_sdf_values_match_nanovdb, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_gradient_accuracy", test_texture_sdf_gradient_accuracy, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_extrapolation", test_texture_sdf_extrapolation, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_array_indexing", test_texture_sdf_array_indexing, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_multi_resolution", test_texture_sdf_multi_resolution, devices=devices)
add_function_test(TestTextureSDF, "test_texture_sdf_in_model", test_texture_sdf_in_model, devices=devices)
add_function_test(TestTextureSDF, "test_empty_texture_sdf_data", test_empty_texture_sdf_data, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
