# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact normals must differentiate the SDF cell at the query point."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.sdf_texture import (
    QuantizationMode,
    TextureSDFData,
    create_texture_sdf_from_primitive,
    texture_sample_sdf,
    texture_sample_sdf_grad_only,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


@wp.kernel
def _sample_contact_gradients(
    sdf: TextureSDFData,
    points: wp.array[wp.vec3],
    gradients: wp.array[wp.vec3],
):
    i = wp.tid()
    gradients[i] = texture_sample_sdf_grad_only(sdf, points[i])


def test_planar_face_near_cell_edge(test, device):
    """Preserve planar normals when a half-voxel stencil crosses an edge."""
    for scale in (0.001, 1.0, 1000.0):
        for paired in (False, True):
            sdf, coarse, subgrid = create_texture_sdf_from_primitive(
                GeoType.BOX,
                (0.5 * scale,) * 3,
                margin=0.05 * scale,
                narrow_band_range=(-1.0 * scale, 1.0 * scale),
                max_resolution=32,
                quantization_mode=QuantizationMode.FLOAT32,
                paired_samples=paired,
                device=device,
            )
            points = []
            expected = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    # This cell lies entirely in the planar face region.
                    # The former half-voxel derivative crosses the neighboring edge.
                    p = np.zeros(3)
                    p[axis] = sign * 0.482
                    p[(axis + 1) % 3] = 0.481
                    points.append(p * scale)
                    normal = np.zeros(3)
                    normal[axis] = sign
                    expected.append(normal)
                    # Include both sides of an internal voxel boundary in an
                    # affine region, without placing the query on a box edge.
                    for offset in (-1.0e-5, 0.0, 1.0e-5):
                        p = np.zeros(3)
                        p[axis] = sign * (0.20625 + offset)
                        points.append(p * scale)
                        expected.append(normal)
            query = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
            gradients = wp.empty(len(points), dtype=wp.vec3, device=device)
            wp.launch(_sample_contact_gradients, dim=len(points), inputs=[sdf, query, gradients], device=device)
            np.testing.assert_allclose(gradients.numpy(), expected, atol=4.0e-6, rtol=4.0e-6)
            # Hold sparse textures alive until all queries have completed.
            test.assertIsNotNone(coarse)
            test.assertIsNotNone(subgrid)


@wp.kernel
def _sample_coarse_gradient(
    sdf: TextureSDFData,
    point: wp.vec3,
    result: wp.array[wp.vec3],
):
    result[0] = texture_sample_sdf_grad_only(sdf, point)


@wp.kernel
def _sample_coarse_difference(sdf: TextureSDFData, point: wp.vec3, result: wp.array[wp.vec3]):
    h = float(0.0001)
    dx = texture_sample_sdf(sdf, point + wp.vec3(h, 0.0, 0.0)) - texture_sample_sdf(sdf, point - wp.vec3(h, 0.0, 0.0))
    dy = texture_sample_sdf(sdf, point + wp.vec3(0.0, h, 0.0)) - texture_sample_sdf(sdf, point - wp.vec3(0.0, h, 0.0))
    dz = texture_sample_sdf(sdf, point + wp.vec3(0.0, 0.0, h)) - texture_sample_sdf(sdf, point - wp.vec3(0.0, 0.0, h))
    result[1] = wp.vec3(dx, dy, dz) / (2.0 * h)


def test_coarse_gradient_matches_stored_value(test, device):
    """Use coarse-cell spacing when differentiating a coarse fallback."""
    sdf, coarse, subgrid = create_texture_sdf_from_primitive(
        GeoType.BOX,
        (0.5, 0.5, 0.5),
        max_resolution=32,
        quantization_mode=QuantizationMode.FLOAT32,
        device=device,
    )
    result = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(_sample_coarse_gradient, dim=1, inputs=[sdf, wp.vec3(0.2, 0.03, 0.05), result], device=device)
    wp.launch(_sample_coarse_difference, dim=1, inputs=[sdf, wp.vec3(0.2, 0.03, 0.05), result], device=device)
    values = result.numpy()
    np.testing.assert_allclose(values[0], values[1], atol=0.0003, rtol=0.0003)
    test.assertIsNotNone(coarse)
    test.assertIsNotNone(subgrid)


class TestContactSDFGradients(unittest.TestCase):
    """Exercise the gradient sampler selected by mesh contact generation."""


add_function_test(
    TestContactSDFGradients,
    "test_planar_face_near_cell_edge",
    test_planar_face_near_cell_edge,
    devices=get_cuda_test_devices(),
)

add_function_test(
    TestContactSDFGradients,
    "test_coarse_gradient_matches_stored_value",
    test_coarse_gradient_matches_stored_value,
    devices=get_cuda_test_devices(),
)

if __name__ == "__main__":
    unittest.main()
