# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_contact import (
    mesh_sdf_contact_passes_surface_filter,
    mesh_sdf_contact_search_precision,
)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_surface_filter_kernel(out: wp.array[wp.int32]):
    out[0] = int(mesh_sdf_contact_passes_surface_filter(0.05, 0.1, 0.5, 1.0))
    out[1] = int(mesh_sdf_contact_passes_surface_filter(0.06, 0.1, 0.5, 1.0))
    out[2] = int(mesh_sdf_contact_passes_surface_filter(-0.01, 0.1, 0.5, 1.0))
    out[3] = int(mesh_sdf_contact_passes_surface_filter(10.0, 0.1, 0.5, 0.0))


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


class TestSDFContact(unittest.TestCase):
    def test_mesh_sdf_contact_surface_filter_uses_voxel_radius(self) -> None:
        device = wp.get_preferred_device()
        passes = wp.empty(4, dtype=wp.int32, device=device)

        wp.launch(_mesh_sdf_contact_surface_filter_kernel, dim=1, inputs=[passes], device=device)

        np.testing.assert_array_equal(passes.numpy(), np.array([1, 0, 1, 1], dtype=np.int32))

    def test_mesh_sdf_contact_search_precision_uses_inner_envelope(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(4, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_search_precision_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(values.numpy(), np.array([0.0, 0.001, 0.005, 0.005], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
