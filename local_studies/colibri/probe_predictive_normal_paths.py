# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare normal encoding when predictive reduction reuses or allocates."""

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_and_reduce_predictive_contact,
    export_contact_to_buffer,
)


@wp.kernel
def compare_paths(
    normals: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    reused: GlobalContactReducerData,
    allocated: GlobalContactReducerData,
    ids: wp.array2d[wp.int32],
):
    i = wp.tid()
    n = normals[i]
    first = export_contact_to_buffer(2 * i, 2 * i + 1, wp.vec3(0.0), n, 0.01, i, reused)
    ids[i, 0] = export_and_reduce_predictive_contact(
        2 * i,
        2 * i + 1,
        wp.vec3(0.0),
        n,
        0.01,
        0.0,
        0.0,
        0.0,
        i,
        poses,
        velocities,
        angular,
        0.1,
        0.02,
        first,
        reused,
    )
    ids[i, 1] = export_and_reduce_predictive_contact(
        2 * i,
        2 * i + 1,
        wp.vec3(0.0),
        n,
        0.01,
        0.0,
        0.0,
        0.0,
        i,
        poses,
        velocities,
        angular,
        0.1,
        0.02,
        -1,
        allocated,
    )


if __name__ == "__main__":
    count = 128
    rng = np.random.default_rng(42)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals = normals.astype(np.float32)
    normals[0] = [0.9856598377, -0.00226877094, 0.16872920096]
    velocity = np.zeros((2 * count, 3), dtype=np.float32)
    velocity[1::2] = -normals
    device = "cuda:0"
    reused = GlobalContactReducer(capacity=2 * count, device=device, deterministic=True)
    allocated = GlobalContactReducer(capacity=2 * count, device=device, deterministic=True)
    indices = wp.full((count, 2), -1, dtype=wp.int32, device=device)
    wp.launch(
        compare_paths,
        count,
        [
            wp.array(normals, dtype=wp.vec3, device=device),
            wp.array([wp.transform_identity()] * (2 * count), dtype=wp.transform, device=device),
            wp.array(velocity, dtype=wp.vec3, device=device),
            wp.zeros(2 * count, dtype=wp.vec3, device=device),
            reused.get_data_struct(),
            allocated.get_data_struct(),
            indices,
        ],
        device=device,
    )
    ids = indices.numpy()
    assert np.all(ids > 0)
    a = reused.normal.numpy()[ids[:, 0]]
    b = allocated.normal.numpy()[ids[:, 1]]
    different = np.any(a != b, axis=1)
    print("normal paths differing", int(different.sum()), "of", count)
    print("max encoded difference", float(np.max(np.abs(a - b))))
    print("first witness", normals[np.flatnonzero(different)[0]] if different.any() else None)
    np.testing.assert_array_equal(a, b)
