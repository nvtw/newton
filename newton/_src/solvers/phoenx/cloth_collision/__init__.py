# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX cloth-collision plumbing.

Pieces specific to making cloth triangles flow through the standard
collision pipeline:

* :mod:`broadphase_filter` -- the ``@wp.func`` that drops cloth
  triangle pairs sharing a node (adjacent triangles + self-pairs),
  registered with the broadphase via the ``filter_func`` hook.
* :mod:`triangle_aabb` -- per-step kernel that fills the cloth
  triangle slots ``[S, S+T)`` of the concatenated AABB array from
  particle positions.
"""

from newton._src.solvers.phoenx.cloth_collision.broadphase_filter import (
    PhoenxBroadphaseFilterData,
    phoenx_broadphase_filter,
)
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline
from newton._src.solvers.phoenx.cloth_collision.triangle_aabb import (
    compute_cloth_triangle_aabbs_kernel,
    launch_cloth_triangle_aabbs,
)
from newton._src.solvers.phoenx.cloth_collision.triangle_shape_data import (
    compute_cloth_triangle_shape_data_kernel,
    launch_cloth_triangle_shape_data,
)

__all__ = [
    "PhoenxBroadphaseFilterData",
    "PhoenxCollisionPipeline",
    "compute_cloth_triangle_aabbs_kernel",
    "compute_cloth_triangle_shape_data_kernel",
    "launch_cloth_triangle_aabbs",
    "launch_cloth_triangle_shape_data",
    "phoenx_broadphase_filter",
]
