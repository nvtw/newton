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

__all__ = [
    "PhoenxBroadphaseFilterData",
    "phoenx_broadphase_filter",
]
