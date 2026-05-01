# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX cloth-collision plumbing.

Pieces specific to making cloth triangles flow through the standard
collision pipeline:

* :mod:`broadphase_filter` -- the ``@wp.func`` that drops cloth
  triangle pairs sharing a node (adjacent triangles + self-pairs),
  registered with the broadphase via the ``filter_func`` hook.
* :mod:`triangle_stamp` -- single fused per-step kernel that fills
  the cloth-triangle slots ``[S, S+T)`` of the unified shape arrays
  (``shape_type`` / ``shape_transform`` / ``shape_data`` /
  ``shape_auxiliary``) and the broadphase AABB arrays in one pass.
* :mod:`pipeline` -- :class:`PhoenxCollisionPipeline`, the
  :class:`~newton._src.sim.collide.CollisionPipeline` subclass that
  wires the broadphase filter + the stamp kernel into the standard
  rigid pipeline.
"""

from newton._src.solvers.phoenx.cloth_collision.broadphase_filter import (
    PhoenxBroadphaseFilterData,
    phoenx_broadphase_filter,
)
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline
from newton._src.solvers.phoenx.cloth_collision.triangle_stamp import (
    compute_cloth_triangle_stamp_kernel,
    launch_cloth_triangle_stamp,
)

__all__ = [
    "PhoenxBroadphaseFilterData",
    "PhoenxCollisionPipeline",
    "compute_cloth_triangle_stamp_kernel",
    "launch_cloth_triangle_stamp",
    "phoenx_broadphase_filter",
]
