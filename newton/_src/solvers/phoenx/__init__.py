# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Runtime creation and removal of bodies, shapes, joints, and articulations in PhoenX.

Reserve rows with :meth:`~newton.ModelBuilder.reserve_dynamic_capacity`, then pass
``solver.dynamics.data`` to your kernels and call the ``add_*`` and ``remove_*`` functions
there. Changes take effect at the next :meth:`~newton.solvers.SolverPhoenX.step`, so creation
and removal can be captured in a CUDA graph together with the simulation. Every function
returns ``-1`` or ``False`` when a request cannot be met and counts it in
:attr:`Dynamics.error_count`.

.. experimental::

    Runtime creation may change without a deprecation period.
"""

from newton._src.solvers.phoenx.dynamic import (
    BodyDesc,
    Dynamics,
    DynamicsData,
    JointDesc,
    ShapeDesc,
    add_articulation,
    add_body,
    add_joint,
    add_shape,
    articulation_body,
    articulation_joint,
    body_desc,
    is_articulation_active,
    is_body_active,
    joint_desc,
    joint_wrench,
    remove_articulation,
    remove_body,
    remove_joint,
    remove_shape,
    shape_desc,
)
from newton._src.solvers.phoenx.solver import SolverPhoenX  # noqa: F401  (canonical: newton.solvers)

__all__ = [
    "BodyDesc",
    "Dynamics",
    "DynamicsData",
    "JointDesc",
    "ShapeDesc",
    "add_articulation",
    "add_body",
    "add_joint",
    "add_shape",
    "articulation_body",
    "articulation_joint",
    "body_desc",
    "is_articulation_active",
    "is_body_active",
    "joint_desc",
    "joint_wrench",
    "remove_articulation",
    "remove_body",
    "remove_joint",
    "remove_shape",
    "shape_desc",
]
