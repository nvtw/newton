# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Full-coordinate joint row builders for PhoenX articulation DVI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointRowBlock:
    """Compact Jacobian and violation rows for one body-pair joint."""

    jacobian: np.ndarray
    violation: np.ndarray


def ball_socket_rows(
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
) -> JointRowBlock:
    """Build three full-coordinate point-lock rows."""
    jac = np.zeros((3, 12), dtype=np.float64)
    violation = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(parent_anchor_world, dtype=np.float64)
    _fill_spherical_rows(jac, violation, 0, parent_anchor_world, child_anchor_world, parent_com_world, child_com_world)
    return JointRowBlock(jacobian=jac, violation=violation.copy())


def revolute_rows(
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
    parent_axis_world: np.ndarray,
    child_axis_world: np.ndarray,
) -> JointRowBlock:
    """Build five hard rows for a revolute joint."""
    jac = np.zeros((5, 12), dtype=np.float64)
    violation = np.zeros(5, dtype=np.float64)
    anchor_error = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(parent_anchor_world, dtype=np.float64)
    violation[:3] = anchor_error
    _fill_spherical_rows(jac, violation, 0, parent_anchor_world, child_anchor_world, parent_com_world, child_com_world)

    axis_parent = _normalize(parent_axis_world)
    axis_child = _normalize(child_axis_world)
    tangent0, tangent1 = orthonormal_pair(axis_parent)
    swing_error = np.cross(axis_parent, axis_child)
    _fill_perpendicular_rotation_rows(jac, violation, 3, tangent0, tangent1, swing_error)
    return JointRowBlock(jacobian=jac, violation=violation)


def prismatic_rows(
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
    parent_axis_world: np.ndarray,
    angular_error_world: np.ndarray,
) -> JointRowBlock:
    """Build five hard rows for a prismatic joint."""
    jac = np.zeros((5, 12), dtype=np.float64)
    violation = np.zeros(5, dtype=np.float64)
    axis_parent = _normalize(parent_axis_world)
    tangent0, tangent1 = orthonormal_pair(axis_parent)
    anchor_error = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(parent_anchor_world, dtype=np.float64)
    _fill_perpendicular_position_rows(
        jac,
        violation,
        0,
        tangent0,
        tangent1,
        anchor_error,
        parent_anchor_world,
        child_anchor_world,
        parent_com_world,
        child_com_world,
    )
    _fill_locked_orientation_rows(jac, violation, 2, angular_error_world)
    return JointRowBlock(jacobian=jac, violation=violation)


def fixed_rows(
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
    angular_error_world: np.ndarray,
) -> JointRowBlock:
    """Build six hard rows for a fixed joint."""
    jac = np.zeros((6, 12), dtype=np.float64)
    violation = np.zeros(6, dtype=np.float64)
    anchor_error = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(parent_anchor_world, dtype=np.float64)
    violation[:3] = anchor_error
    _fill_spherical_rows(jac, violation, 0, parent_anchor_world, child_anchor_world, parent_com_world, child_com_world)
    _fill_locked_orientation_rows(jac, violation, 3, angular_error_world)
    return JointRowBlock(jacobian=jac, violation=violation)


def cylindrical_rows(
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
    parent_axis_world: np.ndarray,
    child_axis_world: np.ndarray,
) -> JointRowBlock:
    """Build four hard rows for a cylindrical joint."""
    jac = np.zeros((4, 12), dtype=np.float64)
    violation = np.zeros(4, dtype=np.float64)
    axis_parent = _normalize(parent_axis_world)
    axis_child = _normalize(child_axis_world)
    tangent0, tangent1 = orthonormal_pair(axis_parent)
    anchor_error = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(parent_anchor_world, dtype=np.float64)
    _fill_perpendicular_position_rows(
        jac,
        violation,
        0,
        tangent0,
        tangent1,
        anchor_error,
        parent_anchor_world,
        child_anchor_world,
        parent_com_world,
        child_com_world,
    )
    swing_error = np.cross(axis_parent, axis_child)
    _fill_perpendicular_rotation_rows(jac, violation, 2, tangent0, tangent1, swing_error)
    return JointRowBlock(jacobian=jac, violation=violation)


def orthonormal_pair(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to ``axis``."""
    a = _normalize(axis)
    abs_axis = np.abs(a)
    seed_index = int(np.argmin(abs_axis))
    seed = np.zeros(3, dtype=np.float64)
    seed[seed_index] = 1.0
    t0 = _normalize(np.cross(a, seed))
    t1 = np.cross(a, t0)
    return t0, t1


def _fill_spherical_rows(
    jacobian: np.ndarray,
    violation: np.ndarray,
    row0: int,
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
) -> None:
    r_parent = np.asarray(parent_anchor_world, dtype=np.float64) - np.asarray(parent_com_world, dtype=np.float64)
    r_child = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(child_com_world, dtype=np.float64)
    for axis in range(3):
        e = np.zeros(3, dtype=np.float64)
        e[axis] = 1.0
        row = row0 + axis
        jacobian[row, 0:3] = -e
        jacobian[row, 3:6] = -np.cross(r_parent, e)
        jacobian[row, 6:9] = e
        jacobian[row, 9:12] = np.cross(r_child, e)
        violation[row] = (
            np.asarray(child_anchor_world, dtype=np.float64)[axis]
            - np.asarray(parent_anchor_world, dtype=np.float64)[axis]
        )


def _fill_perpendicular_rotation_rows(
    jacobian: np.ndarray,
    violation: np.ndarray,
    row0: int,
    tangent0: np.ndarray,
    tangent1: np.ndarray,
    swing_error: np.ndarray,
) -> None:
    for local, tangent in enumerate((tangent0, tangent1)):
        row = row0 + local
        t = np.asarray(tangent, dtype=np.float64)
        jacobian[row, 3:6] = -t
        jacobian[row, 9:12] = t
        violation[row] = float(np.dot(swing_error, t))


def _fill_perpendicular_position_rows(
    jacobian: np.ndarray,
    violation: np.ndarray,
    row0: int,
    tangent0: np.ndarray,
    tangent1: np.ndarray,
    anchor_error: np.ndarray,
    parent_anchor_world: np.ndarray,
    child_anchor_world: np.ndarray,
    parent_com_world: np.ndarray,
    child_com_world: np.ndarray,
) -> None:
    r_parent = np.asarray(parent_anchor_world, dtype=np.float64) - np.asarray(parent_com_world, dtype=np.float64)
    r_child = np.asarray(child_anchor_world, dtype=np.float64) - np.asarray(child_com_world, dtype=np.float64)
    for local, tangent in enumerate((tangent0, tangent1)):
        row = row0 + local
        t = np.asarray(tangent, dtype=np.float64)
        jacobian[row, 0:3] = -t
        jacobian[row, 3:6] = -np.cross(r_parent, t)
        jacobian[row, 6:9] = t
        jacobian[row, 9:12] = np.cross(r_child, t)
        violation[row] = float(np.dot(anchor_error, t))


def _fill_locked_orientation_rows(
    jacobian: np.ndarray,
    violation: np.ndarray,
    row0: int,
    angular_error_world: np.ndarray,
) -> None:
    error = np.asarray(angular_error_world, dtype=np.float64)
    for axis in range(3):
        e = np.zeros(3, dtype=np.float64)
        e[axis] = 1.0
        row = row0 + axis
        jacobian[row, 3:6] = -e
        jacobian[row, 9:12] = e
        violation[row] = error[axis]


def _normalize(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    length = float(np.linalg.norm(a))
    if length <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector {a}")
    return a / length
