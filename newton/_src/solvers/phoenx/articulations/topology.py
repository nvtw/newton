# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Host-side articulation topology for PhoenX full-coordinate DVI."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from newton._src.solvers.phoenx.constraints.constraint_joint import (
    DRIVE_MODE_OFF,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)


def d6_constraint_row_count(linear_dof_count: int, angular_dof_count: int) -> int:
    """Return full-coordinate equality row count for a D6-style joint.

    Args:
        linear_dof_count: Number of free linear axes.
        angular_dof_count: Number of free angular axes.

    Returns:
        Number of locked equality rows in maximal coordinates.
    """
    n_linear = int(linear_dof_count)
    n_angular = int(angular_dof_count)
    if not (0 <= n_linear <= 3 and 0 <= n_angular <= 3):
        raise ValueError(
            f"D6 row counts require linear_dof_count and angular_dof_count in [0, 3], got ({n_linear}, {n_angular})"
        )

    linear_rows = 3 if n_linear == 0 else (0 if n_linear == 3 else 2)
    angular_rows = 3 if n_angular == 0 else (0 if n_angular == 3 else 2)
    return linear_rows + angular_rows


def joint_constraint_row_count(joint_mode: int) -> int:
    """Return equality row count for a PhoenX ADBS joint mode.

    CABLE contributes only its hard ball-socket stretch rows here; bend and
    twist remain soft rows in the iterative path until the DVI path grows
    compliant rows.

    Args:
        joint_mode: One of the PhoenX ``JOINT_MODE_*`` integer tags.

    Returns:
        Number of hard equality rows represented by the mode.
    """
    mode = int(joint_mode)
    if mode == int(JOINT_MODE_REVOLUTE):
        return 5
    if mode == int(JOINT_MODE_PRISMATIC):
        return 5
    if mode == int(JOINT_MODE_BALL_SOCKET):
        return 3
    if mode == int(JOINT_MODE_FIXED):
        return 6
    if mode == int(JOINT_MODE_CABLE):
        return 3
    if mode == int(JOINT_MODE_UNIVERSAL):
        return 4
    if mode == int(JOINT_MODE_CYLINDRICAL):
        return 4
    if mode == int(JOINT_MODE_PLANAR):
        return 3
    return 0


def joint_drive_row_count(
    joint_mode: int,
    drive_mode: int,
    stiffness_drive: float,
    damping_drive: float,
) -> int:
    """Return the DVI axial drive row count for a PhoenX ADBS joint.

    The direct solve stores all rows for one joint in a six-row block. Driven
    revolute, prismatic, and universal joints have enough room for one scalar
    axial row in addition to their equality rows. Limit rows remain projected
    iterative constraints until the DVI path grows inequality support.

    Args:
        joint_mode: One of the PhoenX ``JOINT_MODE_*`` integer tags.
        drive_mode: One of the PhoenX ``DRIVE_MODE_*`` integer tags.
        stiffness_drive: Position-drive stiffness.
        damping_drive: Velocity-drive damping.

    Returns:
        ``1`` when the DVI topology should allocate an axial drive row,
        otherwise ``0``.
    """
    mode = int(joint_mode)
    if mode not in (int(JOINT_MODE_REVOLUTE), int(JOINT_MODE_PRISMATIC), int(JOINT_MODE_UNIVERSAL)):
        return 0
    if int(drive_mode) == int(DRIVE_MODE_OFF):
        return 0
    if float(stiffness_drive) <= 0.0 and float(damping_drive) <= 0.0:
        return 0
    return 1


def _normalize_static_bodies(
    bodies: np.ndarray,
    static_body_indices: Iterable[int] | np.ndarray | None,
) -> np.ndarray:
    normalized = np.asarray(bodies, dtype=np.int32).copy()
    if static_body_indices is None:
        normalized[normalized < 0] = -1
        return normalized

    static_indices = np.asarray(list(static_body_indices), dtype=np.int32)
    if static_indices.size == 0:
        normalized[normalized < 0] = -1
        return normalized

    static_mask = np.isin(normalized, static_indices)
    normalized[static_mask] = -1
    normalized[normalized < 0] = -1
    return normalized


@dataclass(frozen=True)
class ArticulationTopology:
    """Compact host topology for the direct articulation solve.

    Attributes:
        body1: Normalized parent body per PhoenX joint. Static bodies are
            represented by ``-1``.
        body2: Normalized child body per PhoenX joint. Static bodies are
            represented by ``-1``.
        joint_mode: PhoenX joint mode per joint.
        row_counts: DVI row count per joint, including enabled axial drive rows.
        drive_row_mask: Whether each joint contributes an axial drive row.
        active_joint_indices: Raw joint indices with at least one DVI row.
        active_row_counts: Row counts for active joints in active-block order.
        active_block_offsets: Prefix sum of ``active_row_counts``.
        row_to_active_block: Compact row to active block index.
        total_rows: Total hard equality rows.
    """

    body1: np.ndarray
    body2: np.ndarray
    joint_mode: np.ndarray
    row_counts: np.ndarray
    drive_row_mask: np.ndarray
    active_joint_indices: np.ndarray
    active_row_counts: np.ndarray
    active_block_offsets: np.ndarray
    row_to_active_block: np.ndarray
    total_rows: int

    @classmethod
    def from_host(
        cls,
        body1: np.ndarray,
        body2: np.ndarray,
        joint_mode: np.ndarray,
        *,
        static_body_indices: Iterable[int] | np.ndarray | None = None,
        enabled_joint_mask: np.ndarray | None = None,
        drive_mode: np.ndarray | None = None,
        stiffness_drive: np.ndarray | None = None,
        damping_drive: np.ndarray | None = None,
    ) -> ArticulationTopology:
        """Create topology from PhoenX joint init arrays.

        Args:
            body1: Parent body index per joint.
            body2: Child body index per joint.
            joint_mode: PhoenX ADBS joint mode per joint.
            static_body_indices: Body indices to treat as fixed world anchors.
            enabled_joint_mask: Optional mask selecting the joint columns
                owned by the articulation DVI solve. Disabled columns keep
                their raw topology but contribute zero rows.
            drive_mode: Optional per-joint drive mode. Driven revolute,
                prismatic, and universal joints get one axial drive row.
            stiffness_drive: Optional per-joint drive stiffness.
            damping_drive: Optional per-joint drive damping.

        Returns:
            Compact topology with static bodies normalized to ``-1``.
        """
        body1_np = _normalize_static_bodies(np.asarray(body1, dtype=np.int32), static_body_indices)
        body2_np = _normalize_static_bodies(np.asarray(body2, dtype=np.int32), static_body_indices)
        mode_np = np.asarray(joint_mode, dtype=np.int32).copy()

        if body1_np.shape != body2_np.shape or body1_np.shape != mode_np.shape:
            raise ValueError(
                "body1, body2, and joint_mode must have identical shape, "
                f"got {body1_np.shape}, {body2_np.shape}, {mode_np.shape}"
            )

        equality_row_counts = np.asarray([joint_constraint_row_count(mode) for mode in mode_np], dtype=np.int32)
        drive_row_mask = np.zeros_like(equality_row_counts, dtype=bool)
        if drive_mode is not None:
            drive_np = np.asarray(drive_mode, dtype=np.int32)
            if drive_np.shape != equality_row_counts.shape:
                raise ValueError(f"drive_mode must have shape {equality_row_counts.shape}, got {drive_np.shape}")
            stiffness_np = (
                np.zeros_like(equality_row_counts, dtype=np.float32)
                if stiffness_drive is None
                else np.asarray(stiffness_drive, dtype=np.float32)
            )
            damping_np = (
                np.zeros_like(equality_row_counts, dtype=np.float32)
                if damping_drive is None
                else np.asarray(damping_drive, dtype=np.float32)
            )
            if stiffness_np.shape != equality_row_counts.shape:
                raise ValueError(
                    f"stiffness_drive must have shape {equality_row_counts.shape}, got {stiffness_np.shape}"
                )
            if damping_np.shape != equality_row_counts.shape:
                raise ValueError(f"damping_drive must have shape {equality_row_counts.shape}, got {damping_np.shape}")
            drive_rows = np.asarray(
                [
                    joint_drive_row_count(mode, drive, stiffness, damping)
                    for mode, drive, stiffness, damping in zip(mode_np, drive_np, stiffness_np, damping_np, strict=True)
                ],
                dtype=np.int32,
            )
            drive_row_mask = drive_rows.astype(bool)
            row_counts = equality_row_counts + drive_rows
        else:
            row_counts = equality_row_counts.copy()

        if enabled_joint_mask is not None:
            mask = np.asarray(enabled_joint_mask, dtype=bool)
            if mask.shape != row_counts.shape:
                raise ValueError(f"enabled_joint_mask must have shape {row_counts.shape}, got {mask.shape}")
            row_counts = np.where(mask, row_counts, 0).astype(np.int32)
            drive_row_mask = np.logical_and(drive_row_mask, mask)
        active_joint_indices = np.nonzero(row_counts > 0)[0].astype(np.int32)
        active_row_counts = row_counts[active_joint_indices].astype(np.int32)
        active_block_offsets = np.zeros(active_row_counts.size + 1, dtype=np.int32)
        if active_row_counts.size:
            np.cumsum(active_row_counts, out=active_block_offsets[1:])
        total_rows = int(active_block_offsets[-1])
        row_to_active_block = (
            np.repeat(np.arange(active_row_counts.size, dtype=np.int32), active_row_counts)
            if total_rows > 0
            else np.zeros(0, dtype=np.int32)
        )

        return cls(
            body1=body1_np,
            body2=body2_np,
            joint_mode=mode_np,
            row_counts=row_counts,
            drive_row_mask=drive_row_mask.astype(bool),
            active_joint_indices=active_joint_indices,
            active_row_counts=active_row_counts,
            active_block_offsets=active_block_offsets,
            row_to_active_block=row_to_active_block,
            total_rows=total_rows,
        )

    @property
    def joint_count(self) -> int:
        """Number of PhoenX joint columns in the topology."""
        return int(self.joint_mode.size)

    @property
    def active_joint_count(self) -> int:
        """Number of hard equality blocks."""
        return int(self.active_joint_indices.size)

    @property
    def active_body1(self) -> np.ndarray:
        """Parent body per active equality block."""
        return self.body1[self.active_joint_indices]

    @property
    def active_body2(self) -> np.ndarray:
        """Child body per active equality block."""
        return self.body2[self.active_joint_indices]
