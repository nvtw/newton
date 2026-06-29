# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Uncoloured Jacobi dispatcher for the simple PhoenX flavor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .contacts import (
    CONTACT_ROW_STRIDE,
    assemble_contact_scalar_rows_kernel,
    clear_contact_lambdas_kernel,
    writeback_contact_lambdas_kernel,
)
from .joints import JOINT_ROW_STRIDE, assemble_joint_scalar_rows_kernel
from .rows import (
    apply_body_velocity_deltas_kernel,
    clear_row_multipliers_kernel,
    scalar_row_container_zeros,
    snapshot_body_velocities_kernel,
    snapshot_row_multipliers_kernel,
    solve_scalar_rows_jacobi_kernel,
)

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


_ROW_BLOCK_DIM = 256


class SimplePhoenXDispatcher:
    """One-thread-per-equation Jacobi dispatcher with atomic delta fan-in."""

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world
        self.block_dim = _ROW_BLOCK_DIM
        self._joint_row_count = int(world.num_joints) * JOINT_ROW_STRIDE
        self._contact_row_offset = self._joint_row_count
        self._contact_row_count = int(world.rigid_contact_max) * CONTACT_ROW_STRIDE
        self._row_count = max(1, self._joint_row_count + self._contact_row_count)
        self.rows = scalar_row_container_zeros(self._row_count, device=world.device)
        self._multiplier_snapshot = wp.zeros(self._row_count, dtype=wp.float32, device=world.device)
        self._velocity_snapshot = wp.zeros(world.num_bodies, dtype=wp.vec3f, device=world.device)
        self._angular_velocity_snapshot = wp.zeros(world.num_bodies, dtype=wp.vec3f, device=world.device)
        self._delta_velocity = wp.zeros(world.num_bodies, dtype=wp.vec3f, device=world.device)
        self._delta_angular_velocity = wp.zeros(world.num_bodies, dtype=wp.vec3f, device=world.device)

    def begin_step(self) -> None:
        if self._contact_row_count == 0:
            return
        w = self._world
        wp.launch(
            clear_row_multipliers_kernel,
            dim=self._contact_row_count,
            inputs=[self.rows, wp.int32(self._contact_row_offset)],
            block_dim=self.block_dim,
            device=w.device,
        )
        if w._contact_views is not None:
            wp.launch(
                clear_contact_lambdas_kernel,
                dim=w.rigid_contact_max,
                inputs=[w._contact_views.rigid_contact_count],
                outputs=[w._contact_container],
                block_dim=self.block_dim,
                device=w.device,
            )

    def solve(self, idt: wp.float32) -> None:
        if self._joint_row_count == 0 and self._contact_row_count == 0:
            return
        w = self._world
        if self._joint_row_count > 0:
            wp.launch(
                assemble_joint_scalar_rows_kernel,
                dim=self._joint_row_count,
                inputs=[w.constraints, w.bodies, wp.int32(w.num_joints), idt],
                outputs=[self.rows],
                block_dim=self.block_dim,
                device=w.device,
            )
        if self._contact_row_count > 0 and w._contact_views is not None:
            wp.launch(
                assemble_contact_scalar_rows_kernel,
                dim=self._contact_row_count,
                inputs=[
                    w._contact_cols,
                    w._contact_views,
                    w._contact_container,
                    w._cid_of_contact_cur,
                    wp.int32(w._contact_offset),
                    w.bodies,
                    wp.int32(self._contact_row_offset),
                    idt,
                ],
                outputs=[self.rows],
                block_dim=self.block_dim,
                device=w.device,
            )

        for _ in range(w.solver_iterations):
            wp.launch(
                snapshot_body_velocities_kernel,
                dim=w.num_bodies,
                inputs=[w.bodies],
                outputs=[
                    self._velocity_snapshot,
                    self._angular_velocity_snapshot,
                    self._delta_velocity,
                    self._delta_angular_velocity,
                ],
                block_dim=self.block_dim,
                device=w.device,
            )
            wp.launch(
                snapshot_row_multipliers_kernel,
                dim=self._row_count,
                inputs=[self.rows],
                outputs=[self._multiplier_snapshot],
                block_dim=self.block_dim,
                device=w.device,
            )
            wp.launch(
                solve_scalar_rows_jacobi_kernel,
                dim=self._row_count,
                inputs=[
                    self.rows,
                    w.bodies,
                    self._velocity_snapshot,
                    self._angular_velocity_snapshot,
                    self._multiplier_snapshot,
                    wp.float32(w.sor_boost),
                ],
                outputs=[self._delta_velocity, self._delta_angular_velocity],
                block_dim=self.block_dim,
                device=w.device,
            )
            wp.launch(
                apply_body_velocity_deltas_kernel,
                dim=w.num_bodies,
                inputs=[w.bodies, self._delta_velocity, self._delta_angular_velocity],
                block_dim=self.block_dim,
                device=w.device,
            )
        if self._contact_row_count > 0 and w._contact_views is not None:
            wp.launch(
                writeback_contact_lambdas_kernel,
                dim=w.rigid_contact_max,
                inputs=[
                    w._contact_views.rigid_contact_count,
                    wp.int32(self._contact_row_offset),
                    self.rows,
                ],
                outputs=[w._contact_container],
                block_dim=self.block_dim,
                device=w.device,
            )

    def relax(self, idt: wp.float32) -> None:
        # Small Jacobi substeps replace the separate TGS velocity-relax pass.
        pass


__all__ = ["SimplePhoenXDispatcher"]
