# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Optional diagnostics for the PhoenX simulation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_cloth_bending import CLOTH_BENDING_TIME_US_OFFSET
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import CLOTH_TRIANGLE_TIME_US_OFFSET
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_TIME_US_OFFSET,
    contact_pair_wrench_kernel,
    contact_per_contact_error_kernel,
    contact_per_contact_wrench_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_CONSTRAINT_TIME_US_OFFSET
from newton._src.solvers.phoenx.constraints.constraint_soft_hexahedron import SOFT_HEX_TIME_US_OFFSET
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import SOFT_TET_TIME_US_OFFSET
from newton._src.solvers.phoenx.simulation_kernels import (
    _constraint_gather_errors_kernel,
    _constraint_gather_wrenches_kernel,
    _reduce_constraint_time_us_kernel,
    _reduce_contact_time_us_kernel,
    _zero_constraint_time_us_kernel,
    _zero_contact_time_us_kernel,
)

__all__ = ["StepReport"]


@dataclass
class StepReport:
    """Diagnostic snapshot. Triggers D2H copies; not graph-capture safe."""

    num_colors: int
    """Graph colour count from the last PGS. Multi-world: max across worlds."""

    color_sizes: list[int]
    """Element count per colour. Multi-world: sum across worlds per index."""

    per_world_num_colors: list[int] | None
    """Per-world colour counts; None for single-world."""

    per_world_color_sizes: list[list[int]] | None
    """Per-world per-colour element counts; None for single-world."""

    num_contact_columns: int
    """Active contact columns from the last step."""

    num_joints: int
    """Joint constraint columns (static for the world's lifetime)."""

    num_active_constraints: int
    """Active cids, including joints, deformables, and contacts."""

    max_body_degree: int
    """Max constraints incident to any body or particle node."""

    time_us_total_joints: float | None = None
    """Total wall-clock microseconds spent in joint dispatches."""

    time_us_total_cloth_triangles: float | None = None
    """Total wall-clock microseconds spent in cloth-triangle dispatches."""

    time_us_total_cloth_bending: float | None = None
    """Total wall-clock microseconds spent in cloth-bending dispatches."""

    time_us_total_soft_tetrahedra: float | None = None
    """Total wall-clock microseconds spent in soft-tet dispatches."""

    time_us_total_contacts: float | None = None
    """Total wall-clock microseconds spent in contact dispatches."""

    time_us_total_soft_hexahedra: float | None = None
    """Total wall-clock microseconds spent in soft-hex dispatches."""

    overflow_size: int = 0
    """Constraint count in the mass-splitting overflow bucket."""

    color_group_sizes: list[int] | None = None
    """Constraint counts per sequential color group, when enabled."""


def gather_column_timers(world: Any, num_contact_columns: int) -> dict[str, float]:
    """Reduce per-column timers and copy the six totals to the host."""
    if world._column_timer_totals is None:
        world._column_timer_totals = wp.zeros(6, dtype=wp.float32, device=world.device)
    else:
        world._column_timer_totals.zero_()
    if world._contact_offset > 0:
        soft_tet_time_off = (
            int(SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET) if world._soft_tet_uses_neohookean else int(SOFT_TET_TIME_US_OFFSET)
        )
        wp.launch(
            _reduce_constraint_time_us_kernel,
            dim=world._contact_offset,
            inputs=[
                world.constraints,
                wp.int32(JOINT_CONSTRAINT_TIME_US_OFFSET),
                wp.int32(CLOTH_TRIANGLE_TIME_US_OFFSET),
                wp.int32(CLOTH_BENDING_TIME_US_OFFSET),
                wp.int32(soft_tet_time_off),
                wp.int32(SOFT_HEX_TIME_US_OFFSET),
                wp.int32(world.num_joints),
                wp.int32(world.num_cloth_triangles),
                wp.int32(world.num_cloth_bending),
                wp.int32(world.num_soft_tetrahedra),
                wp.int32(world.num_soft_hexahedra),
                world._column_timer_totals,
            ],
            device=world.device,
        )
    if num_contact_columns > 0 and world.max_contact_columns > 0:
        wp.launch(
            _reduce_contact_time_us_kernel,
            dim=num_contact_columns,
            inputs=[
                world._contact_cols,
                wp.int32(num_contact_columns),
                wp.int32(CONTACT_TIME_US_OFFSET),
                world._column_timer_totals,
            ],
            device=world.device,
        )
    totals = world._column_timer_totals.numpy()
    return {
        "time_us_total_joints": float(totals[0]),
        "time_us_total_cloth_triangles": float(totals[1]),
        "time_us_total_cloth_bending": float(totals[2]),
        "time_us_total_soft_tetrahedra": float(totals[3]),
        "time_us_total_contacts": float(totals[4]),
        "time_us_total_soft_hexahedra": float(totals[5]),
    }


def zero_column_timers(world: Any) -> None:
    """Zero every per-column timer at the start of a measured step."""
    if world._contact_offset > 0:
        soft_tet_time_off = (
            int(SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET) if world._soft_tet_uses_neohookean else int(SOFT_TET_TIME_US_OFFSET)
        )
        wp.launch(
            _zero_constraint_time_us_kernel,
            dim=world._contact_offset,
            inputs=[
                world.constraints,
                world._num_active_constraints,
                wp.int32(JOINT_CONSTRAINT_TIME_US_OFFSET),
                wp.int32(CLOTH_TRIANGLE_TIME_US_OFFSET),
                wp.int32(CLOTH_BENDING_TIME_US_OFFSET),
                wp.int32(soft_tet_time_off),
                wp.int32(SOFT_HEX_TIME_US_OFFSET),
                wp.int32(world.num_joints),
                wp.int32(world.num_cloth_triangles),
                wp.int32(world.num_cloth_bending),
                wp.int32(world.num_soft_tetrahedra),
                wp.int32(world.num_soft_hexahedra),
            ],
            device=world.device,
        )
    if world.max_contact_columns > 0:
        wp.launch(
            _zero_contact_time_us_kernel,
            dim=world.max_contact_columns,
            inputs=[world._contact_cols, wp.int32(world.max_contact_columns), wp.int32(CONTACT_TIME_US_OFFSET)],
            device=world.device,
        )


def gather_constraint_wrenches(world: Any, out: wp.array) -> None:
    """Gather per-column world-frame wrenches on body 2."""
    if world._constraint_capacity == 0:
        return
    out.zero_()
    if world.substep_dt <= 0.0:
        return
    contact_views = world._active_contact_views()
    idt = wp.float32(1.0 / world.substep_dt)
    wp.launch(
        _constraint_gather_wrenches_kernel,
        dim=world._constraint_capacity,
        inputs=[
            world.constraints,
            world._contact_cols,
            world.bodies,
            wp.int32(world._constraint_capacity),
            wp.int32(world.num_joints),
            idt,
            world._contact_container,
            contact_views,
        ],
        outputs=[out],
        device=world.device,
    )
    direct = getattr(world, "_direct_equality_system", None)
    if direct is not None:
        direct.gather_constraint_wrenches(out, idt)


def gather_constraint_errors(world: Any, out: wp.array) -> None:
    """Gather per-column position-level residuals."""
    if world._constraint_capacity == 0:
        return
    out.zero_()
    wp.launch(
        _constraint_gather_errors_kernel,
        dim=world._constraint_capacity,
        inputs=[
            world.constraints,
            world._contact_cols,
            world.bodies,
            wp.int32(world._constraint_capacity),
            wp.int32(world.num_joints),
        ],
        outputs=[out],
        device=world.device,
    )


def num_colors_used(world: Any) -> int:
    """Return the number of graph colors from the last PGS."""
    if world._color_group_data is not None:
        return int(world._color_group_data["num_colors"].numpy()[0])
    if world.step_layout == "single_world":
        return int(world._partitioner.num_colors.numpy()[0])
    return int(world._world_num_colors.numpy().max(initial=0))


def step_report(world: Any) -> StepReport:
    """Build a diagnostic snapshot of the last simulation step."""
    num_contact_columns = (
        int(world._ingest_scratch.num_contact_columns.numpy()[0])
        if world._contact_views is not None and world._ingest_scratch is not None
        else 0
    )
    num_active = (
        int(world._num_active_constraints.numpy()[0])
        if world._num_active_constraints is not None
        else world._contact_offset + num_contact_columns
    )
    timer_kwargs = gather_column_timers(world, num_contact_columns) if world.enable_column_timers else {}

    num_nodes = world.num_bodies + world.num_particles
    if num_active > 0 and num_nodes > 0:
        ends = world._partitioner._adjacency_section_end_indices.numpy()
        n_nodes = min(int(num_nodes), int(ends.shape[0]))
        if n_nodes > 0:
            degrees = ends[:n_nodes].astype(np.int64, copy=False)
            degrees[1:] = degrees[1:] - degrees[:-1]
            max_body_degree = int(degrees.max(initial=0))
        else:
            max_body_degree = 0
    else:
        max_body_degree = 0

    if world.step_layout == "single_world":
        nc = num_colors_used(world)
        if nc > 0:
            starts = (
                world._color_group_data["starts"]
                if world._color_group_data is not None
                else world._partitioner.color_starts
            ).numpy()
            color_sizes = [int(starts[c + 1] - starts[c]) for c in range(nc)]
        else:
            color_sizes = []
        group_sizes = None
        overflow_size = 0
        if world._color_group_data is not None:
            width = world.mass_splitting_color_group_size
            group_sizes = [sum(color_sizes[i : i + width]) for i in range(0, nc, width)]
        elif world.mass_splitting_enabled and world.max_colored_partitions is not None:
            if nc > world.max_colored_partitions:
                overflow_size = color_sizes[world.max_colored_partitions]
        return StepReport(
            num_colors=nc,
            overflow_size=overflow_size,
            color_group_sizes=group_sizes,
            color_sizes=color_sizes,
            per_world_num_colors=None,
            per_world_color_sizes=None,
            num_contact_columns=num_contact_columns,
            num_joints=world.num_joints,
            num_active_constraints=num_active,
            max_body_degree=max_body_degree,
            **timer_kwargs,
        )

    nc_per_world = world._world_num_colors.numpy().astype(np.int32, copy=False)
    starts_2d = world._world_color_starts.numpy().astype(np.int32, copy=False)
    per_world_num_colors = [int(n) for n in nc_per_world]
    per_world_color_sizes = []
    max_nc = 0
    for world_index, count in enumerate(per_world_num_colors):
        row = starts_2d[world_index]
        sizes = [int(row[color + 1] - row[color]) for color in range(count)]
        per_world_color_sizes.append(sizes)
        max_nc = max(max_nc, count)
    aggregate = [0] * max_nc
    for sizes in per_world_color_sizes:
        for color, size in enumerate(sizes):
            aggregate[color] += size
    return StepReport(
        num_colors=max_nc,
        color_sizes=aggregate,
        per_world_num_colors=per_world_num_colors,
        per_world_color_sizes=per_world_color_sizes,
        num_contact_columns=num_contact_columns,
        num_joints=world.num_joints,
        num_active_constraints=num_active,
        max_body_degree=max_body_degree,
        **timer_kwargs,
    )


def gather_contact_wrenches(world: Any, out: wp.array) -> None:
    """Gather per-contact wrenches from the last substep."""
    if world.max_contact_columns == 0:
        out.zero_()
        return
    out.zero_()
    if world.substep_dt <= 0.0 or world._contact_views is None:
        return
    wp.launch(
        contact_per_contact_wrench_kernel,
        dim=world.max_contact_columns,
        inputs=[
            world._contact_cols,
            world.bodies,
            world._contact_container,
            world._contact_views,
            wp.int32(world.max_contact_columns),
            wp.float32(1.0 / world.substep_dt),
        ],
        outputs=[out],
        device=world.device,
    )


def gather_contact_pair_wrenches(
    world: Any,
    wrenches: wp.array,
    body1: wp.array,
    body2: wp.array,
    contact_count: wp.array,
) -> None:
    """Gather one wrench summary per contact column."""
    if world.max_contact_columns == 0:
        return
    if world.substep_dt <= 0.0 or world._contact_views is None:
        wrenches.zero_()
        body1.fill_(-1)
        body2.fill_(-1)
        contact_count.zero_()
        return
    wp.launch(
        contact_pair_wrench_kernel,
        dim=world.max_contact_columns,
        inputs=[
            world._contact_cols,
            world.bodies,
            world._contact_container,
            world._contact_views,
            wp.int32(world.max_contact_columns),
            wp.float32(1.0 / world.substep_dt),
        ],
        outputs=[wrenches, body1, body2, contact_count],
        device=world.device,
    )


def gather_contact_errors(world: Any, out: wp.array) -> None:
    """Gather per-contact position-level residuals."""
    if world.max_contact_columns == 0:
        out.zero_()
        return
    out.zero_()
    if world._contact_views is None:
        return
    wp.launch(
        contact_per_contact_error_kernel,
        dim=world.max_contact_columns,
        inputs=[
            world._contact_cols,
            world.bodies,
            world._contact_container,
            world._contact_views,
            wp.int32(world.max_contact_columns),
        ],
        outputs=[out],
        device=world.device,
    )
