# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen materialized RHS scheduling probe, never a physical trajectory."""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.bilateral_joint import _dot_double
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import _ms_load_body_pair
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton.examples.phoenx.example_phoenx_colibri import Example


@wp.struct
class MaterializedRHS:
    count: wp.array[wp.int32]
    j0: wp.array2d[wp.spatial_vector]
    j1: wp.array2d[wp.spatial_vector]
    t0: wp.array[wp.spatial_vector]
    t1: wp.array[wp.spatial_vector]
    dynamic: wp.array2d[wp.bool]
    accumulated: wp.array2d[wp.float32]
    mass: wp.array2d[wp.float32]
    reference: wp.array2d[wp.float32]
    bias: wp.array2d[wp.float32]


@wp.kernel
def materialize_rhs(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copies: CopyStateContainer,
    num_bodies: wp.int32,
    slabs: wp.array[wp.int32],
    out: MaterializedRHS,
):
    cid = wp.tid()
    data = constraints.bilateral
    count = data.row_count[cid]
    if data.valid[cid] == 0:
        count = 0
    out.count[cid] = count
    if count > 0:
        b0 = constraint_get_body1(constraints, cid)
        b1 = constraint_get_body2(constraints, cid)
        v0, v1, w0, w1, im0, im1, ii0, ii1, slot0, slot1 = _ms_load_body_pair(
            bodies, particles, copies, b0, b1, slabs[cid], num_bodies
        )
        out.t0[cid] = wp.spatial_vector(v0, w0)
        out.t1[cid] = wp.spatial_vector(v1, w1)
        structural = data.structural_index[cid]
        for i in range(count):
            row = data.row_indices[cid, i]
            local = data.row_local[row]
            out.j0[cid, i] = data.wrench0[structural, local]
            out.j1[cid, i] = data.wrench1[structural, local]
            out.dynamic[cid, i] = data.row_dynamic[row]
            out.accumulated[cid, i] = data.accumulated[row]
            out.mass[cid, i] = data.dynamic_mass[row]
            out.reference[cid, i] = data.reference[row]
            out.bias[cid, i] = data.bias[structural, local]


@wp.func
def row_rhs(rows: MaterializedRHS, cid: wp.int32, i: wp.int32):
    residual = _dot_double(rows.j0[cid, i], rows.t0[cid])
    residual += _dot_double(rows.j1[cid, i], rows.t1[cid])
    if rows.dynamic[cid, i]:
        residual += wp.float64(rows.accumulated[cid, i]) / wp.float64(rows.mass[cid, i])
        residual -= wp.float64(rows.reference[cid, i])
    else:
        residual += wp.float64(rows.bias[cid, i])
    return -residual


@wp.kernel
def scalar_materialized_rhs(rows: MaterializedRHS, output: wp.array2d[wp.float64]):
    cid = wp.tid()
    for i in range(rows.count[cid]):
        output[cid, i] = row_rhs(rows, cid, i)


@wp.kernel
def eight_lane_materialized_rhs(rows: MaterializedRHS, output: wp.array2d[wp.float64]):
    cid, i = wp.tid()
    if i < rows.count[cid]:
        output[cid, i] = row_rhs(rows, cid, i)


def main():
    sys.argv = [sys.argv[0], "--viewer", "null", "--num-frames", "330"]
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    for _ in range(330):
        example.step()
    world = example.solver.world
    n = world.num_joints
    rows = MaterializedRHS()
    rows.count = wp.zeros(n, dtype=wp.int32, device=world.device)
    for name in ("j0", "j1"):
        setattr(rows, name, wp.zeros((n, 6), dtype=wp.spatial_vector, device=world.device))
    for name in ("t0", "t1"):
        setattr(rows, name, wp.zeros(n, dtype=wp.spatial_vector, device=world.device))
    rows.dynamic = wp.zeros((n, 6), dtype=wp.bool, device=world.device)
    for name in ("accumulated", "mass", "reference", "bias"):
        setattr(rows, name, wp.zeros((n, 6), dtype=wp.float32, device=world.device))
    groups = world._color_group_data
    ids, starts = groups["ids"].numpy(), groups["starts"].numpy()
    count = int(groups["num_colors"].numpy()[0])
    slabs = np.zeros(n, np.int32)
    for color in range(count):
        for cid in ids[starts[color] : starts[color + 1]]:
            if cid < n:
                slabs[cid] = color // world.mass_splitting_color_group_size
    wp.launch(
        materialize_rhs,
        n,
        [
            world.constraints,
            world.bodies,
            world._particles_or_sentinel(),
            world._copy_state,
            world.num_bodies,
            wp.array(slabs, device=world.device),
            rows,
        ],
        device=world.device,
    )
    reference = None
    report = {
        "scope": "Materialized RHS only; excludes gather, colors, solves and scatter",
        "joint_count": n,
        "active_rows": int(rows.count.numpy().sum()),
        "variants": {},
    }
    for name, kernel, dim in (
        ("scalar", scalar_materialized_rhs, n),
        ("eight_lanes", eight_lane_materialized_rhs, (n, 8)),
    ):
        output = wp.zeros((n, 6), dtype=wp.float64, device=world.device)
        wp.launch(kernel, dim, [rows, output], block_dim=32, device=world.device)
        actual = output.numpy()
        if reference is None:
            reference = actual
        else:
            assert actual.tobytes() == reference.tobytes()
        start, end = wp.Event(world.device, enable_timing=True), wp.Event(world.device, enable_timing=True)
        with wp.ScopedCapture(device=world.device) as capture:
            wp.record_event(start, external=True)
            for _ in range(100):
                wp.launch(kernel, dim, [rows, output], block_dim=32, device=world.device)
            wp.record_event(end, external=True)
        samples = []
        for _ in range(30):
            wp.capture_launch(capture.graph)
            samples.append(wp.get_event_elapsed_time(start, end) / 100)
        assert output.numpy().tobytes() == reference.tobytes()
        report["variants"][name] = {
            "mean_us": float(np.mean(samples[5:]) * 1000),
            "median_us": float(np.median(samples[5:]) * 1000),
            "bit_exact": True,
        }
    Path("/tmp/colibri_cooperative_rhs.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
