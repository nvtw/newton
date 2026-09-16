# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record the first full-scene preparation differences without graph capture."""

import argparse
import types
from pathlib import Path

import numpy as np

from local_studies.colibri import batch_aware_tail, contact_chunks
from local_studies.colibri.bilateral_pgs import install_fused
from local_studies.colibri.parallel_prepare_split import install
from local_studies.colibri.phoenx_scene import Example
from local_studies.colibri.temporal_schedule import install_temporal_schedule
from newton._src.solvers.phoenx import solver_phoenx, solver_phoenx_kernels
from newton.viewer import ViewerNull

parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
solver_phoenx.NUM_INNER_WHILE_ITERATIONS = 1
solver_phoenx_kernels._make_singleworld_persistent_kernel = batch_aware_tail._make_singleworld_persistent_kernel
solver_phoenx_kernels._make_singleworld_fused_kernel = batch_aware_tail._make_singleworld_fused_kernel
contact_chunks.reserve_capacity()
install_temporal_schedule()
scene_args = argparse.Namespace(
    body_count=36,
    mode="maximal",
    layout="single_world",
    no_graph=True,
    substeps=30,
    iterations=1,
    outer_substeps=2,
    contact_gap=0.001,
    source_contact_offsets=True,
    mesh_cylinders=True,
    mass_splitting=True,
    max_colored_partitions=8,
    mass_splitting_batch_size=2,
    contact_offset_map="/tmp/colibri_physx_exact_contact_offsets_m.json",
    speculative_contact_gap_max=0.005,
)
example = Example(ViewerNull(), scene_args)
install_fused(example.solver, mass_splitting=True)
contact_chunks.install(example.solver, chunk_size=6)
world = example.solver.world
prepare = world._singleworld_kernels()[0]
if args.parallel:
    install(example.solver)
original = world._singleworld_head_plus_tail_sweep
saved = {}
counter = [0]


def snapshot(prefix):
    cc = world._contact_container
    arrays = {
        "columns": world._contact_cols.data,
        "derived": cc.derived,
        "impulses": cc.impulses,
        "anchors": cc.lambdas,
        "inverse_mass": world.bodies.inverse_mass,
        "inverse_inertia_world": world.bodies.inverse_inertia_world,
        "position": world.bodies.position,
        "orientation": world.bodies.orientation,
        "velocity": world.bodies.velocity,
        "angular_velocity": world.bodies.angular_velocity,
        "copy_velocity": world._copy_state.velocity,
        "copy_angular_velocity": world._copy_state.angular_velocity,
        "copy_counts": world._copy_state.count_per_node,
        "active_count": world._num_active_constraints,
        "color_starts": world._partitioner.color_starts,
        "color_ids": world._partitioner.element_ids_by_color,
        "color_count": world._partitioner.num_colors,
        "column_count": world._ingest_scratch.num_contact_columns,
    }
    for name, array in arrays.items():
        saved[prefix + name] = array.numpy()
    saved[prefix + "constraint_bodies"] = world._elements.numpy()["bodies"]
    saved[prefix + "dimensions"] = np.asarray([world.num_bodies, world.num_joints, world._contact_offset])


def sweep(self, head, tail, idt, contact_container=None):
    step = counter[0]
    if head is not prepare:
        return original(head, tail, idt, contact_container)
    if step < 5:
        snapshot(f"{step}_before_")
    result = original(head, tail, idt, contact_container)
    if step < 5:
        snapshot(f"{step}_after_")
    counter[0] += 1
    return result


world._singleworld_head_plus_tail_sweep = types.MethodType(sweep, world)
example.simulate()
np.savez(args.output, **saved)
print(f"Saved {counter[0]} preparation calls to {args.output}")
