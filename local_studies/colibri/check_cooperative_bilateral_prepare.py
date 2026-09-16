"""Byte-exact local preparation checks; launch only in an assigned GPU window."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.cooperative_bilateral_prepare import launch
from newton._src.solvers.phoenx.body import BodyContainer, inertia_sym6
from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks
from newton._src.solvers.phoenx.constraints.bilateral_joint_data import BilateralJointData, Mat66d, Vec6d
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.tests import test_bilateral_preparation as existing

FIELDS = ("response0", "response1", "lower", "diagonal", "valid")


def compare(constraints, bodies, copies, count, block_dim):
    data = constraints.bilateral
    initial = {name: getattr(data, name).numpy().copy() for name in FIELDS}
    results = []
    for cooperative in (False, True):
        for name, value in initial.items():
            getattr(data, name).assign(value)
        if cooperative:
            launch(constraints, bodies, copies, count, "cuda:0", block_dim)
        else:
            wp.launch(prepare_bilateral_joint_blocks, count, [constraints, bodies, copies], device="cuda:0")
        results.append({name: getattr(data, name).numpy() for name in FIELDS})
    for name in FIELDS:
        if results[0][name].tobytes() != results[1][name].tobytes():
            raise AssertionError(f"{name}: cooperative preparation differs for count={count},block={block_dim}")


def snapshot_fixture(path, ragged=False, invalid=False):
    s = np.load(path)
    device = "cuda:0"
    constraints = ConstraintContainer()
    bodies = BodyContainer()
    copies = CopyStateContainer()
    data = BilateralJointData()
    count = 7 if ragged else len(s["joint_row_count"])
    base = int(np.argmax(s["joint_row_count"]))
    if ragged:
        assert s["joint_row_count"][base] == 6
    constraints.data = wp.array(
        np.repeat(s["joint_data"][:, base : base + 1], count, axis=1) if ragged else s["joint_data"],
        dtype=wp.float32,
        device=device,
    )
    bodies.inverse_mass = wp.array(s["inverse_mass"], dtype=wp.float32, device=device)
    bodies.inverse_inertia_world = wp.array(s["inverse_inertia"], dtype=inertia_sym6, device=device)
    copies.count_per_node = wp.array(s["copy_count"], dtype=wp.int32, device=device)
    copies.highest_index_in_use = wp.array(np.array([int(np.max(s["copy_count"]) > 1)], dtype=np.int32), device=device)
    for name, dtype in (
        ("row_count", wp.int32),
        ("row_indices", wp.int32),
        ("structural_index", wp.int32),
        ("row_local", wp.int32),
        ("row_dynamic", wp.bool),
        ("wrench0", wp.spatial_vector),
        ("wrench1", wp.spatial_vector),
        ("dynamic_mass", wp.float32),
    ):
        value = s["joint_" + name]
        if ragged:
            if name == "row_count":
                value = np.arange(7, dtype=np.int32)
            elif name == "row_indices":
                value = np.repeat(value[base : base + 1], count, axis=0)
            elif name == "structural_index":
                value = np.repeat(value[base : base + 1], count)
        if invalid and name in ("wrench0", "wrench1"):
            value = np.zeros_like(value)
        if invalid and name == "row_dynamic":
            value = np.zeros_like(value)
        setattr(data, name, wp.array(value, dtype=dtype, device=device))
    for name, dtype, shape in (
        ("response0", wp.spatial_vector, (count, 6, 6)),
        ("response1", wp.spatial_vector, (count, 6, 6)),
        ("lower", Mat66d, (count, 6, 6)),
        ("diagonal", Vec6d, (count, 6)),
        ("valid", wp.int32, (count,)),
    ):
        scalar = np.int32 if name == "valid" else np.float64 if name in ("lower", "diagonal") else np.float32
        value = np.full(shape, 17, dtype=scalar)
        setattr(data, name, wp.array(value, dtype=dtype, device=device))
    constraints.bilateral = data
    return constraints, bodies, copies, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="/tmp/colibri_current_native_velocity_iterations_snapshot.npz")
    parser.add_argument("--output", default="/tmp/cooperative_bilateral_prepare_checks.json")
    args = parser.parse_args()
    wp.init()
    reports = []
    for block in (32, 64):
        original_launch = wp.launch

        def redirect(kernel, *positional, _block=block, _original=original_launch, **kwargs):
            if kernel is prepare_bilateral_joint_blocks:
                dim = kwargs.get("dim", positional[0] if positional else None)
                inputs = kwargs.get("inputs", positional[1] if len(positional) > 1 else None)
                return launch(inputs[0], inputs[1], inputs[2], dim, kwargs.get("device", "cuda:0"), _block)
            return _original(kernel, *positional, **kwargs)

        wp.launch = redirect
        try:
            existing.TestBilateralPreparation().test_fixed_bounds_match_active_rows_and_mass_scales()
        finally:
            wp.launch = original_launch
        reports.append({"block_dim": block, "existing_cases": 84})
        for ragged, invalid in ((False, False), (True, False), (True, True), (False, True)):
            fixture = snapshot_fixture(args.snapshot, ragged, invalid)
            compare(*fixture, block)
            reports.append(
                {"block_dim": block, "joints": fixture[-1], "ragged": ragged, "invalid": invalid, "byte_exact": True}
            )
    Path(args.output).write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
