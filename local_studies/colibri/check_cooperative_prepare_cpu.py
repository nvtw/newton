"""Byte-exact local preparation checks; launch only in an assigned GPU window."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, inertia_sym6
from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks
from newton._src.solvers.phoenx.constraints.bilateral_joint_data import BilateralJointData, Mat66d, Vec6d
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer


def snapshot_fixture(path, ragged=False, invalid=False):
    s = np.load(path)
    device = "cpu"
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


wp.init()
results = []
for ragged, invalid in ((False, False), (True, False), (True, True)):
    constraints, bodies, copies, count = snapshot_fixture(
        "/tmp/colibri_current_native_velocity_iterations_snapshot.npz", ragged, invalid
    )
    wp.launch(prepare_bilateral_joint_blocks, count, [constraints, bodies, copies], device="cpu")
    d = constraints.bilateral
    values = {name: getattr(d, name).numpy() for name in ("response0", "response1", "lower", "diagonal", "valid")}
    assert all(np.isfinite(value).all() for value in values.values())
    row_count = d.row_count.numpy()
    if invalid:
        assert not values["valid"].any()
    else:
        assert np.array_equal(values["valid"], (row_count > 0).astype(np.int32))
    if ragged:
        assert np.all(values["response0"][0] == 17)
        assert np.all(values["diagonal"][0] == 17)
    max_relative_error = 0.0
    if not invalid:
        indices, local, structural = d.row_indices.numpy(), d.row_local.numpy(), d.structural_index.numpy()
        w0, w1 = d.wrench0.numpy(), d.wrench1.numpy()
        dynamic, mass = d.row_dynamic.numpy(), d.dynamic_mass.numpy()
        for cid, active_count in enumerate(row_count):
            n = int(active_count)
            if not n:
                continue
            matrix = np.zeros((n, n))
            for i in range(n):
                row = indices[cid, i]
                for j in range(i + 1):
                    value = np.dot(
                        w0[structural[cid], local[row]].astype(np.float64),
                        values["response0"][cid, j].astype(np.float64),
                    ) + np.dot(
                        w1[structural[cid], local[row]].astype(np.float64),
                        values["response1"][cid, j].astype(np.float64),
                    )
                    if i == j and dynamic[row]:
                        value += 1.0 / float(mass[row])
                    matrix[i, j] = matrix[j, i] = value
            lower = values["lower"][cid, :n, :n]
            reconstructed = (lower * values["diagonal"][cid, :n]) @ lower.T
            relative = np.max(np.abs(reconstructed - matrix)) / max(np.max(np.abs(matrix)), 1e-300)
            max_relative_error = max(max_relative_error, relative)
            assert relative < 2e-14
    results.append(
        {
            "joints": count,
            "ragged": ragged,
            "invalid": invalid,
            "finite": True,
            "max_relative_factor_error": max_relative_error,
        }
    )
Path("/tmp/cooperative_prepare_cpu_smoke.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
