# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Execute the owned maximal kernel on a one-root contact; never use a helper-only gate."""
# ruff: noqa: PLC0415 -- staged callbacks must load before any Newton import.

import argparse
import json
import sys
from pathlib import Path


def main():
    """Check actual total-normal capacity and broken-state writes on CPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--case", choices=("all", "capacity", "broken"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    device = args.device
    if args.staged:
        from . import friction_break_state
        from .native_conditioned_two_body import STAGE, OwnedFinder

        sys.meta_path.insert(0, OwnedFinder())
        previous = friction_break_state.sources

        def sources():
            original, modified = previous()
            for name in modified:
                modified[name] = (STAGE / "newton/_src/solvers/phoenx/constraints" / (name + ".py")).read_text()
            return original, modified

        friction_break_state.sources = sources
        friction_break_state.install()

    import numpy as np
    import warp as wp

    from newton._src.solvers.phoenx import solver_phoenx
    from newton._src.solvers.phoenx.articulations import maximal_contact_gs as owned
    from newton._src.solvers.phoenx.articulations.maximal_contact_response import MaximalContactResponseData
    from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData
    from newton._src.solvers.phoenx.body import body_container_zeros
    from newton._src.solvers.phoenx.constraints.constraint_contact import contact_column_container_zeros
    from newton._src.solvers.phoenx.constraints.contact_container import contact_container_zeros

    wp.config.enable_backward = False
    # One root and exactly one lane: no other thread exists to synchronize.
    # Arithmetic/kernel identity stay unchanged; this is not a CUDA barrier test.
    assert owned._sync_tree.native_snippet == "__syncthreads();"
    if device == "cpu":
        owned._sync_tree.native_snippet = "/* CPU single-lane fixture: barrier is vacuous. */"
    assert solver_phoenx.iterate_maximal_contact_runs_kernel is owned.iterate_maximal_contact_runs_kernel
    tree = MaximalTreeProjectorData()
    tree.body_count = wp.array([1], dtype=wp.int32, device=device)
    tree.max_depth = wp.array([0], dtype=wp.int32, device=device)
    tree.body_slot = wp.array([[1]], dtype=wp.int32, device=device)
    tree.depth = wp.zeros((1, 1), dtype=wp.int32, device=device)
    tree.child_start = wp.zeros((1, 2), dtype=wp.int32, device=device)
    tree.dynamic_row = wp.array([[-1]], dtype=wp.int32, device=device)
    response = MaximalContactResponseData()
    response.body_articulation = wp.array([-1, 0], dtype=wp.int32, device=device)
    response.body_lane = wp.array([-1, 0], dtype=wp.int32, device=device)
    response.mobility = wp.array(np.eye(6, dtype=np.float32)[None, None], dtype=wp.spatial_matrixf, device=device)
    for field in ("impulse", "rhs", "parent_rhs", "velocity"):
        setattr(response, field, wp.zeros((1, 1), dtype=wp.spatial_vectorf, device=device))
    response.joint_velocity = wp.zeros((1, 1), device=device)
    response.contact_active = wp.zeros(1, dtype=wp.int32, device=device)
    bodies = body_container_zeros(2, device=device)
    columns = contact_column_container_zeros(1, device=device)
    data = columns.data.numpy()
    data.view(np.int32)[1:3, 0] = [1, 0]
    data[3:5, 0] = [1, 1]
    data.view(np.int32)[5:7, 0] = [0, 1]
    columns.data.assign(data)
    contacts = contact_container_zeros(1, device=device)
    contacts.lambdas = wp.zeros((13, 1), dtype=wp.float32, device=device)
    mobility = wp.array(np.array([[1], [1], [1], [0], [0], [0]], dtype=np.float32), device=device)
    schedule = wp.array([0], dtype=wp.int32, device=device)
    ends = wp.array([1], dtype=wp.int32, device=device)
    dynamic = wp.zeros(1, dtype=wp.float32, device=device)
    results = []
    for label, old_t, tangential_speed, old_break, expected_t, expected_break in (
        ("interior_retains_actual_normal_capacity", 0.75, 0.0, 1.0, 0.75, 0.0),
        ("actual_sliding_sets_broken", 0.0, 2.0, 0.0, 1.0, 1.0),
    ):
        if args.case == "capacity" and label != "interior_retains_actual_normal_capacity":
            continue
        if args.case == "broken" and label != "actual_sliding_sets_broken":
            continue
        bias = -0.5 if label == "interior_retains_actual_normal_capacity" else 0.0
        # Normal velocity exactly balances the penetrating soft-row bias.
        velocity = np.zeros((2, 3), dtype=np.float32)
        velocity[1] = [tangential_speed, 0, -bias - 0.05829954519867897 / 0.9417003989219666]
        bodies.velocity.assign(velocity)
        bodies.angular_velocity.zero_()
        state = np.zeros((13, 1), dtype=np.float32)
        state[:6, 0] = [0, 0, -1, 1, 0, 0]
        state[12, 0] = old_break
        contacts.lambdas.assign(state)
        contacts.impulses.assign(np.array([[1], [old_t], [0]], dtype=np.float32))
        derived = np.zeros((16, 1), dtype=np.float32)
        derived[0:4, 0] = [1, 1, 1, bias]
        contacts.derived.assign(derived)
        wp.launch(
            owned.iterate_maximal_contact_runs_kernel,
            dim=1 if device == "cpu" else 64,
            block_dim=64,
            inputs=[tree, response, bodies, dynamic, columns, contacts, 3600.0, 1.0, schedule, ends, mobility, True],
            device=device,
        )
        impulse = contacts.impulses.numpy()[:, 0]
        broken = float(contacts.lambdas.numpy()[12, 0])
        actual_velocity = bodies.velocity.numpy()[1]
        result = {"case": label, "impulse": impulse.tolist(), "broken": broken, "velocity": actual_velocity.tolist()}
        results.append(result)
        print(json.dumps(result), flush=True)
        np.testing.assert_allclose(impulse, [1, expected_t, 0], rtol=0, atol=2e-6)
        assert broken == expected_break, result
        np.testing.assert_allclose(
            actual_velocity, velocity[1] + [-float(impulse[1] - old_t), 0, float(impulse[0] - 1)], rtol=0, atol=2e-6
        )
    report = {
        "staged": args.staged,
        "device": device,
        "cpu_barrier_adapter": device == "cpu",
        "actual_module": owned.__file__,
        "results": results,
        "scope": (
            "Actual owned kernel arithmetic/scatter; test-only CPU single-lane barrier no-op."
            if device == "cpu"
            else "Unmodified actual owned CUDA kernel; one root/one contact and full64-lane block."
        ),
    }
    Path(
        "/tmp/colibri_owned_actual_kernel_"
        + ("staged" if args.staged else "canonical")
        + "_"
        + device.replace(":", "_")
        + ".json"
    ).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
