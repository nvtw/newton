# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Preallocated device-copy snapshots of the final colored substep phases."""

import json
import os
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def mark_phase(counts: wp.array[wp.int32], index: int):
    counts[index] += 1


def install(solver, output):
    """Capture without changing the graph; validate entire saved trajectory."""
    w = solver.world
    assert solver.joint_solver == "block_pgs" and not w.mass_splitting_enabled
    assert not w._colored_contact_headers and not w._colored_contact_rows, (
        "This trace requires canonical ordinary-row storage"
    )
    cc = w._contact_container
    arrays = {
        "position": w.bodies.position,
        "orientation": w.bodies.orientation,
        "velocity": w.bodies.velocity,
        "angular_velocity": w.bodies.angular_velocity,
        "inverse_mass": w.bodies.inverse_mass,
        "inverse_inertia": w.bodies.inverse_inertia_world,
        "headers": w._contact_cols.data,
        "lambdas": cc.lambdas,
        "derived": cc.derived,
        "impulses": cc.impulses,
        "contact_count": w._cc_valid_count,
    }
    arrays["joint_data"] = w.constraints.data
    arrays["column_count"] = w._ingest_scratch.num_contact_columns
    for name in ("section_end", "velocity", "angular_velocity"):
        arrays["copy_" + name] = getattr(w._copy_state, name)
    for name in ("element_ids_by_color", "color_starts", "num_colors", "sweep_direction"):
        arrays["color_" + name] = getattr(w._partitioner, name)
    bilateral = w.constraints.bilateral
    for name in (
        "row_count",
        "row_indices",
        "structural_index",
        "row_local",
        "row_dynamic",
        "wrench0",
        "wrench1",
        "bias",
        "reference",
        "dynamic_mass",
        "accumulated",
    ):
        arrays["joint_" + name] = getattr(bilateral, name)
    phases = ("biased", "relax", "integrate")
    counts = wp.zeros(6, dtype=wp.int32, device=w.bodies.position.device)
    # Allocate everything before capture: wp.zeros inside graph construction
    # would replay resets and corrupt prior snapshots.
    slots = {
        phase + "_" + side: {key: wp.empty_like(value) for key, value in arrays.items()}
        for phase in phases
        for side in ("before", "after")
    }

    labels = list(slots)

    def record(label):
        wp.launch(mark_phase, 1, inputs=[counts, labels.index(label)], device=counts.device)
        for key, target in slots[label].items():
            wp.copy(target, arrays[key])

    def wrap(old, phase):
        def call(*args, **kwargs):
            record(phase + "_before")
            result = old(*args, **kwargs)
            record(phase + "_after")
            return result

        return call

    w._integrate_positions = wrap(w._integrate_positions, "integrate")
    dispatcher = w._dispatcher
    dispatcher_type = type(dispatcher)

    def wrap_dispatch(old, phase):
        def call(instance, *args, **kwargs):
            if instance is not dispatcher:
                return old(instance, *args, **kwargs)
            record(phase + "_before")
            result = old(instance, *args, **kwargs)
            record(phase + "_after")
            return result

        return call

    for name, phase in (("solve", "biased"), ("relax", "relax")):
        setattr(dispatcher_type, name, wrap_dispatch(getattr(dispatcher_type, name), phase))

    def finish():
        result = {
            phase + "." + name: array.numpy().copy()
            for phase, values in slots.items()
            for name, array in values.items()
        }
        result["phase_counts"] = counts.numpy()
        assert np.all(result["phase_counts"] > 0), "Unwritten phase buffers"
        result["dt"] = np.array([1.0 / 3600])
        result["num_joints"] = np.array([w.num_joints])
        np.savez_compressed(Path(output).with_suffix(".phases.npz"), **result)
        reference = Path(os.environ["COLIBRI_COLORED_REFERENCE"])
        actual = np.load(Path(output).with_suffix(".npz"))
        baseline = np.load(reference)
        checked = []
        for name in ("q_history", "qd_history", "history_times"):
            assert actual[name].tobytes() == baseline[name].tobytes(), name
            checked.append(name)
        Path(output).with_suffix(".phases.json").write_text(
            json.dumps(
                {
                    "reference": str(reference),
                    "trajectory_byteexact": checked,
                    "colored_contact_headers": w._colored_contact_headers,
                    "colored_contact_rows": w._colored_contact_rows,
                    "phase_counts": result["phase_counts"].tolist(),
                    "scope": "Final invocation of each phase; no allocations inside capture, no arithmetic mutation",
                },
                indent=2,
            )
        )

    return finish
