"""Read-only device ring around owned contact solves and contact refresh.

Install after native conditioned construction. No arithmetic or history mutation;
the saved trajectory must independently match the corresponding baseline prefix.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

SLOTS = 1024
POINTS = 128
FIELDS = 104


@wp.kernel
def capture(
    counter: wp.array[wp.int32],
    phase: int,
    history: wp.array3d[wp.float32],
    lambdas: wp.array2d[wp.float32],
    derived: wp.array2d[wp.float32],
    impulses: wp.array2d[wp.float32],
    headers: wp.array2d[wp.float32],
    count: wp.array[wp.int32],
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    velocity: wp.array[wp.vec3f],
    angular: wp.array[wp.vec3f],
    bias_velocity: wp.array[wp.spatial_vector],
):
    point, field = wp.tid()
    slot = counter[0] % 1024
    value = wp.float32(0.0)
    if field < 27:
        value = lambdas[field, point]
    elif field < 43:
        value = derived[field - 27, point]
    elif field < 46:
        value = impulses[field - 43, point]
    elif field < 78:
        value = headers[field - 46, point]
    elif point < position.shape[0]:
        if field < 81:
            value = position[point][field - 78]
        elif field < 85:
            value = orientation[point][field - 81]
        elif field < 88:
            value = velocity[point][field - 85]
        elif field < 91:
            value = angular[point][field - 88]
    if field >= 96 and field < 102 and point < bias_velocity.shape[0]:
        value = bias_velocity[point][field - 96]
    if field == 92:
        value = wp.float32(phase)
    elif field == 93:
        value = wp.float32(count[0])
    elif field == 94:
        value = wp.float32(counter[0])
    history[slot, point, field] = value


@wp.kernel
def advance(counter: wp.array[wp.int32]):
    counter[0] += 1


def install(solver, output):
    w = solver.world
    assert solver._direct_tree_contacts
    cc = w._contact_container
    assert cc.lambdas.shape[0] == 27 and w.bodies.position.shape[0] <= POINTS
    assert w._contact_cols.data.shape[0] == 32
    history = wp.zeros((SLOTS, POINTS, FIELDS), dtype=wp.float32, device=w.bodies.position.device)
    counter = wp.zeros(1, dtype=wp.int32, device=w.bodies.position.device)

    def record(phase):
        wp.launch(
            capture,
            dim=(POINTS, FIELDS),
            inputs=[
                counter,
                phase,
                history,
                cc.lambdas,
                cc.derived,
                cc.impulses,
                w._contact_cols.data,
                w._cc_valid_count,
                w.bodies.position,
                w.bodies.orientation,
                w.bodies.velocity,
                w.bodies.angular_velocity,
                solver._direct_equality_system.bias_velocity,
            ],
            device=history.device,
        )
        wp.launch(advance, dim=1, inputs=[counter], device=history.device)

    old = w._solve_maximal_articulated_contacts

    def solve(*, use_bias, refresh_mobility):
        record(10 if use_bias else 20)
        old(use_bias=use_bias, refresh_mobility=refresh_mobility)
        record(11 if use_bias else 21)

    w._solve_maximal_articulated_contacts = solve
    direct = solver._direct_equality_system
    apply = direct.apply_bias_velocity

    def apply_bias(scale):
        record(40 if scale < 0 else 42)
        apply(scale)
        record(41 if scale < 0 else 43)

    direct.apply_bias_velocity = apply_bias

    update = w._ingest_and_warmstart_contacts

    def update_contacts(*args, **kwargs):
        record(30)
        result = update(*args, **kwargs)
        record(31)
        return result

    w._ingest_and_warmstart_contacts = update_contacts

    def finish():
        count = int(counter.numpy()[0])
        ring = history.numpy().copy()
        ordered = ring[np.arange(max(0, count - SLOTS), count) % SLOTS]
        assert np.all(ordered[:, 0, 93] <= POINTS), "Contact capture capacity exceeded"
        path = Path(output).with_suffix(".friction_history.npz")
        np.savez_compressed(path, history=ordered, event_count=np.array([count]))
        reference = Path(
            os.environ.get("COLIBRI_FRICTION_REFERENCE_TRAJECTORY", "/tmp/colibri_native_direct_owned_fixed3600.npz")
        )
        baseline = np.load(reference)
        actual = np.load(Path(output).with_suffix(".npz"))
        checked = []
        for name in ("q_history", "qd_history", "history_times"):
            a, b = actual[name], baseline[name]
            assert a.tobytes() == b[: len(a)].tobytes(), name
            checked.append(name)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "trajectory_prefix_byteexact": checked,
                    "reference_trajectory": str(reference),
                    "owned_module_file": sys.modules[
                        "newton._src.solvers.phoenx.articulations.maximal_contact_gs"
                    ].__file__,
                    "events": count,
                    "saved_events": len(ordered),
                    "phase_ids": {
                        "10": "biased_before",
                        "11": "biased_after",
                        "20": "relax_before",
                        "21": "relax_after",
                        "30": "refresh_before",
                        "31": "refresh_after",
                        "40": "subtract_bias_before",
                        "41": "subtract_bias_after",
                        "42": "restore_bias_before",
                        "43": "restore_bias_after",
                    },
                    "scope": "Latest two frames approximately; copies only. Final broken bit is latest solve state, not an OR over phases.",
                },
                indent=2,
            )
        )

    return finish
