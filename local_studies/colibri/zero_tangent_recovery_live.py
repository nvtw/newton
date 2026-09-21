"""Diagnostic removal of tangential positional recovery on corrected owned contacts.

Physical friction coefficients, normal recovery, joint recovery and cone projection
remain native. This is an experiment, not a proposed global recovery policy.
Use COLIBRI_STAGE_RUNNER under native_conditioned_two_body; --cpu-test checks
actual contact-container storage independently without installing a live overlay.
"""

import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_set_bias_t1,
    cc_set_bias_t2,
    contact_container_zeros,
)
from newton.solvers import SolverPhoenX


@wp.kernel(enable_backward=False)
def toggle_recovery(
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    scheduled: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    saved: wp.array[wp.vec2f],
    restore: wp.bool,
):
    """Save/zero or restore only the tangent targets of uniquely owned rows."""
    row = wp.tid()
    if ends.shape[0] > 0 and row < ends[ends.shape[0] - 1]:
        column = scheduled[row]
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        for offset in range(count):
            point = first + offset
            if restore:
                cc_set_bias_t1(contacts, point, saved[point][0])
                cc_set_bias_t2(contacts, point, saved[point][1])
            else:
                saved[point] = wp.vec2f(cc_get_bias_t1(contacts, point), cc_get_bias_t2(contacts, point))
                cc_set_bias_t1(contacts, point, wp.float32(0.0))
                cc_set_bias_t2(contacts, point, wp.float32(0.0))


@wp.kernel
def setup_test(columns: ContactColumnContainer):
    """Leave a prefix and tail unowned and an invalid schedule tail inactive."""
    contact_set_contact_first(columns, 0, 1)
    contact_set_contact_count(columns, 0, 2)


def check_storage():
    """Check untouched arrays and bitwise restoration on real CPU containers."""
    contacts = contact_container_zeros(4, device="cpu")
    columns = contact_column_container_zeros(2, device="cpu")
    wp.launch(setup_test, dim=1, inputs=[columns], device="cpu")
    for name in ("lambdas", "derived", "impulses"):
        array = getattr(contacts, name)
        array.assign(np.arange(array.size, dtype=np.float32).reshape(array.shape) * np.float32(0.125))
    saved = wp.zeros(4, dtype=wp.vec2f, device="cpu")
    scheduled = wp.array([0, 999999], dtype=wp.int32, device="cpu")
    ends = wp.array([1], dtype=wp.int32, device="cpu")
    inputs = [columns, contacts, scheduled, ends, saved]
    for iteration in range(2):
        values = contacts.derived.numpy()
        values[4, 1] = np.float32(-0.0)
        values[5, 2] = np.float32(-3e-6 * (iteration + 1))
        contacts.derived.assign(values)
        before = {name: getattr(contacts, name).numpy().copy() for name in ("lambdas", "derived", "impulses")}
        column_before = columns.data.numpy().tobytes()
        wp.launch(toggle_recovery, dim=2, inputs=[*inputs, False], device="cpu")
        after = contacts.derived.numpy()
        np.testing.assert_array_equal(after[4:6, 1:3], 0)
        mask = np.ones(after.shape, bool)
        mask[4:6, 1:3] = False
        assert before["derived"][mask].tobytes() == after[mask].tobytes()
        for name in ("lambdas", "impulses"):
            assert before[name].tobytes() == getattr(contacts, name).numpy().tobytes()
        assert columns.data.numpy().tobytes() == column_before
        wp.launch(toggle_recovery, dim=2, inputs=[*inputs, True], device="cpu")
        assert before["derived"].tobytes() == contacts.derived.numpy().tobytes()
    print("PASS: only owned tangent biases zero; exact signed-zero restoration; all other rows/arrays unchanged")


def install():
    """Preallocate save buffers and wrap only biased maximal contact solves."""
    import newton._src.solvers.phoenx.solver_phoenx as solver_module

    import newton._src.solvers.phoenx.articulations.maximal_contact_gs as owned
    from local_studies.colibri.check_owned_friction_binding import check

    gate = check(owned)
    assert solver_module.iterate_maximal_contact_runs_kernel is owned.iterate_maximal_contact_runs_kernel
    constructor = SolverPhoenX.__init__
    metadata = {
        "owned_binding": gate,
        "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executed_toggle_sha256": hashlib.sha256(inspect.getsource(toggle_recovery.func).encode()).hexdigest(),
        "scope": __doc__,
        "extra_kernels_per_biased_contact_call": 2,
        "normal_joint_recovery_and_friction_parameters_unchanged": True,
        "unbiased_calls_unmodified": True,
        "preallocated_before_capture": True,
    }

    def construct(solver, *args, **kwargs):
        constructor(solver, *args, **kwargs)
        world = solver.world
        assert world._direct_tree_contacts, "Bounded corrected direct-tree diagnostic"
        capacity = world._contact_container.derived.shape[1]
        saved = wp.zeros(capacity, dtype=wp.vec2f, device=world.device)
        original = world._solve_maximal_articulated_contacts

        def solve(*, use_bias, refresh_mobility):
            if not use_bias:
                return original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            schedule = world._maximal_contact_schedule
            if schedule is None or schedule.section_end.size == 0:
                return original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            contacts = world._contact_container
            assert contacts.derived.shape[1] <= capacity, "Diagnostic save-buffer capacity changed"
            inputs = [world._contact_cols, contacts, schedule.columns, schedule.section_end, saved]
            wp.launch(toggle_recovery, dim=schedule.columns.shape[0], inputs=[*inputs, False], device=world.device)
            try:
                return original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            finally:
                wp.launch(toggle_recovery, dim=schedule.columns.shape[0], inputs=[*inputs, True], device=world.device)

        world._solve_maximal_articulated_contacts = solve

    SolverPhoenX.__init__ = construct
    return metadata


def main():
    """Run the unchanged public harness or the bounded CPU storage check."""
    if "--cpu-test" in sys.argv:
        check_storage()
        return
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    metadata = install()
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        output.with_suffix(".zero_tangent_recovery.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
