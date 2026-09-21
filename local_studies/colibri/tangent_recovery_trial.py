"""Local tangent-only recovery trial overlay for corrected native two-body runs.

Set COLIBRI_STAGE_RUNNER to this module. The ordinary normal solve, response,
joint recovery application, drive, and material friction remain unchanged.
"""

import hashlib
import inspect
import json
import os
import runpy
import sys
from pathlib import Path

import warp as wp
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_normal,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
    cc_set_bias_t1,
    cc_set_bias_t2,
)


@wp.kernel(enable_backward=False)
def shift_tangent_target(
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    scheduled: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    recovery: wp.array[wp.spatial_vector],
    saved: wp.array[wp.vec2f],
    restore: wp.bool,
):
    """Temporarily add only the actual restored tangential point velocity."""
    row = wp.tid()
    if row < ends[ends.shape[0] - 1]:
        column = scheduled[row]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        for offset in range(count):
            k = first + offset
            if restore:
                cc_set_bias_t1(contacts, k, saved[k][0])
                cc_set_bias_t2(contacts, k, saved[k][1])
            else:
                old = wp.vec2f(cc_get_bias_t1(contacts, k), cc_get_bias_t2(contacts, k))
                saved[k] = old
                r0 = cc_get_r0(contacts, k)
                r1 = cc_get_r1(contacts, k)
                v0 = wp.spatial_top(recovery[body0]) + wp.cross(wp.spatial_bottom(recovery[body0]), r0)
                v1 = wp.spatial_top(recovery[body1]) + wp.cross(wp.spatial_bottom(recovery[body1]), r1)
                t0 = cc_get_tangent1(contacts, k)
                t1 = wp.cross(cc_get_normal(contacts, k), t0)
                cc_set_bias_t1(contacts, k, old[0] + wp.dot(t0, v1 - v0))
                cc_set_bias_t2(contacts, k, old[1] + wp.dot(t1, v1 - v0))


def install():
    """Wrap only the native maximal/direct contact split boundary."""
    import newton._src.solvers.phoenx.solver_phoenx as solver_module

    import newton._src.solvers.phoenx.articulations.maximal_contact_gs as owned
    from local_studies.colibri.check_owned_friction_binding import check

    gate = check(owned)
    assert solver_module.iterate_maximal_contact_runs_kernel is owned.iterate_maximal_contact_runs_kernel
    original = PhoenXWorld._solve_maximal_articulated_contacts
    apply_original = DirectEqualitySystem.apply_bias_velocity
    active = [None]
    buffers = {}

    def solve(solver, *, use_bias, refresh_mobility):
        assert active[0] is None
        assert solver._direct_tree_contacts, "Diagnostic supports direct-tree contacts only"
        direct = solver._direct_equality_system
        assert direct is not None and direct.enabled
        key = id(solver)
        if key not in buffers:
            buffers[key] = wp.zeros(solver._contact_container.derived.shape[1], dtype=wp.vec2f, device=solver.device)
        active[0] = solver
        try:
            return original(solver, use_bias=use_bias, refresh_mobility=refresh_mobility)
        finally:
            active[0] = None

    def apply(direct, scale):
        solver = active[0]
        if solver is None:
            return apply_original(direct, scale)
        assert direct is solver._direct_equality_system and scale in (-1.0, 1.0)
        schedule = solver._maximal_contact_schedule
        # Save/restore exact old target values; never subtract the increment later.
        wp.launch(
            shift_tangent_target,
            dim=schedule.columns.shape[0],
            inputs=[
                solver._contact_cols,
                solver._contact_container,
                schedule.columns,
                schedule.section_end,
                direct.bias_velocity,
                buffers[id(solver)],
                wp.bool(scale > 0),
            ],
            device=solver.device,
        )
        return apply_original(direct, scale)

    PhoenXWorld._solve_maximal_articulated_contacts = solve
    DirectEqualitySystem.apply_bias_velocity = apply
    return {
        "owned_binding": gate,
        "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executed_shift_kernel_sha256": hashlib.sha256(
            inspect.getsource(shift_tangent_target.func).encode()
        ).hexdigest(),
        "normal_callback_unchanged": True,
        "extra_kernels_per_biased_contact_call": 2,
        "scope": "Tangent trial includes restored hard-joint recovery; reference diagnostic only.",
    }


def main():
    """Run the same public harness after the corrected native wrapper installs."""
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    metadata = install()
    try:
        if os.environ.get("COLIBRI_RECOVERY_FIXTURE") == "1":
            import unittest

            suite = unittest.defaultTestLoader.loadTestsFromName(
                "newton._src.solvers.phoenx.tests.test_maximal_contact_mobility."
                "TestMaximalContactMobility.test_joint_position_recovery_does_not_launch_internal_contact"
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            assert result.testsRun == 1 and result.wasSuccessful()
        else:
            runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        output.with_suffix(".tangent_recovery.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
