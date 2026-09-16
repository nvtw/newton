"""Isolated native-joint/normal + literal PhysX patch-friction hybrid runner.

Use COLIBRI_STAGE_RUNNER after the corrected native_conditioned_two_body wrapper.
No production modification. Not full PhysX: native soft normals, gravity timing,
joint factorization, geometry and integration remain. Source-derived p8 is an
explicit diagnostic friction relaxation factor, not a production SOR change.
"""

import hashlib
import importlib.util
import inspect
import json
import math
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import physx_patch_reference_live_kernels as kernels
from newton._src.solvers.phoenx import solver_phoenx as solver_module
from newton._src.solvers.phoenx.articulations import maximal_contact_gs as native
from newton._src.solvers.phoenx.constraints.constraint_contact import contact_column_container_zeros
from newton._src.solvers.phoenx.constraints.contact_container import contact_container_zeros
from newton.solvers import SolverPhoenX


def clone_kernels():
    source = inspect.getsource(native.iterate_maximal_contact_runs_kernel.func)
    source = source[source.index("def iterate_maximal_contact_runs_kernel(") :]
    normal = source.replace("def iterate_maximal_contact_runs_kernel(", "def normal_only(")
    start = normal.index("                    normal_delta = wp.dot(normal_impulse, normal)")
    end = normal.index("                response.contact_active[articulation]", start)
    normal = normal[:start] + "                    impulse = normal_impulse\n" + normal[end:]
    friction = source.replace("def iterate_maximal_contact_runs_kernel(", "def patch_friction(")
    friction = friction.replace(
        "    use_bias: wp.bool,",
        "    use_bias: wp.bool,\n    p8: wp.float32,\n    velocity_multiplier: wp.float32,\n    ledger: wp.array[wp.spatial_vector],",
    )
    start = friction.index("                    normal_impulse = -cc_get_normal_lambda")
    end = friction.index("                    rhs0 = tangent_velocity0", start)
    friction = (
        friction[:start]
        + """                    normal_impulse = wp.vec3f(0.0)
                    normal_delta = wp.float32(0.0)
                    normal_load = cc_get_normal_lambda(contacts, contact)
"""
        + friction[end:]
    )
    start = friction.index("                    tangents = contact_project_friction_metric_with_break(")
    end = friction.index("                    contacts.lambdas[CC_FRICTION_BROKEN", start)
    friction = (
        friction[:start]
        + """                    trial0 = (lambda0 - (cc_get_bias_t1(contacts, contact) if use_bias else wp.float32(0.0)) * (p8 * effective0)) - tangent_velocity0 * (velocity_multiplier * effective0)
                    trial1 = (lambda1 - (cc_get_bias_t2(contacts, contact) if use_bias else wp.float32(0.0)) * (p8 * effective1)) - tangent_velocity1 * (velocity_multiplier * effective1)
                    length = wp.sqrt(trial0 * trial0 + trial1 * trial1)
                    broken = wp.float32(0.0)
                    ratio = wp.float32(1.0)
                    if length > friction_static * normal_load:
                        broken = wp.float32(1.0)
                        ratio = wp.min(friction_dynamic * normal_load, length) / length
                    tangents = wp.vec3f(trial0 * ratio, trial1 * ratio, broken)
"""
        + friction[end:]
    )
    marker = (
        "                _set_spatial_impulse(tree, response, articulation, body0, -impulse, -wp.cross(r0, impulse))"
    )
    assert friction.count(marker) == 1
    friction = friction.replace(
        marker,
        """                force0 = -impulse
                torque0 = -wp.cross(r0, impulse)
                torque1 = wp.cross(r1, impulse)
                ledger[body0] += wp.spatial_vectorf(force0[0],force0[1],force0[2],torque0[0],torque0[1],torque0[2])
                ledger[body1] += wp.spatial_vectorf(impulse[0],impulse[1],impulse[2],torque1[0],torque1[1],torque1[2])
"""
        + marker,
    )
    full = (
        Path(native.__file__).read_text()
        + "\n\n@wp.kernel(enable_backward=False)\n"
        + normal
        + "\n\n@wp.kernel(enable_backward=False)\n"
        + friction
    )
    # The source GPU friction equations are under the retained PhysX BSD notice.
    license_text = Path(__file__).with_name("physx_patch_reference.py").read_text().split('"""', 1)[0]
    full = license_text + full
    directory = Path(tempfile.mkdtemp(prefix="colibri_physx_patch_"))
    path = directory / "kernels.py"
    path.write_text(full)
    spec = importlib.util.spec_from_file_location("colibri_physx_patch_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def install():
    module, path = clone_kernels()
    states = {}
    constructor = SolverPhoenX.__init__

    def construct(solver, *args, **kwargs):
        constructor(solver, *args, **kwargs)
        w = solver.world
        assert solver._direct_tree_contacts and w.bodies.position.shape[0] == 4
        assert w.solver_iterations == 1 and w.sor_boost == 1
        device = w.bodies.position.device
        nb = w.bodies.position.shape[0]
        cap = w._contact_cols.data.shape[1]
        state = kernels.PatchState()
        for name in ("count", "broken", "claims"):
            setattr(state, name, wp.zeros(nb, dtype=int, device=device))
        for name in ("refresh", "error"):
            setattr(state, name, wp.zeros(1, dtype=int, device=device))
        for name in ("local0", "local1", "r0", "r1"):
            setattr(state, name, wp.zeros((nb, 2), dtype=wp.vec3f, device=device))
        for name in ("normal0", "normal1", "normal", "tangent", "linear", "angular"):
            setattr(state, name, wp.zeros(nb, dtype=wp.vec3f, device=device))
        state.initial = wp.zeros((nb, 2), dtype=wp.vec2f, device=device)
        data = {
            "solver": solver,
            "state": state,
            "columns": contact_column_container_zeros(cap, device=device),
            "contacts": contact_container_zeros(nb * 2, device=device),
            "mobility": wp.zeros((6, nb * 2), dtype=float, device=device),
            "ledger": wp.zeros(nb, dtype=wp.spatial_vector, device=device),
            "history": wp.zeros((128, nb, 25), dtype=float, device=device),
            "counter": wp.zeros(1, dtype=int, device=device),
        }
        direct = solver._direct_equality_system
        assert direct.accumulated_impulse.shape[0] == 6 and direct.row_wrench0.shape == (1, 6)
        data["joint_history"] = wp.zeros((128, 6, 15), dtype=float, device=device)
        data["endpoint_history"] = wp.zeros((128, nb * 2, 8), dtype=float, device=device)
        states[id(w)] = data
        wp.launch(kernels.mark_refresh, dim=1, inputs=[state], device=device)
        ingest = w._ingest_and_warmstart_contacts

        def refresh(*a, **kw):
            result = ingest(*a, **kw)
            wp.launch(kernels.mark_refresh, dim=1, inputs=[state], device=device)
            return result

        w._ingest_and_warmstart_contacts = refresh
        integrate = w._integrate_positions

        def motion():
            wp.launch(
                kernels.accumulate_motion, dim=nb, inputs=[state, w.bodies, wp.float32(w.substep_dt)], device=device
            )
            return integrate()

        w._integrate_positions = motion
        original = w._solve_maximal_articulated_contacts

        def solve(*, use_bias, refresh_mobility):
            cc = w._contact_container
            schedule = w._maximal_contact_schedule
            projector = w._maximal_tree_projector
            response = w._maximal_contact_response
            wp.launch(kernels.begin_refresh, dim=nb, inputs=[state], device=device)
            wp.launch(
                kernels.refresh_patches,
                dim=cap,
                inputs=[
                    state,
                    w.bodies,
                    w._contact_cols,
                    cc,
                    w._ingest_scratch.num_contact_columns,
                    data["contacts"],
                    wp.float32(0.04),
                    wp.float32(0.025),
                ],
                device=device,
            )
            wp.launch(kernels.finish_refresh, dim=1, inputs=[state], device=device)
            saved = solver_module.iterate_maximal_contact_runs_kernel
            solver_module.iterate_maximal_contact_runs_kernel = module.normal_only
            try:
                original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            finally:
                solver_module.iterate_maximal_contact_runs_kernel = saved
            wp.launch(
                kernels.prepare_anchors,
                dim=cap,
                inputs=[
                    state,
                    w._contact_cols,
                    cc,
                    w._ingest_scratch.num_contact_columns,
                    data["columns"],
                    data["contacts"],
                    wp.float32(1 / w.substep_dt),
                ],
                device=device,
            )
            args = [
                projector.data,
                response.data,
                w.bodies,
                data["columns"],
                data["contacts"],
                schedule.columns,
                schedule.section_end,
                data["mobility"],
            ]
            wp.launch(
                native.refresh_maximal_contact_mobility_kernel,
                dim=projector.launch_dim,
                block_dim=projector.block_dim,
                inputs=args,
                device=device,
            )
            wp.launch(
                kernels.record_before,
                dim=nb,
                inputs=[w.bodies, data["history"], data["counter"], data["ledger"]],
                device=device,
            )
            wp.launch(
                kernels.record_joint,
                dim=6,
                inputs=[
                    data["counter"],
                    data["joint_history"],
                    direct.accumulated_impulse,
                    direct.row_wrench0,
                    direct.row_wrench1,
                    direct.row_dynamic,
                    False,
                ],
                device=device,
            )
            wp.launch(
                kernels.record_endpoints,
                dim=cap,
                inputs=[
                    state,
                    w.bodies,
                    w._contact_cols,
                    data["contacts"],
                    w._ingest_scratch.num_contact_columns,
                    data["counter"],
                    data["endpoint_history"],
                ],
                device=device,
            )
            wp.launch(
                module.patch_friction,
                dim=projector.launch_dim,
                block_dim=projector.block_dim,
                inputs=[
                    projector.data,
                    response.data,
                    w.bodies,
                    projector.dynamic_accumulated_impulse,
                    data["columns"],
                    data["contacts"],
                    wp.float32(1 / w.substep_dt),
                    wp.float32(1),
                    schedule.columns,
                    schedule.section_end,
                    data["mobility"],
                    wp.bool(use_bias),
                    wp.float32(min(0.8, 0.9, 2 / math.sqrt(30))),
                    wp.float32(
                        float(
                            os.environ.get("COLIBRI_PATCH_VELOCITY_MULTIPLIER", str(min(0.8, 0.9, 2 / math.sqrt(30))))
                        )
                    ),
                    data["ledger"],
                ],
                device=device,
            )
            wp.launch(kernels.writeback_broken, dim=nb, inputs=[state, data["contacts"]], device=device)
            wp.launch(
                kernels.record_after,
                dim=nb,
                inputs=[w.bodies, data["history"], data["counter"], data["ledger"]],
                device=device,
            )
            wp.launch(
                kernels.record_joint,
                dim=6,
                inputs=[
                    data["counter"],
                    data["joint_history"],
                    direct.accumulated_impulse,
                    direct.row_wrench0,
                    direct.row_wrench1,
                    direct.row_dynamic,
                    True,
                ],
                device=device,
            )
            wp.launch(kernels.advance_record, dim=1, inputs=[data["counter"]], device=device)

        w._solve_maximal_articulated_contacts = solve

    SolverPhoenX.__init__ = construct
    return states, path


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    states, path = install()
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        for data in states.values():
            state = data["state"]
            count = int(data["counter"].numpy()[0])
            history = data["history"].numpy().copy()
            history = history[np.arange(max(0, count - 128), count) % 128]
            solver = data["solver"]
            np.savez_compressed(
                output.with_suffix(".patch.npz"),
                history=history,
                joint_history=data["joint_history"].numpy()[np.arange(max(0, count - 128), count) % 128],
                endpoint_history=data["endpoint_history"].numpy()[np.arange(max(0, count - 128), count) % 128],
                event_count=np.array([count]),
                body_mass=solver.model.body_mass.numpy(),
                body_inertia=solver.model.body_inertia.numpy(),
                patch_count=state.count.numpy(),
                broken=state.broken.numpy(),
                error=state.error.numpy(),
                anchors=data["contacts"].lambdas.numpy(),
                impulses=data["contacts"].impulses.numpy(),
            )
            assert not np.any(state.error.numpy()), "Unsupported patch geometry/ownership"
        output.with_suffix(".patch.json").write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "generated_source": str(path),
                    "generated_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "p8": min(0.8, 0.9, 2 / math.sqrt(30)),
                    "velocity_multiplier": float(
                        os.environ.get("COLIBRI_PATCH_VELOCITY_MULTIPLIER", str(min(0.8, 0.9, 2 / math.sqrt(30))))
                    ),
                    "bias_multiplier": min(0.8, 0.9, 2 / math.sqrt(30)),
                    "friction_offset_m": 0.04,
                    "correlation_distance_m": 0.025,
                    "normal_policy": "native soft conditioned, all normal rows before patch friction",
                    "momentum_claim": "None: independent/frozen anchor levers require saved per-phase ledger audit",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
