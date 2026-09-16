"""Isolated complete two-body response cache replacing native point tree walks.

Contact projection expressions come from the actually bound corrected owned
module. Larger components retain the original callback. This is not a new
contact model, a convergence claim, or an accepted production optimization.
"""

import ast
import importlib.util
import inspect
import json
import os
import re
import runpy
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.cached_component_response import (
    allocate,
    build_cache,
    compare_rows,
    compare_scalars,
    compare_vectors,
)
from newton._src.solvers.phoenx.articulations import maximal_contact_gs as gs
from newton._src.solvers.phoenx.articulations.maximal_contact_response import MaximalContactResponseData
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData
from newton._src.solvers.phoenx.body import BodyContainer
from newton.solvers import SolverPhoenX


def load_kernels():
    source = Path(gs.__file__).read_text()
    tree = ast.parse(source)
    spans = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def function(name):
        node = spans[name]
        return "\n".join(source.splitlines()[node.decorator_list[0].lineno - 1 : node.end_lineno])

    fallback = []
    for name in ("iterate_maximal_contact_runs_kernel", "refresh_maximal_contact_mobility_kernel"):
        text = function(name).replace("def " + name + "(", "def fallback_" + name + "(")
        text = text.replace(
            "    response: MaximalContactResponseData,",
            "    response: MaximalContactResponseData,\n    cache: CachedComponentData,",
        )
        text = text.replace(
            "    begin = wp.int32(0)", "    if cache.valid[articulation] != 0:\n        return\n    begin = wp.int32(0)"
        )
        fallback.append(text)
    iterate = function("iterate_maximal_contact_runs_kernel")
    iterate = iterate.replace(
        "    response: MaximalContactResponseData,",
        "    response: MaximalContactResponseData,\n    cache: CachedComponentData,",
    )
    iterate = iterate.replace("articulation = tid // wp.int32(_TREE_WIDTH)", "articulation = tid")
    iterate = iterate.replace("lane = tid - articulation * wp.int32(_TREE_WIDTH)", "lane = wp.int32(0)")
    iterate = iterate.replace(
        "if lane < tree.body_count[articulation]:\n                response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)",
        "for target in range(tree.body_count[articulation]):\n                response.impulse[articulation, target] = wp.spatial_vectorf(0.0)",
    )
    iterate = iterate.replace("            _sync_tree()\n", "")
    iterate = iterate.replace(
        "    begin = wp.int32(0)", "    if cache.valid[articulation] == 0:\n        return\n    begin = wp.int32(0)"
    )
    iterate = iterate.replace(
        "_apply_accumulated_impulse(\n                    tree,\n                    response,",
        "apply_cached(\n                    tree,\n                    response,\n                    cache,",
    )
    assert "_apply_accumulated_impulse(" not in iterate and "_sync_tree()" not in iterate
    # Reuse the native exact point-law implementation, including its bound
    # total-normal capacity and latest-solve broken state writes.
    assert "contact_project_friction_metric_with_break" in iterate

    if os.environ.get("COLIBRI_CACHE_REGISTER") == "1":
        local_setup = """    root_body = tree.body_slot[articulation, 0]
    child_body = tree.body_slot[articulation, 1]
    root_v = bodies.velocity[root_body]
    root_w = bodies.angular_velocity[root_body]
    child_v = bodies.velocity[child_body]
    child_w = bodies.angular_velocity[child_body]
    root_mobility = response.mobility[articulation, 0]
    child_inertia = tree.articulated[articulation, 1]
    motion = tree.motion[articulation, 1]
    shift = tree.shift[articulation, 1]
    inverse_d = tree.inverse_d[articulation, 1]
    dynamic_row = tree.dynamic_row[articulation, 1]
    generalized_mass = tree.generalized_mass[articulation, 1]
    local_accumulated = wp.float32(0.0)
    if dynamic_row >= 0:
        local_accumulated = dynamic_accumulated_impulse[dynamic_row]
"""
        iterate = iterate.replace(
            "    for scheduled in range(begin, end):", local_setup + "    for scheduled in range(begin, end):", 1
        )
        iterate = iterate.replace(
            "        friction_static = contact_get_friction(columns, column)",
            "        external_v0 = bodies.velocity[body0]\n        external_w0 = bodies.angular_velocity[body0]\n        external_v1 = bodies.velocity[body1]\n        external_w1 = bodies.angular_velocity[body1]\n        friction_static = contact_get_friction(columns, column)",
        )
        iterate = iterate.replace(
            "            for target in range(tree.body_count[articulation]):\n                response.impulse[articulation, target] = wp.spatial_vectorf(0.0)\n",
            "",
        )
        iterate = iterate.replace(
            "_contact_row_velocity(tree, response, bodies, body0, body1, r0, r1,",
            "row_velocity_local(root_v, root_w, child_v, child_w, external_v0, external_w0, external_v1, external_w1, root_body, child_body, motion, shift, body0, body1, r0, r1,",
        )
        tail = iterate.index("                response.contact_active[articulation] =")
        iterate = (
            iterate[:tail]
            + """            if impulse[0] != 0.0 or impulse[1] != 0.0 or impulse[2] != 0.0:
                root_impulse = wp.spatial_vectorf(0.0)
                child_impulse = wp.spatial_vectorf(0.0)
                torque0 = -wp.cross(r0, impulse)
                torque1 = wp.cross(r1, impulse)
                wrench0 = wp.spatial_vectorf(-impulse[0], -impulse[1], -impulse[2], torque0[0], torque0[1], torque0[2])
                wrench1 = wp.spatial_vectorf(impulse[0], impulse[1], impulse[2], torque1[0], torque1[1], torque1[2])
                if body0 == root_body:
                    root_impulse += wrench0
                if body1 == root_body:
                    root_impulse += wrench1
                if body0 == child_body:
                    child_impulse += wrench0
                if body1 == child_body:
                    child_impulse += wrench1
                root_delta, child_delta, joint_delta = serial_pair_local(root_mobility, child_inertia, motion, shift, inverse_d, root_impulse, child_impulse)
                root_v += wp.spatial_top(root_delta)
                root_w += wp.spatial_bottom(root_delta)
                child_v += wp.spatial_top(child_delta)
                child_w += wp.spatial_bottom(child_delta)
                if dynamic_row >= 0:
                    local_accumulated -= generalized_mass * joint_delta
    bodies.velocity[root_body] = root_v
    bodies.angular_velocity[root_body] = root_w
    bodies.velocity[child_body] = child_v
    bodies.angular_velocity[child_body] = child_w
    if dynamic_row >= 0:
        dynamic_accumulated_impulse[dynamic_row] = local_accumulated
"""
        )
        assert "_contact_row_velocity(" not in iterate and "apply_cached(" not in iterate

    if os.environ.get("COLIBRI_CACHE_POINT_VECTOR") == "1":
        assert os.environ.get("COLIBRI_CACHE_REGISTER") == "1"
        prefix = """                point_velocity = external_relative_point_velocity(root_v, root_w, child_w,
                    external_v0, external_w0, external_v1, external_w1, root_body, child_body,
                    motion, shift, body0, body1, r0, r1)
"""
        for name, direction in [
            ("normal_velocity", "normal"),
            ("tangent_velocity0", "tangent0"),
            ("tangent_velocity1", "tangent1"),
        ]:
            replacement = "                " + name + " = wp.dot(point_velocity, " + direction + ")"
            if name == "normal_velocity":
                replacement = prefix + replacement
            iterate, count = re.subn(
                r"                " + name + r" = row_velocity_local\([^\n]*\)", replacement, iterate
            )
            assert count == 1, (name, count)
        assert "row_velocity_local(" not in iterate
        if os.environ.get("COLIBRI_CACHE_COMPENSATED") == "1":
            iterate = iterate.replace("external_relative_point_velocity(", "compensated_relative_point_velocity(")
            iterate = iterate.replace("wp.dot(point_velocity,", "project_compensated_point_velocity(point_velocity,")

    refresh = function("refresh_maximal_contact_mobility_kernel")
    refresh = refresh.replace(
        "    response: MaximalContactResponseData,",
        "    response: MaximalContactResponseData,\n    cache: CachedComponentData,",
    )
    refresh = refresh.replace("articulation = tid // wp.int32(_TREE_WIDTH)", "articulation = tid")
    refresh = refresh.replace("lane = tid - articulation * wp.int32(_TREE_WIDTH)", "lane = wp.int32(0)")
    refresh = refresh.replace(
        "    begin = wp.int32(0)", "    if cache.valid[articulation] == 0:\n        return\n    begin = wp.int32(0)"
    )
    refresh = refresh.replace(
        "_write_exact_contact_mobility(\n                tree,\n                response,",
        "_write_cached_mobility(\n                tree,\n                response,\n                cache,",
    )
    write = """
@wp.func
def _write_cached_mobility(tree: MaximalTreeProjectorData, response: MaximalContactResponseData,
    cache: CachedComponentData, bodies: BodyContainer, contacts: ContactContainer,
    body0: wp.int32, body1: wp.int32, contact: wp.int32, mobility: wp.array2d[wp.float32]):
    n = cc_get_normal(contacts, contact)
    t = cc_get_tangent1(contacts, contact)
    s = wp.cross(n, t)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    a = cached_cross(tree, response, cache, body0, body1, r0, r1, n, n)
    b = cached_cross(tree, response, cache, body0, body1, r0, r1, t, t)
    c = cached_cross(tree, response, cache, body0, body1, r0, r1, s, s)
    mobility[0, contact] = 0.0
    mobility[1, contact] = 0.0
    mobility[2, contact] = 0.0
    if a > 1.e-12:
        mobility[0, contact] = 1.0 / a
    if b > 1.e-12:
        mobility[1, contact] = 1.0 / b
    if c > 1.e-12:
        mobility[2, contact] = 1.0 / c
    mobility[3, contact] = cached_cross(tree, response, cache, body0, body1, r0, r1, n, t)
    mobility[4, contact] = cached_cross(tree, response, cache, body0, body1, r0, r1, n, s)
    mobility[5, contact] = cached_cross(tree, response, cache, body0, body1, r0, r1, t, s)
"""
    warm = function("warm_start_maximal_contact_runs_kernel")
    # Aggregate all point wrenches, then ONE native factor application. No
    # dependence on a potentially stale cache before equality factorization.
    start = warm.index("    for scheduled in range(begin, end):")
    original_warm_loop = warm[start:]
    warm_header = warm[:start]
    warm = (
        warm[:start]
        + """    if lane < tree.body_count[articulation]:
        response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)
    _sync_tree()
    if lane == 0:
        for scheduled in range(begin, end):
            column = scheduled_column[scheduled]
            body0 = contact_get_body1(columns, column)
            body1 = contact_get_body2(columns, column)
            first = contact_get_contact_first(columns, column)
            count = contact_get_contact_count(columns, column)
            for offset in range(count):
                contact = first + offset
                n = cc_get_normal(contacts, contact)
                t = cc_get_tangent1(contacts, contact)
                impulse = cc_get_normal_lambda(contacts, contact) * n + cc_get_tangent1_lambda(contacts, contact) * t + cc_get_tangent2_lambda(contacts, contact) * wp.cross(n, t)
                r0 = cc_get_r0(contacts, contact)
                r1 = cc_get_r1(contacts, contact)
                _set_spatial_impulse(tree, response, articulation, body0, -impulse, -wp.cross(r0, impulse))
                _set_spatial_impulse(tree, response, articulation, body1, impulse, wp.cross(r1, impulse))
    _sync_tree()
    _apply_accumulated_impulse(tree, response, bodies, dynamic_accumulated_impulse, articulation, lane)
"""
    )
    aggregate_loop = warm[len(warm_header) :]
    warm = (
        warm_header
        + """    unsupported = wp.bool(False)
    for scheduled in range(begin, end):
        column = scheduled_column[scheduled]
        b0 = contact_get_body1(columns, column)
        b1 = contact_get_body2(columns, column)
        a0 = response.body_articulation[b0]
        a1 = response.body_articulation[b1]
        if (a0 == articulation and a1 == articulation) or (a0 != articulation and bodies.inverse_mass[b0] > 0.0) or (a1 != articulation and bodies.inverse_mass[b1] > 0.0):
            unsupported = True
    if unsupported:
"""
        + textwrap.indent(original_warm_loop, "    ")
        + "\n    else:\n"
        + textwrap.indent(aggregate_loop, "    ")
    )
    replacements = {
        "iterate_maximal_contact_runs_kernel": iterate,
        "refresh_maximal_contact_mobility_kernel": refresh,
        "warm_start_maximal_contact_runs_kernel": warm,
    }
    lines = source.splitlines()
    for name in sorted(replacements, key=lambda x: spans[x].lineno, reverse=True):
        node = spans[name]
        lines[node.decorator_list[0].lineno - 1 : node.end_lineno] = replacements[name].splitlines()
    source = "\n".join(lines) + "\n" + "\n".join(fallback)
    marker = "@wp.kernel(enable_backward=False)\ndef refresh_maximal_contact_mobility_kernel"
    source = source.replace(marker, write + "\n" + marker)
    source += """
@wp.kernel(enable_backward=False)
def validate_external(tree: MaximalTreeProjectorData, response: MaximalContactResponseData,
    cache: CachedComponentData, bodies: BodyContainer, columns: ContactColumnContainer,
    scheduled_column: wp.array[wp.int32], section_end: wp.array[wp.int32]):
    articulation = wp.tid()
    begin = wp.int32(0)
    if articulation > 0:
        begin = section_end[articulation - 1]
    for scheduled in range(begin, section_end[articulation]):
        column = scheduled_column[scheduled]
        b0 = contact_get_body1(columns, column)
        b1 = contact_get_body2(columns, column)
        a0 = response.body_articulation[b0]
        a1 = response.body_articulation[b1]
        if (a0 == articulation and a1 == articulation) or (a0 != articulation and bodies.inverse_mass[b0] > 0.0) or (a1 != articulation and bodies.inverse_mass[b1] > 0.0):
            cache.valid[articulation] = 0
"""

    source = source.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\nfrom local_studies.colibri.cached_component_response import CachedComponentData, apply_cached, cached_cross\nfrom local_studies.colibri.register_component_response import row_velocity_local, serial_pair_local\nfrom local_studies.colibri.external_point_velocity import external_relative_point_velocity\nfrom local_studies.colibri.external_point_velocity_compensated import compensated_relative_point_velocity, project_compensated_point_velocity",
    )
    if os.environ.get("COLIBRI_CACHE_POINT_VECTOR") == "1":
        guard = "(a0 == articulation and a1 == articulation) or (a0 != articulation and bodies.inverse_mass[b0] > 0.0) or (a1 != articulation and bodies.inverse_mass[b1] > 0.0)"
        assert source.count(guard) == 2
        source = source.replace(
            guard,
            guard
            + " or (a0 != articulation and bodies.motion_type[b0] != wp.int32(0)) or (a1 != articulation and bodies.motion_type[b1] != wp.int32(0))",
        )
    directory = Path(tempfile.mkdtemp(prefix="colibri_cached_component_"))
    path = directory / "cached_contact_gs.py"
    path.write_text(source)
    name = gs.__package__ + ".cached_contact_gs"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install():
    kernels = load_kernels()
    constructor = SolverPhoenX.__init__
    states = []

    def construct(solver, *args, **kwargs):
        constructor(solver, *args, **kwargs)
        w = solver.world
        projector = w._maximal_tree_projector
        if projector is None or not w._direct_tree_contacts:
            return
        counts = projector.data.body_count.numpy().copy()
        if np.any(counts != 2):
            return  # Explicit unmodified full-component fallback.
        if int(np.count_nonzero(w.bodies.inverse_mass.numpy() > 0)) != 2:
            raise ValueError("First cache gate supports a complete two-body world only")
        assert w.sor_boost == 1.0
        device = w.device
        cache = allocate(len(counts), device)
        shadow = BodyContainer()
        for name in BodyContainer.vars:
            setattr(shadow, name, getattr(w.bodies, name))
        shadow.velocity = wp.empty_like(w.bodies.velocity)
        shadow.angular_velocity = wp.empty_like(w.bodies.angular_velocity)
        scratch_lambda = wp.empty_like(projector.dynamic_accumulated_impulse)
        original_warm = w._warm_start_owned_contacts
        original_solve = w._solve_maximal_articulated_contacts
        states.append(
            {
                "world": w,
                "cache": cache,
                "source": inspect.getsourcefile(kernels.iterate_maximal_contact_runs_kernel.func),
            }
        )

        def warm():
            if not w._contact_input_active_this_step:
                return
            if w._direct_contact_response is not None:
                return original_warm()
            response = w._maximal_contact_response
            schedule = w._maximal_contact_schedule
            projector.factor_contact_response()
            response.compute_mobility()
            wp.launch(
                kernels.warm_start_maximal_contact_runs_kernel,
                dim=projector.launch_dim,
                block_dim=projector.block_dim,
                inputs=[
                    projector.data,
                    response.data,
                    w.bodies,
                    projector.dynamic_accumulated_impulse,
                    w._contact_cols,
                    w._contact_container,
                    schedule.columns,
                    schedule.section_end,
                ],
                device=device,
            )

        def solve(*, use_bias, refresh_mobility):
            response = w._maximal_contact_response
            schedule = w._maximal_contact_schedule
            if response is None or schedule is None:
                return original_solve(use_bias=use_bias, refresh_mobility=refresh_mobility)
            if refresh_mobility:
                response.compute_mobility()
                wp.launch(
                    build_cache,
                    dim=projector.launch_dim,
                    block_dim=projector.block_dim,
                    inputs=[projector.data, response.data, cache],
                    device=device,
                )
                wp.launch(
                    kernels.validate_external,
                    dim=len(counts),
                    inputs=[
                        projector.data,
                        response.data,
                        cache,
                        w.bodies,
                        w._contact_cols,
                        schedule.columns,
                        schedule.section_end,
                    ],
                    device=device,
                )
                wp.launch(
                    kernels.fallback_refresh_maximal_contact_mobility_kernel,
                    dim=projector.launch_dim,
                    block_dim=projector.block_dim,
                    inputs=[
                        projector.data,
                        response.data,
                        cache,
                        w.bodies,
                        w._contact_cols,
                        w._contact_container,
                        schedule.columns,
                        schedule.section_end,
                        schedule.mobility,
                    ],
                    device=device,
                )
                wp.launch(
                    kernels.refresh_maximal_contact_mobility_kernel,
                    dim=len(counts),
                    inputs=[
                        projector.data,
                        response.data,
                        cache,
                        w.bodies,
                        w._contact_cols,
                        w._contact_container,
                        schedule.columns,
                        schedule.section_end,
                        schedule.mobility,
                    ],
                    device=device,
                )
            iterations = w.solver_iterations if use_bias else w._active_velocity_iterations
            direct = w._direct_equality_system
            split = use_bias and iterations > 0 and direct is not None and direct.enabled
            if split:
                direct.compute_bias_velocity()
                direct.apply_bias_velocity(-1.0)
            wp.copy(shadow.velocity, w.bodies.velocity)
            wp.copy(shadow.angular_velocity, w.bodies.angular_velocity)
            wp.copy(scratch_lambda, projector.dynamic_accumulated_impulse)
            for _ in range(iterations):
                wp.launch(
                    kernels.fallback_iterate_maximal_contact_runs_kernel,
                    dim=projector.launch_dim,
                    block_dim=projector.block_dim,
                    inputs=[
                        projector.data,
                        response.data,
                        cache,
                        shadow,
                        scratch_lambda,
                        w._contact_cols,
                        w._contact_container,
                        wp.float32(1.0 / w.substep_dt),
                        wp.float32(w.sor_boost),
                        schedule.columns,
                        schedule.section_end,
                        schedule.mobility,
                        wp.bool(use_bias),
                    ],
                    device=device,
                )
                wp.launch(
                    kernels.iterate_maximal_contact_runs_kernel,
                    dim=len(counts),
                    inputs=[
                        projector.data,
                        response.data,
                        cache,
                        shadow,
                        scratch_lambda,
                        w._contact_cols,
                        w._contact_container,
                        wp.float32(1.0 / w.substep_dt),
                        wp.float32(w.sor_boost),
                        schedule.columns,
                        schedule.section_end,
                        schedule.mobility,
                        wp.bool(use_bias),
                    ],
                    device=device,
                )
            wp.copy(w.bodies.velocity, shadow.velocity)
            wp.copy(w.bodies.angular_velocity, shadow.angular_velocity)
            wp.copy(projector.dynamic_accumulated_impulse, scratch_lambda)
            if split:
                direct.apply_bias_velocity(1.0)

        if os.environ.get("COLIBRI_CACHE_COMPARE") == "1":
            arrays = [
                w.bodies.velocity,
                w.bodies.angular_velocity,
                projector.dynamic_accumulated_impulse,
                w._contact_container.impulses,
                w._contact_container.lambdas,
            ]
            before = [wp.empty_like(a) for a in arrays]
            proposed = [wp.empty_like(a) for a in arrays]
            errors = wp.zeros(16, dtype=float, device=device)
            states[-1]["errors"] = errors

            def compare_solve(*, use_bias, refresh_mobility):
                for dst, src in zip(before, arrays, strict=False):
                    wp.copy(dst, src)
                solve(use_bias=use_bias, refresh_mobility=refresh_mobility)
                for dst, src in zip(proposed, arrays, strict=False):
                    wp.copy(dst, src)
                for dst, src in zip(arrays, before, strict=False):
                    wp.copy(dst, src)
                # Always rebuild native mobility: prior comparison phases may
                # leave either implementation's schedule storage behind.
                original_solve(use_bias=use_bias, refresh_mobility=True)
                for slot, (a, b) in enumerate(zip(proposed, arrays, strict=False)):
                    kernel = compare_rows if a.ndim == 2 else compare_scalars
                    if slot < 2:
                        kernel = compare_vectors
                    wp.launch(kernel, dim=a.shape, inputs=[a, b, errors, slot], device=device)

            def compare_warm():
                for dst, src in zip(before, arrays, strict=False):
                    wp.copy(dst, src)
                warm()
                for dst, src in zip(proposed, arrays, strict=False):
                    wp.copy(dst, src)
                for dst, src in zip(arrays, before, strict=False):
                    wp.copy(dst, src)
                original_warm()
                for slot in range(3):
                    a, b = proposed[slot], arrays[slot]
                    kernel = compare_vectors if slot < 2 else compare_scalars
                    wp.launch(kernel, dim=a.shape, inputs=[a, b, errors, slot + 5], device=device)

            w._solve_maximal_articulated_contacts = compare_solve
            w._warm_start_owned_contacts = compare_warm
        else:
            w._warm_start_owned_contacts = warm
            w._solve_maximal_articulated_contacts = solve

    SolverPhoenX.__init__ = construct
    return states


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    states = install()
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        for data in states:
            cache = data["cache"]
            arrays = {"velocity": cache.velocity.numpy(), "joint": cache.joint.numpy(), "valid": cache.valid.numpy()}
            world = data["world"]
            for prefix, cls, value in [
                ("tree", MaximalTreeProjectorData, world._maximal_tree_projector.data),
                ("response", MaximalContactResponseData, world._maximal_contact_response.data),
                ("bodies", BodyContainer, world.bodies),
            ]:
                for name in cls.vars:
                    field = getattr(value, name)
                    if isinstance(field, wp.array) and field.size > 0:
                        arrays[prefix + "__" + name] = field.numpy().copy()
            np.savez_compressed(output.with_suffix(".cached_response.npz"), **arrays)
        output.with_suffix(".cached_response.json").write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "register_state": os.environ.get("COLIBRI_CACHE_REGISTER") == "1",
                    "point_vector_fp32": os.environ.get("COLIBRI_CACHE_POINT_VECTOR") == "1",
                    "compensated_fp32": os.environ.get("COLIBRI_CACHE_COMPENSATED") == "1",
                    "comparison_errors": [s["errors"].numpy().tolist() for s in states if "errors" in s],
                    "comparison_error_names": [
                        "solve_v",
                        "solve_omega",
                        "solve_drive_lambda",
                        "solve_contact_lambda",
                        "solve_history",
                        "warm_v",
                        "warm_omega",
                        "warm_drive_lambda",
                    ],
                    "comparison_layout": "first8 absolute max errors; next8 absolute max native value over all compared phases",
                    "active_worlds": len(states),
                    "source": [s["source"] for s in states],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
