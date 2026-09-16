"""Local unbiased-final six-wrench correction with original point cones and native response.

Bounded two-body/ground study. No biased changes, no clipping/SOR, no history writes.
Every rejected proposal leaves bodies, impulses, drive and contact history untouched.
"""

import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.full_wrench_candidate import propose
from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponseData,
    apply_maximal_contact_impulse_thread,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData, _sync_tree
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton.solvers import SolverPhoenX


@wp.kernel(enable_backward=False)
def prepare(
    bodies: BodyContainer,
    response: MaximalContactResponseData,
    columns: ContactColumnContainer,
    cc: ContactContainer,
    mobility: wp.array[wp.spatial_matrixf],
    velocity: wp.array[wp.spatial_vectorf],
    maps: wp.array2d[wp.spatial_vectorf],
    coefficients: wp.array[float],
    eligible: wp.array[int],
    interval: wp.array[int],
    selection: wp.array[int],
):
    interval[0] = 0
    interval[1] = 0
    selection[0] = -1
    for col in range(columns.data.shape[1]):
        first = contact_get_contact_first(columns, col)
        count = contact_get_contact_count(columns, col)
        b0 = contact_get_body1(columns, col)
        b1 = contact_get_body2(columns, col)
        if count > 0 and first >= 0 and b0 == 1 and b1 == 0:
            if selection[0] >= 0:
                selection[0] = -2
            else:
                selection[0] = col
                interval[0] = first
                interval[1] = count
    if selection[0] >= 0:
        articulation = response.body_articulation[1]
        lane = response.body_lane[1]
        selection[1] = articulation
        selection[2] = lane
        mobility[0] = response.mobility[articulation, lane]
        velocity[0] = wp.spatial_vectorf(
            bodies.velocity[1][0],
            bodies.velocity[1][1],
            bodies.velocity[1][2],
            bodies.angular_velocity[1][0],
            bodies.angular_velocity[1][1],
            bodies.angular_velocity[1][2],
        )
        for k in range(interval[0], interval[0] + interval[1]):
            eligible[k] = wp.int32(cc.derived[3, k] <= 0.0)
            coefficients[k] = columns.data[3, selection[0]]
            n = wp.vec3f(cc.lambdas[0, k], cc.lambdas[1, k], cc.lambdas[2, k])
            t = wp.vec3f(cc.lambdas[3, k], cc.lambdas[4, k], cc.lambdas[5, k])
            r = wp.vec3f(cc.derived[9, k], cc.derived[10, k], cc.derived[11, k])
            for axis in range(3):
                direction = n
                if axis == 1:
                    direction = t
                if axis == 2:
                    direction = wp.cross(n, t)
                torque = wp.cross(r, direction)
                maps[k, axis] = -wp.spatial_vectorf(
                    direction[0], direction[1], direction[2], torque[0], torque[1], torque[2]
                )


@wp.kernel
def force_reject(status: wp.array[int]):
    status[0] = -8


@wp.func
def predicted_velocity(body: int, bodies: BodyContainer, response: MaximalContactResponseData, articulation: int):
    value = wp.spatial_vectorf(
        bodies.velocity[body][0],
        bodies.velocity[body][1],
        bodies.velocity[body][2],
        bodies.angular_velocity[body][0],
        bodies.angular_velocity[body][1],
        bodies.angular_velocity[body][2],
    )
    if response.body_articulation[body] == articulation:
        value += response.velocity[articulation, response.body_lane[body]]
    return value


@wp.kernel(enable_backward=False)
def verify_apply(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    cc: ContactContainer,
    columns: ContactColumnContainer,
    candidate: wp.array2d[float],
    eligible: wp.array[int],
    interval: wp.array[int],
    selection: wp.array[int],
    status: wp.array[int],
    maps: wp.array2d[wp.spatial_vectorf],
    row0: wp.array2d[wp.spatial_vectorf],
    row1: wp.array2d[wp.spatial_vectorf],
    dynamic: wp.array[wp.bool],
    dynamic_mass: wp.array[float],
    reference: wp.array[float],
    accumulated: wp.array[float],
    counters: wp.array[int],
    residuals: wp.array[float],
    history: wp.array3d[float],
    counter: wp.array[int],
):
    lane = wp.tid()
    articulation = selection[1]
    event = counter[0] % 64
    if lane < 4:
        for d in range(3):
            history[event, lane, d] = bodies.velocity[lane][d]
            history[event, lane, 3 + d] = bodies.angular_velocity[lane][d]
            history[event, lane, 6 + d] = bodies.position[lane][d]
            history[event, lane, 19 + d] = 0.0
            history[event, lane, 22 + d] = 0.0
        for d in range(4):
            history[event, lane, 9 + d] = bodies.orientation[lane][d]
    if lane == 0:
        counters[0] += 1
        if selection[0] < 0:
            status[0] = -8
    _sync_tree()
    if selection[0] >= 0 and status[0] == 1:
        if lane < tree.body_count[articulation]:
            response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)
        _sync_tree()
        if lane == 0:
            force = wp.spatial_vectorf(0.0)
            error = wp.spatial_vectorf(0.0)
            for k in range(interval[0], interval[0] + interval[1]):
                if eligible[k] != 0:
                    for r in range(3):
                        add = maps[k, r] * (candidate[r, k] - cc.impulses[r, k]) - error
                        new = force + add
                        error = (new - force) - add
                        force = new
            response.impulse[articulation, selection[2]] = force
        _sync_tree()
        apply_maximal_contact_impulse_thread(articulation, lane, tree, response)
        if lane == 0:
            reason = wp.int32(1)
            maxnormal = wp.float32(0.0)
            maxtangent = wp.float32(0.0)
            minspec = wp.float32(1.0e30)
            for col in range(columns.data.shape[1]):
                first = contact_get_contact_first(columns, col)
                count = contact_get_contact_count(columns, col)
                if count > 0 and first >= 0:
                    b0 = contact_get_body1(columns, col)
                    b1 = contact_get_body2(columns, col)
                    pv0 = predicted_velocity(b0, bodies, response, articulation)
                    pv1 = predicted_velocity(b1, bodies, response, articulation)
                    for axis in range(6):
                        if not wp.isfinite(pv0[axis]) or not wp.isfinite(pv1[axis]):
                            reason = -6
                    for k in range(first, first + count):
                        r0 = wp.vec3f(cc.derived[9, k], cc.derived[10, k], cc.derived[11, k])
                        r1 = wp.vec3f(cc.derived[12, k], cc.derived[13, k], cc.derived[14, k])
                        rel = (
                            wp.spatial_top(pv1)
                            + wp.cross(wp.spatial_bottom(pv1), r1)
                            - wp.spatial_top(pv0)
                            - wp.cross(wp.spatial_bottom(pv0), r0)
                        )
                        n = wp.vec3f(cc.lambdas[0, k], cc.lambdas[1, k], cc.lambdas[2, k])
                        t = wp.vec3f(cc.lambdas[3, k], cc.lambdas[4, k], cc.lambdas[5, k])
                        vn = wp.dot(n, rel)
                        if cc.derived[3, k] > 0.0:
                            margin = vn + cc.derived[3, k]
                            minspec = wp.min(minspec, margin)
                            if margin < -1.0e-8:
                                reason = -7
                        else:
                            ln = cc.impulses[0, k]
                            lt0 = cc.impulses[1, k]
                            lt1 = cc.impulses[2, k]
                            if col == selection[0]:
                                ln = candidate[0, k]
                                lt0 = candidate[1, k]
                                lt1 = candidate[2, k]
                            nr = wp.max(-vn, wp.float32(0.0))
                            if ln > 0.0:
                                nr = wp.abs(vn)
                            maxnormal = wp.max(maxnormal, nr)
                            if not wp.isfinite(nr) or nr > 1.0e-8:
                                reason = -4
                            mu = columns.data[3, col]
                            if (
                                not wp.isfinite(mu)
                                or mu < 0.0
                                or ln < 0.0
                                or lt0 * lt0 + lt1 * lt1 > (mu * ln) * (mu * ln)
                            ):
                                reason = -3
                            if mu > 0.0 and ln > 0.0:
                                tangent = wp.max(wp.abs(wp.dot(t, rel)), wp.abs(wp.dot(wp.cross(n, t), rel)))
                                maxtangent = wp.max(maxtangent, tangent)
                                if not wp.isfinite(tangent) or tangent > 1.0e-8:
                                    reason = -5
            pv0 = predicted_velocity(1, bodies, response, articulation)
            pv1 = predicted_velocity(2, bodies, response, articulation)
            maxjoint = wp.float32(0.0)
            for row in range(dynamic.shape[0]):
                value = wp.dot(row0[0, row], pv0) + wp.dot(row1[0, row], pv1)
                if dynamic[row]:
                    delta = wp.float32(0.0)
                    for l in range(tree.body_count[articulation]):
                        if tree.dynamic_row[articulation, l] == row:
                            delta = -tree.generalized_mass[articulation, l] * response.joint_velocity[articulation, l]
                    value += (accumulated[row] + delta) / dynamic_mass[row] - reference[row]
                maxjoint = wp.max(maxjoint, wp.abs(value))
                if not wp.isfinite(value) or wp.abs(value) > 1.0e-8:
                    reason = -6
            residuals[0] = maxnormal
            residuals[1] = maxtangent
            residuals[2] = maxjoint
            residuals[3] = minspec
            status[0] = reason
            if reason == 1:
                ground_force = wp.vec3f(0.0)
                ground_torque = wp.vec3f(0.0)
                for k in range(interval[0], interval[0] + interval[1]):
                    if eligible[k] != 0:
                        n = wp.vec3f(cc.lambdas[0, k], cc.lambdas[1, k], cc.lambdas[2, k])
                        t = wp.vec3f(cc.lambdas[3, k], cc.lambdas[4, k], cc.lambdas[5, k])
                        f = (
                            (candidate[0, k] - cc.impulses[0, k]) * n
                            + (candidate[1, k] - cc.impulses[1, k]) * t
                            + (candidate[2, k] - cc.impulses[2, k]) * wp.cross(n, t)
                        )
                        rg = wp.vec3f(cc.derived[12, k], cc.derived[13, k], cc.derived[14, k])
                        ground_force += f
                        ground_torque += wp.cross(rg, f)
                        for r in range(3):
                            cc.impulses[r, k] = candidate[r, k]
                for d in range(3):
                    history[event, 0, 19 + d] = ground_force[d]
                    history[event, 0, 22 + d] = ground_torque[d]
                    history[event, 1, 19 + d] = response.impulse[articulation, selection[2]][d]
                    history[event, 1, 22 + d] = response.impulse[articulation, selection[2]][3 + d]
        _sync_tree()
        if status[0] == 1 and lane < tree.body_count[articulation]:
            body = tree.body_slot[articulation, lane]
            dv = response.velocity[articulation, lane]
            bodies.velocity[body] += wp.spatial_top(dv)
            bodies.angular_velocity[body] += wp.spatial_bottom(dv)
            row = tree.dynamic_row[articulation, lane]
            if row >= 0:
                accumulated[row] -= (
                    tree.generalized_mass[articulation, lane] * response.joint_velocity[articulation, lane]
                )
    _sync_tree()
    if lane < 4:
        for d in range(3):
            history[event, lane, 13 + d] = bodies.velocity[lane][d]
            history[event, lane, 16 + d] = bodies.angular_velocity[lane][d]
    if lane == 0:
        if status[0] == 1:
            counters[1] += 1
        else:
            counters[-status[0] + 2] += 1
        counter[0] += 1


def install():
    states = []
    constructor = SolverPhoenX.__init__

    def construct(solver, *args, **kwargs):
        constructor(solver, *args, **kwargs)
        w = solver.world
        assert solver._direct_tree_contacts and w.bodies.position.shape[0] == 4 and w.sor_boost == 1
        cc = w._contact_container
        cap = cc.lambdas.shape[1]
        device = cc.lambdas.device
        direct = solver._direct_equality_system
        assert direct.accumulated_impulse.shape[0] == 6 and direct.row_wrench0.shape == (1, 6)
        data = {
            "solver": solver,
            "mobility": wp.zeros(1, dtype=wp.spatial_matrixf, device=device),
            "velocity": wp.zeros(1, dtype=wp.spatial_vectorf, device=device),
            "maps": wp.zeros((cap, 3), dtype=wp.spatial_vectorf, device=device),
            "coefficients": wp.zeros(cap, dtype=float, device=device),
            "eligible": wp.zeros(cap, dtype=int, device=device),
            "interval": wp.zeros(2, dtype=int, device=device),
            "selection": wp.zeros(3, dtype=int, device=device),
            "candidate": wp.zeros((3, cap), dtype=float, device=device),
            "status": wp.zeros(1, dtype=int, device=device),
            "requested": wp.zeros(1, dtype=wp.spatial_vectorf, device=device),
            "counters": wp.zeros(11, dtype=int, device=device),
            "residuals": wp.zeros(4, dtype=float, device=device),
            "history": wp.zeros((64, 4, 25), dtype=float, device=device),
            "counter": wp.zeros(1, dtype=int, device=device),
        }
        states.append(data)
        original = w._solve_maximal_articulated_contacts

        def solve(*, use_bias, refresh_mobility):
            original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            if not use_bias:
                response = w._maximal_contact_response.data
                tree = w._maximal_tree_projector.data
                cc = w._contact_container
                cols = w._contact_cols
                wp.launch(
                    prepare,
                    dim=1,
                    inputs=[
                        w.bodies,
                        response,
                        cols,
                        cc,
                        data["mobility"],
                        data["velocity"],
                        data["maps"],
                        data["coefficients"],
                        data["eligible"],
                        data["interval"],
                        data["selection"],
                    ],
                    device=device,
                )
                wp.launch(
                    propose,
                    dim=1,
                    inputs=[
                        data["mobility"],
                        data["velocity"],
                        data["maps"],
                        cc.impulses,
                        data["coefficients"],
                        data["eligible"],
                        data["interval"],
                        data["candidate"],
                        data["status"],
                        data["requested"],
                    ],
                    device=device,
                )
                if os.environ.get("COLIBRI_FULL_WRENCH_FORCE_REJECT") == "1":
                    wp.launch(force_reject, dim=1, inputs=[data["status"]], device=device)
                wp.launch(
                    verify_apply,
                    dim=64,
                    block_dim=64,
                    inputs=[
                        tree,
                        response,
                        w.bodies,
                        cc,
                        cols,
                        data["candidate"],
                        data["eligible"],
                        data["interval"],
                        data["selection"],
                        data["status"],
                        data["maps"],
                        direct.row_wrench0,
                        direct.row_wrench1,
                        direct.row_dynamic,
                        direct.dynamic_mass,
                        direct.velocity_reference,
                        direct.accumulated_impulse,
                        data["counters"],
                        data["residuals"],
                        data["history"],
                        data["counter"],
                    ],
                    device=device,
                )

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
            n = int(data["counter"].numpy()[0])
            history = data["history"].numpy()[np.arange(max(0, n - 64), n) % 64]
            np.savez_compressed(
                output.with_suffix(".full_wrench.npz"),
                history=history,
                counters=data["counters"].numpy(),
                residuals=data["residuals"].numpy(),
                body_mass=data["solver"].model.body_mass.numpy(),
                body_inertia=data["solver"].model.body_inertia.numpy(),
                candidate=data["candidate"].numpy(),
                mobility=data["mobility"].numpy(),
                status=data["status"].numpy(),
            )
            output.with_suffix(".full_wrench.json").write_text(
                json.dumps(
                    {
                        "scope": __doc__,
                        "counters": data["counters"].numpy().tolist(),
                        "counter_names": [
                            "attempt",
                            "accepted",
                            "unknown",
                            "no_load",
                            "mobility",
                            "cone",
                            "normal",
                            "tangent",
                            "joint_drive",
                            "predictive",
                            "unsupported",
                        ],
                        "last_residuals": data["residuals"].numpy().tolist(),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
