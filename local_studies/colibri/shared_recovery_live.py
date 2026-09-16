"""Target-only FP32 projection; native point cones, geometry and ordering unchanged.

Weights are preceding completed solve normal impulses in the SAME contact generation.
Every ingest invalidates them. Unknown/ill-conditioned/no-load projections retain
original recovery. This changes numerical stabilization, not physical response rank.
"""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton.solvers import SolverPhoenX


@wp.kernel
def invalidate(valid: wp.array[int]):
    valid[0] = 0


@wp.kernel
def remember(cc: ContactContainer, weights: wp.array[float], valid: wp.array[int]):
    k = wp.tid()
    weights[k] = wp.max(cc.impulses[0, k], wp.float32(0.0))
    if k == 0:
        valid[0] = 1


@wp.func
def field(t: wp.vec3f, a: wp.vec3f, b: wp.vec3f, spin: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(wp.dot(t, a), wp.dot(t, b), wp.dot(t, spin))


@wp.kernel
def project_targets(
    cc: ContactContainer,
    columns: ContactColumnContainer,
    weights: wp.array[float],
    valid: wp.array[int],
    saved: wp.array2d[float],
    diagnostics: wp.array2d[float],
):
    col = wp.tid()
    diagnostics[0, col] = 0.0
    first = contact_get_contact_first(columns, col)
    count = contact_get_contact_count(columns, col)
    if count > 0 and first >= 0 and first + count <= weights.shape[0]:
        for k in range(first, first + count):
            saved[0, k] = cc.derived[4, k]
            saved[1, k] = cc.derived[5, k]
        mu = columns.data[3, col]
        if valid[0] != 0 and mu > 0.0:
            total = wp.float32(0.0)
            center = wp.vec3f(0.0)
            for k in range(first, first + count):
                r = wp.vec3f(cc.derived[9, k], cc.derived[10, k], cc.derived[11, k])
                total += weights[k]
                center += weights[k] * r
            if total > 0.0:
                center = center / total
                normal = wp.vec3f(cc.lambdas[0, first], cc.lambdas[1, first], cc.lambdas[2, first])
                a = wp.normalize(wp.vec3f(cc.lambdas[3, first], cc.lambdas[4, first], cc.lambdas[5, first]))
                b = wp.cross(normal, a)
                gram = wp.mat33f(0.0)
                rhs = wp.vec3f(0.0)
                compatible = wp.bool(True)
                for k in range(first, first + count):
                    n = wp.vec3f(cc.lambdas[0, k], cc.lambdas[1, k], cc.lambdas[2, k])
                    if n[0] != normal[0] or n[1] != normal[1] or n[2] != normal[2]:
                        compatible = False
                    t = wp.vec3f(cc.lambdas[3, k], cc.lambdas[4, k], cc.lambdas[5, k])
                    r = wp.vec3f(cc.derived[9, k], cc.derived[10, k], cc.derived[11, k])
                    spin = wp.cross(normal, r - center)
                    f0 = field(t, a, b, spin)
                    f1 = field(wp.cross(n, t), a, b, spin)
                    w = weights[k] / total
                    gram += w * (wp.outer(f0, f0) + wp.outer(f1, f1))
                    rhs += w * (f0 * saved[0, k] + f1 * saved[1, k])
                diagnostics[0, col] = -1.0
                determinant = wp.determinant(gram)
                if compatible and determinant > 0.0:
                    inverse = wp.inverse(gram)
                    norm_g = wp.float32(0.0)
                    norm_inv = wp.float32(0.0)
                    for row in range(3):
                        norm_g = wp.max(norm_g, wp.abs(gram[row, 0]) + wp.abs(gram[row, 1]) + wp.abs(gram[row, 2]))
                        norm_inv = wp.max(
                            norm_inv, wp.abs(inverse[row, 0]) + wp.abs(inverse[row, 1]) + wp.abs(inverse[row, 2])
                        )
                    condition = norm_g * norm_inv
                    coefficient = inverse * rhs
                    residual = gram * coefficient - rhs
                    scale = norm_g * wp.length(coefficient) + wp.length(rhs)
                    error = wp.length(residual)
                    diagnostics[1, col] = condition
                    diagnostics[2, col] = error
                    # Numerical projection accuracy gate only; fallback changes NO physical modes.
                    if (
                        wp.isfinite(condition)
                        and condition * wp.float32(64.0 * 1.1920928955078125e-7) < 0.01
                        and error <= wp.float32(64.0 * 1.1920928955078125e-7) * scale
                    ):
                        change = wp.float32(0.0)
                        for k in range(first, first + count):
                            n = wp.vec3f(cc.lambdas[0, k], cc.lambdas[1, k], cc.lambdas[2, k])
                            t = wp.vec3f(cc.lambdas[3, k], cc.lambdas[4, k], cc.lambdas[5, k])
                            r = wp.vec3f(cc.derived[9, k], cc.derived[10, k], cc.derived[11, k])
                            spin = wp.cross(normal, r - center)
                            target0 = wp.dot(field(t, a, b, spin), coefficient)
                            target1 = wp.dot(field(wp.cross(n, t), a, b, spin), coefficient)
                            change = wp.max(
                                change, wp.max(wp.abs(target0 - saved[0, k]), wp.abs(target1 - saved[1, k]))
                            )
                            cc.derived[4, k] = target0
                            cc.derived[5, k] = target1
                        diagnostics[0, col] = 1.0
                        diagnostics[3, col] = change
                        diagnostics[4, col] = total


@wp.kernel
def save_targets(cc: ContactContainer, saved: wp.array2d[float]):
    k = wp.tid()
    saved[0, k] = cc.derived[4, k]
    saved[1, k] = cc.derived[5, k]


@wp.kernel
def restore_targets(cc: ContactContainer, saved: wp.array2d[float]):
    k = wp.tid()
    cc.derived[4, k] = saved[0, k]
    cc.derived[5, k] = saved[1, k]


def install():
    states = []
    constructor = SolverPhoenX.__init__

    def construct(solver, *args, **kwargs):
        constructor(solver, *args, **kwargs)
        w = solver.world
        assert solver._direct_tree_contacts and w.solver_iterations == 1 and w.sor_boost == 1
        cc = w._contact_container
        device = cc.lambdas.device
        cap = cc.lambdas.shape[1]
        cols = w._contact_cols.data.shape[1]
        data = {
            "solver": solver,
            "weights": wp.zeros(cap, dtype=float, device=device),
            "valid": wp.zeros(1, dtype=int, device=device),
            "saved": wp.zeros((2, cap), dtype=float, device=device),
            "diagnostics": wp.zeros((5, cols), dtype=float, device=device),
        }
        states.append(data)
        ingest = w._ingest_and_warmstart_contacts

        def refresh(*a, **kw):
            result = ingest(*a, **kw)
            wp.launch(invalidate, dim=1, inputs=[data["valid"]], device=device)
            return result

        w._ingest_and_warmstart_contacts = refresh
        original = w._solve_maximal_articulated_contacts

        def solve(*, use_bias, refresh_mobility):
            current = w._contact_container
            if use_bias:
                wp.launch(save_targets, dim=cap, inputs=[current, data["saved"]], device=device)
                wp.launch(
                    project_targets,
                    dim=cols,
                    inputs=[
                        current,
                        w._contact_cols,
                        data["weights"],
                        data["valid"],
                        data["saved"],
                        data["diagnostics"],
                    ],
                    device=device,
                )
            original(use_bias=use_bias, refresh_mobility=refresh_mobility)
            if use_bias:
                wp.launch(restore_targets, dim=cap, inputs=[current, data["saved"]], device=device)
            wp.launch(remember, dim=cap, inputs=[current, data["weights"], data["valid"]], device=device)

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
            np.savez_compressed(
                output.with_suffix(".shared_recovery.npz"),
                weights=data["weights"].numpy(),
                diagnostics=data["diagnostics"].numpy(),
            )
        output.with_suffix(".shared_recovery.json").write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "weights": "Previous completed solve, invalidated every ingest",
                    "physics": "Native current normal ordering, per-point cones, common-point metric projection and history reset unchanged",
                    "conditioning": "FP32 condition*64eps<.01 and backward residual gate; otherwise original targets",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
