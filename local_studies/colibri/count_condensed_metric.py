"""CPU attribution of the exact native metric helper during condensed PGS."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_d6_frozen_rows import contact_sweep
from local_studies.colibri.coupled_support_online import assemble_snapshot


def _make_counted_metric(scalar_type):
    relative_tolerance = 2.0e-7 if scalar_type == wp.float32 else 1.0e-7

    @wp.func
    def impl(
        mobility_t1: wp.float32,
        mobility_t1t2: wp.float32,
        mobility_t2: wp.float32,
        rhs_t1: wp.float32,
        rhs_t2: wp.float32,
        lambda_t1_old: wp.float32,
        lambda_t2_old: wp.float32,
        static_radius: wp.float32,
        dynamic_radius: wp.float32,
    ) -> wp.vec4f:
        """Minimize tangent kinetic energy within the Coulomb disk.

        Sliding friction must oppose the resulting slip velocity. For an
        anisotropic mobility, radial clamping of the unconstrained impulse
        does not satisfy this condition. The disk boundary minimizer solves
        (K + alpha I) lambda = K lambda_old - velocity, with alpha >= 0.
        """
        scale = wp.max(mobility_t1, mobility_t2)
        result = wp.vec2f(0.0)
        root_iterations = wp.int32(0)
        kind = wp.int32(0)
        if scale > wp.float32(0.0) and static_radius > wp.float32(0.0):
            kind = wp.int32(1)
            a = scalar_type(mobility_t1) / scalar_type(scale)
            c = scalar_type(mobility_t2) / scalar_type(scale)
            b = scalar_type(mobility_t1t2) / scalar_type(scale)
            v1 = scalar_type(rhs_t1) / scalar_type(scale)
            v2 = scalar_type(rhs_t2) / scalar_type(scale)
            target1 = a * scalar_type(lambda_t1_old) + b * scalar_type(lambda_t2_old) - v1
            target2 = b * scalar_type(lambda_t1_old) + c * scalar_type(lambda_t2_old) - v2
            determinant = scalar_type(a) * scalar_type(c) - scalar_type(b) * scalar_type(b)
            if determinant > scalar_type(0.0):
                result = wp.vec2f(
                    wp.float32(
                        (scalar_type(c) * scalar_type(target1) - scalar_type(b) * scalar_type(target2)) / determinant
                    ),
                    wp.float32(
                        (scalar_type(a) * scalar_type(target2) - scalar_type(b) * scalar_type(target1)) / determinant
                    ),
                )
            else:
                # A rank-one mobility has one movable tangent direction.
                inverse_trace_squared = scalar_type(1.0) / ((a + c) * (a + c))
                result = wp.vec2f(
                    wp.float32((a * target1 + b * target2) * inverse_trace_squared),
                    wp.float32((b * target1 + c * target2) * inverse_trace_squared),
                )
            if wp.length_sq(result) > static_radius * static_radius:
                kind = wp.int32(2)
                result = wp.vec2f(0.0)
                if dynamic_radius > wp.float32(0.0):
                    lower = scalar_type(0.0)
                    radius = scalar_type(dynamic_radius)
                    target_x = scalar_type(target1)
                    target_y = scalar_type(target2)
                    upper = wp.sqrt(target_x * target_x + target_y * target_y) / radius
                    alpha = upper
                    old_x = scalar_type(lambda_t1_old)
                    old_y = scalar_type(lambda_t2_old)
                    old_length_sq = old_x * old_x + old_y * old_y
                    if old_length_sq > scalar_type(0.0):
                        # A previous sliding impulse gives a coherent KKT
                        # multiplier estimate. It only initializes the root
                        # search; the same bracket and accuracy still apply.
                        old_length = wp.sqrt(old_length_sq)
                        old_k_old = old_x * (a * old_x + b * old_y) + old_y * (b * old_x + c * old_y)
                        estimate = (old_x * target_x + old_y * target_y) / (
                            radius * old_length
                        ) - old_k_old / old_length_sq
                        if estimate > lower and estimate < upper:
                            alpha = estimate
                    # The norm bound supplies a feasible upper bracket. Newton
                    # usually converges in a few iterations; bisection safeguards
                    # nearly singular blocks without changing their mobility.
                    for _iteration in range(40):
                        root_iterations += wp.int32(1)
                        aa = scalar_type(a) + alpha
                        cc = scalar_type(c) + alpha
                        bb = scalar_type(b)
                        det = aa * cc - bb * bb
                        x = (cc * target_x - bb * target_y) / det
                        y = (aa * target_y - bb * target_x) / det
                        length = wp.sqrt(x * x + y * y)
                        error = length - radius
                        # Newton approaches this convex decreasing norm from
                        # below its root after the first step. Accept convergence
                        # from either side; otherwise positive roundoff can force
                        # all forty iterations despite an accurate solution.
                        if wp.abs(error) <= scalar_type(relative_tolerance) * radius:
                            boundary_scale = radius / length
                            result = wp.vec2f(wp.float32(x * boundary_scale), wp.float32(y * boundary_scale))
                            break
                        if error > scalar_type(0.0):
                            lower = alpha
                        else:
                            upper = alpha
                            result = wp.vec2f(wp.float32(x), wp.float32(y))
                        derivative = -(cc * x * x - scalar_type(2.0) * bb * x * y + aa * y * y) / (det * length)
                        proposal = alpha - error / derivative
                        if proposal <= lower or proposal >= upper:
                            proposal = scalar_type(0.5) * (lower + upper)
                        alpha = proposal
        if wp.static(scalar_type == wp.float64):
            kind += wp.int32(4)
        return wp.vec4f(result[0], result[1], wp.float32(root_iterations), wp.float32(kind))

    return impl


_counted_metric_float32 = _make_counted_metric(wp.float32)
_counted_metric_float64 = _make_counted_metric(wp.float64)


@wp.func
def counted_metric(
    mobility_t1: wp.float32,
    mobility_t1t2: wp.float32,
    mobility_t2: wp.float32,
    rhs_t1: wp.float32,
    rhs_t2: wp.float32,
    lambda_t1_old: wp.float32,
    lambda_t2_old: wp.float32,
    static_radius: wp.float32,
    dynamic_radius: wp.float32,
) -> wp.vec4f:
    """Use float64 only when the tangent mobility is ill-conditioned."""
    scale = wp.max(mobility_t1, mobility_t2)
    well_conditioned = wp.bool(False)
    if scale > wp.float32(0.0):
        a = mobility_t1 / scale
        b = mobility_t1t2 / scale
        c = mobility_t2 / scale
        # Bound the determinant cancellation before choosing float32.
        # Tiny positive eigenvalues retain the precise physical solve.
        well_conditioned = a * c - b * b > wp.float32(0.01) * (a + c) * (a + c)
    if well_conditioned:
        return _counted_metric_float32(
            mobility_t1,
            mobility_t1t2,
            mobility_t2,
            rhs_t1,
            rhs_t2,
            lambda_t1_old,
            lambda_t2_old,
            static_radius,
            dynamic_radius,
        )
    return _counted_metric_float64(
        mobility_t1,
        mobility_t1t2,
        mobility_t2,
        rhs_t1,
        rhs_t2,
        lambda_t1_old,
        lambda_t2_old,
        static_radius,
        dynamic_radius,
    )


@wp.kernel(enable_backward=False)
def counted_sweep(
    c: wp.array2d[wp.float64],
    wct: wp.array2d[wp.float64],
    h: wp.array2d[wp.float64],
    bias: wp.array[wp.float64],
    gamma: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    first: int,
    count: int,
    size: int,
    stats: wp.array[wp.int32],
):
    """Apply original-order normal and production metric tangent updates."""
    for p in range(first, count):
        row = 3 * p
        vn = bias[row]
        for j in range(size):
            vn += c[row, j] * velocity[j]
        oldn = lam[row]
        newn = wp.max(wp.float64(0.0), oldn - (vn + gamma[p] * oldn) / (h[row, row] + gamma[p]))
        dn = newn - oldn
        lam[row] = newn
        for j in range(size):
            velocity[j] += wct[j, row] * dn
        vt1 = bias[row + 1]
        vt2 = bias[row + 2]
        for j in range(size):
            vt1 += c[row + 1, j] * velocity[j]
            vt2 += c[row + 2, j] * velocity[j]
        old1 = lam[row + 1]
        old2 = lam[row + 2]
        radius = wp.float32(mu[p] * newn)
        tangent = counted_metric(
            wp.float32(h[row + 1, row + 1]),
            wp.float32(h[row + 1, row + 2]),
            wp.float32(h[row + 2, row + 2]),
            wp.float32(vt1),
            wp.float32(vt2),
            wp.float32(old1),
            wp.float32(old2),
            radius,
            radius,
        )
        kind = wp.int32(tangent[3])
        stats[kind] += 1
        precision = kind // 4
        stats[8 + precision * 41 + wp.int32(tangent[2])] += 1
        lam[row + 1] = wp.float64(tangent[0])
        lam[row + 2] = wp.float64(tangent[1])
        for j in range(size):
            velocity[j] += wct[j, row + 1] * (lam[row + 1] - old1) + wct[j, row + 2] * (lam[row + 2] - old2)


def main():
    """Require byte-identical original/instrumented outputs at every sweep."""
    wp.init()
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    reports = []
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        c, g = a["C"], a["P"] @ a["C"].T
        h = c @ g
        start = a["vbar"] + g @ a["old"]
        arrays = [
            wp.array(x, dtype=wp.float64, device="cpu")
            for x in (c, g, h, a["rhs"] - c @ a["vbar"], a["gamma"], a["mu"])
        ]
        la = wp.array(a["old"], dtype=wp.float64, device="cpu")
        va = wp.array(start, dtype=wp.float64, device="cpu")
        lb = wp.array(a["old"], dtype=wp.float64, device="cpu")
        vb = wp.array(start, dtype=wp.float64, device="cpu")
        stats = wp.zeros(90, dtype=wp.int32, device="cpu")
        for iteration in range(32):
            wp.launch(contact_sweep, dim=1, inputs=[*arrays, la, va, 0, len(a["mu"]), 12], device="cpu")
            wp.launch(counted_sweep, dim=1, inputs=[*arrays, lb, vb, 0, len(a["mu"]), 12, stats], device="cpu")
            assert la.numpy().tobytes() == lb.numpy().tobytes()
            assert va.numpy().tobytes() == vb.numpy().tobytes()
        count = stats.numpy()
        precision = []
        for fp in range(2):
            hist = count[8 + 41 * fp : 8 + 41 * (fp + 1)]
            precision.append(
                dict(
                    bits=32 + 32 * fp,
                    calls=int(count[4 * fp : 4 * fp + 3].sum()),
                    inactive=int(count[4 * fp]),
                    sticking=int(count[4 * fp + 1]),
                    sliding=int(count[4 * fp + 2]),
                    root_iterations_histogram={str(i): int(n) for i, n in enumerate(hist) if n},
                    root_iterations_total=int(np.arange(41) @ hist),
                    maximum_iterations=int(np.flatnonzero(hist)[-1]) if hist.any() else 0,
                )
            )
        assert sum(p["calls"] for p in precision) == 32 * len(a["mu"])
        reports.append(
            dict(
                phase=phase,
                sweeps=32,
                points=len(a["mu"]),
                precision=precision,
                every_sweep_impulse_velocity_byte_equal=True,
            )
        )
    path = Path("/tmp/colibri_condensed_metric_counts.json")
    path.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
