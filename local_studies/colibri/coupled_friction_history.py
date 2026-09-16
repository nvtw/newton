"""Evaluate the final coupled friction trial using the installed history overlay."""

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric_with_break


@wp.kernel(enable_backward=False)
def _trials(
    block: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
    impulse: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    result: wp.array2d[wp.float32],
):
    """Return the actual static-break branch, including a transition back to stick."""
    p = wp.tid()
    r = 3 * p
    radius = wp.float32(mu[p] * impulse[r])
    value = contact_project_friction_metric_with_break(
        wp.float32(block[p, 0]),
        wp.float32(block[p, 1]),
        wp.float32(block[p, 2]),
        wp.float32(rhs[r + 1]),
        wp.float32(rhs[r + 2]),
        wp.float32(impulse[r + 1]),
        wp.float32(impulse[r + 2]),
        radius,
        radius,
    )
    for j in range(3):
        result[p, j] = value[j]


def final_break_flags(a, lam, velocity):
    """Preserve the original material law; classify only eligible owned points."""
    count = len(a["mu"])
    if not count:
        return np.zeros(0, dtype=np.float32)
    h = a["C"] @ a["P"] @ a["C"].T
    block = np.array(
        [[h[3 * p + 1, 3 * p + 1], h[3 * p + 1, 3 * p + 2], h[3 * p + 2, 3 * p + 2]] for p in range(count)]
    )
    rhs = a["rhs"] - a["C"] @ a["vbar"] + a["C"] @ velocity
    arrays = [wp.array(x, dtype=wp.float64, device="cpu") for x in (block, rhs, lam, a["mu"])]
    result = wp.zeros((count, 3), dtype=wp.float32, device="cpu")
    wp.launch(_trials, dim=count, inputs=[*arrays, result], device="cpu")
    return result.numpy()[:, 2]
