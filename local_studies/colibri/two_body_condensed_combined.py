"""Combine validated inactive-contact shortcut and fixed FP64 reduction tree."""

import numpy as np
import warp as wp

from local_studies.colibri.single_block_support_reference import coupled_response
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source_lane, 32);
#else
    return value;
#endif
""")
def shuffle(value: wp.float64, source_lane: wp.int32) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    value += __shfl_down_sync(0xffffffffu, value, 8, 16);
    value += __shfl_down_sync(0xffffffffu, value, 4, 16);
    value += __shfl_down_sync(0xffffffffu, value, 2, 16);
    value += __shfl_down_sync(0xffffffffu, value, 1, 16);
#endif
    return value;
""")
def reduce16(value: wp.float64) -> wp.float64: ...


@wp.kernel(enable_backward=False)
def prepare_packed(
    c: wp.array2d[wp.float64],
    g: wp.array2d[wp.float64],
    blocks: wp.array2d[wp.float64],
    cp: wp.array3d[wp.float64],
    gp: wp.array3d[wp.float64],
    count: int,
):
    """Pack point-local physical DOFs without changing mobility arithmetic."""
    point = wp.tid()
    if point < count:
        for i in range(3):
            for dof in range(12):
                cp[point, i, dof] = c[3 * point + i, dof]
                gp[point, i, dof] = g[dof, 3 * point + i]
            for j in range(3):
                value = wp.float64(0)
                for dof in range(12):
                    value += c[3 * point + i, dof] * g[dof, 3 * point + j]
                blocks[point, 3 * i + j] = value


@wp.kernel(enable_backward=False)
def warp_sweeps(
    c: wp.array3d[wp.float64],
    g: wp.array3d[wp.float64],
    blocks: wp.array2d[wp.float64],
    baseline: wp.array[wp.float64],
    bias: wp.array[wp.float64],
    gamma: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    impulse: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    count: int,
    sweeps: int,
):
    """One lane per velocity DOF; lane zero projects each successive contact."""
    lane = wp.tid()
    v = wp.float64(0)
    if lane < 12:
        v = baseline[lane]
        for row in range(3 * count):
            v += g[row // 3, row % 3, lane] * impulse[row]
    for _sweep in range(sweeps):
        for point in range(count):
            row = 3 * point
            cn = wp.float64(0)
            ct1 = wp.float64(0)
            ct2 = wp.float64(0)
            gn = wp.float64(0)
            gt1 = wp.float64(0)
            gt2 = wp.float64(0)
            if lane < 12:
                cn = c[point, 0, lane]
            vn = bias[row]
            vn += reduce16(cn * v)
            inactive = wp.float64(0)
            if lane == 0:
                if (
                    vn >= wp.float64(0)
                    and impulse[row] == wp.float64(0)
                    and impulse[row + 1] == wp.float64(0)
                    and impulse[row + 2] == wp.float64(0)
                ):
                    inactive = wp.float64(1)
            inactive = shuffle(inactive, 0)
            if inactive > wp.float64(0):
                continue
            if lane < 12:
                ct1 = c[point, 1, lane]
                ct2 = c[point, 2, lane]
                gn = g[point, 0, lane]
                gt1 = g[point, 1, lane]
                gt2 = g[point, 2, lane]
            dn = wp.float64(0)
            if lane == 0:
                oldn = impulse[row]
                newn = wp.max(wp.float64(0), oldn - (vn + gamma[point] * oldn) / (blocks[point, 0] + gamma[point]))
                impulse[row] = newn
                dn = newn - oldn
            dn = shuffle(dn, 0)
            v += gn * dn
            vt1 = bias[row + 1]
            vt2 = bias[row + 2]
            vt1 += reduce16(ct1 * v)
            vt2 += reduce16(ct2 * v)
            delta1 = wp.float64(0)
            delta2 = wp.float64(0)
            if lane == 0:
                old1 = impulse[row + 1]
                old2 = impulse[row + 2]
                radius = wp.float32(mu[point] * impulse[row])
                tangent = contact_project_friction_metric(
                    wp.float32(blocks[point, 4]),
                    wp.float32(blocks[point, 5]),
                    wp.float32(blocks[point, 8]),
                    wp.float32(vt1),
                    wp.float32(vt2),
                    wp.float32(old1),
                    wp.float32(old2),
                    radius,
                    radius,
                )
                impulse[row + 1] = wp.float64(tangent[0])
                impulse[row + 2] = wp.float64(tangent[1])
                delta1 = impulse[row + 1] - old1
                delta2 = impulse[row + 2] - old2
            delta1 = shuffle(delta1, 0)
            delta2 = shuffle(delta2, 0)
            v += gt1 * delta1 + gt2 * delta2
    if lane < 12:
        velocity[lane] = v


def solve(a, sweeps=32):
    """Use the existing 128-lane Cholesky response then bounded contact sweeps."""
    w, b, c = a["W"], a["B"], a["C"]
    n = len(a["mu"])
    rows = 3 * n
    assert w.shape == (12, 12) and b.shape == (6, 12) and n <= 128

    def arr(x):
        return wp.array(np.asarray(x, dtype=np.float64), dtype=wp.float64, device="cuda:0")

    rhs = np.column_stack([a["targets"] - b @ a["free"], b @ w @ c.T])
    y = wp.zeros(rhs.shape, dtype=wp.float64, device="cuda:0")
    g = wp.zeros((12, rows), dtype=wp.float64, device="cuda:0")
    base = wp.zeros(12, dtype=wp.float64, device="cuda:0")
    velocity = wp.zeros(12, dtype=wp.float64, device="cuda:0")
    impulse = arr(a["old"])
    c_gpu = arr(c)
    wp.launch(
        coupled_response,
        dim=128,
        block_dim=128,
        inputs=[
            arr(a["K"]),
            arr(rhs),
            arr(w @ c.T),
            arr(w @ b.T),
            arr(a["free"]),
            c_gpu,
            arr(np.zeros(rows)),
            -1,
            rows,
            wp.zeros((6, 6), dtype=wp.float64, device="cuda:0"),
            y,
            g,
            base,
            arr(np.r_[a["old"], 0.0]),
            velocity,
        ],
        device="cuda:0",
    )
    blocks = wp.zeros((n, 9), dtype=wp.float64, device="cuda:0")
    packed_c = wp.zeros((n, 3, 12), dtype=wp.float64, device="cuda:0")
    packed_g = wp.zeros((n, 3, 12), dtype=wp.float64, device="cuda:0")
    if n:
        wp.launch(prepare_packed, dim=n, inputs=[c_gpu, g, blocks, packed_c, packed_g, n], device="cuda:0")
    wp.launch(
        warp_sweeps,
        dim=32,
        block_dim=32,
        inputs=[
            packed_c,
            packed_g,
            blocks,
            base,
            arr(a["rhs"] - c @ a["vbar"]),
            arr(a["gamma"]),
            arr(a["mu"]),
            impulse,
            velocity,
            n,
            sweeps,
        ],
        device="cuda:0",
    )
    lam = impulse.numpy()
    v = velocity.numpy()
    ys = y.numpy()
    joint = ys[:, 0] - ys[:, 1:] @ lam
    return lam, v, joint, {"response": g.numpy(), "baseline": base.numpy(), "joint_solved": ys}
