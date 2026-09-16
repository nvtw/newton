"""Bounded GPU contact PGS on exact six-row joint-conditioned response.

Reference setup uses host arrays. GPU stores 12-by-contact-row response and
six point-local coefficients, never a dense contact-by-contact mobility.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot
from local_studies.colibri.single_block_support_reference import coupled_response
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric

vec12d = wp.types.vector(length=12, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def prepare_local_blocks(
    c: wp.array2d[wp.float64], g: wp.array2d[wp.float64], blocks: wp.array2d[wp.float64], count: int
):
    """Form only each point's exact condensed normal and tangent responses."""
    point = wp.tid()
    if point < count:
        for i in range(3):
            for j in range(3):
                value = wp.float64(0)
                for k in range(12):
                    value += c[3 * point + i, k] * g[k, 3 * point + j]
                blocks[point, 3 * i + j] = value


@wp.kernel(enable_backward=False)
def condensed_sweeps(
    c: wp.array2d[wp.float64],
    g: wp.array2d[wp.float64],
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
    """Sequential original-order PGS; physical responses preserve all joint rows."""
    v = vec12d(wp.float64(0))
    for i in range(12):
        value = baseline[i]
        for row in range(3 * count):
            value += g[i, row] * impulse[row]
        v[i] = value
    for _sweep in range(sweeps):
        for point in range(count):
            row = 3 * point
            vn = bias[row]
            for i in range(12):
                vn += c[row, i] * v[i]
            oldn = impulse[row]
            newn = wp.max(wp.float64(0), oldn - (vn + gamma[point] * oldn) / (blocks[point, 0] + gamma[point]))
            impulse[row] = newn
            for i in range(12):
                v[i] += g[i, row] * (newn - oldn)
            vt1 = bias[row + 1]
            vt2 = bias[row + 2]
            for i in range(12):
                vt1 += c[row + 1, i] * v[i]
                vt2 += c[row + 2, i] * v[i]
            old1 = impulse[row + 1]
            old2 = impulse[row + 2]
            radius = wp.float32(mu[point] * newn)
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
            for i in range(12):
                v[i] += g[i, row + 1] * (impulse[row + 1] - old1) + g[i, row + 2] * (impulse[row + 2] - old2)
    for i in range(12):
        velocity[i] = v[i]


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
    if n:
        wp.launch(prepare_local_blocks, dim=n, inputs=[c_gpu, g, blocks, n], device="cuda:0")
    wp.launch(
        condensed_sweeps,
        dim=1,
        inputs=[
            c_gpu,
            g,
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


def main():
    """Gate frozen physical responses before any live integration or speed claim."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/colibri_two_body_condensed_gpu.json")
    args = parser.parse_args()
    wp.init()
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    records = []
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        for sweeps in (16, 32):
            lam, v, joint, gpu = solve(a, sweeps)
            evaluate, *_ = natural_map_evaluator(a["A"], a["rhs"], a["gamma"], a["mu"])
            residual = float(np.max(abs(evaluate(lam)[0])))
            joint_error = float(np.max(abs(a["B"] @ v + a["diagonal"] * joint - a["targets"])))
            physical = a["free"] + a["W"] @ (a["C"].T @ lam + a["B"].T @ joint)
            response_error = float(np.max(abs(v - physical)))
            assert joint_error < 1e-9 and response_error < 1e-9
            assert residual < (2e-6 if sweeps == 16 else 1e-8)
            records.append(
                {
                    "phase": phase,
                    "sweeps": sweeps,
                    "contact_residual": residual,
                    "joint_residual": joint_error,
                    "response_error": response_error,
                }
            )
            np.savez(
                Path(args.output).with_suffix("." + phase + str(sweeps) + ".npz"),
                lam=lam,
                velocity=v,
                joint=joint,
                **gpu,
            )
    Path(args.output).write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
