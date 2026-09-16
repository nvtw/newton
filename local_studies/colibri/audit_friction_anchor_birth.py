"""CPU replay of actual newborn material anchors and independent FP64 reconstruction."""

import json
import os
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def replay(
    pos: wp.array[wp.vec3],
    q: wp.array[wp.quat],
    com: wp.array[wp.vec3],
    bodies: wp.array[wp.vec2i],
    a: wp.array2d[wp.float32],
    result: wp.array2d[wp.float32],
):
    k = wp.tid()
    i = bodies[k][0]
    j = bodies[k][1]
    p0 = pos[i] + wp.quat_rotate(q[i], wp.vec3(a[6, k], a[7, k], a[8, k]) - com[i])
    p1 = pos[j] + wp.quat_rotate(q[j], wp.vec3(a[9, k], a[10, k], a[11, k]) - com[j])
    n = wp.vec3(a[0, k], a[1, k], a[2, k])
    t = wp.vec3(a[3, k], a[4, k], a[5, k])
    delta = p1 - p0
    result[k, 0] = wp.float32(0.08) * wp.dot(delta, t) * wp.float32(3600.0)
    result[k, 1] = wp.float32(0.08) * wp.dot(delta, wp.cross(n, t)) * wp.float32(3600.0)
    for d in range(3):
        result[k, d + 2] = delta[d]


def rotation(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    device = os.environ.get("ANCHOR_REPLAY_DEVICE", "cpu")
    x = np.load("/tmp/colibri_analytical_birth_capture.birth.npz")
    count = int(x["rigid_contact_count"][0])
    ids = (
        np.column_stack(
            [x["shape_body"][x["rigid_contact_shape0"][:count]], x["shape_body"][x["rigid_contact_shape1"][:count]]]
        ).astype(np.int32)
        + 1
    )
    out = wp.zeros((count, 5), device=device)
    wp.launch(
        replay,
        count,
        [
            wp.array(x["position"], dtype=wp.vec3, device=device),
            wp.array(x["orientation"], dtype=wp.quat, device=device),
            wp.array(x["body_com"], dtype=wp.vec3, device=device),
            wp.array(ids, dtype=wp.vec2i, device=device),
            wp.array(x["lambdas"], device=device),
            out,
        ],
        device=device,
    )
    actual = x["derived"][4:6, :count].T.astype(float)
    cpu = out.numpy().copy()
    stored = []
    rebuilt = []
    records = []
    for k, (i, j) in enumerate(ids):
        p0 = x["position"][i].astype(float)
        p1 = x["position"][j].astype(float)
        r0 = rotation(x["orientation"][i])
        r1 = rotation(x["orientation"][j])
        c0 = x["body_com"][i].astype(float)
        c1 = x["body_com"][j].astype(float)
        n = x["lambdas"][:3, k].astype(float)
        t = x["lambdas"][3:6, k].astype(float)
        basis = np.array([t, np.cross(n, t)])
        a0 = x["lambdas"][6:9, k].astype(float)
        a1 = x["lambdas"][9:12, k].astype(float)
        delta = p1 + r1 @ (a1 - c1) - p0 - r0 @ (a0 - c0)
        stored.append(288 * basis @ delta)
        w0 = p0 + r0 @ (x["rigid_contact_point0"][k].astype(float) - c0)
        w1 = p1 + r1 @ (x["rigid_contact_point1"][k].astype(float) - c1)
        point = 0.5 * (w0 + w1 + (float(x["rigid_contact_margin0"][k]) - float(x["rigid_contact_margin1"][k])) * n)
        b0 = c0 + r0.T @ (point - p0)
        b1 = c1 + r1.T @ (point - p1)
        delta64 = p1 + r1 @ (b1 - c1) - p0 - r0 @ (b0 - c0)
        rebuilt.append(288 * basis @ delta64)
        records.append(
            {
                "point": k,
                "bodies": [int(i), int(j)],
                "native_bias": actual[k].tolist(),
                "cpu_replay_bias": cpu[k, :2].tolist(),
                "stored_anchors_fp64_bias": stored[-1].tolist(),
                "rebuilt_fp64_bias": rebuilt[-1].tolist(),
            }
        )
    report = {
        "device": device,
        "count": count,
        "nonzero_native_components": int(np.count_nonzero(actual)),
        "cpu_replay_bitexact": bool(actual.astype(np.float32).tobytes() == cpu[:, :2].tobytes()),
        "cpu_replay_max_difference": float(abs(actual - cpu[:, :2]).max()),
        "native_max_bias": float(abs(actual).max()),
        "stored_anchors_fp64_max_bias": float(np.max(np.abs(stored))),
        "rebuilt_fp64_max_bias": float(np.max(np.abs(rebuilt))),
        "implied_displacement_quantum_m": float(2.0**-30),
        "records": records,
    }
    Path("/tmp/colibri_anchor_birth_audit_" + device.replace(":", "_") + ".json").write_text(
        json.dumps(report, indent=2)
    )
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
