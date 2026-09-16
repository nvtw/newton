"""Actual FP32 material-reference helper on captured birth geometry and known motion."""

import os

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_DWORDS_PER_CONTACT,
    ContactContainer,
    cc_get_friction_reference_delta,
    cc_set_friction_reference_pose,
)


@wp.kernel(enable_backward=False)
def check(
    cc: ContactContainer,
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    com: wp.array[wp.vec3],
    ids: wp.array[wp.vec2i],
    out: wp.array2d[wp.vec3],
):
    k = wp.tid()
    i = ids[k][0]
    j = ids[k][1]
    p0 = position[i]
    p1 = position[j]
    q0 = orientation[i]
    q1 = orientation[j]
    cc_set_friction_reference_pose(cc, k, p0, q0, com[i], p1, q1, com[j])
    out[k, 0] = cc_get_friction_reference_delta(cc, k, p0, q0, com[i], p1, q1, com[j])
    shift = wp.vec3(0.00001, -0.00002, 0.00003)
    out[k, 1] = cc_get_friction_reference_delta(cc, k, p0, q0, com[i], p1 + shift, q1, com[j])
    # Actual represented FP32 translation is the expected motion.
    out[k, 2] = (p1 + shift) - p1


def main():
    device = os.environ.get("ANCHOR_REPLAY_DEVICE", "cpu")
    x = np.load("/tmp/colibri_analytical_birth_capture.birth.npz")
    count = int(x["rigid_contact_count"][0])
    ids = np.empty((count, 2), dtype=np.int32)
    h = x["headers"].view(np.int32)
    for col in range(int(x["column_count"][0])):
        start, num = h[5:7, col]
        ids[start : start + num] = h[1:3, col]
    cc = ContactContainer()
    data = np.zeros((CC_DWORDS_PER_CONTACT, count), dtype=np.float32)
    data[:13] = x["lambdas"][:13, :count]
    cc.lambdas = wp.array(data, device=device)
    out = wp.empty((count, 3), dtype=wp.vec3, device=device)
    wp.launch(
        check,
        count,
        [
            cc,
            wp.array(x["position"], dtype=wp.vec3, device=device),
            wp.array(x["orientation"], dtype=wp.quat, device=device),
            wp.array(x["body_com"], dtype=wp.vec3, device=device),
            wp.array(ids, dtype=wp.vec2i, device=device),
            out,
        ],
        device=device,
    )
    result = out.numpy()
    np.testing.assert_array_equal(result[:, 0], np.zeros((count, 3)))
    np.testing.assert_allclose(result[:, 1], result[:, 2], rtol=0, atol=5e-11)
    print(
        "BIRTH_REFERENCE_PASS",
        device,
        count,
        "birth_max",
        np.max(abs(result[:, 0])),
        "translation_error",
        np.max(abs(result[:, 1] - result[:, 2])),
        flush=True,
    )


if __name__ == "__main__":
    main()
