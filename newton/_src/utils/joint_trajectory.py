# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample authored joint trajectories into drive targets between their samples."""

from __future__ import annotations

import warp as wp


@wp.func
def hermite_sample(
    q0: wp.float32, qd0: wp.float32, q1: wp.float32, qd1: wp.float32, phase: wp.float32, sample_dt: wp.float32
) -> wp.vec2:
    """Position and velocity ``phase`` of the way from sample 0 to sample 1, ``sample_dt`` apart.

    The cubic Hermite through both samples' positions and velocities is continuous
    in position and velocity across samples, so a servo following it sees no steps.
    """
    s = phase
    s2 = s * s
    s3 = s2 * s
    q = (
        (2.0 * s3 - 3.0 * s2 + 1.0) * q0
        + (s3 - 2.0 * s2 + s) * sample_dt * qd0
        + (3.0 * s2 - 2.0 * s3) * q1
        + (s3 - s2) * sample_dt * qd1
    )
    qd = (6.0 * s2 - 6.0 * s) * (q0 - q1) / sample_dt + (3.0 * s2 - 4.0 * s + 1.0) * qd0 + (3.0 * s2 - 2.0 * s) * qd1
    return wp.vec2(q, qd)


@wp.kernel(enable_backward=False)
def _sample_joint_trajectories_kernel(
    rows: wp.array2d[wp.int32],
    phase: wp.float32,
    sample_dt: wp.float32,
    trajectory_q: wp.array2d[wp.float32],
    trajectory_qd: wp.array2d[wp.float32],
    coordinates: wp.array2d[wp.int32],
    dofs: wp.array2d[wp.int32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
):
    item, axis = wp.tid()
    row = rows[item, 0]
    following = rows[item, 1]
    q = trajectory_q[row, axis]
    qd = trajectory_qd[row, axis]
    if following != row:
        sample = hermite_sample(q, qd, trajectory_q[following, axis], trajectory_qd[following, axis], phase, sample_dt)
        q = sample[0]
        qd = sample[1]
    target_q[coordinates[item, axis]] = q
    target_qd[dofs[item, axis]] = qd


def sample_joint_trajectories(
    rows: wp.array2d[wp.int32],
    phase: float,
    sample_dt: float,
    trajectory_q: wp.array2d[wp.float32],
    trajectory_qd: wp.array2d[wp.float32],
    coordinates: wp.array2d[wp.int32],
    dofs: wp.array2d[wp.int32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
) -> None:
    """Write drive targets of several joint groups ``phase`` of the way between trajectory samples.

    Item ``i`` of ``rows`` names its group's current and next sample row of the
    ``trajectory_q`` and ``trajectory_qd`` tables (samples ``sample_dt`` apart; the
    same row twice holds the sample). ``coordinates[i, axis]`` and ``dofs[i, axis]``
    index ``target_q`` and ``target_qd``. Graph-capturable; ``rows`` may change on the
    device between launches.
    """
    wp.launch(
        _sample_joint_trajectories_kernel,
        dim=coordinates.shape,
        inputs=[rows, float(phase), float(sample_dt), trajectory_q, trajectory_qd, coordinates, dofs],
        outputs=[target_q, target_qd],
        device=target_q.device,
    )
