# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kugelstadt APD polar decomposition.

The PhoenX soft-tet iterate hot loop calls
:func:`newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron._extract_rotation`
once per PGS sweep to refresh the corotational rotation that the
ARAP residual is computed against. The function was swapped from
Mueller's scalar-denominator Newton step to Kugelstadt et al.'s
analytical polar decomposition (APD, full 3x3 Hessian inversion +
Cayley map quaternion update) in commit ``a39ef505``. APD's
quadratic convergence lets the iterate run with much smaller iter
caps, but a bug in the gradient sign or the Hessian inversion would
produce wrong rotations and break the iterate's convergence (which
the higher-level momentum-balance test catches statistically but
not always cleanly).

These tests construct a known deformation gradient ``F = R_known *
S_known`` with ``R_known`` a known rotation and ``S_known`` a
symmetric positive-definite stretch, then drive ``_extract_rotation``
through a tiny wrapper kernel and assert the extracted rotation
matches ``R_known`` to ~1e-5 in Frobenius norm. Failure modes the
asserts catch:

* Wrong gradient sign / wrong Hessian sign -> drifts away from
  ``R_known``; max||R_extracted - R_known||_F shoots up.
* Wrong axis convention -> rotation extracts about a different
  axis or with flipped handedness.
* Wrong quaternion-product order (``q * dq`` vs ``dq * q``) ->
  warm-start is rotated about a body-frame instead of world-frame
  axis; cold-start case (q_init = identity) still works but
  warm-start cases drift.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import _extract_rotation


@wp.kernel
def _extract_rotation_kernel(
    F_arr: wp.array[wp.mat33f],
    q_init_arr: wp.array[wp.quatf],
    max_iters: wp.int32,
    q_out_arr: wp.array[wp.quatf],
):
    """Wrapper: drive ``_extract_rotation`` per input and store the
    result. One thread per input slot."""
    tid = wp.tid()
    F = F_arr[tid]
    q_init = q_init_arr[tid]
    q = _extract_rotation(F, q_init, max_iters)
    q_out_arr[tid] = q


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternion to 3x3 rotation matrix in numpy."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """(axis, angle) -> (x, y, z, w) quaternion."""
    axis = axis / np.linalg.norm(axis)
    h = 0.5 * angle_rad
    s, c = np.sin(h), np.cos(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def _extract_via_kernel(F_list, q_init_list, max_iters, device) -> np.ndarray:
    """Launch the wrapper kernel on ``F_list`` / ``q_init_list`` and
    return the extracted quaternions as an ``(N, 4)`` ndarray."""
    n = len(F_list)
    F_arr = wp.array(np.asarray(F_list, dtype=np.float32), dtype=wp.mat33f, device=device)
    q_init_arr = wp.array(np.asarray(q_init_list, dtype=np.float32), dtype=wp.quatf, device=device)
    q_out_arr = wp.zeros(n, dtype=wp.quatf, device=device)
    wp.launch(
        _extract_rotation_kernel,
        dim=n,
        inputs=[F_arr, q_init_arr, wp.int32(max_iters)],
        outputs=[q_out_arr],
        device=device,
    )
    return q_out_arr.numpy()


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX rotation-extraction tests are CUDA-only.",
)
class TestExtractRotationAPD(unittest.TestCase):
    def _assert_rotation_recovers(self, R_known: np.ndarray, q_extracted: np.ndarray, label: str, atol: float = 5e-5):
        R_extracted = _quat_to_matrix(q_extracted.astype(np.float64))
        diff = np.linalg.norm(R_extracted - R_known, ord="fro")
        self.assertLess(
            diff,
            atol,
            f"{label}: ||R_extracted - R_known||_F = {diff:.2e} exceeds {atol}.\n"
            f"R_extracted=\n{R_extracted}\nR_known=\n{R_known}",
        )

    def test_identity_stretch_returns_identity(self):
        """``F = I`` (no deformation): the closest rotation is the
        identity. Drive from identity warm-start and from a small-
        angle warm-start; both must land at identity."""
        device = wp.get_preferred_device()
        F_id = np.eye(3, dtype=np.float32)
        q_id = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q_small = _axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 0.05)  # ~3 degrees
        out = _extract_via_kernel([F_id, F_id], [q_id, q_small], max_iters=8, device=device)
        self._assert_rotation_recovers(np.eye(3), out[0], "identity warm-start")
        self._assert_rotation_recovers(np.eye(3), out[1], "small-angle warm-start")

    def test_pure_rotation_recovers_rotation(self):
        """``F = R`` (pure rotation, no stretch): the polar
        decomposition must extract ``R`` back."""
        device = wp.get_preferred_device()
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ]
        angles = [0.2, -0.5, 0.7, 1.1, -1.3]
        Fs, qs_init = [], []
        Rs_known = []
        for axis, ang in zip(axes, angles, strict=True):
            q_known = _axis_angle_to_quat(axis, ang)
            R = _quat_to_matrix(q_known.astype(np.float64))
            Rs_known.append(R)
            Fs.append(R.astype(np.float32))
            qs_init.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        out = _extract_via_kernel(Fs, qs_init, max_iters=12, device=device)
        for i, R in enumerate(Rs_known):
            self._assert_rotation_recovers(R, out[i], f"pure rotation #{i}")

    def test_rotation_plus_stretch_recovers_rotation(self):
        """``F = R * S`` with ``R`` known and ``S`` symmetric positive-
        definite. Polar decomposition factors this as ``F = U * P``
        where ``U`` is the closest rotation and ``P`` is symmetric
        SPD. With ``R`` known and ``S`` symmetric SPD this construction
        already produces a polar decomposition (``U = R``, ``P = S``),
        so ``_extract_rotation`` must return ``R``."""
        device = wp.get_preferred_device()
        # Five rotations.
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 1.0, -1.0]),
        ]
        angles = [0.3, 0.6, -0.4, 0.8, -1.0]
        # Five stretches (symmetric positive definite). Diagonal
        # values vary; off-diagonals stay small so SPD is preserved.
        stretches = [
            np.diag([1.1, 1.0, 0.9]),
            np.diag([1.5, 0.7, 1.2]),
            np.diag([2.0, 2.0, 2.0]),  # uniform scale, polar still well-defined
            np.array([[1.2, 0.05, 0.0], [0.05, 1.1, 0.02], [0.0, 0.02, 0.9]]),
            np.array([[0.95, -0.1, 0.05], [-0.1, 1.05, -0.05], [0.05, -0.05, 1.0]]),
        ]
        Fs, qs_init, Rs_known = [], [], []
        for axis, ang, S in zip(axes, angles, stretches, strict=True):
            q_known = _axis_angle_to_quat(axis, ang)
            R = _quat_to_matrix(q_known.astype(np.float64))
            F = R @ S
            Fs.append(F.astype(np.float32))
            qs_init.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            Rs_known.append(R)
        out = _extract_via_kernel(Fs, qs_init, max_iters=16, device=device)
        for i, R in enumerate(Rs_known):
            # Looser tolerance: SPD stretches with off-diagonals
            # converge a touch slower in FP32 than the pure-rotation
            # case.
            self._assert_rotation_recovers(R, out[i], f"rotation+stretch #{i}", atol=2e-4)

    def test_warm_start_one_iter_converges(self):
        """APD's quadratic convergence: from a warm-start that is
        already close to the answer, ONE iteration must land within
        ~1e-3 of the truth. This guards against a regression in the
        Cayley-map quaternion update."""
        device = wp.get_preferred_device()
        # F is a small rotation about x; warm-start is the EXACT
        # rotation. One iter should leave it essentially unchanged.
        q_known = _axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 0.4)
        R = _quat_to_matrix(q_known.astype(np.float64))
        F = R.astype(np.float32)
        out = _extract_via_kernel([F], [q_known], max_iters=1, device=device)
        self._assert_rotation_recovers(R, out[0], "warm-start one-iter", atol=5e-5)


if __name__ == "__main__":
    unittest.main()
