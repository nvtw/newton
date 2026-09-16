# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independently integrate normalized FP32 increments into the real-pose audit."""

import types
from pathlib import Path

import numpy as np

from . import check_real_birth_differential as original
from .check_friction_differential_reference import conjugate, multiply
from .check_normalized_rotation_increment import normalized_increment


def normalized_rotate32(q, point):
    """Rotate small vectors without rounding an absolute normalized quaternion."""
    cross = np.cross(q[:3], point)
    result = point + (np.float32(2) / np.dot(q, q)) * (q[3] * cross + np.cross(q[:3], cross))
    assert result.dtype == np.float32
    return result


def candidate(p0, p1, q0, q1, n0, n1, r0, r1, anchor):
    """Retain fourteen raw reference values and factor all normalization differences."""
    dq0, dq1 = r0 - q0, r1 - q1
    qr = multiply(conjugate(q0), q1)
    dqr = multiply(conjugate(dq0), q1) + multiply(conjugate(r0), dq1)
    dt = normalized_rotate32(conjugate(r0), (n1 - p1) - (n0 - p0))
    dt += normalized_increment(conjugate(q0), conjugate(dq0), p1 - p0)
    result = normalized_rotate32(r0, dt + normalized_increment(qr, dqr, anchor))
    assert result.dtype == np.float32
    return result


def direct(args, unit=False):
    """Use the independently normalized absolute-transform oracle in every comparison."""
    return original.direct(args, unit=True)


def main():
    """Run precisely the original controls without changing its frozen implementation."""
    namespace = dict(original.main.__globals__)
    namespace.update(candidate=candidate, direct=direct)
    namespace["Path"] = lambda name: Path(name.replace("real_birth_differential", "real_birth_normalized_differential"))
    run = types.FunctionType(original.main.__code__, namespace)
    run()


if __name__ == "__main__":
    main()
