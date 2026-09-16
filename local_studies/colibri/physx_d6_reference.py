# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
# Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
# Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
#

"""FP32 CPU translation of a bounded PhysX GPU TGS D6 row path.

Source: PhysX ed6e5ca2474c9c80ad4f4826591b88476779c6ef.
Supports two dynamic rigid bodies, unit mass scales, bilateral locked rows
and force-mode twist springs. No limits, restitution, acceleration drive,
kinematics, splitting, or unresolved zero-response rows. This is a prepared-row
reference, not the complete PhysX SDK, importer, island schedule or integrator.
"""

import copy
from dataclasses import dataclass

import numpy as np

F = np.float32
ZERO = F(0)


def dot(a, b):
    """Use explicit FP32 three-component accumulation."""
    p = np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)
    return F(F(p[0] + p[1]) + p[2])


@dataclass
class Row:
    """Source row fields before TGS coefficient preparation."""

    linear0: np.ndarray
    angular0: np.ndarray
    linear1: np.ndarray
    angular1: np.ndarray
    error: np.float32
    target: np.float32
    hint: int
    angular: bool
    spring: bool = False
    stiffness: np.float32 = ZERO
    damping: np.float32 = ZERO


def constants(row, response, dt, erp):
    """Translate force-spring/hard branches of compute1dConstraintSolverConstantsTGS."""
    dt, response = F(dt), F(response)
    if response <= 0:
        raise ValueError("No physical response; no silent mode removal")
    if row.spring:
        a = F(dt * F(dt * row.stiffness + row.damping))
        b = F(dt * F(row.damping * row.target))
        x = F(F(1) / F(F(1) + a * response))
        bias_scale = F(-x * row.stiffness * dt)
        return np.array([bias_scale * row.error, bias_scale, -x * a, x * b], dtype=np.float32)
    bias_scale = F(-F(F(1) / dt) * F(erp))
    return np.array([row.error * bias_scale, bias_scale, -1, row.target], dtype=np.float32)


def prepare(rows, inv_mass, sqrt_inv_inertia, dt, sim_dt, erp=0.5, preprocessing=True):
    """Stable hint sort, source mass-metric Gram-Schmidt, TGS coefficients."""
    rows = sorted(copy.deepcopy(rows), key=lambda row: row.hint)
    im = np.asarray(inv_mass, dtype=np.float32)
    si = np.asarray(sqrt_inv_inertia, dtype=np.float32)
    original = copy.deepcopy(rows)
    transform = np.eye(len(rows), dtype=np.float32)
    angular = [[si[0] @ r.angular0, si[1] @ r.angular1] for r in rows]
    if preprocessing:
        for group in (4, 8):
            indices = [i for i, r in enumerate(rows) if r.hint >> 8 == group]
            for index, i in enumerate(indices):
                r = rows[i]
                for j in indices[:index]:
                    base = rows[j]
                    s0 = r.linear1 * (base.linear1 * im[1]) + r.linear0 * (base.linear0 * im[0])
                    s1 = angular[i][1] * angular[j][1] + angular[i][0] * angular[j][0]
                    a = base.linear0 * (base.linear0 * im[0]) + base.linear1 * (base.linear1 * im[1])
                    b = angular[j][0] * angular[j][0] + angular[j][1] * angular[j][1]
                    denominator = F(F((a + b)[0] + (a + b)[1]) + (a + b)[2])
                    if denominator <= 0:
                        raise ValueError("Dependent hard basis: unsupported, no rank cutoff")
                    # Source stores normalized mass-weighted rows before taking this dot.
                    reciprocal = F(F(1) / denominator)
                    s0 = r.linear1 * (base.linear1 * im[1] * reciprocal) + r.linear0 * (
                        base.linear0 * im[0] * reciprocal
                    )
                    s1 = angular[i][1] * (angular[j][1] * reciprocal) + angular[i][0] * (angular[j][0] * reciprocal)
                    p = s0 + s1
                    coefficient = F(F(p[0] + p[1]) + p[2])
                    for name in ("linear0", "angular0", "linear1", "angular1"):
                        setattr(r, name, getattr(r, name) - getattr(base, name) * coefficient)
                    r.error = F(r.error - base.error * coefficient)
                    r.target = F(r.target - base.target * coefficient)
                    angular[i][0] -= angular[j][0] * coefficient
                    angular[i][1] -= angular[j][1] * coefficient
                    transform[i] -= transform[j] * coefficient
    prepared = []
    basis = []
    for i, r in enumerate(rows):
        response = F(
            F(im[0] * dot(r.linear0, r.linear0) + dot(angular[i][0], angular[i][0]))
            + F(im[1] * dot(r.linear1, r.linear1) + dot(angular[i][1], angular[i][1]))
        )
        coeff = constants(r, response, dt, erp)
        if r.spring:
            max_bias = F(np.finfo(np.float32).max)
        else:
            max_bias = F((0.75 if r.angular else 15.0) / sim_dt)
        prepared.append(
            {
                "row": r,
                "coeff": coeff,
                "applied": F(0),
                "max_bias": max_bias,
                "ortho_target": preprocessing and r.hint == 2048,
            }
        )
        if preprocessing and r.hint == 1024:
            basis.append((angular[i][0].copy(), angular[i][1].copy(), F(1 / response), r.error))
    return {
        "rows": prepared,
        "basis": basis,
        "inv_mass": im,
        "sqrt_inv_inertia": si,
        "transform": transform,
        "original": original,
    }


def solve(block, velocity, arms, lin_delta=None, ang_delta=None, elapsed=0):
    """Translate solve1DBlockTGS ordered row updates in mixed mass coordinates.

    Angular velocity and ang_delta use PhysX square-root-inertia coordinates.
    Arms are the already rotated current levers; lin_delta is accumulated COM
    displacement and arm motion together. No hidden pose integration occurs.
    """
    v = np.asarray(velocity, dtype=np.float32).copy()
    arms = np.asarray(arms, dtype=np.float32)
    ld = np.zeros((2, 3), dtype=np.float32) if lin_delta is None else np.asarray(lin_delta, dtype=np.float32)
    ad = np.zeros((2, 3), dtype=np.float32) if ang_delta is None else np.asarray(ang_delta, dtype=np.float32)
    si, im = block["sqrt_inv_inertia"], block["inv_mass"]
    ledger = []
    for entry in block["rows"]:
        row = entry["row"]
        init_bias, bias_scale, multiplier, target = entry["coeff"]
        a0 = (row.angular0 if row.angular else np.zeros(3, dtype=np.float32)) + np.cross(arms[0], row.linear0)
        a1 = (row.angular1 if row.angular else np.zeros(3, dtype=np.float32)) + np.cross(arms[1], row.linear1)
        a0 = si[0] @ a0
        a1 = si[1] @ a1
        if entry["ortho_target"]:
            delta0 = np.zeros(3, dtype=np.float32)
            delta1 = np.zeros(3, dtype=np.float32)
            projected_error = F(0)
            for b0, b1, recip, error in block["basis"]:
                projection = F(F(dot(a0, b0) + dot(a1, b1)) * recip)
                delta0 += b0 * projection
                delta1 += b1 * projection
                e = F(error + F(dot(b0, ad[0]) - dot(b1, ad[1])))
                projected_error = F(projected_error + e * projection)
            a0 -= delta0
            a1 -= delta1
            init_bias = F(init_bias - bias_scale * projected_error)
        if bias_scale == 0:
            init_bias = F(0)
        delta_ang = F(F(dot(a0, ad[0]) - dot(a1, ad[1])) * F(1 if row.angular else 0))
        motion = F(F(dot(row.linear0, ld[0]) - dot(row.linear1, ld[1])) + delta_ang)
        error_delta = motion if row.spring else F(motion - target * F(elapsed))
        response = F(
            F(im[0] * dot(row.linear0, row.linear0) + dot(a0, a0))
            + F(im[1] * dot(row.linear1, row.linear1) + dot(a1, a1))
        )
        if response <= 0:
            raise ValueError("No response; zero-response mode unsupported")
        reciprocal = F(1 / response)
        vm = multiplier if row.spring else F(reciprocal * multiplier)
        bias = F(np.clip(F(init_bias + error_delta * bias_scale), -entry["max_bias"], entry["max_bias"]))
        constant = F(bias + target) if row.spring else F(reciprocal * F(bias + target))
        normal_v = F(F(dot(v[0, 0], row.linear0) + dot(v[0, 1], a0)) - F(dot(v[1, 0], row.linear1) + dot(v[1, 1], a1)))
        old = entry["applied"]
        updated = F(old + F(vm * normal_v + constant))
        df = F(updated - old)
        entry["applied"] = updated
        v[0, 0] += row.linear0 * F(df * im[0])
        v[1, 0] -= row.linear1 * F(df * im[1])
        v[0, 1] += a0 * df
        v[1, 1] -= a1 * df
        ledger.append(
            np.array(
                [
                    np.r_[row.linear0 * df, np.linalg.solve(si[0], a0) * df],
                    np.r_[-row.linear1 * df, -np.linalg.solve(si[1], a1) * df],
                ],
                dtype=np.float32,
            )
        )
    return v, np.asarray(ledger, dtype=np.float32)


def conclude(block):
    """Translate conclude1DBlockTGS for ordinary hard rows and force springs."""
    for entry in block["rows"]:
        entry["coeff"][1] = F(0)
        if entry["row"].spring:
            entry["coeff"][:] = F(0)


def aligned_hinge_rows(arms, twist_delta, stiffness, damping):
    """Emit the aligned-frame hinge subset of GPU D6JointSolverPrep.

    twist_delta is the X component of target-conjugate * relative quaternion.
    Locked angular Jacobians at identity are half-axis, not unit-axis.
    """
    axes = np.eye(3, dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    rows = [
        Row(
            z.copy(),
            axes[0].copy(),
            z.copy(),
            axes[0].copy(),
            F(-2 * F(twist_delta)),
            F(0),
            0,
            True,
            True,
            F(stiffness),
            F(damping),
        )
    ]
    for axis in axes:
        rows.append(
            Row(axis.copy(), np.cross(arms[0], axis), axis.copy(), np.cross(arms[1], axis), F(0), F(0), 2048, False)
        )
    for axis in axes[1:]:
        rows.append(Row(z.copy(), axis * F(0.5), z.copy(), axis * F(0.5), F(0), F(0), 1024, True))
    return rows
