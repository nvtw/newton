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
"""Isolated FP32 algebra translation of the PhysX GPU TGS patch-friction subset.

No torsional friction, material targets, dominance, or kinematic velocities.
Caller must run ALL normal rows first and supply their accumulated normal sum.
Physical world angular velocities replace PhysX sqrt-inertia coordinates; this
is algebra-equivalent, not instruction/rounding-identical. Diagnostic p8 is
source-fixed min(.8, preparation_bias_coefficient), NOT a production SOR change.
"""

from dataclasses import dataclass

import numpy as np

F = np.float32


@dataclass
class Patch:
    local0: np.ndarray
    local1: np.ndarray
    normal0: np.ndarray
    normal1: np.ndarray
    broken: bool = False


def correlate(patch, positions, rotations, normal, count, distance):
    if patch is None or patch.broken or len(patch.local0) == 0 or len(patch.local0) > count:
        return False
    n0 = rotations[0].T @ normal
    if np.dot(patch.normal0, n0) <= F(0.999):
        return False
    relative_rotation = rotations[0].T @ rotations[1]
    if np.dot(patch.normal0, relative_rotation @ patch.normal1) <= F(0.999):
        return False
    relative_position = rotations[0].T @ (positions[1] - positions[0])
    for a, b in zip(patch.local0, patch.local1, strict=True):
        if abs(np.dot(a - (relative_position + relative_rotation @ b), patch.normal0)) >= F(distance):
            return False
    return True


def refresh(points, gaps, normal, positions, rotations, friction_offset, correlation_distance, previous=None):
    """Preserve a correlated patch, then source-order farthest two-anchor growth."""
    points = np.asarray(points, dtype=F)
    gaps = np.asarray(gaps, dtype=F)
    keep = correlate(previous, positions, rotations, normal, len(points), correlation_distance)
    old = previous if keep else None
    if old is not None and len(old.local0) == 2:
        span = old.local0[0] - old.local0[1]
        diagonal = np.max(points, axis=0) - np.min(points, axis=0)
        if F(4) * np.dot(span, span) >= np.dot(diagonal, diagonal):
            return Patch(old.local0.copy(), old.local1.copy(), old.normal0.copy(), old.normal1.copy())
        old = None
    anchors = []
    if old is not None and len(old.local0) == 1:
        anchors.append(positions[0] + rotations[0] @ old.local0[0])
    distance = F(0)
    for point, gap in zip(points, gaps, strict=True):
        if gap >= F(friction_offset):
            continue
        if not anchors:
            anchors.append(point.copy())
        elif len(anchors) == 1:
            delta = point - anchors[0]
            distance = np.dot(delta, delta)
            if distance > F(1e-8):
                anchors.append(point.copy())
        else:
            d0 = np.dot(point - anchors[0], point - anchors[0])
            d1 = np.dot(point - anchors[1], point - anchors[1])
            if d0 > d1:
                if d0 > distance:
                    anchors[1] = point.copy()
                    distance = d0
            elif d1 > distance:
                anchors[0] = point.copy()
                distance = d1
    local0 = np.array([rotations[0].T @ (a - positions[0]) for a in anchors], dtype=F).reshape(-1, 3)
    local1 = np.array([rotations[1].T @ (a - positions[1]) for a in anchors], dtype=F).reshape(-1, 3)
    if old is not None and len(old.local0) == 1 and anchors:
        local0[0] = old.local0[0]
        local1[0] = old.local1[0]
    return Patch(local0, local1, rotations[0].T @ normal, rotations[1].T @ normal)


def prepare(
    patch,
    positions,
    rotations,
    normal,
    linear,
    inverse_mass,
    inverse_inertia,
    step_dt,
    preparation_bias_coefficient,
    solver_offset_slop=0,
):
    """Prepare independent anchor levers and scalar tangent responses once/refresh."""
    relative = linear[0] - linear[1]
    tangent = relative - normal * np.dot(normal, relative)
    fallback = (
        np.array([0, -normal[2], normal[1]], dtype=F)
        if abs(normal[0]) < F(0.70710678)
        else np.array([-normal[1], normal[0], 0], dtype=F)
    )
    if np.dot(tangent, tangent) <= F(0.0001):
        tangent = fallback
    tangent = tangent / F(np.sqrt(np.dot(tangent, tangent)))
    tangents = np.array([tangent, np.cross(normal, tangent)], dtype=F)
    levers = np.array(
        [[rotations[0] @ a, rotations[1] @ b] for a, b in zip(patch.local0, patch.local1, strict=True)], dtype=F
    ).reshape(-1, 2, 3)
    crosses = np.zeros((len(levers), 2, 2, 3), dtype=F)
    initial = np.zeros((len(levers), 2), dtype=F)
    response = np.zeros_like(initial)
    for k, (a, b) in enumerate(levers):
        initial[k] = tangents @ ((a + positions[0]) - (b + positions[1]))
        for t in range(2):
            for body in range(2):
                cross = np.cross(levers[k, body], tangents[t])
                cross[np.abs(cross) < F(solver_offset_slop)] = F(0)
                crosses[k, t, body] = cross
            response[k, t] = inverse_mass.sum() + sum(
                (np.dot(crosses[k, t, b], inverse_inertia[b] @ crosses[k, t, b]) for b in range(2)), start=F(0)
            )
    return {
        "tangents": tangents,
        "levers": levers,
        "crosses": crosses,
        "initial": initial,
        "response": response,
        "impulse": np.zeros_like(initial),
        "bias_coefficient": F(1) / F(step_dt),
        "p8": min(F(0.8), F(preparation_bias_coefficient)),
        "broken": False,
    }


def solve(
    prepared,
    linear,
    angular,
    inverse_mass,
    inverse_inertia,
    normal_sum,
    static_mu,
    dynamic_mu,
    linear_delta,
    angular_delta,
):
    """Literal pairwise diagonal trial, radial disk clamp, latest-solve broken bit.

    Equal/opposite linear impulses use independently authored endpoint levers.
    Returned wrenches expose the angular-momentum defect when endpoints differ.
    """
    n = len(prepared["levers"])
    linear = linear.copy()
    angular = angular.copy()
    ledger = np.zeros((2, 6), dtype=F)
    broken = False
    if not n:
        return linear, angular, ledger
    radius = F(static_mu) * F(normal_sum) * F(0.5 if n == 2 else 1)
    dynamic = F(dynamic_mu) * F(normal_sum) * F(0.5 if n == 2 else 1)
    for k in range(n):
        axes = prepared["tangents"]
        crosses = prepared["crosses"][k]
        relative = linear[0] - linear[1]
        speed = axes @ relative + crosses[:, 0] @ angular[0] - crosses[:, 1] @ angular[1]
        error = (
            prepared["initial"][k]
            + axes @ (linear_delta[0] - linear_delta[1])
            + crosses[:, 0] @ angular_delta[0]
            - crosses[:, 1] @ angular_delta[1]
        )
        multiplier = np.divide(
            prepared["p8"], prepared["response"][k], out=np.zeros(2, dtype=F), where=prepared["response"][k] > 0
        )
        trial = (prepared["impulse"][k] - error * prepared["bias_coefficient"] * multiplier) - speed * multiplier
        length = F(np.sqrt(np.dot(trial, trial)))
        clamp = length > radius
        broken = broken or bool(clamp)
        ratio = min(dynamic, length) / length if clamp else F(1)
        updated = trial * ratio
        delta = updated - prepared["impulse"][k]
        for t in range(2):
            impulse = axes[t] * delta[t]
            for body, sign in ((0, F(1)), (1, F(-1))):
                torque = sign * crosses[t, body] * delta[t]
                linear[body] += sign * inverse_mass[body] * impulse
                angular[body] += inverse_inertia[body] @ torque
                ledger[body, :3] += sign * impulse
                ledger[body, 3:] += torque
        prepared["impulse"][k] = updated
    prepared["broken"] = broken
    return linear, angular, ledger


def writeback(patch, prepared):
    """GPU writeback marks the patch only if the latest header is broken."""
    if prepared["broken"]:
        patch.broken = True
