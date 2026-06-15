# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified local-block pipeline benchmark.

This is a larger PhoenX unification prototype than the rows3-only sidecars.
It compares a typed local solve against a max-width descriptor path for the
main block shapes currently used by rigid contacts and ADBS joints:

* 3-row contact cone at a point;
* 3-row point-anchor dense block;
* 3-row angular dense block;
* 4-row two-anchor tangent dense block;
* 1-row linear / angular axial row.

The ``sidecar4`` path is the full-unification stress test: every block fetches
four prepacked Jacobian rows and runs the same residual / projection / apply
sequence. Smaller blocks disable unused rows through zero Jacobians and fixed
lambda bounds. Contact projection still needs its cone projection, but the
velocity fetch and ``M^-1 J^T`` apply are shape-agnostic. ``grouped_split``
keeps compact typed math but shape-buckets block IDs inside each color, testing
whether lower branch divergence can beat the extra indirection / locality cost.
``hybrid`` uses a setup-time per-color policy to pick compact split or sidecar4.
``auto_best`` reports the host-side tournament winner across graph-capture-safe
kernel choices, modeling the setup-time selector we would want in production
instead of hard-coding a semantic policy. ``--real-scenes`` extracts current
PhoenX colored graphs and joint modes so the same local-block candidates can be
measured against real H1 / DR-Legs / tower operation mixes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.experimental.bench_rigid_rows3_sidecar import (
    _bench,
    _max_err,
)
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    VELOCITY_ROWS3_PROJECT_CONTACT_CONE,
    VelocityRows3Op,
    block_solve_velocity_rows3_op,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_JOINT_MODE,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)

_OP_CONTACT3 = wp.constant(wp.int32(1))
_OP_POINT3 = wp.constant(wp.int32(2))
_OP_ANGULAR3 = wp.constant(wp.int32(3))
_OP_TANGENT4 = wp.constant(wp.int32(4))
_OP_SCALAR_LINEAR = wp.constant(wp.int32(5))
_OP_SCALAR_ANGULAR = wp.constant(wp.int32(6))

_OP_CONTACT3_HOST = 1
_OP_POINT3_HOST = 2
_OP_ANGULAR3_HOST = 3
_OP_TANGENT4_HOST = 4
_OP_SCALAR_LINEAR_HOST = 5
_OP_SCALAR_ANGULAR_HOST = 6

_COLOR_POLICY_SPLIT = wp.constant(wp.int32(0))
_COLOR_POLICY_SIDECAR4 = wp.constant(wp.int32(1))

_COLOR_POLICY_SPLIT_HOST = 0
_COLOR_POLICY_SIDECAR4_HOST = 1
_MIXED_SCENE = "mixed_h1_tower"

_JOINT_MODE_REVOLUTE_HOST = int(JOINT_MODE_REVOLUTE)
_JOINT_MODE_PRISMATIC_HOST = int(JOINT_MODE_PRISMATIC)
_JOINT_MODE_BALL_SOCKET_HOST = int(JOINT_MODE_BALL_SOCKET)
_JOINT_MODE_FIXED_HOST = int(JOINT_MODE_FIXED)
_JOINT_MODE_CABLE_HOST = int(JOINT_MODE_CABLE)
_JOINT_MODE_UNIVERSAL_HOST = int(JOINT_MODE_UNIVERSAL)
_JOINT_MODE_CYLINDRICAL_HOST = int(JOINT_MODE_CYLINDRICAL)
_JOINT_MODE_PLANAR_HOST = int(JOINT_MODE_PLANAR)
_JOINT_MODE_OFFSET_HOST = int(_OFF_JOINT_MODE)

_REAL_SCENE_ALIASES = {
    "h1": "h1",
    "h1_flat": "h1",
    "g1": "g1",
    "g1_flat": "g1",
    "dr_legs": "dr_legs",
    "drlegs": "dr_legs",
    "tower": "tower",
}


@dataclass(frozen=True)
class ScenePreset:
    blocks_per_color: tuple[int, ...]
    contact_ratio: float
    tangent4_ratio: float
    angular3_ratio: float
    scalar_ratio: float


_SCENE_PRESETS: dict[str, ScenePreset] = {
    # Robot-shaped mixes: mostly joint blocks, a few contacts.
    "h1": ScenePreset((18, 14, 9, 5), 0.12, 0.12, 0.22, 0.30),
    "g1": ScenePreset((28, 21, 13, 6), 0.00, 0.18, 0.24, 0.34),
    "dr_legs": ScenePreset((24, 21, 15, 8, 3), 0.08, 0.16, 0.22, 0.34),
    # Large single-world contact scene.
    "tower": ScenePreset((96, 91, 86, 80, 74, 68, 62, 56, 50, 44, 38, 32), 1.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class ScheduleHost:
    block_ids: np.ndarray
    color_starts: np.ndarray
    world_color_starts: np.ndarray
    blocks: int
    colors: int


@dataclass(frozen=True)
class RealSceneSpec:
    label: str
    scene: str
    worlds: int
    step_layout: str


def _build_schedule(preset: ScenePreset, worlds: int) -> ScheduleHost:
    block_ids: list[int] = []
    color_starts: list[int] = [0]
    world_color_starts: list[int] = [0]
    block = 0
    for _world in range(worlds):
        for count in preset.blocks_per_color:
            for _ in range(count):
                block_ids.append(block)
                block += 1
            color_starts.append(len(block_ids))
        world_color_starts.append(len(color_starts) - 1)
    return ScheduleHost(
        block_ids=np.asarray(block_ids, dtype=np.int32),
        color_starts=np.asarray(color_starts, dtype=np.int32),
        world_color_starts=np.asarray(world_color_starts, dtype=np.int32),
        blocks=block,
        colors=len(color_starts) - 1,
    )


@wp.func
def _rows4_dot(
    row0: wp.vec3f,
    row1: wp.vec3f,
    row2: wp.vec3f,
    row3: wp.vec3f,
    x: wp.vec3f,
) -> wp.vec4f:
    return wp.vec4f(wp.dot(row0, x), wp.dot(row1, x), wp.dot(row2, x), wp.dot(row3, x))


@wp.func
def _rows4_t_mul(
    row0: wp.vec3f,
    row1: wp.vec3f,
    row2: wp.vec3f,
    row3: wp.vec3f,
    d: wp.vec4f,
) -> wp.vec3f:
    return row0 * d[0] + row1 * d[1] + row2 * d[2] + row3 * d[3]


@wp.func
def _vec3_from4(v: wp.vec4f) -> wp.vec3f:
    return wp.vec3f(v[0], v[1], v[2])


@wp.func
def _mat33_from44(m: wp.mat44f) -> wp.mat33f:
    return wp.mat33f(
        m[0, 0],
        m[0, 1],
        m[0, 2],
        m[1, 0],
        m[1, 1],
        m[1, 2],
        m[2, 0],
        m[2, 1],
        m[2, 2],
    )


@wp.func
def _diag3_from44(m: wp.mat44f) -> wp.vec3f:
    return wp.vec3f(m[0, 0], m[1, 1], m[2, 2])


@wp.func
def _solve_bounded4(
    k_inv: wp.mat44f,
    residual: wp.vec4f,
    lambda_old: wp.vec4f,
    mass_coeff: wp.vec4f,
    impulse_coeff: wp.vec4f,
    lambda_min: wp.vec4f,
    lambda_max: wp.vec4f,
) -> wp.vec4f:
    d_unsoft = -(k_inv @ residual)
    candidate = wp.vec4f(
        lambda_old[0] + mass_coeff[0] * d_unsoft[0] - impulse_coeff[0] * lambda_old[0],
        lambda_old[1] + mass_coeff[1] * d_unsoft[1] - impulse_coeff[1] * lambda_old[1],
        lambda_old[2] + mass_coeff[2] * d_unsoft[2] - impulse_coeff[2] * lambda_old[2],
        lambda_old[3] + mass_coeff[3] * d_unsoft[3] - impulse_coeff[3] * lambda_old[3],
    )
    return wp.vec4f(
        wp.clamp(candidate[0], lambda_min[0], lambda_max[0]),
        wp.clamp(candidate[1], lambda_min[1], lambda_max[1]),
        wp.clamp(candidate[2], lambda_min[2], lambda_max[2]),
        wp.clamp(candidate[3], lambda_min[3], lambda_max[3]),
    )


@wp.func
def _solve_bounded3(
    k_inv: wp.mat33f,
    residual: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.vec3f,
    impulse_coeff: wp.vec3f,
    lambda_min: wp.vec3f,
    lambda_max: wp.vec3f,
) -> wp.vec3f:
    d_unsoft = -(k_inv @ residual)
    candidate = wp.vec3f(
        lambda_old[0] + mass_coeff[0] * d_unsoft[0] - impulse_coeff[0] * lambda_old[0],
        lambda_old[1] + mass_coeff[1] * d_unsoft[1] - impulse_coeff[1] * lambda_old[1],
        lambda_old[2] + mass_coeff[2] * d_unsoft[2] - impulse_coeff[2] * lambda_old[2],
    )
    return wp.vec3f(
        wp.clamp(candidate[0], lambda_min[0], lambda_max[0]),
        wp.clamp(candidate[1], lambda_min[1], lambda_max[1]),
        wp.clamp(candidate[2], lambda_min[2], lambda_max[2]),
    )


@wp.func
def _solve_scalar_bounded(
    k_inv: wp.float32,
    residual: wp.float32,
    lambda_old: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> wp.float32:
    d_unsoft = -(k_inv * residual)
    candidate = lambda_old + mass_coeff * d_unsoft - impulse_coeff * lambda_old
    return wp.clamp(candidate, lambda_min, lambda_max)


@wp.func
def _make_contact_projection(
    k_diag: wp.vec3f,
    residual: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.vec3f,
    impulse_coeff: wp.vec3f,
    lambda_min: wp.vec3f,
    lambda_max: wp.vec3f,
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
) -> wp.vec3f:
    op = VelocityRows3Op()
    op.k_inv = k_diag
    op.residual = residual
    op.lambda_old = lambda_old
    op.mass_coeff = mass_coeff
    op.impulse_coeff = impulse_coeff
    op.lambda_min = lambda_min
    op.lambda_max = lambda_max
    op.projection_mode = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
    op.friction_static = friction_static
    op.friction_kinetic = friction_kinetic
    return block_solve_velocity_rows3_op(op, wp.float32(1.0)).lambda_new


@wp.kernel
def _init_blocks_kernel(
    op_kind: wp.array[wp.int32],
    op_kind_seed: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    axis3: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    r2: wp.array[wp.vec3f],
    r3: wp.array[wp.vec3f],
    jla0: wp.array[wp.vec3f],
    jla1: wp.array[wp.vec3f],
    jla2: wp.array[wp.vec3f],
    jla3: wp.array[wp.vec3f],
    jaa0: wp.array[wp.vec3f],
    jaa1: wp.array[wp.vec3f],
    jaa2: wp.array[wp.vec3f],
    jaa3: wp.array[wp.vec3f],
    jlb0: wp.array[wp.vec3f],
    jlb1: wp.array[wp.vec3f],
    jlb2: wp.array[wp.vec3f],
    jlb3: wp.array[wp.vec3f],
    jab0: wp.array[wp.vec3f],
    jab1: wp.array[wp.vec3f],
    jab2: wp.array[wp.vec3f],
    jab3: wp.array[wp.vec3f],
    k_inv: wp.array[wp.mat44f],
    bias: wp.array[wp.vec4f],
    lambda_old: wp.array[wp.vec4f],
    mass_coeff: wp.array[wp.vec4f],
    impulse_coeff: wp.array[wp.vec4f],
    lambda_min: wp.array[wp.vec4f],
    lambda_max: wp.array[wp.vec4f],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    period: wp.int32,
    contact_slots: wp.int32,
    tangent4_slots: wp.int32,
    angular3_slots: wp.int32,
    scalar_slots: wp.int32,
):
    tid = wp.tid()
    phase = wp.float32((tid * wp.int32(29)) & wp.int32(255)) * wp.float32(0.00390625)

    va = wp.vec3f(phase, wp.float32(0.2) - phase, wp.float32(0.1) + wp.float32(0.5) * phase)
    wa = wp.vec3f(wp.float32(0.3) * phase, wp.float32(0.4) - phase, wp.float32(0.2) * phase)
    vb = wp.vec3f(wp.float32(0.15) - phase, wp.float32(0.1) + phase, wp.float32(0.25) * phase)
    wb = wp.vec3f(wp.float32(0.2) + wp.float32(0.25) * phase, -wp.float32(0.1) * phase, phase)
    v_a[tid] = va
    w_a[tid] = wa
    v_b[tid] = vb
    w_b[tid] = wb
    inv_m_a[tid] = wp.float32(0.8) + wp.float32(0.1) * phase
    inv_m_b[tid] = wp.float32(0.7) + wp.float32(0.2) * phase
    inv_i_a[tid] = wp.mat33f(
        wp.float32(0.60),
        wp.float32(0.02),
        wp.float32(0.00),
        wp.float32(0.02),
        wp.float32(0.70),
        wp.float32(0.01),
        wp.float32(0.00),
        wp.float32(0.01),
        wp.float32(0.80),
    )
    inv_i_b[tid] = wp.mat33f(
        wp.float32(0.90),
        -wp.float32(0.01),
        wp.float32(0.02),
        -wp.float32(0.01),
        wp.float32(0.65),
        wp.float32(0.00),
        wp.float32(0.02),
        wp.float32(0.00),
        wp.float32(0.75),
    )

    n = wp.normalize(wp.vec3f(wp.float32(1.0), wp.float32(0.2), wp.float32(0.1)))
    t1 = wp.normalize(wp.vec3f(-wp.float32(0.2), wp.float32(1.0), wp.float32(0.0)))
    t2 = wp.normalize(wp.cross(n, t1))
    e0 = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    e1 = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    e2 = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    zero = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    rr0 = wp.vec3f(wp.float32(0.10), wp.float32(0.03) + wp.float32(0.02) * phase, -wp.float32(0.05))
    rr1 = wp.vec3f(-wp.float32(0.08), wp.float32(0.04), wp.float32(0.06) + wp.float32(0.01) * phase)
    rr2 = wp.vec3f(wp.float32(0.14), -wp.float32(0.04), wp.float32(0.03) + wp.float32(0.01) * phase)
    rr3 = wp.vec3f(-wp.float32(0.11), wp.float32(0.02), -wp.float32(0.07))
    r0[tid] = rr0
    r1[tid] = rr1
    r2[tid] = rr2
    r3[tid] = rr3

    _ = period
    _ = contact_slots
    _ = tangent4_slots
    _ = angular3_slots
    _ = scalar_slots
    kind = op_kind_seed[tid]
    op_kind[tid] = kind

    k = wp.mat44f(
        wp.float32(0.38) + wp.float32(0.02) * phase,
        wp.float32(0.015),
        wp.float32(0.006),
        wp.float32(0.004),
        wp.float32(0.015),
        wp.float32(0.47) + wp.float32(0.02) * phase,
        wp.float32(0.012),
        wp.float32(0.003),
        wp.float32(0.006),
        wp.float32(0.012),
        wp.float32(0.58) + wp.float32(0.02) * phase,
        wp.float32(0.010),
        wp.float32(0.004),
        wp.float32(0.003),
        wp.float32(0.010),
        wp.float32(0.52) + wp.float32(0.02) * phase,
    )
    b = wp.vec4f(wp.float32(0.02) - wp.float32(0.03) * phase, wp.float32(0.01), -wp.float32(0.015), wp.float32(0.006))
    lam = wp.vec4f(wp.float32(0.05), wp.float32(0.01), wp.float32(0.015), -wp.float32(0.02))
    mc = wp.vec4f(wp.float32(0.90), wp.float32(0.95), wp.float32(1.0), wp.float32(0.88))
    ic = wp.vec4f(wp.float32(0.06), wp.float32(0.04), wp.float32(0.0), wp.float32(0.02))
    lo = wp.vec4f(-BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
    hi = wp.vec4f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
    fs = wp.float32(0.0)
    fk = wp.float32(0.0)

    a0 = e0
    a1 = e1
    a2 = e2
    a3 = zero
    jl0 = zero
    jl1 = zero
    jl2 = zero
    jl3 = zero
    ja0 = zero
    ja1 = zero
    ja2 = zero
    ja3 = zero
    jb0 = zero
    jb1 = zero
    jb2 = zero
    jb3 = zero
    jc0 = zero
    jc1 = zero
    jc2 = zero
    jc3 = zero

    if kind == _OP_CONTACT3:
        a0 = n
        a1 = t1
        a2 = t2
        k = wp.mat44f(
            wp.float32(0.35),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.45),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.55),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
        mc = wp.vec4f(wp.float32(0.82), wp.float32(1.0), wp.float32(1.0), wp.float32(0.0))
        ic = wp.vec4f(wp.float32(0.11), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        lo = wp.vec4f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, lam[3])
        hi = wp.vec4f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, lam[3])
        fs = wp.float32(0.8)
        fk = wp.float32(0.6)
        jl0 = -n
        jl1 = -t1
        jl2 = -t2
        ja0 = -wp.cross(rr0, n)
        ja1 = -wp.cross(rr0, t1)
        ja2 = -wp.cross(rr0, t2)
        jb0 = n
        jb1 = t1
        jb2 = t2
        jc0 = wp.cross(rr1, n)
        jc1 = wp.cross(rr1, t1)
        jc2 = wp.cross(rr1, t2)
    elif kind == _OP_TANGENT4:
        a0 = t1
        a1 = t2
        a2 = t1
        a3 = t2
        jl0 = -t1
        jl1 = -t2
        jl2 = -t1
        jl3 = -t2
        ja0 = wp.cross(t1, rr0)
        ja1 = wp.cross(t2, rr0)
        ja2 = wp.cross(t1, rr2)
        ja3 = wp.cross(t2, rr2)
        jb0 = t1
        jb1 = t2
        jb2 = t1
        jb3 = t2
        jc0 = wp.cross(rr1, t1)
        jc1 = wp.cross(rr1, t2)
        jc2 = wp.cross(rr3, t1)
        jc3 = wp.cross(rr3, t2)
    elif kind == _OP_ANGULAR3:
        lo = wp.vec4f(-wp.float32(0.6), wp.float32(0.0), -wp.float32(0.25), lam[3])
        hi = wp.vec4f(wp.float32(0.6), BLOCK_LAMBDA_INF, wp.float32(0.25), lam[3])
        ja0 = -e0
        ja1 = -e1
        ja2 = -e2
        jc0 = e0
        jc1 = e1
        jc2 = e2
        k[3, 0] = wp.float32(0.0)
        k[3, 1] = wp.float32(0.0)
        k[3, 2] = wp.float32(0.0)
        k[0, 3] = wp.float32(0.0)
        k[1, 3] = wp.float32(0.0)
        k[2, 3] = wp.float32(0.0)
        k[3, 3] = wp.float32(0.0)
        lo[3] = lam[3]
        hi[3] = lam[3]
    elif kind == _OP_SCALAR_LINEAR:
        a0 = n
        k = wp.mat44f(
            wp.float32(0.42),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
        lo = wp.vec4f(wp.float32(0.0), lam[1], lam[2], lam[3])
        hi = wp.vec4f(BLOCK_LAMBDA_INF, lam[1], lam[2], lam[3])
        jl0 = -n
        ja0 = -wp.cross(rr0, n)
        jb0 = n
        jc0 = wp.cross(rr1, n)
    elif kind == _OP_SCALAR_ANGULAR:
        a0 = n
        k = wp.mat44f(
            wp.float32(0.44),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
        lo = wp.vec4f(-wp.float32(0.4), lam[1], lam[2], lam[3])
        hi = wp.vec4f(wp.float32(0.4), lam[1], lam[2], lam[3])
        ja0 = -n
        jc0 = n
    else:
        k[3, 0] = wp.float32(0.0)
        k[3, 1] = wp.float32(0.0)
        k[3, 2] = wp.float32(0.0)
        k[0, 3] = wp.float32(0.0)
        k[1, 3] = wp.float32(0.0)
        k[2, 3] = wp.float32(0.0)
        k[3, 3] = wp.float32(0.0)
        lo[3] = lam[3]
        hi[3] = lam[3]
        jl0 = -e0
        jl1 = -e1
        jl2 = -e2
        ja0 = wp.cross(e0, rr0)
        ja1 = wp.cross(e1, rr0)
        ja2 = wp.cross(e2, rr0)
        jb0 = e0
        jb1 = e1
        jb2 = e2
        jc0 = wp.cross(rr1, e0)
        jc1 = wp.cross(rr1, e1)
        jc2 = wp.cross(rr1, e2)

    axis0[tid] = a0
    axis1[tid] = a1
    axis2[tid] = a2
    axis3[tid] = a3
    jla0[tid] = jl0
    jla1[tid] = jl1
    jla2[tid] = jl2
    jla3[tid] = jl3
    jaa0[tid] = ja0
    jaa1[tid] = ja1
    jaa2[tid] = ja2
    jaa3[tid] = ja3
    jlb0[tid] = jb0
    jlb1[tid] = jb1
    jlb2[tid] = jb2
    jlb3[tid] = jb3
    jab0[tid] = jc0
    jab1[tid] = jc1
    jab2[tid] = jc2
    jab3[tid] = jc3
    k_inv[tid] = k
    bias[tid] = b
    lambda_old[tid] = lam
    mass_coeff[tid] = mc
    impulse_coeff[tid] = ic
    lambda_min[tid] = lo
    lambda_max[tid] = hi
    friction_static[tid] = fs
    friction_kinetic[tid] = fk


@wp.kernel(enable_backward=False)
def _solve_sidecar4_world_loop_kernel(
    block_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    op_kind: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    jla0: wp.array[wp.vec3f],
    jla1: wp.array[wp.vec3f],
    jla2: wp.array[wp.vec3f],
    jla3: wp.array[wp.vec3f],
    jaa0: wp.array[wp.vec3f],
    jaa1: wp.array[wp.vec3f],
    jaa2: wp.array[wp.vec3f],
    jaa3: wp.array[wp.vec3f],
    jlb0: wp.array[wp.vec3f],
    jlb1: wp.array[wp.vec3f],
    jlb2: wp.array[wp.vec3f],
    jlb3: wp.array[wp.vec3f],
    jab0: wp.array[wp.vec3f],
    jab1: wp.array[wp.vec3f],
    jab2: wp.array[wp.vec3f],
    jab3: wp.array[wp.vec3f],
    k_inv: wp.array[wp.mat44f],
    bias: wp.array[wp.vec4f],
    lambda_old: wp.array[wp.vec4f],
    mass_coeff: wp.array[wp.vec4f],
    impulse_coeff: wp.array[wp.vec4f],
    lambda_min: wp.array[wp.vec4f],
    lambda_max: wp.array[wp.vec4f],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec4f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                block = block_ids[cursor]
                va = v_a[block]
                wa = w_a[block]
                vb = v_b[block]
                wb = w_b[block]
                residual = (
                    _rows4_dot(jla0[block], jla1[block], jla2[block], jla3[block], va)
                    + _rows4_dot(jaa0[block], jaa1[block], jaa2[block], jaa3[block], wa)
                    + _rows4_dot(jlb0[block], jlb1[block], jlb2[block], jlb3[block], vb)
                    + _rows4_dot(jab0[block], jab1[block], jab2[block], jab3[block], wb)
                    + bias[block]
                )

                lambda_new = _solve_bounded4(
                    k_inv[block],
                    residual,
                    lambda_old[block],
                    mass_coeff[block],
                    impulse_coeff[block],
                    lambda_min[block],
                    lambda_max[block],
                )
                if op_kind[block] == _OP_CONTACT3:
                    contact_lambda = _make_contact_projection(
                        _diag3_from44(k_inv[block]),
                        _vec3_from4(residual),
                        _vec3_from4(lambda_old[block]),
                        _vec3_from4(mass_coeff[block]),
                        _vec3_from4(impulse_coeff[block]),
                        _vec3_from4(lambda_min[block]),
                        _vec3_from4(lambda_max[block]),
                        friction_static[block],
                        friction_kinetic[block],
                    )
                    lambda_new = wp.vec4f(contact_lambda[0], contact_lambda[1], contact_lambda[2], lambda_old[block][3])
                delta = lambda_new - lambda_old[block]
                out_va[block] = va + inv_m_a[block] * _rows4_t_mul(
                    jla0[block], jla1[block], jla2[block], jla3[block], delta
                )
                out_wa[block] = wa + inv_i_a[block] @ _rows4_t_mul(
                    jaa0[block], jaa1[block], jaa2[block], jaa3[block], delta
                )
                out_vb[block] = vb + inv_m_b[block] * _rows4_t_mul(
                    jlb0[block], jlb1[block], jlb2[block], jlb3[block], delta
                )
                out_wb[block] = wb + inv_i_b[block] @ _rows4_t_mul(
                    jab0[block], jab1[block], jab2[block], jab3[block], delta
                )
                out_lambda[block] = lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_hybrid_world_loop_kernel(
    block_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    color_policy: wp.array[wp.int32],
    op_kind: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    r2: wp.array[wp.vec3f],
    r3: wp.array[wp.vec3f],
    jla0: wp.array[wp.vec3f],
    jla1: wp.array[wp.vec3f],
    jla2: wp.array[wp.vec3f],
    jla3: wp.array[wp.vec3f],
    jaa0: wp.array[wp.vec3f],
    jaa1: wp.array[wp.vec3f],
    jaa2: wp.array[wp.vec3f],
    jaa3: wp.array[wp.vec3f],
    jlb0: wp.array[wp.vec3f],
    jlb1: wp.array[wp.vec3f],
    jlb2: wp.array[wp.vec3f],
    jlb3: wp.array[wp.vec3f],
    jab0: wp.array[wp.vec3f],
    jab1: wp.array[wp.vec3f],
    jab2: wp.array[wp.vec3f],
    jab3: wp.array[wp.vec3f],
    k_inv: wp.array[wp.mat44f],
    bias: wp.array[wp.vec4f],
    lambda_old: wp.array[wp.vec4f],
    mass_coeff: wp.array[wp.vec4f],
    impulse_coeff: wp.array[wp.vec4f],
    lambda_min: wp.array[wp.vec4f],
    lambda_max: wp.array[wp.vec4f],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec4f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            use_sidecar = color_policy[color] == _COLOR_POLICY_SIDECAR4
            cursor = start + local_tid
            while cursor < end:
                block = block_ids[cursor]
                va = v_a[block]
                wa = w_a[block]
                vb = v_b[block]
                wb = w_b[block]
                lam_old = lambda_old[block]
                lam_new = lam_old
                va_out = va
                wa_out = wa
                vb_out = vb
                wb_out = wb
                if use_sidecar:
                    residual4_side = (
                        _rows4_dot(jla0[block], jla1[block], jla2[block], jla3[block], va)
                        + _rows4_dot(jaa0[block], jaa1[block], jaa2[block], jaa3[block], wa)
                        + _rows4_dot(jlb0[block], jlb1[block], jlb2[block], jlb3[block], vb)
                        + _rows4_dot(jab0[block], jab1[block], jab2[block], jab3[block], wb)
                        + bias[block]
                    )
                    lam_new = _solve_bounded4(
                        k_inv[block],
                        residual4_side,
                        lam_old,
                        mass_coeff[block],
                        impulse_coeff[block],
                        lambda_min[block],
                        lambda_max[block],
                    )
                    if op_kind[block] == _OP_CONTACT3:
                        contact_lambda = _make_contact_projection(
                            _diag3_from44(k_inv[block]),
                            _vec3_from4(residual4_side),
                            _vec3_from4(lam_old),
                            _vec3_from4(mass_coeff[block]),
                            _vec3_from4(impulse_coeff[block]),
                            _vec3_from4(lambda_min[block]),
                            _vec3_from4(lambda_max[block]),
                            friction_static[block],
                            friction_kinetic[block],
                        )
                        lam_new = wp.vec4f(contact_lambda[0], contact_lambda[1], contact_lambda[2], lam_old[3])
                    delta_side = lam_new - lam_old
                    va_out = va + inv_m_a[block] * _rows4_t_mul(
                        jla0[block], jla1[block], jla2[block], jla3[block], delta_side
                    )
                    wa_out = wa + inv_i_a[block] @ _rows4_t_mul(
                        jaa0[block], jaa1[block], jaa2[block], jaa3[block], delta_side
                    )
                    vb_out = vb + inv_m_b[block] * _rows4_t_mul(
                        jlb0[block], jlb1[block], jlb2[block], jlb3[block], delta_side
                    )
                    wb_out = wb + inv_i_b[block] @ _rows4_t_mul(
                        jab0[block], jab1[block], jab2[block], jab3[block], delta_side
                    )
                else:
                    kind = op_kind[block]
                    if kind == _OP_CONTACT3:
                        n = axis0[block]
                        t1 = axis1[block]
                        t2 = axis2[block]
                        rr0 = r0[block]
                        rr1 = r1[block]
                        rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                        residual3_contact = wp.vec3f(wp.dot(n, rel), wp.dot(t1, rel), wp.dot(t2, rel)) + _vec3_from4(
                            bias[block]
                        )
                        contact_lambda = _make_contact_projection(
                            _diag3_from44(k_inv[block]),
                            residual3_contact,
                            _vec3_from4(lam_old),
                            _vec3_from4(mass_coeff[block]),
                            _vec3_from4(impulse_coeff[block]),
                            _vec3_from4(lambda_min[block]),
                            _vec3_from4(lambda_max[block]),
                            friction_static[block],
                            friction_kinetic[block],
                        )
                        delta3_contact = contact_lambda - _vec3_from4(lam_old)
                        impulse_contact = delta3_contact[0] * n + delta3_contact[1] * t1 + delta3_contact[2] * t2
                        va_out = va - inv_m_a[block] * impulse_contact
                        wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse_contact)
                        vb_out = vb + inv_m_b[block] * impulse_contact
                        wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse_contact)
                        lam_new = wp.vec4f(contact_lambda[0], contact_lambda[1], contact_lambda[2], lam_old[3])
                    elif kind == _OP_TANGENT4:
                        t1 = axis0[block]
                        t2 = axis1[block]
                        rr0 = r0[block]
                        rr1 = r1[block]
                        rr2 = r2[block]
                        rr3 = r3[block]
                        jv1 = -va + wp.cross(rr0, wa) + vb - wp.cross(rr1, wb)
                        jv2 = -va + wp.cross(rr2, wa) + vb - wp.cross(rr3, wb)
                        residual4_tangent = (
                            wp.vec4f(wp.dot(t1, jv1), wp.dot(t2, jv1), wp.dot(t1, jv2), wp.dot(t2, jv2)) + bias[block]
                        )
                        lam_new = _solve_bounded4(
                            k_inv[block],
                            residual4_tangent,
                            lam_old,
                            mass_coeff[block],
                            impulse_coeff[block],
                            lambda_min[block],
                            lambda_max[block],
                        )
                        d4 = lam_new - lam_old
                        imp1 = d4[0] * t1 + d4[1] * t2
                        imp2 = d4[2] * t1 + d4[3] * t2
                        total = imp1 + imp2
                        va_out = va - inv_m_a[block] * total
                        wa_out = wa - inv_i_a[block] @ (wp.cross(rr0, imp1) + wp.cross(rr2, imp2))
                        vb_out = vb + inv_m_b[block] * total
                        wb_out = wb + inv_i_b[block] @ (wp.cross(rr1, imp1) + wp.cross(rr3, imp2))
                    elif kind == _OP_SCALAR_LINEAR:
                        n = axis0[block]
                        rr0 = r0[block]
                        rr1 = r1[block]
                        rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                        residual_scalar = wp.dot(n, rel) + bias[block][0]
                        l0 = _solve_scalar_bounded(
                            k_inv[block][0, 0],
                            residual_scalar,
                            lam_old[0],
                            mass_coeff[block][0],
                            impulse_coeff[block][0],
                            lambda_min[block][0],
                            lambda_max[block][0],
                        )
                        d = l0 - lam_old[0]
                        impulse = d * n
                        va_out = va - inv_m_a[block] * impulse
                        wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse)
                        vb_out = vb + inv_m_b[block] * impulse
                        wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse)
                        lam_new = wp.vec4f(l0, lam_old[1], lam_old[2], lam_old[3])
                    elif kind == _OP_SCALAR_ANGULAR:
                        n = axis0[block]
                        residual_scalar_ang = wp.dot(n, wb - wa) + bias[block][0]
                        l0 = _solve_scalar_bounded(
                            k_inv[block][0, 0],
                            residual_scalar_ang,
                            lam_old[0],
                            mass_coeff[block][0],
                            impulse_coeff[block][0],
                            lambda_min[block][0],
                            lambda_max[block][0],
                        )
                        d = l0 - lam_old[0]
                        impulse = d * n
                        wa_out = wa - inv_i_a[block] @ impulse
                        wb_out = wb + inv_i_b[block] @ impulse
                        lam_new = wp.vec4f(l0, lam_old[1], lam_old[2], lam_old[3])
                    else:
                        a0 = axis0[block]
                        a1 = axis1[block]
                        a2 = axis2[block]
                        if kind == _OP_ANGULAR3:
                            residual3_ang = wp.vec3f(wp.dot(a0, wb - wa), wp.dot(a1, wb - wa), wp.dot(a2, wb - wa))
                            residual3_ang = residual3_ang + _vec3_from4(bias[block])
                            lambda3 = _solve_bounded3(
                                _mat33_from44(k_inv[block]),
                                residual3_ang,
                                _vec3_from4(lam_old),
                                _vec3_from4(mass_coeff[block]),
                                _vec3_from4(impulse_coeff[block]),
                                _vec3_from4(lambda_min[block]),
                                _vec3_from4(lambda_max[block]),
                            )
                            d3 = lambda3 - _vec3_from4(lam_old)
                            impulse_ang = d3[0] * a0 + d3[1] * a1 + d3[2] * a2
                            wa_out = wa - inv_i_a[block] @ impulse_ang
                            wb_out = wb + inv_i_b[block] @ impulse_ang
                            lam_new = wp.vec4f(lambda3[0], lambda3[1], lambda3[2], lam_old[3])
                        else:
                            rr0 = r0[block]
                            rr1 = r1[block]
                            rel = -va + wp.cross(rr0, wa) + vb - wp.cross(rr1, wb)
                            residual3_point = rel + _vec3_from4(bias[block])
                            lambda3 = _solve_bounded3(
                                _mat33_from44(k_inv[block]),
                                residual3_point,
                                _vec3_from4(lam_old),
                                _vec3_from4(mass_coeff[block]),
                                _vec3_from4(impulse_coeff[block]),
                                _vec3_from4(lambda_min[block]),
                                _vec3_from4(lambda_max[block]),
                            )
                            d3 = lambda3 - _vec3_from4(lam_old)
                            impulse_point = d3[0] * a0 + d3[1] * a1 + d3[2] * a2
                            va_out = va - inv_m_a[block] * impulse_point
                            wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse_point)
                            vb_out = vb + inv_m_b[block] * impulse_point
                            wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse_point)
                            lam_new = wp.vec4f(lambda3[0], lambda3[1], lambda3[2], lam_old[3])

                out_va[block] = va_out
                out_wa[block] = wa_out
                out_vb[block] = vb_out
                out_wb[block] = wb_out
                out_lambda[block] = lam_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_split_world_loop_kernel(
    block_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    op_kind: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    r2: wp.array[wp.vec3f],
    r3: wp.array[wp.vec3f],
    k_inv: wp.array[wp.mat44f],
    bias: wp.array[wp.vec4f],
    lambda_old: wp.array[wp.vec4f],
    mass_coeff: wp.array[wp.vec4f],
    impulse_coeff: wp.array[wp.vec4f],
    lambda_min: wp.array[wp.vec4f],
    lambda_max: wp.array[wp.vec4f],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec4f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                block = block_ids[cursor]
                kind = op_kind[block]
                va = v_a[block]
                wa = w_a[block]
                vb = v_b[block]
                wb = w_b[block]
                lam_old = lambda_old[block]
                lam_new = lam_old
                va_out = va
                wa_out = wa
                vb_out = vb
                wb_out = wb
                if kind == _OP_CONTACT3:
                    n = axis0[block]
                    t1 = axis1[block]
                    t2 = axis2[block]
                    rr0 = r0[block]
                    rr1 = r1[block]
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual3_contact = wp.vec3f(wp.dot(n, rel), wp.dot(t1, rel), wp.dot(t2, rel)) + _vec3_from4(
                        bias[block]
                    )
                    contact_lambda = _make_contact_projection(
                        _diag3_from44(k_inv[block]),
                        residual3_contact,
                        _vec3_from4(lam_old),
                        _vec3_from4(mass_coeff[block]),
                        _vec3_from4(impulse_coeff[block]),
                        _vec3_from4(lambda_min[block]),
                        _vec3_from4(lambda_max[block]),
                        friction_static[block],
                        friction_kinetic[block],
                    )
                    delta3 = contact_lambda - _vec3_from4(lam_old)
                    impulse = delta3[0] * n + delta3[1] * t1 + delta3[2] * t2
                    va_out = va - inv_m_a[block] * impulse
                    wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse)
                    vb_out = vb + inv_m_b[block] * impulse
                    wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse)
                    lam_new = wp.vec4f(contact_lambda[0], contact_lambda[1], contact_lambda[2], lam_old[3])
                elif kind == _OP_TANGENT4:
                    t1 = axis0[block]
                    t2 = axis1[block]
                    rr0 = r0[block]
                    rr1 = r1[block]
                    rr2 = r2[block]
                    rr3 = r3[block]
                    jv1 = -va + wp.cross(rr0, wa) + vb - wp.cross(rr1, wb)
                    jv2 = -va + wp.cross(rr2, wa) + vb - wp.cross(rr3, wb)
                    residual4 = (
                        wp.vec4f(wp.dot(t1, jv1), wp.dot(t2, jv1), wp.dot(t1, jv2), wp.dot(t2, jv2)) + bias[block]
                    )
                    lam_new = _solve_bounded4(
                        k_inv[block],
                        residual4,
                        lam_old,
                        mass_coeff[block],
                        impulse_coeff[block],
                        lambda_min[block],
                        lambda_max[block],
                    )
                    d4 = lam_new - lam_old
                    imp1 = d4[0] * t1 + d4[1] * t2
                    imp2 = d4[2] * t1 + d4[3] * t2
                    total = imp1 + imp2
                    va_out = va - inv_m_a[block] * total
                    wa_out = wa - inv_i_a[block] @ (wp.cross(rr0, imp1) + wp.cross(rr2, imp2))
                    vb_out = vb + inv_m_b[block] * total
                    wb_out = wb + inv_i_b[block] @ (wp.cross(rr1, imp1) + wp.cross(rr3, imp2))
                elif kind == _OP_SCALAR_LINEAR:
                    n = axis0[block]
                    rr0 = r0[block]
                    rr1 = r1[block]
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual_scalar = wp.dot(n, rel) + bias[block][0]
                    l0 = _solve_scalar_bounded(
                        k_inv[block][0, 0],
                        residual_scalar,
                        lam_old[0],
                        mass_coeff[block][0],
                        impulse_coeff[block][0],
                        lambda_min[block][0],
                        lambda_max[block][0],
                    )
                    d = l0 - lam_old[0]
                    impulse = d * n
                    va_out = va - inv_m_a[block] * impulse
                    wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse)
                    vb_out = vb + inv_m_b[block] * impulse
                    wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse)
                    lam_new = wp.vec4f(l0, lam_old[1], lam_old[2], lam_old[3])
                elif kind == _OP_SCALAR_ANGULAR:
                    n = axis0[block]
                    residual_scalar_ang = wp.dot(n, wb - wa) + bias[block][0]
                    l0 = _solve_scalar_bounded(
                        k_inv[block][0, 0],
                        residual_scalar_ang,
                        lam_old[0],
                        mass_coeff[block][0],
                        impulse_coeff[block][0],
                        lambda_min[block][0],
                        lambda_max[block][0],
                    )
                    d = l0 - lam_old[0]
                    impulse = d * n
                    wa_out = wa - inv_i_a[block] @ impulse
                    wb_out = wb + inv_i_b[block] @ impulse
                    lam_new = wp.vec4f(l0, lam_old[1], lam_old[2], lam_old[3])
                else:
                    a0 = axis0[block]
                    a1 = axis1[block]
                    a2 = axis2[block]
                    if kind == _OP_ANGULAR3:
                        residual3 = wp.vec3f(wp.dot(a0, wb - wa), wp.dot(a1, wb - wa), wp.dot(a2, wb - wa))
                        residual3 = residual3 + _vec3_from4(bias[block])
                        lambda3 = _solve_bounded3(
                            _mat33_from44(k_inv[block]),
                            residual3,
                            _vec3_from4(lam_old),
                            _vec3_from4(mass_coeff[block]),
                            _vec3_from4(impulse_coeff[block]),
                            _vec3_from4(lambda_min[block]),
                            _vec3_from4(lambda_max[block]),
                        )
                        d3 = lambda3 - _vec3_from4(lam_old)
                        impulse = d3[0] * a0 + d3[1] * a1 + d3[2] * a2
                        wa_out = wa - inv_i_a[block] @ impulse
                        wb_out = wb + inv_i_b[block] @ impulse
                        lam_new = wp.vec4f(lambda3[0], lambda3[1], lambda3[2], lam_old[3])
                    else:
                        rr0 = r0[block]
                        rr1 = r1[block]
                        rel = -va + wp.cross(rr0, wa) + vb - wp.cross(rr1, wb)
                        residual3 = rel + _vec3_from4(bias[block])
                        lambda3 = _solve_bounded3(
                            _mat33_from44(k_inv[block]),
                            residual3,
                            _vec3_from4(lam_old),
                            _vec3_from4(mass_coeff[block]),
                            _vec3_from4(impulse_coeff[block]),
                            _vec3_from4(lambda_min[block]),
                            _vec3_from4(lambda_max[block]),
                        )
                        d3 = lambda3 - _vec3_from4(lam_old)
                        impulse = d3[0] * a0 + d3[1] * a1 + d3[2] * a2
                        va_out = va - inv_m_a[block] * impulse
                        wa_out = wa - inv_i_a[block] @ wp.cross(rr0, impulse)
                        vb_out = vb + inv_m_b[block] * impulse
                        wb_out = wb + inv_i_b[block] @ wp.cross(rr1, impulse)
                        lam_new = wp.vec4f(lambda3[0], lambda3[1], lambda3[2], lam_old[3])

                out_va[block] = va_out
                out_wa[block] = wa_out
                out_vb[block] = vb_out
                out_wb[block] = wb_out
                out_lambda[block] = lam_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


def _alloc_vec(count: int, device: wp.context.Devicelike):
    return wp.empty(count, dtype=wp.vec3f, device=device)


def _alloc_vec4(count: int, device: wp.context.Devicelike):
    return wp.empty(count, dtype=wp.vec4f, device=device)


def _alloc_mat33(count: int, device: wp.context.Devicelike):
    return wp.empty(count, dtype=wp.mat33f, device=device)


def _alloc_mat44(count: int, device: wp.context.Devicelike):
    return wp.empty(count, dtype=wp.mat44f, device=device)


def _parse_scenes(value: str) -> tuple[str, ...]:
    scenes = tuple(raw.strip() for raw in value.split(",") if raw.strip())
    choices = (*tuple(_SCENE_PRESETS), _MIXED_SCENE)
    for scene in scenes:
        if scene not in _SCENE_PRESETS and scene != _MIXED_SCENE:
            raise ValueError(f"unknown scene {scene!r}; choices={choices}")
    return scenes


def _parse_stride_value(value: str) -> int | str:
    item = value.strip().lower()
    return "auto" if item == "auto" else int(item)


def _slot_counts(preset: ScenePreset, period: int) -> tuple[int, int, int, int]:
    contact = int(round(max(0.0, min(1.0, preset.contact_ratio)) * float(period)))
    tangent4 = int(round(max(0.0, min(1.0, preset.tangent4_ratio)) * float(period)))
    angular3 = int(round(max(0.0, min(1.0, preset.angular3_ratio)) * float(period)))
    scalar = int(round(max(0.0, min(1.0, preset.scalar_ratio)) * float(period)))
    total = contact + tangent4 + angular3 + scalar
    if total > period:
        scalar = max(0, scalar - (total - period))
    return contact, tangent4, angular3, scalar


def _kind_from_lane(
    lane: int,
    contact_slots: int,
    tangent4_slots: int,
    angular3_slots: int,
    scalar_slots: int,
) -> int:
    if lane < contact_slots:
        return _OP_CONTACT3_HOST
    if lane < contact_slots + tangent4_slots:
        return _OP_TANGENT4_HOST
    if lane < contact_slots + tangent4_slots + angular3_slots:
        return _OP_ANGULAR3_HOST
    if lane < contact_slots + tangent4_slots + angular3_slots + scalar_slots:
        return _OP_SCALAR_LINEAR_HOST if (lane & 1) == 0 else _OP_SCALAR_ANGULAR_HOST
    return _OP_POINT3_HOST


def _seed_op_kinds_for_schedule(
    schedule: ScheduleHost,
    period: int,
    contact_slots: int,
    tangent4_slots: int,
    angular3_slots: int,
    scalar_slots: int,
) -> np.ndarray:
    op_kind = np.empty(schedule.blocks, dtype=np.int32)
    for block_id in range(schedule.blocks):
        op_kind[block_id] = _kind_from_lane(
            block_id % period, contact_slots, tangent4_slots, angular3_slots, scalar_slots
        )
    return op_kind


def _color_policy_from_contact_slots(contact_slots: int, period: int, threads_per_world: int) -> int:
    if threads_per_world <= 8:
        return _COLOR_POLICY_SPLIT_HOST
    return _COLOR_POLICY_SPLIT_HOST if contact_slots * 2 >= period else _COLOR_POLICY_SIDECAR4_HOST


def _uniform_color_policy(
    schedule: ScheduleHost, contact_slots: int, period: int, threads_per_world: int
) -> np.ndarray:
    policy = _color_policy_from_contact_slots(contact_slots, period, threads_per_world)
    return np.full(schedule.colors, policy, dtype=np.int32)


def _parse_step_layout(raw: str, *, scene: str, worlds: int) -> str:
    normalized = raw.strip().lower().replace("-", "_")
    if not normalized:
        return "single_world" if scene == "tower" and worlds == 1 else "multi_world"
    if normalized in ("single", "single_world"):
        return "single_world"
    if normalized in ("multi", "multi_world"):
        return "multi_world"
    raise ValueError(f"unknown real-scene layout {raw!r}; use single or multi")


def _parse_real_scene_spec(value: str) -> RealSceneSpec:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("real scene spec must not be empty")
    parts = normalized.split(":")
    if len(parts) > 3:
        raise ValueError(f"real scene spec {value!r} has too many ':' fields")
    scene_key = parts[0]
    if scene_key not in _REAL_SCENE_ALIASES:
        choices = ", ".join(sorted(_REAL_SCENE_ALIASES))
        raise ValueError(f"unknown real scene {scene_key!r}; choices={choices}")
    scene = _REAL_SCENE_ALIASES[scene_key]
    default_worlds = 1 if scene == "tower" else 64
    worlds = int(parts[1]) if len(parts) >= 2 and parts[1] else default_worlds
    if worlds <= 0:
        raise ValueError(f"real scene worlds must be positive, got {worlds}")
    step_layout = _parse_step_layout(parts[2] if len(parts) == 3 else "", scene=scene, worlds=worlds)
    layout_label = "single" if step_layout == "single_world" else "multi"
    return RealSceneSpec(f"real_{scene}_{worlds}_{layout_label}", scene, worlds, step_layout)


def _parse_real_scenes(value: str) -> tuple[RealSceneSpec, ...]:
    return tuple(_parse_real_scene_spec(raw.strip()) for raw in value.split(",") if raw.strip())


def _ops_for_joint_mode(mode: int) -> tuple[int, ...]:
    if mode == _JOINT_MODE_REVOLUTE_HOST:
        return (_OP_POINT3_HOST, _OP_ANGULAR3_HOST, _OP_SCALAR_ANGULAR_HOST)
    if mode == _JOINT_MODE_PRISMATIC_HOST:
        return (_OP_TANGENT4_HOST, _OP_SCALAR_LINEAR_HOST)
    if mode == _JOINT_MODE_BALL_SOCKET_HOST:
        return (_OP_POINT3_HOST,)
    if mode == _JOINT_MODE_FIXED_HOST:
        return (_OP_POINT3_HOST, _OP_ANGULAR3_HOST, _OP_SCALAR_LINEAR_HOST)
    if mode == _JOINT_MODE_CABLE_HOST:
        return (_OP_POINT3_HOST, _OP_ANGULAR3_HOST, _OP_SCALAR_ANGULAR_HOST)
    if mode == _JOINT_MODE_UNIVERSAL_HOST:
        return (_OP_POINT3_HOST, _OP_SCALAR_ANGULAR_HOST)
    if mode == _JOINT_MODE_CYLINDRICAL_HOST:
        return (_OP_TANGENT4_HOST, _OP_SCALAR_LINEAR_HOST)
    if mode == _JOINT_MODE_PLANAR_HOST:
        return (_OP_SCALAR_LINEAR_HOST, _OP_ANGULAR3_HOST)
    return (_OP_POINT3_HOST,)


def _op_kind_rank(kind: int) -> int:
    if kind == _OP_CONTACT3_HOST:
        return 0
    if kind == _OP_TANGENT4_HOST:
        return 1
    if kind == _OP_ANGULAR3_HOST:
        return 2
    if kind == _OP_SCALAR_LINEAR_HOST:
        return 3
    if kind == _OP_SCALAR_ANGULAR_HOST:
        return 4
    return 5


def _color_policy_from_ops(ops: list[int], threads_per_world: int) -> int:
    if not ops or threads_per_world <= 8:
        return _COLOR_POLICY_SPLIT_HOST
    contacts = sum(1 for op in ops if op == _OP_CONTACT3_HOST)
    if contacts == 0:
        return _COLOR_POLICY_SPLIT_HOST
    return _COLOR_POLICY_SPLIT_HOST if contacts * 2 >= len(ops) else _COLOR_POLICY_SIDECAR4_HOST


def _kind_counts_text(op_kind: np.ndarray) -> str:
    labels = {
        _OP_CONTACT3_HOST: "contact3",
        _OP_POINT3_HOST: "point3",
        _OP_ANGULAR3_HOST: "angular3",
        _OP_TANGENT4_HOST: "tangent4",
        _OP_SCALAR_LINEAR_HOST: "scalar_lin",
        _OP_SCALAR_ANGULAR_HOST: "scalar_ang",
    }
    values, counts = np.unique(op_kind, return_counts=True)
    return ",".join(f"{labels.get(int(v), str(int(v)))}:{int(c)}" for v, c in zip(values, counts, strict=True))


def _real_world_color_ranges(world) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    if world.step_layout == "single_world":
        eids = world._partitioner.element_ids_by_color.numpy()
        starts = world._partitioner.color_starts.numpy()
        num_colors = int(world._partitioner.num_colors.numpy()[0])
        return eids, [[(int(starts[color]), int(starts[color + 1])) for color in range(num_colors)]]

    eids = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy()
    ranges_by_world: list[list[tuple[int, int]]] = []
    for world_id in range(world.num_worlds):
        base = int(csr[world_id])
        ranges: list[tuple[int, int]] = []
        for color in range(int(num_colors_per_world[world_id])):
            ranges.append((base + int(starts[world_id, color]), base + int(starts[world_id, color + 1])))
        ranges_by_world.append(ranges)
    return eids, ranges_by_world


def _build_real_scene(spec: RealSceneSpec, args: argparse.Namespace):
    if spec.scene == "h1":
        return h1_flat.build(
            spec.worlds,
            "phoenx",
            args.real_substeps,
            args.real_solver_iterations,
            step_layout=spec.step_layout,
            prepare_refresh_stride=args.real_prepare_refresh_stride,
        )
    if spec.scene == "g1":
        return g1_flat.build(
            spec.worlds,
            "phoenx",
            args.real_substeps,
            args.real_solver_iterations,
            step_layout=spec.step_layout,
            prepare_refresh_stride=args.real_prepare_refresh_stride,
        )
    if spec.scene == "dr_legs":
        return dr_legs.build(
            spec.worlds,
            "phoenx",
            args.real_substeps,
            args.real_solver_iterations,
            step_layout=spec.step_layout,
            prepare_refresh_stride=args.real_prepare_refresh_stride,
        )
    if spec.scene == "tower":
        return tower.build(
            num_worlds=spec.worlds,
            solver_name="phoenx",
            substeps=args.real_substeps,
            solver_iterations=args.real_solver_iterations,
            step_layout=spec.step_layout,
            prepare_refresh_stride=args.real_prepare_refresh_stride,
        )
    raise ValueError(f"unknown real scene {spec.scene!r}")


def _build_real_schedule_and_metadata(
    spec: RealSceneSpec, args: argparse.Namespace
) -> tuple[ScheduleHost, np.ndarray, np.ndarray, str]:
    handle = _build_real_scene(spec, args)
    for _ in range(args.real_prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()
    world = _extract_solver(handle).world
    active = int(world._num_active_constraints.numpy()[0])
    family = world._element_family.numpy()
    eids, ranges_by_world = _real_world_color_ranges(world)
    constraint_words = world.constraints.data.numpy()
    joint_mode_words = np.ascontiguousarray(
        constraint_words[_JOINT_MODE_OFFSET_HOST, : max(1, int(world.num_joints))],
        dtype=np.float32,
    ).view(np.int32)

    block_ids: list[int] = []
    op_kind: list[int] = []
    color_starts: list[int] = [0]
    world_color_starts: list[int] = [0]
    color_policy: list[int] = []
    unsupported = 0
    block = 0
    for ranges in ranges_by_world:
        for start, end in ranges:
            color_ops: list[int] = []
            for cursor in range(start, end):
                eid = int(eids[cursor])
                if eid < 0 or eid >= active:
                    continue
                fam = int(family[eid])
                if fam == 1:
                    ops = (_OP_CONTACT3_HOST,)
                elif fam == 0 and eid < int(world.num_joints):
                    ops = _ops_for_joint_mode(int(joint_mode_words[eid]))
                else:
                    unsupported += 1
                    continue
                for op in ops:
                    block_ids.append(block)
                    op_kind.append(op)
                    color_ops.append(op)
                    block += 1
            color_starts.append(len(block_ids))
            color_policy.append(_color_policy_from_ops(color_ops, int(args.threads_per_world)))
        world_color_starts.append(len(color_starts) - 1)

    if not block_ids:
        block_ids = [0]
        op_kind = [_OP_POINT3_HOST]
        color_starts = [0, 1]
        world_color_starts = [0, 1]
        color_policy = [_COLOR_POLICY_SIDECAR4_HOST]
        block = 1

    schedule = ScheduleHost(
        block_ids=np.asarray(block_ids, dtype=np.int32),
        color_starts=np.asarray(color_starts, dtype=np.int32),
        world_color_starts=np.asarray(world_color_starts, dtype=np.int32),
        blocks=block,
        colors=len(color_starts) - 1,
    )
    op_kind_host = np.asarray(op_kind, dtype=np.int32)
    label = f"real_ops={{{_kind_counts_text(op_kind_host)}}},unsupported={unsupported}"
    return schedule, op_kind_host, np.asarray(color_policy, dtype=np.int32), label


def _build_mixed_schedule_and_metadata(
    worlds: int, period: int, threads_per_world: int
) -> tuple[ScheduleHost, np.ndarray, np.ndarray, str]:
    h1 = _SCENE_PRESETS["h1"]
    tower = _SCENE_PRESETS["tower"]
    h1_slots = _slot_counts(h1, period)
    tower_slots = _slot_counts(tower, period)
    block_ids: list[int] = []
    color_starts: list[int] = [0]
    world_color_starts: list[int] = [0]
    color_slots: list[tuple[int, int, int, int]] = []
    color_policy: list[int] = []
    block = 0
    for _world in range(worlds):
        for count in h1.blocks_per_color:
            for _ in range(count):
                block_ids.append(block)
                block += 1
            color_starts.append(len(block_ids))
            color_slots.append(h1_slots)
            color_policy.append(_color_policy_from_contact_slots(h1_slots[0], period, threads_per_world))
        for count in tower.blocks_per_color:
            for _ in range(count):
                block_ids.append(block)
                block += 1
            color_starts.append(len(block_ids))
            color_slots.append(tower_slots)
            color_policy.append(_color_policy_from_contact_slots(tower_slots[0], period, threads_per_world))
        world_color_starts.append(len(color_starts) - 1)

    schedule = ScheduleHost(
        block_ids=np.asarray(block_ids, dtype=np.int32),
        color_starts=np.asarray(color_starts, dtype=np.int32),
        world_color_starts=np.asarray(world_color_starts, dtype=np.int32),
        blocks=block,
        colors=len(color_starts) - 1,
    )
    op_kind = np.empty(schedule.blocks, dtype=np.int32)
    for color, slots in enumerate(color_slots):
        start = int(schedule.color_starts[color])
        end = int(schedule.color_starts[color + 1])
        contact_slots, tangent4_slots, angular3_slots, scalar_slots = slots
        for local, block_id in enumerate(schedule.block_ids[start:end]):
            op_kind[int(block_id)] = _kind_from_lane(
                local % period, contact_slots, tangent4_slots, angular3_slots, scalar_slots
            )
    label = f"mixed[h1={h1_slots},tower={tower_slots}]"
    return schedule, op_kind, np.asarray(color_policy, dtype=np.int32), label


def _build_shape_grouped_schedule(schedule: ScheduleHost, op_kind_host: np.ndarray) -> ScheduleHost:
    block_ids: list[int] = []
    for color in range(schedule.colors):
        start = int(schedule.color_starts[color])
        end = int(schedule.color_starts[color + 1])
        color_blocks = schedule.block_ids[start:end].tolist()
        color_blocks.sort(key=lambda block_id: (_op_kind_rank(int(op_kind_host[int(block_id)])), int(block_id)))
        block_ids.extend(int(block_id) for block_id in color_blocks)
    return ScheduleHost(
        block_ids=np.asarray(block_ids, dtype=np.int32),
        color_starts=schedule.color_starts.copy(),
        world_color_starts=schedule.world_color_starts.copy(),
        blocks=schedule.blocks,
        colors=schedule.colors,
    )


def _run_scene(
    args: argparse.Namespace,
    scene: str,
    device: wp.context.Devicelike,
    *,
    real_spec: RealSceneSpec | None = None,
) -> None:
    period = int(args.period)
    if real_spec is not None:
        schedule, op_kind_seed_host, color_policy_host, slot_label = _build_real_schedule_and_metadata(real_spec, args)
        contact_slots, tangent4_slots, angular3_slots, scalar_slots = (0, 0, 0, 0)
        scene_label = real_spec.label
    elif scene == _MIXED_SCENE:
        schedule, op_kind_seed_host, color_policy_host, slot_label = _build_mixed_schedule_and_metadata(
            int(args.worlds), period, int(args.threads_per_world)
        )
        contact_slots, tangent4_slots, angular3_slots, scalar_slots = (0, 0, 0, 0)
        scene_label = scene
    else:
        preset = _SCENE_PRESETS[scene]
        schedule = _build_schedule(preset, int(args.worlds))
        contact_slots, tangent4_slots, angular3_slots, scalar_slots = _slot_counts(preset, period)
        op_kind_seed_host = _seed_op_kinds_for_schedule(
            schedule, period, contact_slots, tangent4_slots, angular3_slots, scalar_slots
        )
        color_policy_host = _uniform_color_policy(schedule, contact_slots, period, int(args.threads_per_world))
        slot_label = f"slots=(c{contact_slots},t4{tangent4_slots},a3{angular3_slots},s{scalar_slots})"
        scene_label = scene

    blocks = schedule.blocks
    num_worlds = int(schedule.world_color_starts.shape[0] - 1)
    grouped_schedule = _build_shape_grouped_schedule(schedule, op_kind_seed_host)

    block_ids = wp.array(schedule.block_ids, dtype=wp.int32, device=device)
    grouped_block_ids = wp.array(grouped_schedule.block_ids, dtype=wp.int32, device=device)
    color_starts = wp.array(schedule.color_starts, dtype=wp.int32, device=device)
    world_color_starts = wp.array(schedule.world_color_starts, dtype=wp.int32, device=device)
    op_kind_seed = wp.array(op_kind_seed_host, dtype=wp.int32, device=device)
    color_policy = wp.array(color_policy_host, dtype=wp.int32, device=device)

    op_kind = wp.empty(blocks, dtype=wp.int32, device=device)
    v_a = _alloc_vec(blocks, device)
    w_a = _alloc_vec(blocks, device)
    v_b = _alloc_vec(blocks, device)
    w_b = _alloc_vec(blocks, device)
    inv_m_a = wp.empty(blocks, dtype=wp.float32, device=device)
    inv_m_b = wp.empty(blocks, dtype=wp.float32, device=device)
    inv_i_a = _alloc_mat33(blocks, device)
    inv_i_b = _alloc_mat33(blocks, device)
    axis0 = _alloc_vec(blocks, device)
    axis1 = _alloc_vec(blocks, device)
    axis2 = _alloc_vec(blocks, device)
    axis3 = _alloc_vec(blocks, device)
    r0 = _alloc_vec(blocks, device)
    r1 = _alloc_vec(blocks, device)
    r2 = _alloc_vec(blocks, device)
    r3 = _alloc_vec(blocks, device)
    jla0 = _alloc_vec(blocks, device)
    jla1 = _alloc_vec(blocks, device)
    jla2 = _alloc_vec(blocks, device)
    jla3 = _alloc_vec(blocks, device)
    jaa0 = _alloc_vec(blocks, device)
    jaa1 = _alloc_vec(blocks, device)
    jaa2 = _alloc_vec(blocks, device)
    jaa3 = _alloc_vec(blocks, device)
    jlb0 = _alloc_vec(blocks, device)
    jlb1 = _alloc_vec(blocks, device)
    jlb2 = _alloc_vec(blocks, device)
    jlb3 = _alloc_vec(blocks, device)
    jab0 = _alloc_vec(blocks, device)
    jab1 = _alloc_vec(blocks, device)
    jab2 = _alloc_vec(blocks, device)
    jab3 = _alloc_vec(blocks, device)
    k_inv = _alloc_mat44(blocks, device)
    bias = _alloc_vec4(blocks, device)
    lambda_old = _alloc_vec4(blocks, device)
    mass_coeff = _alloc_vec4(blocks, device)
    impulse_coeff = _alloc_vec4(blocks, device)
    lambda_min = _alloc_vec4(blocks, device)
    lambda_max = _alloc_vec4(blocks, device)
    friction_static = wp.empty(blocks, dtype=wp.float32, device=device)
    friction_kinetic = wp.empty(blocks, dtype=wp.float32, device=device)

    split_va = _alloc_vec(blocks, device)
    split_wa = _alloc_vec(blocks, device)
    split_vb = _alloc_vec(blocks, device)
    split_wb = _alloc_vec(blocks, device)
    split_lambda = _alloc_vec4(blocks, device)
    side_va = _alloc_vec(blocks, device)
    side_wa = _alloc_vec(blocks, device)
    side_vb = _alloc_vec(blocks, device)
    side_wb = _alloc_vec(blocks, device)
    side_lambda = _alloc_vec4(blocks, device)
    grouped_va = _alloc_vec(blocks, device)
    grouped_wa = _alloc_vec(blocks, device)
    grouped_vb = _alloc_vec(blocks, device)
    grouped_wb = _alloc_vec(blocks, device)
    grouped_lambda = _alloc_vec4(blocks, device)
    hybrid_va = _alloc_vec(blocks, device)
    hybrid_wa = _alloc_vec(blocks, device)
    hybrid_vb = _alloc_vec(blocks, device)
    hybrid_wb = _alloc_vec(blocks, device)
    hybrid_lambda = _alloc_vec4(blocks, device)

    wp.launch(
        _init_blocks_kernel,
        dim=blocks,
        inputs=[
            op_kind,
            op_kind_seed,
            v_a,
            w_a,
            v_b,
            w_b,
            inv_m_a,
            inv_m_b,
            inv_i_a,
            inv_i_b,
            axis0,
            axis1,
            axis2,
            axis3,
            r0,
            r1,
            r2,
            r3,
            jla0,
            jla1,
            jla2,
            jla3,
            jaa0,
            jaa1,
            jaa2,
            jaa3,
            jlb0,
            jlb1,
            jlb2,
            jlb3,
            jab0,
            jab1,
            jab2,
            jab3,
            k_inv,
            bias,
            lambda_old,
            mass_coeff,
            impulse_coeff,
            lambda_min,
            lambda_max,
            friction_static,
            friction_kinetic,
            int(args.period),
            contact_slots,
            tangent4_slots,
            angular3_slots,
            scalar_slots,
        ],
        device=device,
    )

    split_inputs = [
        block_ids,
        color_starts,
        world_color_starts,
        op_kind,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        r2,
        r3,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        friction_static,
        friction_kinetic,
        split_va,
        split_wa,
        split_vb,
        split_wb,
        split_lambda,
        num_worlds,
        int(args.iterations),
        int(args.threads_per_world),
    ]
    side_inputs = [
        block_ids,
        color_starts,
        world_color_starts,
        op_kind,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        jla0,
        jla1,
        jla2,
        jla3,
        jaa0,
        jaa1,
        jaa2,
        jaa3,
        jlb0,
        jlb1,
        jlb2,
        jlb3,
        jab0,
        jab1,
        jab2,
        jab3,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        friction_static,
        friction_kinetic,
        side_va,
        side_wa,
        side_vb,
        side_wb,
        side_lambda,
        num_worlds,
        int(args.iterations),
        int(args.threads_per_world),
    ]
    grouped_inputs = [
        grouped_block_ids,
        color_starts,
        world_color_starts,
        op_kind,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        r2,
        r3,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        friction_static,
        friction_kinetic,
        grouped_va,
        grouped_wa,
        grouped_vb,
        grouped_wb,
        grouped_lambda,
        num_worlds,
        int(args.iterations),
        int(args.threads_per_world),
    ]
    hybrid_inputs = [
        block_ids,
        color_starts,
        world_color_starts,
        color_policy,
        op_kind,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        r2,
        r3,
        jla0,
        jla1,
        jla2,
        jla3,
        jaa0,
        jaa1,
        jaa2,
        jaa3,
        jlb0,
        jlb1,
        jlb2,
        jlb3,
        jab0,
        jab1,
        jab2,
        jab3,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        friction_static,
        friction_kinetic,
        hybrid_va,
        hybrid_wa,
        hybrid_vb,
        hybrid_wb,
        hybrid_lambda,
        num_worlds,
        int(args.iterations),
        int(args.threads_per_world),
    ]

    def split_run() -> None:
        wp.launch(
            _solve_split_world_loop_kernel,
            dim=max(1, num_worlds * int(args.threads_per_world)),
            inputs=split_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def side_run() -> None:
        wp.launch(
            _solve_sidecar4_world_loop_kernel,
            dim=max(1, num_worlds * int(args.threads_per_world)),
            inputs=side_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def grouped_run() -> None:
        wp.launch(
            _solve_split_world_loop_kernel,
            dim=max(1, num_worlds * int(args.threads_per_world)),
            inputs=grouped_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def hybrid_run() -> None:
        wp.launch(
            _solve_hybrid_world_loop_kernel,
            dim=max(1, num_worlds * int(args.threads_per_world)),
            inputs=hybrid_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    side_colors = int(np.count_nonzero(color_policy_host == _COLOR_POLICY_SIDECAR4_HOST))
    split_colors = int(color_policy_host.size - side_colors)

    split_run()
    side_run()
    grouped_run()
    hybrid_run()
    side_err = _max_err(
        (side_va, split_va),
        (side_wa, split_wa),
        (side_vb, split_vb),
        (side_wb, split_wb),
        (side_lambda, split_lambda),
    )
    grouped_err = _max_err(
        (grouped_va, split_va),
        (grouped_wa, split_wa),
        (grouped_vb, split_vb),
        (grouped_wb, split_wb),
        (grouped_lambda, split_lambda),
    )
    hybrid_err = _max_err(
        (hybrid_va, split_va),
        (hybrid_wa, split_wa),
        (hybrid_vb, split_vb),
        (hybrid_wb, split_wb),
        (hybrid_lambda, split_lambda),
    )
    split_ms, _ = _bench(split_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    side_ms, _ = _bench(side_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    grouped_ms, _ = _bench(grouped_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    hybrid_ms, _ = _bench(hybrid_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    auto_label, auto_ms = min(
        (
            ("split", split_ms),
            ("grouped", grouped_ms),
            ("sidecar4", side_ms),
            ("hybrid", hybrid_ms),
        ),
        key=lambda item: item[1],
    )
    side_speedup = split_ms / side_ms if side_ms > 0.0 else float("nan")
    grouped_speedup = split_ms / grouped_ms if grouped_ms > 0.0 else float("nan")
    hybrid_speedup = split_ms / hybrid_ms if hybrid_ms > 0.0 else float("nan")
    auto_speedup = split_ms / auto_ms if auto_ms > 0.0 else float("nan")
    print(
        f"{scene_label:18s} worlds={num_worlds:5d} blocks={blocks:7d} colors={schedule.colors:5d} "
        f"policy=(side{side_colors},split{split_colors}) {slot_label} "
        f"split={split_ms:8.4f}ms grouped={grouped_ms:8.4f}ms sidecar4={side_ms:8.4f}ms "
        f"hybrid={hybrid_ms:8.4f}ms auto_best={auto_label}:{auto_ms:8.4f}ms "
        f"grouped_speedup={grouped_speedup:6.3f}x sidecar4_speedup={side_speedup:6.3f}x "
        f"hybrid_speedup={hybrid_speedup:6.3f}x auto_speedup={auto_speedup:6.3f}x grouped_err={grouped_err:.6g} "
        f"sidecar4_err={side_err:.6g} hybrid_err={hybrid_err:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", default="h1,g1,dr_legs,tower,mixed_h1_tower")
    parser.add_argument(
        "--real-scenes",
        default="",
        help="Comma-separated real PhoenX scene specs such as h1:64, dr_legs:64, or tower:1:single.",
    )
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--real-substeps", type=int, default=1)
    parser.add_argument("--real-solver-iterations", type=int, default=8)
    parser.add_argument("--real-prepare-refresh-stride", type=_parse_stride_value, default="auto")
    parser.add_argument("--real-prime-frames", type=int, default=1)
    parser.add_argument("--threads-per-world", type=int, default=32)
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    print(
        f"device={device} synthetic_worlds={args.worlds} real_scenes={args.real_scenes or '-'} "
        f"real_prepare={args.real_prepare_refresh_stride} iterations={args.iterations} "
        f"tpw={args.threads_per_world} n_runs={args.n_runs} trials={args.trials}"
    )
    for scene in _parse_scenes(args.scenes):
        _run_scene(args, scene, device)
    for spec in _parse_real_scenes(args.real_scenes):
        _run_scene(args, spec.label, device, real_spec=spec)


if __name__ == "__main__":
    main()
