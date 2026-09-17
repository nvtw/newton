# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load ordered contact rows cooperatively within a CUDA warp."""

import functools

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    block_solve_accumulated_inverse_bounded_1,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_eff_n,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_r0,
    cc_get_r1,
    cc_set_normal_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import (
    ContactTGS,
    NormalRow,
    get_solve_contact_tgs,
    record_impulse,
)
from newton._src.solvers.phoenx.helpers.math_helpers import apply_pair_velocity_impulse


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    unsigned int mask = (0xffffffffu >> (32 - width)) << (threadIdx.x & (32 - width));
    return __shfl_sync(mask, value, source, width);
#else
    return value;
#endif
""")
def shuffle(value: wp.float32, source: wp.int32, width: wp.int32) -> wp.float32: ...


@wp.func
def shuffle_vec(value: wp.vec3f, source: int, width: int):
    return wp.vec3f(
        shuffle(value[0], source, width), shuffle(value[1], source, width), shuffle(value[2], source, width)
    )


@functools.cache
def get_solve_rows_cooperative(record_wrenches: bool = False, *, lanes: int = 8):
    """Build a cooperative solve with optional impulse-wrench accounting."""
    if lanes not in (4, 8, 16, 32):
        raise ValueError("Contact subgroup width must be 4, 8, 16, or 32")
    solve_contact_tgs = get_solve_contact_tgs(record_wrenches)

    @wp.func
    def solve_rows_cooperative(
        state: ContactTGS,
        cc: ContactContainer,
        first: int,
        size: int,
        bodies: BodyContainer,
        a: int,
        b: int,
        v0: wp.vec3f,
        v1: wp.vec3f,
        w0: wp.vec3f,
        w1: wp.vec3f,
        m0: float,
        m1: float,
        i0: wp.mat33f,
        i1: wp.mat33f,
        mu_s: float,
        mu_d: float,
        idt: float,
        biased: bool,
        lane: int,
    ):
        for base in range(0, size, wp.static(lanes)):
            k_load = first + base + lane
            cached = NormalRow()
            prior = float(0.0)
            if base + lane < size:
                if biased:
                    cached = state.normal_rows[k_load]
                else:
                    cached.normal = cc_get_normal(cc, k_load)
                    cached.r0 = cc_get_r0(cc, k_load)
                    cached.r1 = cc_get_r1(cc, k_load)
                    cached.effective_mass = cc_get_eff_n(cc, k_load)
                    cached.bias = cc_get_bias(cc, k_load)
                prior = cc_get_normal_lambda(cc, k_load)
            for source in range(wp.min(wp.static(lanes), size - base)):
                normal = shuffle_vec(cached.normal, source, wp.static(lanes))
                r0 = shuffle_vec(cached.r0, source, wp.static(lanes))
                r1 = shuffle_vec(cached.r1, source, wp.static(lanes))
                effective_mass = shuffle(cached.effective_mass, source, wp.static(lanes))
                bias = shuffle(cached.bias, source, wp.static(lanes))
                old = shuffle(prior, source, wp.static(lanes))
                # Keep identical ordered arithmetic active across the subgroup;
                # only lane zero writes impulses and the final body state.
                if biased or bias <= 0.0:
                    if not biased:
                        bias = 0.0
                    relative = v1 + wp.cross(w1, r1) - v0 - wp.cross(w0, r0)
                    speed = wp.dot(relative, normal)
                    update = block_solve_accumulated_inverse_bounded_1(
                        effective_mass, speed + bias, old, 1.0, 0.0, 1.0, 0.0, BLOCK_LAMBDA_INF
                    )
                    if lane == 0:
                        cc_set_normal_lambda(cc, first + base + source, update.lambda_new)
                    impulse = update.delta * normal
                    if wp.static(record_wrenches) and lane == 0:
                        record_impulse(state, first + base + source, bodies.position[a] + r0, impulse)
                    v0, v1, w0, w1 = apply_pair_velocity_impulse(v0, v1, w0, w1, m0, m1, i0, i1, r0, r1, impulse)
        if lane == 0:
            v0, v1, w0, w1 = solve_contact_tgs(
                state, cc, first, size, bodies, a, b, v0, v1, w0, w1, m0, m1, i0, i1, mu_s, mu_d, idt, biased
            )
        return v0, v1, w0, w1

    return solve_rows_cooperative
